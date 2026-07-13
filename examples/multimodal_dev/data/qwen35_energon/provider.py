# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Energon provider for descriptor-aware packed Qwen3.5-VL data."""

from __future__ import annotations

from examples.multimodal_dev.mdp_parallel_groups import get_pp_cp_local_rank
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_END_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from megatron.core import parallel_state
from megatron.energon import (
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)
from megatron.training import get_args

from .task_encoder import Qwen35EnergonTaskEncoder


class _CyclicDataIterator:
    """Expose an Energon loader as the cyclic iterator Megatron expects."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iterator = self._cycle()

    def _cycle(self):
        while True:
            yield from self._dataloader

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)


def _worker_config(args) -> WorkerConfig:
    workers = int(getattr(args, "num_workers", 2))
    if not parallel_state.is_initialized():
        return WorkerConfig.default_worker_config(workers)
    return WorkerConfig(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
    )


def _tokenizer(args):
    tokenizer_path = getattr(args, "tokenizer_path", None) or getattr(args, "tokenizer_model", None)
    if not tokenizer_path:
        raise ValueError(
            "--tokenizer-path or --tokenizer-model must point to a "
            "Qwen3.5-VL HuggingFace tokenizer with --dataset-provider energon"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if int(image_token_id) != QWEN35_VL_IMAGE_TOKEN_ID:
        raise ValueError(
            "the configured tokenizer is not Qwen3.5-VL: "
            f"<|image_pad|>={image_token_id}, expected {QWEN35_VL_IMAGE_TOKEN_ID}"
        )
    return tokenizer


def _task_encoder(args, tokenizer) -> Qwen35EnergonTaskEncoder:
    if parallel_state.model_parallel_is_initialized():
        cp_size = int(parallel_state.get_context_parallel_world_size())
        cp_rank = int(parallel_state.get_context_parallel_rank())
        pp_size = int(parallel_state.get_pipeline_model_parallel_world_size())
    else:
        cp_size = int(getattr(args, "context_parallel_size", 1))
        cp_rank = 0
        pp_size = int(getattr(args, "pipeline_model_parallel_size", 1))

    inner_scope = str(getattr(args, "mdp_inner_dp_scope", "cp"))
    mdp_requested = bool(getattr(args, "mdp_encoder_mode", True))
    if mdp_requested and inner_scope not in ("cp", "pp_cp"):
        raise ValueError(
            "--mdp-inner-dp-scope must be either cp or pp_cp; "
            f"got {inner_scope!r}"
        )
    partition_vision = mdp_requested and (
        cp_size > 1 or (inner_scope == "pp_cp" and pp_size > 1)
    )
    if partition_vision and inner_scope == "cp" and pp_size != 1:
        raise ValueError("CP-local --mdp-encoder-mode requires PP=1")
    if mdp_requested and inner_scope == "pp_cp" and pp_size <= 1:
        raise ValueError("pp_cp --mdp-encoder-mode requires PP>1")

    prepartition_rank = int(cp_rank) if partition_vision else 0
    prepartition_world = int(cp_size) if partition_vision else 1
    if partition_vision and inner_scope == "pp_cp":
        prepartition_rank = get_pp_cp_local_rank(args, pp_size, cp_size)
        prepartition_world = int(pp_size) * int(cp_size)

    return Qwen35EnergonTaskEncoder(
        tokenizer=tokenizer,
        seq_length=int(getattr(args, "total_seq_length", args.seq_length)),
        image_token_id=int(getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID)),
        video_token_id=int(getattr(args, "video_token_id", QWEN35_VL_VIDEO_TOKEN_ID)),
        vision_start_token_id=int(
            getattr(args, "vision_start_token_id", QWEN35_VL_VISION_START_TOKEN_ID)
        ),
        vision_end_token_id=int(
            getattr(args, "vision_end_token_id", QWEN35_VL_VISION_END_TOKEN_ID)
        ),
        patch_size=int(getattr(args, "patch_size", 16)),
        temporal_patch_size=int(getattr(args, "temporal_patch_size", 2)),
        spatial_merge_size=int(getattr(args, "spatial_merge_size", 2)),
        image_min_pixels=int(getattr(args, "image_min_pixels", 0)),
        image_max_pixels=int(getattr(args, "image_max_pixels", 0)),
        cp_size=int(cp_size),
        mdp_loader_prepartition=partition_vision,
        mdp_loader_prepartition_rank=prepartition_rank,
        mdp_loader_prepartition_world=prepartition_world,
        mdp_loader_prepartition_encoder_stage=True,
        mdp_loader_prepartition_materialize=True,
        mdp_lpt_hidden_size=int(getattr(args, "vision_hidden_size", 1152)),
    )


def train_valid_test_datasets_provider(_train_val_test_num_samples):
    """Return external Energon loaders for Megatron pretraining."""
    args = get_args()
    data_path = getattr(args, "energon_path", None)
    if not data_path:
        raise ValueError("--energon-path is required with --dataset-provider energon")
    if getattr(args, "dataloader_type", "single") != "external":
        raise ValueError("--dataloader-type external is required with --dataset-provider energon")
    if int(getattr(args, "micro_batch_size", 1)) != 1:
        raise ValueError("Qwen3.5-VL Energon packing requires --micro-batch-size 1")
    if int(args.energon_packing_buffer_size) < 1:
        raise ValueError("--energon-packing-buffer-size must be positive")
    if int(args.energon_max_samples_per_sequence) < 1:
        raise ValueError("--energon-max-samples-per-sequence must be positive")
    if int(args.energon_prefetch_factor) < 1:
        raise ValueError("--energon-prefetch-factor must be positive")
    if (
        parallel_state.model_parallel_is_initialized()
        and parallel_state.get_tensor_model_parallel_rank() != 0
    ):
        return None, None, None

    args.use_packed_sequence = True
    worker_config = _worker_config(args)
    tokenizer = _tokenizer(args)
    common = {
        "batch_size": 1,
        "worker_config": worker_config,
        "packing_buffer_size": int(args.energon_packing_buffer_size),
        "max_samples_per_sequence": int(args.energon_max_samples_per_sequence),
    }
    train_dataset = get_train_dataset(
        data_path,
        task_encoder=_task_encoder(args, tokenizer),
        split_part=args.dataset_split,
        shuffle_buffer_size=int(args.energon_shuffle_buffer_size),
        **common,
    )
    train_loader = _CyclicDataIterator(
        get_savable_loader(train_dataset, prefetch_factor=int(args.energon_prefetch_factor))
    )

    validation_loader = None
    if int(getattr(args, "eval_iters", 0)) > 0:
        validation_datasets = get_val_datasets(
            data_path, task_encoder=_task_encoder(args, tokenizer), **common
        )
        if validation_datasets:
            validation_loader = iter(get_loader(validation_datasets[0][0]))
    return train_loader, validation_loader, None
