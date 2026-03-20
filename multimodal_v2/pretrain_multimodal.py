# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Standalone entry point for multimodal_v2 model training (FSDP + EP).

Usage::

    torchrun --nproc_per_node=8 multimodal_v2/pretrain_multimodal.py \\
        --model-arch qwen35_vl \\
        --model-variant proxy \\
        --dataset-provider mock \\
        ... (other megatron args)
"""

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.training import get_args, pretrain
from megatron.training.arguments import core_transformer_config_from_args

from multimodal_v2.arguments import add_multimodal_args
from multimodal_v2.forward_step import forward_step


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    **kwargs,
):
    """Build a multimodal model from ``--model-arch`` / ``--model-variant``.

    The language ``TransformerConfig`` is built from CLI args so that
    parallelism settings, precision, and fusion flags are inherited.
    """
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")

    from multimodal_v2.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]
    model_class = registry["model_class"]
    language_spec_fn = registry["language_spec_fn"]
    vision_config_fn = registry["vision_config_fn"]

    language_config = core_transformer_config_from_args(args)
    language_config.mrope_section = [11, 11, 10]

    vision_config = vision_config_fn(
        num_layers_override=getattr(args, "vision_num_layers", None),
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16

    language_spec = language_spec_fn(
        config=language_config,
        vp_stage=kwargs.get("vp_stage", None),
        pp_rank=None,
    )

    mtp_block_spec = None
    if getattr(args, "mtp_num_layers", None):
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=language_config,
            spec=language_spec,
            use_transformer_engine=(
                args.transformer_impl == "transformer_engine"
            ),
            vp_stage=kwargs.get("vp_stage", None),
            pp_rank=None,
        )

    model = model_class(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        image_token_id=getattr(args, "image_token_id", 248056),
        mtp_block_spec=mtp_block_spec,
        parallel_output=True,
    )

    return model


def datasets_provider(train_val_test_num_samples):
    """Dataset provider dispatcher."""
    args = get_args()
    provider = getattr(args, "dataset_provider", "mock")

    if provider == "mock":
        from multimodal_v2.data.mock import (
            train_valid_test_datasets_provider,
        )
        return train_valid_test_datasets_provider(
            train_val_test_num_samples,
        )

    raise ValueError(f"Unknown dataset provider: {provider}")


if __name__ == "__main__":
    datasets_provider.is_distributed = True

    pretrain(
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={},
        extra_args_provider=add_multimodal_args,
    )
