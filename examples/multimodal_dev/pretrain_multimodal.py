# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Standalone entry point for multimodal_dev model training (FSDP + EP).

This entry point is **model-agnostic**.  All model-specific logic (layer
specs, model construction, FLOPs metadata, dataset generation) is
delegated to factory functions registered in
:data:`multimodal_dev.models.MODEL_REGISTRY`.

Adding a new architecture only requires:

1. Creating a new model package under ``multimodal_dev/models/<arch>/``
   with the appropriate factory functions.
2. Registering an entry in ``MODEL_REGISTRY``.

No changes to this file are necessary.

Usage::

    torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \\
        --model-arch qwen35_vl \\
        --dataset-provider mock \\
        ... (other megatron args)
"""

import dataclasses
import importlib
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
)

from examples.multimodal_dev.arguments import (
    add_multimodal_args,
    encoder_recompute_overrides_from_args,
    validate_encoder_recompute_args,
)
from examples.multimodal_dev.forward_step import (
    forward_step,
    quantized_row_alignment,
    seqlen_alignment_factor,
)
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.utils import start_memory_history_recording


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    **kwargs,
):
    """Build a multimodal model from ``--model-arch``.

    The language ``TransformerConfig`` is built from CLI args so that
    parallelism settings, precision, and fusion flags are inherited.
    Model-specific post-processing and construction are delegated to the
    registry factory functions.
    """
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")

    from examples.multimodal_dev.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]

    # --- language config (generic + model-specific post-processing) ---
    language_config = core_transformer_config_from_args(args)
    if getattr(args, "use_packed_sequence", False) and args.pipeline_model_parallel_size > 1:
        # THD activation length varies per microbatch, so the pipeline
        # scheduler must negotiate shapes at each send/recv instead of sizing
        # its P2P buffers from --seq-length.  Assigned after
        # TransformerConfig.__post_init__, hence the repeated dispatcher check.
        language_config.variable_seq_lengths = True
        if (
            language_config.num_moe_experts is not None
            and language_config.moe_token_dispatcher_type == "allgather"
        ):
            raise ValueError(
                "--use-packed-sequence with pipeline_model_parallel_size > 1 "
                "requires an alltoall MoE token dispatcher; the allgather "
                "dispatcher does not support the variable sequence lengths "
                "the pipeline scheduler needs to negotiate THD shapes."
            )
    post_language_config_fn = registry.get("post_language_config_fn")
    if post_language_config_fn is not None:
        post_language_config_fn(language_config, args)

    # Variable-length THD packs change the P2P tensor shape between
    # microbatches; the pipeline schedule must negotiate shapes per send.
    if getattr(args, "use_packed_sequence", False) and args.pipeline_model_parallel_size > 1:
        language_config.variable_seq_lengths = True

    # --- vision config ---
    vision_config = registry["vision_config_fn"](
        num_layers_override=getattr(args, "vision_num_layers", None),
        variant=getattr(args, "model_variant", None),
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16
    vision_config.apply_rope_fusion = language_config.apply_rope_fusion

    encoder_recompute_overrides = encoder_recompute_overrides_from_args(args)
    if encoder_recompute_overrides:
        vision_config = dataclasses.replace(
            vision_config, **encoder_recompute_overrides
        )

    # --- vision FLOPs metadata ---
    vision_flops_fn = registry.get("vision_flops_fn")
    if vision_flops_fn is not None:
        vision_flops_fn(args, language_config, vision_config)

    # --- build model (fully delegated to the arch factory) ---
    model = registry["model_factory_fn"](
        args=args,
        language_config=language_config,
        vision_config=vision_config,
        pre_process=pre_process,
        post_process=post_process,
        **kwargs,
    )

    return model


def _resolve_provider_fn(provider_fn):
    """Resolve a provider that may be a dotted import path string."""
    if isinstance(provider_fn, str):
        module_path, func_name = provider_fn.rsplit(".", 1)
        provider_fn = getattr(
            importlib.import_module(module_path), func_name,
        )
    return provider_fn


def datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Dataset provider dispatcher.

    Routes to the dataset factory registered for the current
    ``(--model-arch, --dataset-provider)`` combination. ``vp_stage`` is
    accepted for the virtual-pipeline dataset-provider contract; the
    registered datasets are stage-independent.
    """
    del vp_stage
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")
    provider = getattr(args, "dataset_provider", "mock")

    from examples.multimodal_dev.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]
    available = registry.get("dataset_providers", {})

    if provider not in available:
        raise ValueError(
            f"Unknown dataset provider '{provider}' for arch "
            f"'{model_arch}'. Available: {list(available.keys())}"
        )

    provider_fn = _resolve_provider_fn(available[provider])
    return provider_fn(train_val_test_num_samples)


def _mdp_adapter_builder(args):
    """Build the Qwen3.5-VL MDP adapter plus its vision TransformerConfig.

    Mirrors model_provider's vision-config assembly so the MDP encoder is
    built from exactly the same configuration as the native path.
    """
    from examples.multimodal_dev.mdp_adapter import build_mdp_adapter
    from examples.multimodal_dev.models import MODEL_REGISTRY

    registry = MODEL_REGISTRY[getattr(args, "model_arch", "qwen35_vl")]
    language_config = core_transformer_config_from_args(args)
    post_language_config_fn = registry.get("post_language_config_fn")
    if post_language_config_fn is not None:
        post_language_config_fn(language_config, args)
    vision_config = registry["vision_config_fn"](
        num_layers_override=getattr(args, "vision_num_layers", None),
        variant=getattr(args, "model_variant", None),
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16
    vision_config.apply_rope_fusion = language_config.apply_rope_fusion
    vision_config.params_dtype = language_config.params_dtype
    # The encoder DDP derives its gradient prescale from this flag; MDP
    # requires prescale 1 (WORLD sum, normalized once by 1/T_global).
    vision_config.calculate_per_token_loss = language_config.calculate_per_token_loss
    return build_mdp_adapter(args, language_config), vision_config


def _setup_mdp(args):
    """Validate the MDP configuration and register the adapter builder."""
    from megatron.core.mdp import integration as mdp_integration

    if not getattr(args, "use_packed_sequence", False):
        raise RuntimeError(
            "--mdp-enable requires --use-packed-sequence: the dual-THD contract "
            "packs decoder samples into [1, T]"
        )
    if not getattr(args, "use_vanilla_collate_fn", False):
        raise RuntimeError(
            "--mdp-enable requires --use-vanilla-collate-fn: pack_or_pad_batch "
            "consumes the per-sample dict list only the identity collate produces"
        )
    mdp_integration.validate_from_args(args)
    from megatron.core.mdp.checkpoint import assert_supported_checkpoint_config

    assert_supported_checkpoint_config(args)
    mdp_integration.set_adapter_builder(_mdp_adapter_builder)


if __name__ == "__main__":
    datasets_provider.is_distributed = True

    args = parse_and_validate_args(
        extra_args_provider=add_multimodal_args,
        args_defaults={},
    )
    validate_encoder_recompute_args(args)
    # Fail fast on a quantization recipe the collate path cannot align, before
    # the model and datasets are built (the result is cached on the argument values).
    quantized_row_alignment(args)
    if getattr(args, "mdp_enable", False):
        _setup_mdp(args)
    if args.pipeline_model_parallel_size > 1 and not args.use_packed_sequence:
        # The BSHD collate pads to --seq-length and rounds *up* to the CP/SP
        # alignment factor, while the pipeline scheduler sizes its static P2P
        # buffers from --seq-length with floor division.  An unaligned
        # --seq-length would make the two disagree and hang at the first
        # cross-stage send.  The packed/THD path is exempt: it sets
        # ``variable_seq_lengths`` and negotiates shapes instead.
        alignment = seqlen_alignment_factor(
            args.tensor_model_parallel_size,
            args.context_parallel_size,
            args.sequence_parallel,
        )
        if args.seq_length % alignment != 0:
            raise ValueError(
                f"--seq-length ({args.seq_length}) must be divisible by {alignment} "
                f"when pipeline_model_parallel_size > 1: the pipeline scheduler "
                f"sizes its static P2P buffers with floor division, so the padded "
                f"activation length would not match them."
            )
    full_config = pretrain_cfg_container_from_args(args)
    # training.py enables allocator history only on the config-container MODEL
    # flow; this entry uses model_provider, so it enables recording itself, and
    # must stay ahead of pretrain(), which constructs the model.
    start_memory_history_recording(getattr(full_config, "profiling", None))
    pretrain(
        full_config,
        datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider=model_provider,
    )
