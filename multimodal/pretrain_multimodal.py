# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Entry point for multimodal model training.

Usage:
    torchrun --nproc_per_node=8 multimodal/pretrain_multimodal.py \
        --model-arch qwen35_vl \
        --model-variant proxy \
        --dataset-provider mock \
        ... (other megatron args)
"""

import os
import sys

# Add repo root to path for multimodal imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training.arguments import core_transformer_config_from_args

from multimodal.arguments import add_multimodal_args
from multimodal.forward_step import forward_step


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    **kwargs,
):
    """Model provider for multimodal training.

    Builds the model based on --model-arch and --model-variant CLI args.
    The language TransformerConfig is built from CLI args so that parallelism
    settings (TP, PP, SP), precision (bf16/fp16), and fusion flags are
    correctly inherited.
    """
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")

    from multimodal.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]
    model_class = registry["model_class"]
    language_spec_fn = registry["language_spec_fn"]
    vision_config_fn = registry["vision_config_fn"]

    # Build language config from CLI args (picks up TP, PP, SP, bf16,
    # MoE, hybrid-attention, cross-entropy fusion, etc.)
    language_config = core_transformer_config_from_args(args)
    # MRoPE section is architecture-specific and not a CLI arg.
    language_config.mrope_section = [11, 11, 10]

    # Vision config: architecture-specific values, not from CLI.
    # Inherit precision from the language config.
    vision_config = vision_config_fn(
        num_layers_override=getattr(args, "vision_num_layers", None)
    )
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16

    # Build language spec (supports PP slicing)
    language_spec = language_spec_fn(
        config=language_config,
        vp_stage=kwargs.get("vp_stage", None),
        pp_rank=None,
    )

    # Determine which pipeline components to build.
    # Megatron does NOT pass add_encoder/add_decoder; compute from PP rank.
    add_encoder = parallel_state.is_pipeline_first_stage()
    add_decoder = True

    # Build model
    model = model_class(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        image_token_id=getattr(args, "image_token_id", 248056),
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        parallel_output=True,
    )

    return model


def datasets_provider(train_val_test_num_samples):
    """Dataset provider dispatcher."""
    args = get_args()
    dataset_provider = getattr(args, "dataset_provider", "mock")

    if dataset_provider == "mock":
        from multimodal.data.mock import train_valid_test_datasets_provider

        return train_valid_test_datasets_provider(train_val_test_num_samples)
    else:
        raise ValueError(f"Unknown dataset provider: {dataset_provider}")


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
