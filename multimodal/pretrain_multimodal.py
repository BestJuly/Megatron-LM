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

from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain

from multimodal.arguments import add_multimodal_args
from multimodal.forward_step import forward_step


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
):
    """Model provider for multimodal training.

    Builds the model based on --model-arch and --model-variant CLI args.
    """
    args = get_args()
    model_arch = getattr(args, "model_arch", "qwen35_vl")
    model_variant = getattr(args, "model_variant", "proxy")

    from multimodal.models import MODEL_REGISTRY

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    registry = MODEL_REGISTRY[model_arch]
    model_class = registry["model_class"]
    language_config_fn = registry["language_config_fn"]
    language_spec_fn = registry["language_spec_fn"]
    vision_config_fn = registry["vision_config_fn"]

    # Build configs
    language_config = language_config_fn(variant=model_variant)
    vision_config = vision_config_fn()

    # Build language spec (supports PP slicing)
    pp_rank = None
    vp_stage = None
    language_spec = language_spec_fn(
        config=language_config,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )

    # Build model
    model = model_class(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        max_sequence_length=getattr(args, "max_position_embeddings", 262144),
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
