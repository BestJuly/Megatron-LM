# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model provider for Kimi K2.5 VL Vision-Language MIMO model.

Assembles a ``MimoModel`` using components from the local
``examples/mimo/model_providers/kimi_k25/`` package.

The language backbone uses MoE with Multi-Latent Attention (MLA).
The vision encoder (MoonViT3d + PatchMergerMLP) is dynamically loaded
from the HuggingFace model repository.

Supported variants (via ``--model-variant`` CLI arg):
  ``proxy``   4 layers, 16 experts — single-node bring-up
  ``full``    61 layers, 256 experts — production (default)

NOTE: MIMO currently requires PP=1 and CP=1.
"""

from typing import Any, Dict, Optional

import torch

from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import (
    VisionModalitySubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from . import (
    KIMI_K25_IMAGE_TOKEN_ID,
    KIMI_K25_VOCAB_SIZE,
    KimiK25VisionEncoder,
    get_kimi_k25_language_config,
)
from .specs import get_kimi_k25_language_spec


# Default HF model path for Kimi K2.5 VL
_DEFAULT_HF_MODEL_PATH = "moonshotai/Kimi-K2.5"


def _apply_args_to_language_config(cfg):
    """Apply Megatron CLI args to the language MLATransformerConfig.

    Overrides precision, parallelism, and training-time settings from
    ``get_args()`` onto the architecture config returned by
    ``get_kimi_k25_language_config``.
    """
    try:
        from megatron.training import get_args

        args = get_args()
    except (ModuleNotFoundError, AssertionError):
        return cfg

    # Precision
    if getattr(args, "bf16", False):
        cfg.bf16 = True
    if getattr(args, "fp16", False):
        cfg.fp16 = True

    # Parallelism
    cfg.tensor_model_parallel_size = getattr(args, "tensor_model_parallel_size", 1)
    cfg.pipeline_model_parallel_size = getattr(args, "pipeline_model_parallel_size", 1)
    cfg.expert_model_parallel_size = getattr(args, "expert_model_parallel_size", 1)
    cfg.sequence_parallel = getattr(args, "sequence_parallel", False)
    cfg.context_parallel_size = getattr(args, "context_parallel_size", 1)

    # Distributed optimizer (needed for FSDP)
    if getattr(args, "use_distributed_optimizer", False):
        cfg.use_distributed_optimizer = True

    # Meta-device init (needed for FSDP)
    if getattr(args, "init_model_with_meta_device", False):
        cfg.init_model_with_meta_device = True

    # MoE overrides from CLI
    if getattr(args, "moe_token_dispatcher_type", None) is not None:
        cfg.moe_token_dispatcher_type = args.moe_token_dispatcher_type
    if getattr(args, "moe_grouped_gemm", None) is not None:
        cfg.moe_grouped_gemm = args.moe_grouped_gemm

    return cfg


def _get_variant() -> str:
    """Read model variant from CLI args, defaulting to ``'full'``."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), "model_variant", "full") or "full"
    except (ModuleNotFoundError, AssertionError):
        return "full"


def _get_hf_model_path() -> str:
    """Read HF model path from CLI args or environment."""
    import os

    # Check environment variable first
    env_path = os.environ.get("KIMI_K25_HF_MODEL_PATH")
    if env_path:
        return env_path

    # Check CLI args
    try:
        from megatron.training import get_args

        args = get_args()
        path = getattr(args, "hf_model_path", None)
        if path:
            return path
    except (ModuleNotFoundError, AssertionError):
        pass

    return _DEFAULT_HF_MODEL_PATH


def model_provider_kimi_k25_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = KIMI_K25_IMAGE_TOKEN_ID,
):
    """Build a Kimi K2.5 VL Vision-Language MIMO model.

    The model variant is controlled by ``--model-variant`` CLI arg:
      - ``proxy`` : 4 layers, 16 experts  (single-node testing)
      - ``full``  : 61 layers, 256 experts (production)

    Components:
      - HF-loaded MoonViT3d vision encoder (frozen)
      - Kimi K2 MoE+MLA decoder (variant-controlled)

    The vision encoder path is resolved from (in order):
      1. ``KIMI_K25_HF_MODEL_PATH`` environment variable
      2. ``--hf-model-path`` CLI arg (if wired)
      3. Default: ``moonshotai/Kimi-K2.5``
    """
    variant = _get_variant()
    hf_model_path = _get_hf_model_path()

    # --- Language config & spec ---
    language_config = get_kimi_k25_language_config(variant=variant)
    _apply_args_to_language_config(language_config)

    layer_spec = get_kimi_k25_language_spec(language_config)

    # --- Vision encoder ---
    vision_encoder = ModuleSpec(
        module=KimiK25VisionEncoder,
        params={
            "hf_model_path": hf_model_path,
            "language_hidden_size": language_config.hidden_size,
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"kimi_k25_vision": vision_encoder},
            "input_projections": [],
        },
    )

    # --- Language model spec ---
    seq_length = 4096
    try:
        from megatron.training import get_args

        seq_length = getattr(get_args(), "seq_length", 4096)
    except (ModuleNotFoundError, AssertionError):
        pass

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": layer_spec,
            "vocab_size": KIMI_K25_VOCAB_SIZE,
            "max_sequence_length": seq_length,
            "pre_process": pre_process,
            "post_process": post_process,
            "position_embedding_type": "rope",
        },
    )

    # --- Assemble MIMO model ---
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": image_special_token_id},
    )

    mimo_model = MimoModel(mimo_model_config)
    print("*" * 100)
    print_mimo_structure(mimo_model)
    print("*" * 100)

    # --- Load language model checkpoint (optional) ---
    try:
        from megatron.training import get_args

        _args = get_args()
        if _args.language_model_checkpoint is not None:
            load_submodule_ckpt(
                mimo_model.language_model,
                _args.language_model_checkpoint,
            )
            print(
                "Successfully loaded language model checkpoint "
                f"from {_args.language_model_checkpoint}"
            )
    except (ModuleNotFoundError, AssertionError):
        pass

    # --- Freeze vision encoder ---
    for param in (
        mimo_model.modality_submodules.images.encoders.kimi_k25_vision.parameters()
    ):
        param.requires_grad = False

    return mimo_model
