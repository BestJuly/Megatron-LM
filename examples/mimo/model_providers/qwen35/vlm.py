# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model provider for Qwen3.5-VL Vision-Language MIMO model.

Assembles a ``MimoModel`` using components from the local
``examples/mimo/model_providers/qwen35/`` package (a self-contained
duplicate of the qwen35_vl model code).  This file contains only
MIMO-specific assembly logic (PP flags, arg overrides, checkpoint
loading, encoder freezing).

Supported variants (via ``--model-variant`` CLI arg):
  ``proxy``       4 layers, 16 experts — single-node bring-up
  ``397b_a17b``   60 layers, 512 experts — production (default)
  ``9b``          Dense 32-layer 9B model
  ``35b_a3b``     MoE 35B-A3B model
  ``35b_a3b_light``  Reduced 35B-A3B for single-node testing
"""

from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import (
    VisionModalitySubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from . import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VOCAB_SIZE,
    ROTARY_BASE,
    ROTARY_PERCENT,
    Qwen35VLVisionEncoder,
    get_qwen35_vl_language_config,
    get_qwen35_vl_language_spec,
    get_qwen35_vl_vision_config,
    get_qwen35_vl_vision_spec,
)


def _apply_args_to_language_config(cfg):
    """Apply Megatron CLI args to the language TransformerConfig.

    Overrides precision, parallelism, and training-time settings from
    ``get_args()`` onto the architecture config returned by
    ``get_qwen35_vl_language_config``.  This ensures that flags such as
    ``--sequence-parallel``, ``--use-distributed-optimizer``, and
    ``--use-megatron-fsdp`` are reflected in the TransformerConfig that
    GPTModel reads during construction.
    """
    try:
        from megatron.training import get_args

        args = get_args()
        if getattr(args, "bf16", False):
            cfg.bf16 = True
        if getattr(args, "fp16", False):
            cfg.fp16 = True
        # Sequence parallelism — required for correct SP behaviour at TP > 1.
        cfg.sequence_parallel = getattr(args, "sequence_parallel", False)
        # Tensor / expert model parallel sizes — used by GPTModel for
        # MoE routing and ColumnParallelLinear initialisation checks.
        tp = getattr(args, "tensor_model_parallel_size", 1)
        ep = getattr(args, "expert_model_parallel_size", 1)
        if tp > 1:
            cfg.tensor_model_parallel_size = tp
        if ep > 1:
            cfg.expert_model_parallel_size = ep
    except (ModuleNotFoundError, AssertionError):
        pass
    return cfg


def _get_variant() -> str:
    """Read model variant from CLI args, defaulting to ``'397b_a17b'``."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), "model_variant", "397b_a17b") or "397b_a17b"
    except (ModuleNotFoundError, AssertionError):
        return "397b_a17b"


def model_provider_qwen35_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
):
    """Build a Qwen3.5-VL Vision-Language MIMO model.

    The model variant is controlled by ``--model-variant`` CLI arg:
      - ``proxy``     : 4 layers, 16 experts  (single-node testing)
      - ``397b_a17b`` : 60 layers, 512 experts (production)

    Components (all from ``multimodal_v2.models.qwen35_vl``):
      - Megatron-native Qwen3.5 vision encoder (frozen, language-dim output)
      - Qwen3-Next MoE decoder (variant-controlled depth and expert count)
    """
    variant = _get_variant()

    # --- Language config & spec ---
    language_config = get_qwen35_vl_language_config(variant=variant)
    _apply_args_to_language_config(language_config)

    layer_spec = get_qwen35_vl_language_spec(language_config)

    # --- Vision config & encoder ---
    vision_config = get_qwen35_vl_vision_config()
    vision_config.bf16 = language_config.bf16
    vision_config.fp16 = language_config.fp16
    vision_spec = get_qwen35_vl_vision_spec()

    vision_encoder = ModuleSpec(
        module=Qwen35VLVisionEncoder,
        params={
            "config": vision_config,
            "transformer_layer_spec": vision_spec,
            "spatial_merge_size": 2,
            "out_hidden_size": language_config.hidden_size,
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"qwen35_vision": vision_encoder},
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
            "vocab_size": QWEN35_VL_VOCAB_SIZE,
            "max_sequence_length": seq_length,
            "pre_process": pre_process,
            "post_process": post_process,
            "position_embedding_type": "rope",
            "rotary_percent": ROTARY_PERCENT,
            "rotary_base": ROTARY_BASE,
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
        mimo_model.modality_submodules.images.encoders.qwen35_vision.parameters()
    ):
        param.requires_grad = False

    return mimo_model
