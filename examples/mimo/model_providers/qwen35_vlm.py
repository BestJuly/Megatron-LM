# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model provider for a Qwen3.5-397B-A17B style Vision-Language MIMO model.

Assembles a ``MimoModel`` using shared components from
``multimodal_v2.models.qwen35_vl``.  The vision encoder, configs,
and specs are the **single source of truth** in ``multimodal_v2``; this
file contains only MIMO-specific assembly logic (PP flags, arg overrides,
checkpoint loading, encoder freezing).
"""

from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo.utils.model_helpers import load_submodule_ckpt
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import (
    VisionModalitySubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from multimodal_v2.models.qwen35_vl import (
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

    MIMO-specific: overrides precision and training-time settings from
    ``get_args()`` onto the architecture config returned by
    ``get_qwen35_vl_language_config``.
    """
    try:
        from megatron.training import get_args

        args = get_args()
        if getattr(args, "bf16", False):
            cfg.bf16 = True
        if getattr(args, "fp16", False):
            cfg.fp16 = True
    except (ModuleNotFoundError, AssertionError):
        pass
    return cfg


def model_provider_qwen35_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
):
    """Build a Qwen3.5-397B-A17B Vision-Language MIMO model.

    Components (all from ``multimodal_v2.models.qwen35_vl``):
      - Megatron-native Qwen3.5 vision encoder (frozen, 4096-dim output)
      - Qwen3-Next MoE decoder (60 layers, 512 experts, hybrid GDN)
    """

    # --- Language config & spec ---
    language_config = get_qwen35_vl_language_config(
        variant="397b_a17b",
    )
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
        mimo_model
        .modality_submodules
        .images
        .encoders
        .qwen35_vision
        .parameters()
    ):
        param.requires_grad = False

    return mimo_model
