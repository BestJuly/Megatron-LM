# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configuration for Qwen3.8-Flash-Next VL (HF ``Qwen4ExpForConditionalGeneration``).

The vision tower, the multimodal glue (``get_rope_index``, placeholder scatter) and the token
ids are the Qwen3.5-VL ones: HF ``qwen4_exp``'s ``*Vision*`` classes and rope-index helpers
are line-for-line ``qwen3_5`` after renaming, and the official config has the large ViT
(depth 27, hidden 1152, 16 heads, ffn 4304, patch 16, temporal 2, merge 2, 2304 positions,
no deepstack) with ``out_hidden_size`` equal to the text hidden size (2560). The ViT, the MDP
adapter and the collate spec of ``qwen35_vl`` are therefore reused unchanged.

The decoder is the Qwen3.8-Flash-Next ``HybridModel`` (GDN + QSA + MoE, gated residual over
4 streams, Engram/PLE, MTP), built by the default decoder hook from CLI arguments. What
differs from the decoder-only recipe is the position encoding: text RoPE is mRoPE
``[11, 11, 10]``, interleaved, partial 0.25, theta 1e7 (for text-only tokens it reduces to the
1-D RoPE of the decoder-only model).

Variants (``--model-variant``) only choose the vision depth; the decoder shape is CLI args
(``scripts/run_qwen38_vl.sh``, toolkit ``configs/dev/models/qwen38_vl_proxy.yaml``):
    ``full``   27-layer ViT, the released 48-layer decoder (EP64 scale)
    ``proxy``  4-layer ViT, the 8-layer single-node decoder proxy
"""

from typing import Optional

from examples.multimodal_dev.models.qwen35_vl.configuration import get_qwen35_vl_vision_config
from megatron.core.transformer.transformer_config import TransformerConfig

QWEN38_VL_IMAGE_TOKEN_ID: int = 248056
QWEN38_VL_VIDEO_TOKEN_ID: int = 248057
QWEN38_VL_VISION_START_TOKEN_ID: int = 248053
QWEN38_VL_VISION_END_TOKEN_ID: int = 248054
QWEN38_VL_EOS_TOKEN_ID: int = 248044
QWEN38_VL_VOCAB_SIZE: int = 248320

ROTARY_BASE: int = 10_000_000
ROTARY_PERCENT: float = 0.25
MROPE_SECTION: list = [11, 11, 10]

VISION_NUM_LAYERS = {"full": 27, "proxy": 4}


def resolve_image_token_id(args) -> int:
    """``--image-token-id`` when given, else the Qwen3.8-VL placeholder id."""
    value = getattr(args, "image_token_id", None)
    return QWEN38_VL_IMAGE_TOKEN_ID if value is None else int(value)


def get_qwen38_vl_vision_config(
    num_layers_override: Optional[int] = None, variant: Optional[str] = None
) -> TransformerConfig:
    """The Qwen3.5-VL large ViT; depth from ``num_layers_override`` or the variant."""
    if variant is not None and variant not in VISION_NUM_LAYERS:
        raise ValueError(
            f"Unknown Qwen3.8-VL variant '{variant}'. Choose from {list(VISION_NUM_LAYERS)}."
        )
    if num_layers_override is None:
        num_layers_override = VISION_NUM_LAYERS[variant or "full"]
    # variant=None selects the large-ViT dims (hidden 1152 / 16 heads / ffn 4304), the
    # Qwen3.8 tower; the merger output width is set from the decoder hidden size by the model.
    return get_qwen35_vl_vision_config(num_layers_override=num_layers_override, variant=None)
