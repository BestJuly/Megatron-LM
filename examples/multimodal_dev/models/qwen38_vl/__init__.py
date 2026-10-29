# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.8-Flash-Next VL: the Qwen3.5-VL ViT around the Qwen3.8 ``HybridModel`` decoder."""

from examples.multimodal_dev.models.qwen38_vl.configuration import (
    MROPE_SECTION,
    QWEN38_VL_EOS_TOKEN_ID,
    QWEN38_VL_IMAGE_TOKEN_ID,
    QWEN38_VL_VIDEO_TOKEN_ID,
    QWEN38_VL_VISION_END_TOKEN_ID,
    QWEN38_VL_VISION_START_TOKEN_ID,
    QWEN38_VL_VOCAB_SIZE,
    ROTARY_BASE,
    ROTARY_PERCENT,
    get_qwen38_vl_vision_config,
    resolve_image_token_id,
)
from examples.multimodal_dev.models.qwen38_vl.factory import (
    build_model,
    check_qwen38_vl_args,
    post_language_config,
    set_vision_flops_metadata,
)
from examples.multimodal_dev.models.qwen38_vl.model import Qwen38VLModel

__all__ = [
    "Qwen38VLModel",
    "build_model",
    "check_qwen38_vl_args",
    "post_language_config",
    "set_vision_flops_metadata",
    "get_qwen38_vl_vision_config",
    "resolve_image_token_id",
    "MROPE_SECTION",
    "ROTARY_BASE",
    "ROTARY_PERCENT",
    "QWEN38_VL_EOS_TOKEN_ID",
    "QWEN38_VL_IMAGE_TOKEN_ID",
    "QWEN38_VL_VIDEO_TOKEN_ID",
    "QWEN38_VL_VISION_END_TOKEN_ID",
    "QWEN38_VL_VISION_START_TOKEN_ID",
    "QWEN38_VL_VOCAB_SIZE",
]
