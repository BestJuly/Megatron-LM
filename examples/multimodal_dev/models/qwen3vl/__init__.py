# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3-VL hybrid model package — qwen35-VL vision encoder + qwen3 LLM decoder.

The vision encoder, multimodal token vocabulary, MRoPE position scheme, and
``compute_position_ids`` logic come from ``qwen35_vl``. The LLM decoder uses
the ``qwen3`` model path. MRoPE is preserved with
``rotary_percent=0.5`` and ``mrope_section=[11, 11, 10]`` to match qwen3's
``head_dim=128`` while preserving the qwen35-VL [t,h,w] partition ratio.
"""

from examples.multimodal_dev.models.qwen3vl.factory import (
    build_model,
    get_qwen3vl_vision_config,
    post_language_config,
    set_vision_flops_metadata,
)
