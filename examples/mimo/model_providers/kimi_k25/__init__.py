# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Kimi K2.5 VL model components for the MIMO training path.

This package provides model provider, configuration, and vision encoder
for the Kimi K2.5 Vision-Language model in the MIMO framework.

The language backbone uses MoE with Multi-Latent Attention (MLA),
sharing architecture with DeepSeek V2/V3 and Kimi K2.

The vision encoder (MoonViT3d) and projector (PatchMergerMLP) are
dynamically loaded from the HuggingFace model repository since they
are custom modules not yet available as native MCore implementations.
"""

from .configuration import (
    KIMI_K25_IMAGE_TOKEN_ID,
    KIMI_K25_VOCAB_SIZE,
    get_kimi_k25_language_config,
)
from .vision_encoder import KimiK25VisionEncoder

__all__ = [
    "KimiK25VisionEncoder",
    "get_kimi_k25_language_config",
    "KIMI_K25_IMAGE_TOKEN_ID",
    "KIMI_K25_VOCAB_SIZE",
]
