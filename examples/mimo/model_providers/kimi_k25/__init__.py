# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Kimi K2.5 VL model provider for training.

This package provides a standalone KimiK25VLModel whose checkpoint layout
(language_model.*, vision_tower.*, mm_projector.*) is identical to
Megatron-Bridge, enabling direct checkpoint loading.
"""

from .configuration import (
    KIMI_K25_IMAGE_TOKEN_ID,
    KIMI_K25_VOCAB_SIZE,
    get_kimi_k25_language_config,
)
from .model import KimiK25VLModel

__all__ = [
    "KimiK25VLModel",
    "get_kimi_k25_language_config",
    "KIMI_K25_IMAGE_TOKEN_ID",
    "KIMI_K25_VOCAB_SIZE",
]
