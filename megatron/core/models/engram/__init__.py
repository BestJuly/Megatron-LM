# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek Engram modules for Megatron Core."""

from .config import EngramConfig
from .engram import Engram
from .layer_specs import apply_engram_to_layer_spec

__all__ = ["Engram", "EngramConfig", "apply_engram_to_layer_spec"]
