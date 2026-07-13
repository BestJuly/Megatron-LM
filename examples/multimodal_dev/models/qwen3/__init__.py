# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3 text-only MoE model components.

Used by ``MODEL_REGISTRY`` (in ``examples/multimodal_dev/models/__init__.py``)
to provide a ``--model-arch qwen3`` option. Re-exports the registry hooks
from :mod:`.factory`.
"""

from examples.multimodal_dev.models.qwen3.factory import (
    build_model,
    get_qwen3_vision_config,
    post_language_config,
    set_vision_flops_metadata,
)
from examples.multimodal_dev.models.qwen3.specs import get_qwen3_language_spec

__all__ = [
    # Factory hooks (registered in MODEL_REGISTRY)
    "build_model",
    "get_qwen3_vision_config",
    "post_language_config",
    "set_vision_flops_metadata",
    # Spec helper
    "get_qwen3_language_spec",
]
