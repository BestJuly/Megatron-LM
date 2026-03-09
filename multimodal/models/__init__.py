# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model registry: maps --model-arch string to (model_class, config_fn, spec_fn)."""

from multimodal.models.qwen35_vl import (
    Qwen35VLModel,
    get_qwen35_vl_language_config,
    get_qwen35_vl_language_spec,
    get_qwen35_vl_vision_config,
)

MODEL_REGISTRY = {
    "qwen35_vl": {
        "model_class": Qwen35VLModel,
        "language_config_fn": get_qwen35_vl_language_config,
        "language_spec_fn": get_qwen35_vl_language_spec,
        "vision_config_fn": get_qwen35_vl_vision_config,
    },
}
