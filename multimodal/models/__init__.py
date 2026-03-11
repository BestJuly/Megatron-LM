# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model registry: maps --model-arch string to (model_class, spec_fn, vision_config_fn).

The language TransformerConfig is built from CLI args in model_provider via
core_transformer_config_from_args, so language_config_fn is NOT in the registry.
Only model-architecture-specific components that cannot come from CLI args are here.
"""

from multimodal.models.qwen35_vl import (
    Qwen35VLModel,
    get_qwen35_vl_language_spec,
    get_qwen35_vl_vision_config,
)

MODEL_REGISTRY = {
    "qwen35_vl": {
        "model_class": Qwen35VLModel,
        "language_spec_fn": get_qwen35_vl_language_spec,
        "vision_config_fn": get_qwen35_vl_vision_config,
    },
}
