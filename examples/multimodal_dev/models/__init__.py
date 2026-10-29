# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model registry for multimodal_dev training.

Maps ``--model-arch`` to a set of factory functions that fully encapsulate
model-specific logic.  The training entry point (``pretrain_multimodal.py``),
the THD collator (``forward_step.py``) and the MDP adapter dispatcher
(``mdp_adapter.py``) remain model-agnostic — adding a new architecture only
requires a new registry entry (and its backing module) without touching them.

Registry entry fields
---------------------
``model_factory_fn``  *(required)*
    ``(args, language_config, vision_config, **kwargs) -> MegatronModule``
    Builds and returns the complete model instance.

``vision_config_fn``  *(required)*
    ``(num_layers_override=None, variant=None) -> TransformerConfig``
    Returns the vision encoder TransformerConfig.

``post_language_config_fn``  *(optional)*
    ``(language_config, args) -> None``
    Mutates the language TransformerConfig in-place with model-specific
    fields (e.g. ``mrope_section``).

``language_model_builder_fn``  *(optional)*
    ``(args, language_config, *, pre_process, post_process, vp_stage,
    pg_collection, **config_overrides) -> decoder``.
    Builds this stage's language decoder. The entry point binds it and hands
    the factory a ``language_model_builder(**config_overrides)`` callable.
    Default: ``models.base.build_language_model`` (``hybrid_config_from_args``
    + ``HybridModelBuilder``), which honours ``--spec`` and the builder's
    stack features. Override only for a decoder that is not a HybridModel.

``vision_flops_fn``  *(optional)*
    ``(args, language_config, vision_config) -> None``
    Sets vision FLOPs metadata on ``args`` for training throughput logging.
    See ``megatron/training/training.py::vision_encoder_flops`` for the knobs.

``vision_collate_spec_fn``  *(required for packed multimodal training)*
    ``(args) -> VisionCollateSpec``
    Describes the model's vision items to the generic THD collator: image-token
    id, shape-key width, and the payload/output row arithmetic. Must be callable
    on every rank without a model instance.

``mdp_adapter_fn``  *(required for ``--mdp-enable``)*
    ``(args, language_config, vision_config) -> MdpModelAdapter``
    Builds the MDP model adapter, normally a
    ``BaseGridVisionAdapter`` subclass in ``models/<arch>/mdp_adapter.py``.

``dataset_providers``  *(optional)*
    ``Dict[str, str | callable]``
    Maps ``--dataset-provider`` names to callables (or dotted import paths
    resolved lazily) with signature
    ``(train_val_test_num_samples) -> (train_ds, val_ds, test_ds)``.
"""

from examples.multimodal_dev.models.qwen35_vl.configuration import get_qwen35_vl_vision_config
from examples.multimodal_dev.models.qwen35_vl.factory import build_model as _build_qwen35_vl_model
from examples.multimodal_dev.models.qwen35_vl.factory import (
    post_language_config as _qwen35_vl_post_language_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import (
    set_vision_flops_metadata as _qwen35_vl_vision_flops,
)
from examples.multimodal_dev.models.qwen35_vl.mdp_adapter import (
    build_collate_spec as _qwen35_vl_collate_spec,
)
from examples.multimodal_dev.models.qwen35_vl.mdp_adapter import (
    build_mdp_adapter as _qwen35_vl_mdp_adapter,
)
from examples.multimodal_dev.models.qwen38_vl.configuration import get_qwen38_vl_vision_config
from examples.multimodal_dev.models.qwen38_vl.factory import build_model as _build_qwen38_vl_model
from examples.multimodal_dev.models.qwen38_vl.factory import (
    post_language_config as _qwen38_vl_post_language_config,
)
from examples.multimodal_dev.models.qwen38_vl.factory import (
    set_vision_flops_metadata as _qwen38_vl_vision_flops,
)

MODEL_REGISTRY = {
    "qwen35_vl": {
        "model_factory_fn": _build_qwen35_vl_model,
        "vision_config_fn": get_qwen35_vl_vision_config,
        "post_language_config_fn": _qwen35_vl_post_language_config,
        "vision_flops_fn": _qwen35_vl_vision_flops,
        "vision_collate_spec_fn": _qwen35_vl_collate_spec,
        "mdp_adapter_fn": _qwen35_vl_mdp_adapter,
        "dataset_providers": {
            "mock": (
                "examples.multimodal_dev.data.mock"
                ".train_valid_test_datasets_provider"
            ),
            "cord_v2": (
                "examples.multimodal_dev.data.cord_v2"
                ".train_valid_test_datasets_provider"
            ),
            "mdp_mock": (
                "examples.multimodal_dev.data.mdp_mock"
                ".train_valid_test_datasets_provider"
            ),
        },
    },
    # Qwen3.8-Flash-Next VL: Qwen3.5-VL ViT, token ids, collate spec, MDP adapter and data
    # providers (all identical in HF qwen4_exp); the Qwen3.8 decoder comes from the default
    # decoder hook.
    "qwen38_vl": {
        "model_factory_fn": _build_qwen38_vl_model,
        "vision_config_fn": get_qwen38_vl_vision_config,
        "post_language_config_fn": _qwen38_vl_post_language_config,
        "vision_flops_fn": _qwen38_vl_vision_flops,
        "vision_collate_spec_fn": _qwen35_vl_collate_spec,
        "mdp_adapter_fn": _qwen35_vl_mdp_adapter,
        "dataset_providers": {
            "mock": (
                "examples.multimodal_dev.data.mock"
                ".train_valid_test_datasets_provider"
            ),
            "mdp_mock": (
                "examples.multimodal_dev.data.mdp_mock"
                ".train_valid_test_datasets_provider"
            ),
        },
    },
}


def registry_for(args) -> dict:
    """The registry entry selected by ``--model-arch``."""
    model_arch = getattr(args, "model_arch", None)
    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model arch '{model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_arch]
