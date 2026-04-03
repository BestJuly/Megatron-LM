# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Vision encoder wrapper for Kimi K2.5 VL.

The MoonViT3d vision encoder and PatchMergerMLP projector are custom
HuggingFace modules. Since no native MCore implementation exists, we
dynamically load them from the HF model repository via
``transformers.dynamic_module_utils.get_class_from_dynamic_module``.

This approach mirrors the Megatron-Bridge implementation in
``megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl``.

The encoder wraps both the vision tower (MoonViT3d) and the projector
(PatchMergerMLP) into a single module that outputs embeddings in the
language model's hidden dimension, matching the MIMO
``VisionModalitySubmodules`` interface.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

logger = logging.getLogger(__name__)


class KimiK25VisionEncoder(nn.Module):
    """Vision encoder for Kimi K2.5 VL using HF dynamic modules.

    Loads MoonViT3d (vision tower) and PatchMergerMLP (projector)
    from the HuggingFace model repository at construction time.

    The forward method takes pixel_values and grid_thw (as produced
    by the HF Kimi K2.5 processor) and returns projected embeddings
    in the language model's hidden dimension.

    Args:
        hf_model_path: Path or HF hub ID for the Kimi K2.5 VL model
            (e.g., ``"moonshotai/Kimi-K2.5"``).  Required for dynamic
            module loading.
        language_hidden_size: Hidden size of the language model. Used
            only for validation; the actual output dimension is
            determined by PatchMergerMLP's configuration.
    """

    def __init__(
        self,
        hf_model_path: str,
        language_hidden_size: Optional[int] = None,
    ):
        super().__init__()

        if hf_model_path is None:
            raise ValueError(
                "hf_model_path must be set for KimiK25VisionEncoder. "
                "Provide the HuggingFace model path or hub ID."
            )

        self.hf_model_path = hf_model_path

        # Load custom classes from the HF model repository
        MoonViT3dPretrainedModel = get_class_from_dynamic_module(
            "modeling_kimi_k25.MoonViT3dPretrainedModel",
            hf_model_path,
        )
        PatchMergerMLP = get_class_from_dynamic_module(
            "modeling_kimi_k25.PatchMergerMLP",
            hf_model_path,
        )
        VisionTowerConfig = get_class_from_dynamic_module(
            "modeling_kimi_k25.VisionTowerConfig",
            hf_model_path,
        )
        ProjectorConfig = get_class_from_dynamic_module(
            "modeling_kimi_k25.ProjectorConfig",
            hf_model_path,
        )

        # Patch MoonViT3dEncoder to add missing use_deterministic_attn attribute
        import importlib

        _vit_module = importlib.import_module(MoonViT3dPretrainedModel.__module__)
        if not getattr(_vit_module.MoonViT3dEncoder, "_bridge_init_patched", False):
            _orig_encoder_init = _vit_module.MoonViT3dEncoder.__init__

            def _patched_encoder_init(self, *args, **kwargs):
                self.use_deterministic_attn = False
                _orig_encoder_init(self, *args, **kwargs)

            _vit_module.MoonViT3dEncoder.__init__ = _patched_encoder_init
            _vit_module.MoonViT3dEncoder._bridge_init_patched = True

        # Load vision config from HF model path
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        vision_config = hf_config.vision_config

        # Build vision tower and projector
        vision_tower_config = VisionTowerConfig(vision_config)
        projector_config = ProjectorConfig(vision_config)

        self.vision_tower = MoonViT3dPretrainedModel(vision_tower_config)
        self.mm_projector = PatchMergerMLP(projector_config)

        logger.info(
            "KimiK25VisionEncoder initialized from %s "
            "(vision_tower: MoonViT3d, projector: PatchMergerMLP)",
            hf_model_path,
        )

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the vision tower parameters."""
        try:
            return next(self.vision_tower.parameters()).dtype
        except StopIteration:
            return torch.bfloat16

    def forward(
        self,
        pixel_values: Tensor,
        grid_thw: Tensor,
    ) -> Tensor:
        """Encode images through MoonViT3d + PatchMergerMLP.

        Args:
            pixel_values: Image tensor(s) for the vision tower.
                Shape depends on the HF processor output.
            grid_thw: Tensor of shape ``(num_images, 3)`` containing
                ``[temporal, height, width]`` per image in patch-grid
                units.

        Returns:
            Projected visual embeddings of shape
            ``[total_merged_patches, language_hidden_size]``.
        """
        pixel_values = pixel_values.to(self.dtype)
        image_features = self.vision_tower(pixel_values, grid_thw)
        projected = self.mm_projector(image_features)
        return projected
