# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL multimodal model for standalone FSDP + EP training.

Composes a Megatron-native Qwen3.5 vision encoder with a ``HybridModel``
language decoder using MRoPE and hybrid GatedDeltaNet / full-attention
layers.
"""

from typing import Callable, Optional

from torch import Tensor

from examples.multimodal_dev.models.base import MultimodalModel
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
    VISION_KWARGS,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from examples.multimodal_dev.models.qwen35_vl.specs import get_qwen35_vl_vision_spec
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class Qwen35VLModel(MultimodalModel):
    """Qwen3.5-VL multimodal model.

    Args:
        language_config: ``TransformerConfig`` for the language decoder.
        language_model_builder: Zero-argument callable returning this stage's
            decoder (the registry's ``language_model_builder_fn``, bound by the
            entry point). Called *after* the vision encoder is constructed so
            the parameter-initialization RNG stream keeps its historical order.
        vision_config: ``TransformerConfig`` for the vision encoder.
        vision_spec: ``ModuleSpec`` for vision encoder layers.
        image_token_id: Token ID for image placeholders.
        spatial_merge_size: Vision encoder spatial merge factor.
        pre_process: First PP stage flag — build the vision encoder here.
        vp_stage: Virtual pipeline stage index, or ``None`` when VPP is
            disabled.
        build_vision_encoder: Build the in-model vision encoder. MDP sets
            this to ``False``: its replicated encoder domain owns the
            encoder, and an unused in-model copy would still drift under
            weight decay and collide with the ``vision_model.*``
            checkpoint keys.

    The decoder's position embedding must be MRoPE; the factory checks the
    recipe before anything is built.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        language_model_builder: Callable[[], MegatronModule],
        vision_config: TransformerConfig,
        vision_spec: ModuleSpec = None,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        spatial_merge_size: int = 2,
        pre_process: bool = True,
        vp_stage: Optional[int] = None,
        build_vision_encoder: bool = True,
    ):
        if vision_spec is None:
            vision_spec = get_qwen35_vl_vision_spec()

        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = spatial_merge_size

        # Vision encoder lives on the first PP stage only. Under MDP the
        # replicated encoder domain replaces it (build_vision_encoder=False):
        # an unused in-model copy would still drift under weight decay and
        # collide with the vision_model.* checkpoint keys.
        if pre_process and build_vision_encoder:
            vkw = dict(VISION_KWARGS)
            vkw["spatial_merge_size"] = spatial_merge_size
            vkw["out_hidden_size"] = language_config.hidden_size

            vision_encoder = Qwen35VLVisionEncoder(
                config=vision_config,
                transformer_layer_spec=vision_spec,
                in_channels=vkw["in_channels"],
                patch_size=vkw["patch_size"],
                temporal_patch_size=vkw["temporal_patch_size"],
                spatial_merge_size=vkw["spatial_merge_size"],
                out_hidden_size=vkw["out_hidden_size"],
                max_num_positions=vkw["max_num_positions"],
            )
        else:
            vision_encoder = None

        super().__init__(
            language_config=language_config,
            language_model=language_model_builder(),
            vision_encoder=vision_encoder,
            image_token_id=image_token_id,
            pre_process=pre_process,
            vp_stage=vp_stage,
        )

    def compute_position_ids(
        self,
        input_ids: Tensor,
        image_grid_thw: Optional[Tensor] = None,
        packed_seq_params=None,
    ) -> Tensor:
        """Compute 3D MRoPE position IDs for Qwen3.5-VL.

        In THD mode ``input_ids`` is ``[1, T]`` and ``packed_seq_params``
        supplies per-segment boundaries; positions restart at 0 per
        segment. In BSHD mode ``input_ids`` is ``[B, S]`` and
        ``packed_seq_params`` should be ``None``.

        Returns:
            ``[3, B, S]`` position IDs for MRoPE (``[3, 1, T]`` in THD).
        """
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            packed_seq_params=packed_seq_params,
        )
        return position_ids
