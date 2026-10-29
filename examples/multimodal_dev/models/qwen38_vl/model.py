# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.8-Flash-Next VL model: the Qwen3.5-VL ViT and multimodal glue around the
Qwen3.8-Flash-Next ``HybridModel`` decoder.

Everything model-specific on the vision side is identical to Qwen3.5-VL (see
``configuration.py``), so this subclass only fixes the token-id defaults. Decoder features
(gated residual, QSA, Engram/PLE, MTP) come from the decoder the builder hands in.
"""

from typing import Callable, Optional

from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel
from examples.multimodal_dev.models.qwen38_vl.configuration import (
    QWEN38_VL_IMAGE_TOKEN_ID,
    QWEN38_VL_VIDEO_TOKEN_ID,
    QWEN38_VL_VISION_START_TOKEN_ID,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class Qwen38VLModel(Qwen35VLModel):
    """Qwen3.8-Flash-Next VL. Arguments as :class:`Qwen35VLModel`."""

    def __init__(
        self,
        language_config: TransformerConfig,
        language_model_builder: Callable[[], MegatronModule],
        vision_config: TransformerConfig,
        vision_spec: ModuleSpec = None,
        image_token_id: int = QWEN38_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN38_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN38_VL_VISION_START_TOKEN_ID,
        spatial_merge_size: int = 2,
        pre_process: bool = True,
        vp_stage: Optional[int] = None,
        build_vision_encoder: bool = True,
    ):
        super().__init__(
            language_config=language_config,
            language_model_builder=language_model_builder,
            vision_config=vision_config,
            vision_spec=vision_spec,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            spatial_merge_size=spatial_merge_size,
            pre_process=pre_process,
            vp_stage=vp_stage,
            build_vision_encoder=build_vision_encoder,
        )
