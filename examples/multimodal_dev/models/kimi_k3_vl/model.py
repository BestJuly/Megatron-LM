# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kimi-K3 vision-language model: MoonViT tower + Kimi-K3 hybrid decoder.

The decoder is whatever ``HybridModelBuilder`` builds from the recipe (KDA,
MLA, latent MoE, attention residuals, MTP ...); this wrapper only adds the
vision tower on the first pipeline stage and the ``<|media_pad|>`` scatter
inherited from :class:`MultimodalModel`.

Kimi-K3's decoder has no positional embedding (MLA runs NoPE and KDA is
position-free), so the default sequential ``position_ids`` of the base class
are passed through and ignored -- there is no mRoPE index to build.
"""

from typing import Callable, Optional

from examples.multimodal_dev.models.base import MultimodalModel
from examples.multimodal_dev.models.kimi_k3_vl.configuration import KIMI_K3_IMAGE_TOKEN_ID
from examples.multimodal_dev.models.kimi_k3_vl.vision_encoder import KimiK3VLVisionEncoder
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class KimiK3VLModel(MultimodalModel):
    """Kimi-K3 multimodal model.

    Args:
        language_config: ``TransformerConfig`` for the language decoder.
        language_model_builder: Zero-argument callable returning this stage's
            decoder; called after the vision tower is built so the
            parameter-initialization RNG order is vision first.
        vision_config: ``TransformerConfig`` for MoonViT.
        vision_spec: ``ModuleSpec`` for the ViT layers (default: MoonViT spec).
        image_token_id: ``<|media_pad|>`` id.
        pre_process: First PP stage flag -- the vision tower lives there.
        vp_stage: Virtual pipeline stage index or ``None``.
        build_vision_encoder: ``False`` under MDP, whose replicated encoder
            domain owns the vision weights.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        language_model_builder: Callable[[], MegatronModule],
        vision_config: TransformerConfig,
        vision_spec: ModuleSpec = None,
        image_token_id: int = KIMI_K3_IMAGE_TOKEN_ID,
        pre_process: bool = True,
        vp_stage: Optional[int] = None,
        build_vision_encoder: bool = True,
    ):
        if pre_process and build_vision_encoder:
            vision_encoder = KimiK3VLVisionEncoder(
                config=vision_config,
                transformer_layer_spec=vision_spec,
                out_hidden_size=language_config.hidden_size,
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
