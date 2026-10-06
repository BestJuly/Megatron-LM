# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL MDP adapter and vision collate spec.

The adapter is everything model-specific MDP core needs beyond
:class:`~examples.multimodal_dev.models.vision_base.BaseGridVisionAdapter`: the
shared vision-encoder factory and a chunk-oblivious ``encode``. It lives under
``examples/multimodal_dev`` because ``megatron/core/mdp`` must not import model
packages.

Qwen3.5-VL keys each vision item by ``(t, h, w)`` patch-grid dimensions, packs
``t*h*w`` payload rows, and emits ``t*(h/m)*(w/m)`` rows after the spatial merge.
"""

import torch

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    VISION_KWARGS,
    resolve_image_token_id,
)
from examples.multimodal_dev.models.qwen35_vl.specs import get_qwen35_vl_vision_spec
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from examples.multimodal_dev.models.vision_base import (
    BaseGridVisionAdapter,
    VisionCollateSpec,
    grid_workload_fn,
)


def payload_width_of(vision_kwargs: dict) -> int:
    """Flat pixel-payload row width for a Conv3d patch embedding."""
    return (
        vision_kwargs["in_channels"]
        * vision_kwargs["temporal_patch_size"]
        * vision_kwargs["patch_size"] ** 2
    )


def build_collate_spec(args, vision_kwargs: dict = None) -> VisionCollateSpec:
    """Vision collate spec for Qwen3.5-VL.

    Called by the collator on every rank; it must not depend on anything that
    only the first pipeline stage knows.
    """
    kwargs = dict(vision_kwargs or VISION_KWARGS)
    merge = kwargs["spatial_merge_size"]

    def _validate(shape_key):
        t, h, w = shape_key
        if t < 1 or h < 1 or w < 1:
            raise ValueError(f"grid {shape_key} is not positive")
        if h % merge != 0 or w % merge != 0:
            raise ValueError(
                f"grid {shape_key} is not divisible by spatial_merge_size={merge}"
            )

    return VisionCollateSpec(
        image_token_id=resolve_image_token_id(args),
        shape_key_width=3,
        payload_rows_of=lambda k: k[0] * k[1] * k[2],
        output_rows_of=lambda k: k[0] * (k[1] // merge) * (k[2] // merge),
        workload_of=grid_workload_fn(merge),
        validate_shape_key=_validate,
    )


class Qwen35VLMdpAdapter(BaseGridVisionAdapter):
    """MdpModelAdapter implementation for Qwen3.5-VL.

    Args:
        args: Megatron parsed arguments (read for ``image_token_id`` only).
        out_hidden_size: Language decoder hidden size (patch-merger output).
        vision_kwargs: Optional override of the Qwen3.5-VL vision kwargs.
    """

    def __init__(self, args, out_hidden_size: int, vision_kwargs: dict = None):
        self._vision_kwargs = dict(vision_kwargs or VISION_KWARGS)
        self._vision_kwargs["out_hidden_size"] = out_hidden_size
        super().__init__(
            collate_spec=build_collate_spec(args, self._vision_kwargs),
            payload_width=payload_width_of(self._vision_kwargs),
            output_hidden_size=out_hidden_size,
        )

    def build_encoder(self, model_config, *, pg_collection) -> torch.nn.Module:
        """Same factory as the non-MDP path (models/qwen35_vl/model.py)."""
        del pg_collection  # the encoder has no model parallelism in v1
        kwargs = self._vision_kwargs
        return Qwen35VLVisionEncoder(
            config=model_config,
            transformer_layer_spec=get_qwen35_vl_vision_spec(),
            in_channels=kwargs["in_channels"],
            patch_size=kwargs["patch_size"],
            temporal_patch_size=kwargs["temporal_patch_size"],
            spatial_merge_size=kwargs["spatial_merge_size"],
            out_hidden_size=kwargs["out_hidden_size"],
            max_num_positions=kwargs["max_num_positions"],
        )

    def encode(self, encoder: torch.nn.Module, payload: torch.Tensor, layout) -> torch.Tensor:
        """Encoder forward for one (already rebased) chunk sub-layout.

        The encoder builds its vision-only THD ``PackedSeqParams`` internally
        from ``grid_thw`` (one sub-sequence per temporal frame); the decoder
        ``PackedSeqParams`` is never read here.
        """
        # A CPU tensor, deliberately: with the grid cache enabled the encoder
        # consumes grid_thw exclusively as Python lists (tolist), so a device
        # tensor here cost a blocking pageable H2D on the busy compute stream
        # (~2.4 ms/iter measured) followed by D2H readbacks inside the
        # encoder. The encoder moves it to the device itself on the uncached
        # (QWEN35_VL_GRID_CACHE=0) fallback paths that do tensor math on it.
        grid_thw = torch.tensor(
            [segment.shape_key for segment in layout.segments], dtype=torch.long
        )
        return encoder(payload, grid_thw)


def build_mdp_adapter(args, language_config, vision_config=None) -> Qwen35VLMdpAdapter:
    """Registry hook: build the Qwen3.5-VL MDP adapter."""
    del vision_config  # dimensions come from VISION_KWARGS + the decoder width
    return Qwen35VLMdpAdapter(args, out_hidden_size=language_config.hidden_size)
