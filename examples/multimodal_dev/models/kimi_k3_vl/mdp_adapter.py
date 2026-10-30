# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kimi-K3 vision collate spec and MDP adapter.

A Kimi-K3 vision item is keyed by its ``(t, h, w)`` patch grid. It packs
``t*h*w`` raster-ordered rows of ``3*14*14 = 588`` pixels and, because MoonViT
averages over frames before the 2x2 merge, emits ``(h/2)*(w/2)`` rows --
independent of ``t``. All rows of one item form a single attention sequence,
so the attention ``L^2`` term is ``(t*h*w)^2`` per item.
"""

import torch

from examples.multimodal_dev.models.kimi_k3_vl.configuration import (
    VISION_KWARGS,
    resolve_image_token_id,
)
from examples.multimodal_dev.models.kimi_k3_vl.specs import get_kimi_k3_vl_vision_spec
from examples.multimodal_dev.models.kimi_k3_vl.vision_encoder import KimiK3VLVisionEncoder
from examples.multimodal_dev.models.vision_base import BaseGridVisionAdapter, VisionCollateSpec

PAYLOAD_WIDTH = VISION_KWARGS["in_channels"] * VISION_KWARGS["patch_size"] ** 2


def item_output_rows(shape_key, merge: int = VISION_KWARGS["spatial_merge_size"]) -> int:
    """Decoder slots one item fills: ``(h/m)*(w/m)``, frames pooled away."""
    _, h, w = shape_key
    return (h // merge) * (w // merge)


def validate_grid(shape_key, kwargs: dict = VISION_KWARGS) -> None:
    """MoonViT's preconditions on one ``(t, h, w)`` grid."""
    t, h, w = shape_key
    merge = kwargs["spatial_merge_size"]
    if t < 1 or h < 1 or w < 1:
        raise ValueError(f"grid {shape_key} is not positive")
    if h % merge or w % merge:
        raise ValueError(f"grid {shape_key} is not divisible by merge_kernel_size={merge}")
    if t > kwargs["pos_emb_time"]:
        raise ValueError(
            f"grid {shape_key} has {t} frames; MoonViT's time embedding covers "
            f"{kwargs['pos_emb_time']}"
        )
    if h > kwargs["rope_max_side"] or w > kwargs["rope_max_side"]:
        raise ValueError(
            f"grid {shape_key} exceeds MoonViT's {kwargs['rope_max_side']}-patch side limit"
        )


def kimi_workload(shape_keys: torch.Tensor) -> tuple:
    """``(patch_rows, attn_L^2, output_rows)`` with whole-item attention."""
    merge = VISION_KWARGS["spatial_merge_size"]
    rows = (shape_keys[:, 0] * shape_keys[:, 1] * shape_keys[:, 2]).to(torch.float64)
    merged_h = torch.div(shape_keys[:, 1], merge, rounding_mode="floor")
    merged_w = torch.div(shape_keys[:, 2], merge, rounding_mode="floor")
    merged = (merged_h * merged_w).to(torch.float64)
    return rows.sum(), (rows * rows).sum(), merged.sum()


def build_collate_spec(args) -> VisionCollateSpec:
    """Vision collate spec; callable on every rank without a model."""
    return VisionCollateSpec(
        image_token_id=resolve_image_token_id(args),
        shape_key_width=3,
        payload_rows_of=lambda k: k[0] * k[1] * k[2],
        output_rows_of=item_output_rows,
        workload_of=kimi_workload,
        validate_shape_key=validate_grid,
    )


class KimiK3VLMdpAdapter(BaseGridVisionAdapter):
    """``MdpModelAdapter`` for Kimi-K3.

    Args:
        args: Megatron parsed arguments (``image_token_id`` only).
        out_hidden_size: Decoder hidden size (projector output).
    """

    def __init__(self, args, out_hidden_size: int):
        self._out_hidden_size = out_hidden_size
        super().__init__(
            collate_spec=build_collate_spec(args),
            payload_width=PAYLOAD_WIDTH,
            output_hidden_size=out_hidden_size,
        )

    def build_encoder(self, model_config, *, pg_collection) -> torch.nn.Module:
        """Same factory as the native path (``models/kimi_k3_vl/model.py``)."""
        del pg_collection  # the encoder has no model parallelism in v1
        from examples.multimodal_dev.models.kimi_k3_vl.factory import align_vision_norm_epsilon

        align_vision_norm_epsilon(model_config)
        return KimiK3VLVisionEncoder(
            config=model_config,
            transformer_layer_spec=get_kimi_k3_vl_vision_spec(),
            out_hidden_size=self._out_hidden_size,
        )

    def encode(self, encoder: torch.nn.Module, payload: torch.Tensor, layout) -> torch.Tensor:
        """Encoder forward for one rebased chunk sub-layout.

        The grid goes in as Python triples: the encoder only reads their
        values, so a device tensor would just add an H2D copy and readbacks.
        """
        grids = [tuple(int(v) for v in segment.shape_key) for segment in layout.segments]
        return encoder(payload, grids)


def build_mdp_adapter(args, language_config, vision_config=None) -> KimiK3VLMdpAdapter:
    """Registry hook: build the Kimi-K3 MDP adapter."""
    del vision_config  # widths come from the vision config MDP passes to build_encoder
    return KimiK3VLMdpAdapter(args, out_hidden_size=language_config.hidden_size)
