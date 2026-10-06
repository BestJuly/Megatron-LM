# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared vision plumbing for multimodal_dev: collate spec and MDP adapter base.

Every vision encoder in this example family shares the same *data* shape — a flat
``[rows, payload_width]`` patch payload plus one small integer shape key per item
— even when the encoders themselves differ (patch size, merge policy, gated vs
plain MLP, learned vs rotary position embeddings). That shared shape is captured
here once:

* :class:`VisionCollateSpec` is what the generic THD collator in
  ``forward_step.py`` needs to cut a model's vision sidecar. It replaces the
  previous arrangement where the collator read ``args.image_token_id`` and
  ``args.vision_spatial_merge_size`` -- the latter being a side effect of the
  FLOPs-metadata helper, which silently defaulted to 2 for any new model.
* :class:`BaseGridVisionAdapter` implements the model-independent half of
  ``MdpModelAdapter``: sidecar-to-``CapturedMicrobatch`` conversion, the LPT
  cost, and item validation. Subclasses supply only ``build_encoder`` and
  ``encode``.

Neither class assumes a ``(t, h, w)`` reading of the shape key. A model whose
items are keyed by something else only has to supply different
``payload_rows_of`` / ``output_rows_of`` callables and a matching
``shape_key_width``.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterator, Optional

import torch

from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem

#: Fixed integer columns of a ``vision_item_meta`` row before the shape key:
#: ``(sample_index, image_ordinal, payload_row_start)``.
META_PREFIX_COLUMNS = 3


@dataclass(frozen=True)
class VisionCollateSpec:
    """Everything the generic THD collator needs from a model's vision encoder.

    Args:
        image_token_id: Decoder placeholder token whose slots the encoder fills.
        shape_key_width: Number of integers per item shape key; must equal the
            trailing width of ``sample[grid_key]`` and the adapter's
            ``shape_key_width``.
        payload_rows_of: ``shape_key -> `` rows this item occupies in the flat
            pixel payload.
        output_rows_of: ``shape_key -> `` rows the encoder emits for this item.
            This is also the number of decoder image-token slots it fills.
        validate_shape_key: Optional per-item precondition (e.g. a divisibility
            rule); raise ``ValueError`` on violation. Applied by the collator and
            again, as ``MdpConfigurationError``, by the adapter before transport.
        grid_key: Per-sample metadata key holding an ``[N, shape_key_width]``
            integer tensor of shape keys.
        require_contiguous_slots: Whether an item's decoder image-token slots
            must form one unbroken run. True for encoders whose output maps
            directly onto a contiguous block. Set False for layouts that
            interleave separator tokens (per-row newlines, start/end markers)
            between an item's vision tokens -- MDP places embeddings with
            ``masked_scatter`` in mask order, so interleaving is harmless, but
            silently accepting it for a model that does *not* expect it would
            hide a truncation bug.
        workload_of: ``[N, shape_key_width]`` integer tensor of shape keys ->
            ``(patch_rows, attn_squared_sum, output_rows)`` as 0-dim tensors, the
            three per-micro-batch quantities the throughput logger needs. Kept
            here rather than hardcoded in the training loop because the
            attention term depends on how the encoder segments its packed
            sequence. Must stay on the input tensor's device (no ``.item()``).
    """

    image_token_id: int
    payload_rows_of: Callable[[tuple], int]
    output_rows_of: Callable[[tuple], int]
    workload_of: Callable[["torch.Tensor"], tuple]
    shape_key_width: int = 3
    validate_shape_key: Optional[Callable[[tuple], None]] = None
    grid_key: str = "image_grid_thw"
    require_contiguous_slots: bool = True

    def check(self, shape_key: tuple) -> None:
        """Validate one shape key's width and model-specific preconditions."""
        if len(shape_key) != self.shape_key_width:
            raise ValueError(
                f"shape key {shape_key} has width {len(shape_key)}, expected "
                f"{self.shape_key_width}"
            )
        if self.validate_shape_key is not None:
            self.validate_shape_key(shape_key)


def grid_workload_fn(spatial_merge_size: int) -> Callable[[torch.Tensor], tuple]:
    """``workload_of`` for a ``(t, h, w)`` patch grid with frame-wise attention.

    Each temporal frame of each item is one attention sub-sequence of length
    ``h*w``, matching the ``cu_seqlens`` such an encoder builds, so the ``L^2``
    term is ``sum_items t*(h*w)^2``. Output rows follow the spatial merge.
    All arithmetic stays on the input tensor's device.
    """
    merge = spatial_merge_size

    def _workload(shape_keys: torch.Tensor) -> tuple:
        t = shape_keys[:, 0].to(torch.float64)
        frame = (shape_keys[:, 1] * shape_keys[:, 2]).to(torch.float64)
        merged_h = torch.div(shape_keys[:, 1], merge, rounding_mode="floor")
        merged_w = torch.div(shape_keys[:, 2], merge, rounding_mode="floor")
        merged = t * (merged_h * merged_w).to(torch.float64)
        return (t * frame).sum(), (t * frame * frame).sum(), merged.sum()

    return _workload


class BaseGridVisionAdapter:
    """Model-independent half of ``MdpModelAdapter`` for flat-patch encoders.

    Subclasses must implement :meth:`build_encoder` and :meth:`encode`, and are
    the only place that interprets ``shape_key`` as encoder geometry.

    Args:
        collate_spec: The model's :class:`VisionCollateSpec`.
        payload_width: Trailing width of one flat pixel payload row.
        output_hidden_size: Width of one encoder output row (post-projector).
    """

    def __init__(
        self,
        *,
        collate_spec: VisionCollateSpec,
        payload_width: int,
        output_hidden_size: int,
    ) -> None:
        self.collate_spec = collate_spec
        self.payload_width = payload_width
        self.output_hidden_size = output_hidden_size

    @property
    def shape_key_width(self) -> int:
        """Descriptor wire width of this model's shape key."""
        return self.collate_spec.shape_key_width

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def get_batch(self, data_iterator: Iterator) -> Optional[CapturedMicrobatch]:
        """One microbatch through the native THD collation path.

        Requires the vision sidecar (``--mdp-enable`` makes the collator emit
        it), from which the per-item records and decoder positions are cut. The
        pixel payload and the sidecar keys are removed from the replayed model
        payload: pixels never enter the decoder.
        """
        from examples.multimodal_dev.forward_step import get_batch

        batch = get_batch(data_iterator)
        if batch is None:
            return None
        if "vision_item_meta" not in batch:
            raise RuntimeError(
                "MDP adapter needs the vision sidecar; the collator must run with "
                "with_vision_sidecar=True (set by --mdp-enable)."
            )
        meta = batch.pop("vision_item_meta")
        positions = batch.pop("vision_decoder_positions")
        pixels = batch.pop("pixel_values", None)
        spec = self.collate_spec

        items = []
        position_cursor = 0
        # One D2H transfer for the whole positions tensor; slicing the CPU
        # copy per item avoids a device sync per vision item.
        positions_cpu = positions.cpu().tolist()
        for row in meta.cpu().tolist():
            sample_id, ordinal, payload_row_start = (int(v) for v in row[:META_PREFIX_COLUMNS])
            shape_key = tuple(int(v) for v in row[META_PREFIX_COLUMNS:])
            output_rows = spec.output_rows_of(shape_key)
            decoder_positions = tuple(
                positions_cpu[position_cursor : position_cursor + output_rows]
            )
            position_cursor += output_rows
            items.append(
                CapturedVisionItem(
                    sample_id=sample_id,
                    image_ordinal=ordinal,
                    shape_key=shape_key,
                    payload_row_start=payload_row_start,
                    payload_rows=spec.payload_rows_of(shape_key),
                    output_rows=output_rows,
                    decoder_positions=decoder_positions,
                )
            )
        if position_cursor != positions.numel():
            raise RuntimeError(
                f"vision sidecar mismatch: consumed {position_cursor} decoder positions "
                f"of {positions.numel()}"
            )

        packed_seq_params = batch.pop("packed_seq_params", None)
        if pixels is not None and pixels.shape[0] == 0:
            pixels = None
        if pixels is not None and pixels.dtype == torch.float32:
            pixels = pixels.bfloat16()
        return CapturedMicrobatch(
            decoder_packed_seq_params=packed_seq_params,
            vision_items=tuple(items),
            flat_pixel_payload=pixels,
            model_payload=MappingProxyType(batch),
        )

    # ------------------------------------------------------------------
    # Planning cost and validation
    # ------------------------------------------------------------------

    def estimate_cost(self, item: CapturedVisionItem) -> int:
        """Patch rows as the LPT ordering cost; never sizes any buffer."""
        return item.payload_rows

    def validate_item(self, item: CapturedVisionItem) -> None:
        """Re-derive both row counts from the shape key and compare.

        Core has already checked the model-independent invariants; this catches
        a collator and an adapter that disagree about the model's geometry, which
        would otherwise surface as a mis-sized bridge buffer.
        """
        spec = self.collate_spec
        key = (item.sample_id, item.image_ordinal)
        try:
            spec.check(item.shape_key)
        except ValueError as exc:
            raise MdpConfigurationError(f"MDP: item {key} violates: {exc}.") from None
        expected_payload = spec.payload_rows_of(item.shape_key)
        if item.payload_rows != expected_payload:
            raise MdpConfigurationError(
                f"MDP: item {key} violates: payload_rows == payload_rows_of(shape_key) "
                f"({item.payload_rows} != {expected_payload} for {item.shape_key})."
            )
        expected_output = spec.output_rows_of(item.shape_key)
        if item.output_rows != expected_output:
            raise MdpConfigurationError(
                f"MDP: item {key} violates: output_rows == output_rows_of(shape_key) "
                f"({item.output_rows} != {expected_output} for {item.shape_key})."
            )

    # ------------------------------------------------------------------
    # Encoder factory and forward — model-specific
    # ------------------------------------------------------------------

    def build_encoder(self, model_config, *, pg_collection) -> torch.nn.Module:
        """Build the vision encoder through the same factory as the non-MDP path."""
        raise NotImplementedError

    def encode(self, encoder: torch.nn.Module, payload: torch.Tensor, layout):
        """Encoder forward for one (already rebased) chunk sub-layout."""
        raise NotImplementedError
