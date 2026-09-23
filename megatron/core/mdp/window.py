# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP iteration window: capture once, replay per VPP chunk.

The window is the only exit point for pixels and descriptors. It consumes exactly
one real data iterator per iteration and returns ``num_vpp_chunks`` independent
cursors, so the replay iterators never consume additional sampler input.
"""

import threading
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from torch import Tensor

from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.observability import nvtx_phase
from megatron.core.mdp.protocols import CapturedMicrobatch, MdpModelAdapter, VisionDescriptor

# Owner-sharded pixel reading: while ``capture`` fetches one microbatch, this
# holds ``(microbatch_owner_worker_id, my_worker_id)`` so the model's collate
# path (which has no microbatch context in its signature) can skip pixel
# materialization on non-owner workers. Unset outside MDP capture, so the
# native path is unchanged. Thread-local because the
# window-capture prefetch thread may capture the next train window while the
# main thread captures an eval window.
_PIXEL_OWNERSHIP = threading.local()


def pixel_capture_suppressed() -> bool:
    """True when the in-progress capture microbatch is NOT owned by this worker.

    Queried by the model collate path to skip pixel materialization + H2D for
    microbatches whose pixels another planning-group worker owns.
    """
    context = getattr(_PIXEL_OWNERSHIP, "value", None)
    if context is None:
        return False
    owner_worker_id, my_worker_id = context
    return owner_worker_id != my_worker_id


def _decoder_offset_in_sample(item, microbatch_id: int, output_rows: int) -> int:
    """Offset of an item's first decoder row inside its packed sample.

    The descriptor carries a start offset and a row count, not the per-row
    position tuple, so the run must be contiguous and contained in the sample
    the span columns describe. The collator checks contiguity too; this is the
    check at the point where the *plan input* is built, because at decoder CP>1
    a violation here does not fail loudly — it silently routes rows to the wrong
    context-parallel rank.
    """
    where = (
        f"item (mb={microbatch_id}, sample={item.sample_id}, "
        f"ordinal={item.image_ordinal})"
    )
    positions = item.decoder_positions
    start = positions[0]
    if positions[-1] - start != output_rows - 1:
        raise MdpConfigurationError(
            f"MDP: {where} violates: decoder positions are contiguous "
            f"({start}..{positions[-1]} for {output_rows} rows)."
        )
    if item.sample_padded_len <= 0:
        # The adapter did not supply the sample span. That is inert at CP=1,
        # where the split is the identity and the offset is never read; at CP>1
        # the planner's split_item rejects a non-positive span, so an adapter
        # that never learned to emit it fails loudly there rather than routing
        # rows by a fabricated offset here.
        return 0
    offset = start - item.sample_padded_start
    if offset < 0 or offset + output_rows > item.sample_padded_len:
        raise MdpConfigurationError(
            f"MDP: {where} violates: its decoder rows lie inside its sample's "
            f"padded span [{item.sample_padded_start}, "
            f"{item.sample_padded_start + item.sample_padded_len}); got "
            f"[{start}, {start + output_rows})."
        )
    return offset


@dataclass(frozen=True)
class MdpMicrobatchVisionRecord:
    """Endpoint-local record of one vision item.

    ``decoder_positions`` exists only here: absolute token offsets in the current
    decoder microbatch THD ``[1, T_dec]`` (physical layout per
    ``cu_seqlens_q_padded`` when alignment padding exists). Only the endpoint
    needs them when filling the leaf; they never enter descriptors, routes, or
    the plan.
    """

    global_item_id: int
    sample_id: int
    image_ordinal: int
    grid_thw: tuple
    output_rows: int
    decoder_positions: tuple


@dataclass(frozen=True)
class MdpMicrobatchRecord:
    """One decoder microbatch's replay record.

    ``model_payload`` is opaque to core; ``decoder_packed_seq_params`` stays an
    explicit field because core must assert ``qkv_format == "thd"`` (dual-THD
    isolation), and it is never passed to the vision encoder.
    """

    microbatch_id: int
    text_only: bool
    vision_items: tuple
    decoder_packed_seq_params: Any
    model_payload: Mapping[str, Any]


class _ReplayCursor:
    """One VPP chunk's independent cursor over the captured records."""

    def __init__(self, records: Sequence[MdpMicrobatchRecord]) -> None:
        self._records = records
        self._next = 0

    def __iter__(self) -> Iterator[MdpMicrobatchRecord]:
        return self

    def __next__(self) -> MdpMicrobatchRecord:
        if self._next >= len(self._records):
            raise MdpStateError(
                f"MDP: replay cursor violates: at most {len(self._records)} records "
                "per cursor per iteration (cursor overrun)."
            )
        record = self._records[self._next]
        self._next += 1
        return record


class MdpIterationWindow:
    """Captured iteration state: records, descriptors, and the pixel sidecar."""

    def __init__(
        self,
        records: Sequence[MdpMicrobatchRecord],
        descriptors: Sequence[VisionDescriptor],
        sidecar: Mapping[int, Tensor],
        num_vpp_chunks: int,
    ) -> None:
        self._records = tuple(records)
        self._descriptors = tuple(descriptors)
        self._sidecar = dict(sidecar)
        self._num_vpp_chunks = num_vpp_chunks
        self._replayed = False

    @classmethod
    def capture(
        cls,
        data_iterators: Union[Iterator, Sequence[Iterator]],
        *,
        num_microbatches: int,
        adapter: MdpModelAdapter,
        num_vpp_chunks: int,
        lane_id: Optional[int],
        my_worker_id: int,
        num_workers: int,
    ) -> "MdpIterationWindow":
        """Consume one real iterator and build records, descriptors, and sidecar.

        Only the owner endpoint (``lane_id is not None``) emits descriptors;
        ``global_item_id`` values are assigned in ``(microbatch_id, sample_id,
        image_ordinal)`` capture order, so they are stable and unique within the
        planning group (one endpoint per group generates them).

        Every worker materializes and cuts pixels only for the microbatches it owns
        (``microbatch_id % num_workers == my_worker_id``); the ownership context
        set around each ``adapter.get_batch`` call lets the model collate path
        skip pixel materialization on non-owners.
        """
        if isinstance(data_iterators, (list, tuple)):
            iterator = data_iterators[0] if data_iterators else None
        else:
            iterator = data_iterators
        if my_worker_id < 0 or not num_workers or my_worker_id >= num_workers:
            raise MdpConfigurationError(
                "MDP: owner-sharded capture requires 0 <= my_worker_id < num_workers "
                f"(got {my_worker_id}, {num_workers})."
            )

        records = []
        descriptors = []
        sidecar = {}
        next_item_id = 0
        merge = adapter.spatial_merge_size
        for microbatch_id in range(num_microbatches):
            owner_worker_id = microbatch_id % num_workers
            owns_pixels = owner_worker_id == my_worker_id
            try:
                _PIXEL_OWNERSHIP.value = (owner_worker_id, my_worker_id)
                with nvtx_phase("p1_get_batch"):
                    captured = adapter.get_batch(iterator)
            finally:
                _PIXEL_OWNERSHIP.value = None
            if captured is None:
                raise MdpStateError(
                    f"MDP: data iterator violates: {num_microbatches} microbatches per "
                    f"iteration (exhausted at microbatch {microbatch_id})."
                )
            _validate_captured(
                captured,
                microbatch_id,
                merge,
                expect_pixels=owns_pixels,
            )
            vision_records = []
            for item in captured.vision_items:
                t, h, w = item.grid_thw
                output_rows = t * (h // merge) * (w // merge)
                item_id = next_item_id
                next_item_id += 1
                vision_records.append(
                    MdpMicrobatchVisionRecord(
                        global_item_id=item_id,
                        sample_id=item.sample_id,
                        image_ordinal=item.image_ordinal,
                        grid_thw=item.grid_thw,
                        output_rows=output_rows,
                        decoder_positions=item.decoder_positions,
                    )
                )
                if lane_id is not None:
                    if len(item.decoder_positions) != output_rows:
                        raise MdpConfigurationError(
                            f"MDP: item (mb={microbatch_id}, sample={item.sample_id}, "
                            f"ordinal={item.image_ordinal}) violates: "
                            f"len(decoder_positions) == output_rows "
                            f"({len(item.decoder_positions)} != {output_rows})."
                        )
                    offset_in_sample = _decoder_offset_in_sample(
                        item, microbatch_id, output_rows
                    )
                    descriptors.append(
                        VisionDescriptor(
                            global_item_id=item_id,
                            sample_id=item.sample_id,
                            image_ordinal=item.image_ordinal,
                            owner_dp_lane=lane_id,
                            microbatch_id=microbatch_id,
                            estimated_cost_units=adapter.estimate_cost(item),
                            payload_rows=item.payload_rows,
                            output_rows=output_rows,
                            grid_thw=item.grid_thw,
                            owner_worker_id=owner_worker_id,
                            sample_padded_start=item.sample_padded_start,
                            sample_padded_len=item.sample_padded_len,
                            decoder_offset_in_sample=offset_in_sample,
                        )
                    )
                if owns_pixels:
                    sidecar[item_id] = captured.flat_pixel_payload[
                        item.payload_row_start : item.payload_row_start + item.payload_rows
                    ]
            records.append(
                MdpMicrobatchRecord(
                    microbatch_id=microbatch_id,
                    text_only=not captured.vision_items,
                    vision_items=tuple(vision_records),
                    decoder_packed_seq_params=captured.decoder_packed_seq_params,
                    model_payload=captured.model_payload,
                )
            )
        return cls(records, descriptors, sidecar, num_vpp_chunks)

    def replay_iterators(self) -> list:
        """``num_vpp_chunks`` independent cursors; a second call raises."""
        if self._replayed:
            raise MdpStateError(
                "MDP: iteration window violates: replay iterators are created once per "
                "capture."
            )
        self._replayed = True
        return [_ReplayCursor(self._records) for _ in range(self._num_vpp_chunks)]

    def records(self) -> Sequence[MdpMicrobatchRecord]:
        """All captured microbatch records, in microbatch order."""
        return self._records

    def descriptors(self) -> Sequence[VisionDescriptor]:
        """The endpoint's descriptors; empty on non-endpoint members."""
        return self._descriptors

    def payload_sidecar(self) -> Mapping[int, Tensor]:
        """Per-item pixel tensors keyed by ``global_item_id``.

        Contains the items from this worker's owned microbatches only.
        """
        return dict(self._sidecar)

    def release_pixels(self) -> None:
        """Drop the window/owner pixel references after P1.

        If a processed payload is an input to the producer autograd/checkpoint
        graph, that graph owns the underlying storage until P5 backward completes.
        """
        self._sidecar.clear()


def _validate_captured(
    captured: CapturedMicrobatch,
    microbatch_id: int,
    merge: int,
    expect_pixels: Optional[bool] = None,
) -> None:
    """Local consistency checks performed before any bridge collective.

    An owned microbatch must carry pixels for its items and a non-owned one must
    not (its collate pixel branch is suppressed), so both wiring failures are
    diagnosed here.
    """
    params = captured.decoder_packed_seq_params
    if params is not None and getattr(params, "qkv_format", None) != "thd":
        raise MdpConfigurationError(
            f"MDP: microbatch {microbatch_id} violates: decoder_packed_seq_params."
            f"qkv_format == 'thd' (got {getattr(params, 'qkv_format', None)!r})."
        )
    has_pixels = captured.flat_pixel_payload is not None
    has_items = bool(captured.vision_items)
    if expect_pixels is False:
        if has_pixels:
            raise MdpConfigurationError(
                f"MDP: microbatch {microbatch_id} violates: non-owned microbatches "
                "carry no pixel payload under owner-sharded capture (the collate pixel "
                "branch was not suppressed)."
            )
    elif has_items and not has_pixels:
        raise MdpConfigurationError(
            f"MDP: microbatch {microbatch_id} violates: pixel data and grid metadata "
            "either both exist or both are absent (items without pixels)."
        )
    previous = None
    for item in captured.vision_items:
        key = (item.sample_id, item.image_ordinal)
        if previous is not None and key <= previous:
            raise MdpConfigurationError(
                f"MDP: microbatch {microbatch_id} violates: vision_items ordered by "
                f"(sample_id, image_ordinal) without duplicates (at {key})."
            )
        previous = key
        t, h, w = item.grid_thw
        if t < 1 or h < 1 or w < 1:
            raise MdpConfigurationError(
                f"MDP: item {key} in microbatch {microbatch_id} violates: grid_thw is "
                f"positive (got {item.grid_thw})."
            )
        if h % merge != 0 or w % merge != 0:
            raise MdpConfigurationError(
                f"MDP: item {key} in microbatch {microbatch_id} violates: h and w are "
                f"divisible by spatial_merge_size={merge} (got {item.grid_thw})."
            )
        if t * h * w != item.payload_rows:
            raise MdpConfigurationError(
                f"MDP: item {key} in microbatch {microbatch_id} violates: "
                f"payload_rows == t*h*w ({item.payload_rows} != {t * h * w})."
            )
        end = item.payload_row_start + item.payload_rows
        if item.payload_row_start < 0 or (
            has_pixels and end > captured.flat_pixel_payload.shape[0]
        ):
            raise MdpConfigurationError(
                f"MDP: item {key} in microbatch {microbatch_id} violates: payload rows "
                f"[{item.payload_row_start}, {end}) lie inside flat_pixel_payload."
            )
