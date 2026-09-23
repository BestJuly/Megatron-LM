# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP plan data model: routes, layouts, capacity policy, and the plan digest.

Pure-compute module: no ``torch.distributed``, no device tensors. The plan follows
a minimal sufficient data model — anything uniquely derivable (offsets, totals,
frame ``cu_seqlens``) is computed by functions, never duplicated as fields, and
never enters the digest.
"""

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Optional, Sequence

from megatron.core.mdp.errors import MdpPlanError

PLAN_SCHEMA_VERSION = 6


@dataclass(frozen=True)
class RowCapacityPolicy:
    """Plan-level row-capacity policy; every buffer size derives from it.

    Production v1 uses ``alignment_rows=1`` so ``capacity == valid``. Padding rows
    affect allocation only; they never participate in attention and never appear
    in unpacked results.
    """

    alignment_rows: int = 1

    def capacity_of(self, valid_rows: int) -> int:
        """``align_up(valid_rows, alignment_rows)``."""
        if valid_rows < 0:
            raise MdpPlanError(f"MDP: valid_rows={valid_rows} violates: valid_rows >= 0.")
        align = self.alignment_rows
        return (valid_rows + align - 1) // align * align


@dataclass(frozen=True)
class RouteSlice:
    """One routed slice: which rows of which item, produced by which logical
    worker, received by which endpoint rank.

    ``owner_worker_id`` is the logical worker holding the item's pixels at
    dispatch time and is always the PIXEL phase source.

    ``slice_id`` / ``item_row_start`` / ``item_rows`` are the sub-interval
    fields decoder CP needs. At CP=1 every item still has exactly one slice,
    with ``slice_id == 0``, ``item_row_start == 0`` and
    ``item_rows == output_rows``, so the plan is unchanged. At CP>1 the decoder
    sequence is zigzag-sharded and an item's rows split across endpoints; one
    endpoint can hold two disjoint runs of the same item, which is why the
    slice — not the item — is the routed unit and why the transport key
    (:class:`~megatron.core.mdp.bridge.BridgeBufferKey`) carries ``slice_id``.

    ``item_row_start`` is in the item's own output-row space, so it indexes the
    producer's encoder output directly; the destination-side offset lives in the
    endpoint's :class:`MicrobatchLayout`.
    """

    global_item_id: int
    producer_worker_id: int
    endpoint_rank: int
    owner_worker_id: int
    slice_id: int = 0
    item_row_start: int = 0
    item_rows: int = 0


@dataclass(frozen=True)
class EncoderThdSegment:
    """One vision item inside a producer's encoder THD pack.

    ``payload_row_start/payload_rows`` index the PIXEL buffer only;
    ``output_row_start/output_rows`` index the EMBEDDING/GRADIENT buffer only.
    """

    global_item_id: int
    microbatch_id: int
    sample_id: int
    image_ordinal: int
    payload_row_start: int
    payload_rows: int
    output_row_start: int
    output_rows: int
    grid_thw: tuple


def frame_lengths(segments: Sequence[EncoderThdSegment]) -> tuple:
    """Per-frame THD lengths: each ``(t,h,w)`` contributes ``t`` frames of ``h*w``."""
    lengths = []
    for segment in segments:
        t, h, w = segment.grid_thw
        lengths.extend([h * w] * t)
    return tuple(lengths)


def frame_cu_seqlens(segments: Sequence[EncoderThdSegment]) -> tuple:
    """Cumulative frame boundaries; ``(0,)`` for an empty producer.

    This is vision-encoder THD metadata derived from ``grid_thw`` and is unrelated
    to decoder sample ``cu_seqlens``.
    """
    cu = [0]
    for length in frame_lengths(segments):
        cu.append(cu[-1] + length)
    return tuple(cu)


@dataclass(frozen=True)
class EncoderThdLayout:
    """One producer's encoder THD pack, in that producer's plan order.

    Frame boundaries and totals are derived from ``segments[].grid_thw``, never
    stored: chunking requires rebased local boundaries, so a stored whole-producer
    cumulative array only invites misuse.
    """

    producer_worker_id: int
    segments: tuple

    @property
    def total_payload_rows(self) -> int:
        """Total pixel patch rows across segments."""
        return sum(segment.payload_rows for segment in self.segments)

    @property
    def total_output_rows(self) -> int:
        """Total post-merge vision-token rows across segments."""
        return sum(segment.output_rows for segment in self.segments)


def split_encoder_layout(layout: EncoderThdLayout, *, max_payload_rows) -> tuple:
    """Split a producer layout into chunk sub-layouts at complete segment boundaries.

    Every sub-layout rebases ``payload_row_start/output_row_start`` to 0 so it can
    construct that chunk's ``PackedSeqParams`` directly; the adapter never learns
    that chunking exists. If one segment alone exceeds ``max_payload_rows`` its
    chunk is allowed to exceed the limit. ``max_payload_rows=None`` returns a
    one-element tuple.
    """
    if max_payload_rows is None or not layout.segments:
        return (layout,)
    if max_payload_rows <= 0:
        raise MdpPlanError(
            f"MDP: max_payload_rows={max_payload_rows} violates: None or a positive integer."
        )

    chunks = []
    current = []
    current_rows = 0
    for segment in layout.segments:
        if current and current_rows + segment.payload_rows > max_payload_rows:
            chunks.append(current)
            current = []
            current_rows = 0
        current.append(segment)
        current_rows += segment.payload_rows
    if current:
        chunks.append(current)

    sub_layouts = []
    for chunk in chunks:
        payload_base = chunk[0].payload_row_start
        output_base = chunk[0].output_row_start
        rebased = []
        for segment in chunk:
            rebased.append(
                EncoderThdSegment(
                    global_item_id=segment.global_item_id,
                    microbatch_id=segment.microbatch_id,
                    sample_id=segment.sample_id,
                    image_ordinal=segment.image_ordinal,
                    payload_row_start=segment.payload_row_start - payload_base,
                    payload_rows=segment.payload_rows,
                    output_row_start=segment.output_row_start - output_base,
                    output_rows=segment.output_rows,
                    grid_thw=segment.grid_thw,
                )
            )
        sub_layouts.append(
            EncoderThdLayout(producer_worker_id=layout.producer_worker_id, segments=tuple(rebased))
        )
    return tuple(sub_layouts)


@dataclass(frozen=True)
class LayoutSegment:
    """One route slice's row interval inside a microbatch's endpoint leaf.

    ``slice_id`` pairs this segment with its :class:`RouteSlice`; ``output_rows``
    is that slice's row count, not necessarily the whole item's.
    """

    global_item_id: int
    leaf_row_start: int
    output_rows: int
    slice_id: int = 0


@dataclass(frozen=True)
class MicrobatchLayout:
    """One endpoint's leaf layout for one decoder microbatch.

    At CP=1 there is one layout per microbatch, ordered by
    ``(sample_id, image_ordinal)`` — never by LPT assignment order.

    At CP>1 there is one layout per ``(microbatch_id, endpoint_rank)``, and the
    order is the endpoint's **rank-local** row order. That is the order the
    decoder's image-token mask sees after the CP split, and ``masked_scatter``
    consumes the leaf in exactly that order. It coincides with
    ``(sample_id, image_ordinal)`` at CP=1.

    ``text_only`` means this microbatch has no vision rows *for this endpoint*.
    """

    microbatch_id: int
    text_only: bool
    total_output_rows: int
    segments: tuple
    endpoint_rank: int = -1


def compute_plan_digest(
    schema_version: int,
    capacity_policy: RowCapacityPolicy,
    entries: Sequence[tuple],
    cp_size: int = 1,
) -> bytes:
    """Digest of the minimal sufficient set with fixed-width packing plus blake2b.

    ``entries`` are 12-int tuples in ascending ``(global_item_id, slice_id)``
    order: ``(global_item_id, slice_id, producer_worker_id, order_in_producer,
    endpoint_rank, item_row_start, item_rows, payload_rows, output_rows, grid_t,
    grid_h, grid_w)``. ``hash()`` and pickle are forbidden: they vary across
    Python environments and would produce false positives in the cross-rank
    consistency check.

    The slice fields are in the digest deliberately. At decoder CP>1 the split
    is derived arithmetic, and a member that derives it differently would
    otherwise post a mismatched ``all_to_all_single`` — a hang rather than a
    diagnosable error.
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(
        struct.pack("<3q", schema_version, capacity_policy.alignment_rows, cp_size)
    )
    for entry in entries:
        hasher.update(struct.pack("<12q", *entry))
    return hasher.digest()


@dataclass(frozen=True)
class MdpBatchPlan:
    """The immutable per-iteration plan: one source of truth for pixel dispatch,
    embedding return, and reverse gradient routing.

    Indexes are built once at construction; queries never scan all routes.
    """

    schema_version: int
    iteration: int
    outer_dp_rank: int
    capacity_policy: RowCapacityPolicy
    routes: tuple
    layouts: tuple
    encoder_layouts: tuple
    digest: bytes
    _routes_by_producer: dict = field(init=False, repr=False, compare=False)
    _routes_by_endpoint: dict = field(init=False, repr=False, compare=False)
    _routes_by_item: dict = field(init=False, repr=False, compare=False)
    _encoder_layout_by_producer: dict = field(init=False, repr=False, compare=False)
    _layout_by_microbatch: dict = field(init=False, repr=False, compare=False)
    _segment_by_item: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        routes_by_producer = {}
        routes_by_endpoint = {}
        routes_by_item = {}
        seen_slices = set()
        for route in self.routes:
            key = (route.global_item_id, route.slice_id)
            if key in seen_slices:
                raise MdpPlanError(
                    f"MDP: route {key} violates: one route per (item, slice_id)."
                )
            seen_slices.add(key)
            routes_by_producer.setdefault(route.producer_worker_id, []).append(route)
            routes_by_endpoint.setdefault(route.endpoint_rank, []).append(route)
            routes_by_item.setdefault(route.global_item_id, []).append(route)
        encoder_layout_by_producer = {}
        segment_by_item = {}
        for layout in self.encoder_layouts:
            if layout.producer_worker_id in encoder_layout_by_producer:
                raise MdpPlanError(
                    f"MDP: producer_worker_id={layout.producer_worker_id} violates: one "
                    "encoder layout per producer."
                )
            encoder_layout_by_producer[layout.producer_worker_id] = layout
            for segment in layout.segments:
                if segment.global_item_id in segment_by_item:
                    raise MdpPlanError(
                        f"MDP: global_item_id={segment.global_item_id} violates: every "
                        "item is assigned exactly once."
                    )
                segment_by_item[segment.global_item_id] = segment
        layout_by_microbatch = {}
        for layout in self.layouts:
            key = (layout.microbatch_id, layout.endpoint_rank)
            if key in layout_by_microbatch:
                raise MdpPlanError(
                    f"MDP: (microbatch_id, endpoint_rank)={key} violates: one layout "
                    "per microbatch per endpoint."
                )
            layout_by_microbatch[key] = layout
        object.__setattr__(
            self, "_routes_by_producer", {k: tuple(v) for k, v in routes_by_producer.items()}
        )
        object.__setattr__(
            self, "_routes_by_endpoint", {k: tuple(v) for k, v in routes_by_endpoint.items()}
        )
        object.__setattr__(
            self,
            "_routes_by_item",
            {k: tuple(sorted(v, key=lambda r: r.slice_id)) for k, v in routes_by_item.items()},
        )
        object.__setattr__(self, "_encoder_layout_by_producer", encoder_layout_by_producer)
        object.__setattr__(self, "_layout_by_microbatch", layout_by_microbatch)
        object.__setattr__(self, "_segment_by_item", segment_by_item)

    def routes_for_producer(self, worker_id: int) -> Sequence[RouteSlice]:
        """Routes produced by one logical worker."""
        return self._routes_by_producer.get(worker_id, ())

    def routes_for_endpoint(self, rank: int) -> Sequence[RouteSlice]:
        """Route slices received by one endpoint rank."""
        return self._routes_by_endpoint.get(rank, ())

    def routes_for_item(self, item_id: int) -> Sequence[RouteSlice]:
        """One item's route slices in ascending ``slice_id`` order.

        One entry at CP=1; at CP>1 one per endpoint run of the item's rows.
        """
        return self._routes_by_item.get(item_id, ())

    def layouts_for_endpoint(self, rank: int) -> Sequence["MicrobatchLayout"]:
        """This endpoint's leaf layouts, in ascending ``microbatch_id`` order.

        The runtime builds leaves only for its own endpoint; at CP>1 iterating
        ``plan.layouts`` would allocate every CP peer's leaf on every rank.
        """
        return tuple(
            layout
            for layout in self.layouts
            if layout.endpoint_rank == rank
        )

    def encoder_layout_for_producer(self, worker_id: int) -> EncoderThdLayout:
        """One producer's encoder THD layout; empty layout if it has no work."""
        layout = self._encoder_layout_by_producer.get(worker_id)
        if layout is None:
            return EncoderThdLayout(producer_worker_id=worker_id, segments=())
        return layout

    def layout_for_microbatch(
        self, mb_id: int, endpoint_rank: Optional[int] = None
    ) -> MicrobatchLayout:
        """One endpoint's leaf layout for one decoder microbatch.

        ``endpoint_rank`` may be omitted only when the microbatch has exactly one
        layout, which is always the case at CP=1. Omitting it at CP>1 is a
        programming error, not a defaulted-to-cp0 read.
        """
        if endpoint_rank is None:
            matches = [
                layout for layout in self.layouts if layout.microbatch_id == mb_id
            ]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise MdpPlanError(
                    f"MDP: microbatch_id={mb_id} violates: microbatch is part of "
                    "this plan."
                )
            raise MdpPlanError(
                f"MDP: microbatch_id={mb_id} has {len(matches)} endpoint layouts "
                "violates: endpoint_rank is required when the decoder is "
                "context-parallel."
            )
        try:
            return self._layout_by_microbatch[(mb_id, endpoint_rank)]
        except KeyError:
            raise MdpPlanError(
                f"MDP: (microbatch_id={mb_id}, endpoint_rank={endpoint_rank}) violates: "
                "the pair is part of this plan."
            ) from None

    def segment_for_item(self, item_id: int) -> EncoderThdSegment:
        """Build-time dictionary lookup; never a linear scan across routes."""
        try:
            return self._segment_by_item[item_id]
        except KeyError:
            raise MdpPlanError(
                f"MDP: global_item_id={item_id} violates: item is part of this plan."
            ) from None
