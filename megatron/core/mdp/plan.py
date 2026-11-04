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

PLAN_SCHEMA_VERSION = 7


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
    """One routed slice: which item rows, which logical worker produces them,
    which endpoint rank receives them. Item offsets are looked up in the encoder
    layout.

    ``owner_worker_id`` is the logical worker holding the item's pixels at
    dispatch time and is always the PIXEL phase source; PIXEL moves whole items
    (slice 0 stands for the item), EMBEDDING and GRADIENT move slices.

    Without decoder context parallelism every item is one slice (``slice_id`` 0,
    the whole item) to the planning group's endpoint. Under decoder CP an item is
    split into the runs of rows each CP rank owns: slice ``slice_id`` covers item
    rows ``[item_row_start, item_row_start + item_rows)`` and goes straight to the
    PP0 TP source of the owning CP rank. ``item_rows=None`` means the whole item.
    """

    global_item_id: int
    producer_worker_id: int
    endpoint_rank: int
    owner_worker_id: int
    slice_id: int = 0
    item_row_start: int = 0
    item_rows: Optional[int] = None


@dataclass(frozen=True)
class EncoderThdSegment:
    """One vision item inside a producer's encoder THD pack.

    ``payload_row_start/payload_rows`` index the PIXEL buffer only;
    ``output_row_start/output_rows`` index the EMBEDDING/GRADIENT buffer only.
    ``shape_key`` is carried verbatim from the descriptor so the producer can
    rebuild encoder metadata for items whose pixels came from another rank; core
    never interprets it.
    """

    global_item_id: int
    microbatch_id: int
    sample_id: int
    image_ordinal: int
    payload_row_start: int
    payload_rows: int
    output_row_start: int
    output_rows: int
    shape_key: tuple


@dataclass(frozen=True)
class EncoderThdLayout:
    """One producer's encoder THD pack, in that producer's plan order.

    Totals are derived from the segments, never stored: chunking requires rebased
    local boundaries, so a stored whole-producer cumulative array only invites
    misuse. Attention sub-sequence boundaries inside the pack are the adapter's
    business -- it derives them from ``segments[].shape_key`` inside ``encode``.
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
                    shape_key=segment.shape_key,
                )
            )
        sub_layouts.append(
            EncoderThdLayout(producer_worker_id=layout.producer_worker_id, segments=tuple(rebased))
        )
    return tuple(sub_layouts)


@dataclass(frozen=True)
class LayoutSegment:
    """One routed slice's row interval inside an endpoint's microbatch leaf.

    ``output_rows`` is the slice's row count (the whole item without decoder CP).
    """

    global_item_id: int
    leaf_row_start: int
    output_rows: int
    slice_id: int = 0


@dataclass(frozen=True)
class MicrobatchLayout:
    """One endpoint's leaf layout for one decoder microbatch.

    Segments follow the endpoint's local decoder order: ``(sample_id,
    image_ordinal)`` without decoder CP, ascending decoder position under it --
    never LPT assignment order. Without decoder CP every microbatch has one
    layout at the group endpoint (``text_only`` when it carries no item). Under
    decoder CP a vision microbatch has one layout per CP rank that owns rows of
    it, at that rank's PP0 TP source; a rank owning none has no layout.
    ``endpoint_rank=None`` (hand-built layouts) means the group endpoint.
    """

    microbatch_id: int
    text_only: bool
    total_output_rows: int
    segments: tuple
    endpoint_rank: Optional[int] = None


#: Digest slots per entry that precede the model's ``shape_key``.
DIGEST_FIXED_SLOTS = 9


def compute_plan_digest(
    schema_version: int,
    capacity_policy: RowCapacityPolicy,
    entries: Sequence[tuple],
    *,
    shape_key_width: int,
    cp_size: int = 1,
    cp_layout_code: int = 0,
) -> bytes:
    """Digest of the minimal sufficient set with fixed-width packing plus blake2b.

    ``entries`` are ``DIGEST_FIXED_SLOTS + shape_key_width`` int tuples, one per
    routed slice in ascending ``(global_item_id, slice_id)`` order:
    ``(global_item_id, slice_id, producer_worker_id, order_in_producer,
    endpoint_rank, item_row_start, item_rows, payload_rows, output_rows,
    *shape_key)``. The decoder CP size and layout are folded into the header:
    every rank derives the slice split locally, so a divergence there must fail
    the digest check rather than hang the bridge's ``all_to_all_single``.
    ``shape_key_width`` is folded into the header, so two models with different
    shape-key widths can never produce the same digest for structurally different
    plans. ``hash()`` and pickle are forbidden: they vary across Python
    environments and would produce false positives in the cross-rank consistency
    check.
    """
    entry_width = DIGEST_FIXED_SLOTS + shape_key_width
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(
        struct.pack(
            "<5q",
            schema_version,
            capacity_policy.alignment_rows,
            shape_key_width,
            cp_size,
            cp_layout_code,
        )
    )
    packer = struct.Struct(f"<{entry_width}q")
    for entry in entries:
        if len(entry) != entry_width:
            raise MdpPlanError(
                f"MDP: digest entry {entry} violates: exactly {entry_width} slots "
                f"(DIGEST_FIXED_SLOTS + shape_key_width={shape_key_width})."
            )
        hasher.update(packer.pack(*entry))
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
    _encoder_layout_by_producer: dict = field(init=False, repr=False, compare=False)
    _layout_by_microbatch: dict = field(init=False, repr=False, compare=False)
    _segment_by_item: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        routes_by_producer = {}
        routes_by_endpoint = {}
        for route in self.routes:
            routes_by_producer.setdefault(route.producer_worker_id, []).append(route)
            routes_by_endpoint.setdefault(route.endpoint_rank, []).append(route)
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
                    f"MDP: microbatch_id={layout.microbatch_id} violates: one layout per "
                    f"microbatch per endpoint (endpoint_rank={layout.endpoint_rank})."
                )
            layout_by_microbatch[key] = layout
        object.__setattr__(
            self, "_routes_by_producer", {k: tuple(v) for k, v in routes_by_producer.items()}
        )
        object.__setattr__(
            self, "_routes_by_endpoint", {k: tuple(v) for k, v in routes_by_endpoint.items()}
        )
        object.__setattr__(self, "_encoder_layout_by_producer", encoder_layout_by_producer)
        object.__setattr__(self, "_layout_by_microbatch", layout_by_microbatch)
        object.__setattr__(self, "_segment_by_item", segment_by_item)

    def routes_for_producer(self, worker_id: int) -> Sequence[RouteSlice]:
        """Routes produced by one logical worker."""
        return self._routes_by_producer.get(worker_id, ())

    def routes_for_endpoint(self, rank: int) -> Sequence[RouteSlice]:
        """Routes received by one endpoint rank."""
        return self._routes_by_endpoint.get(rank, ())

    def encoder_layout_for_producer(self, worker_id: int) -> EncoderThdLayout:
        """One producer's encoder THD layout; empty layout if it has no work."""
        layout = self._encoder_layout_by_producer.get(worker_id)
        if layout is None:
            return EncoderThdLayout(producer_worker_id=worker_id, segments=())
        return layout

    def layout_for_microbatch(
        self, mb_id: int, endpoint_rank: Optional[int] = None
    ) -> MicrobatchLayout:
        """The leaf layout of one decoder microbatch at one endpoint.

        ``endpoint_rank=None`` addresses the microbatch's only layout (the
        single-endpoint plan without decoder CP); it is an error when the
        microbatch is split across several endpoints.
        """
        if endpoint_rank is None:
            matches = [
                layout for (mb, _), layout in self._layout_by_microbatch.items() if mb == mb_id
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise MdpPlanError(
                    f"MDP: microbatch_id={mb_id} violates: one layout when no endpoint is "
                    f"named (it has {len(matches)}, one per decoder CP rank)."
                )
        try:
            return self._layout_by_microbatch[(mb_id, endpoint_rank)]
        except KeyError:
            raise MdpPlanError(
                f"MDP: microbatch_id={mb_id} violates: microbatch has a layout at "
                f"endpoint_rank={endpoint_rank} in this plan."
            ) from None

    def layouts_for_endpoint(self, endpoint_rank: Optional[int]) -> Sequence[MicrobatchLayout]:
        """The layouts one endpoint receives, in microbatch order."""
        return tuple(
            layout for layout in self.layouts if layout.endpoint_rank == endpoint_rank
        )

    def segment_for_item(self, item_id: int) -> EncoderThdSegment:
        """Build-time dictionary lookup; never a linear scan across routes."""
        try:
            return self._segment_by_item[item_id]
        except KeyError:
            raise MdpPlanError(
                f"MDP: global_item_id={item_id} violates: item is part of this plan."
            ) from None
