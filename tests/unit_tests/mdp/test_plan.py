# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-compute tests for the MDP plan data model and digest. No distributed
state, no CUDA."""

import pytest

from megatron.core.mdp.errors import MdpPlanError
from megatron.core.mdp.plan import (
    PLAN_SCHEMA_VERSION,
    EncoderThdLayout,
    EncoderThdSegment,
    LayoutSegment,
    MdpBatchPlan,
    MicrobatchLayout,
    RouteSlice,
    RowCapacityPolicy,
    compute_plan_digest,
    frame_cu_seqlens,
    frame_lengths,
    split_encoder_layout,
)


def _segment(item_id, grid, payload_start=0, output_start=0, mb=0, sample=0, ordinal=0):
    t, h, w = grid
    return EncoderThdSegment(
        global_item_id=item_id,
        microbatch_id=mb,
        sample_id=sample,
        image_ordinal=ordinal,
        payload_row_start=payload_start,
        payload_rows=t * h * w,
        output_row_start=output_start,
        output_rows=t * (h // 2) * (w // 2),
        grid_thw=grid,
    )


# ------------------------- capacity policy -------------------------


def test_capacity_policy_alignment():
    identity = RowCapacityPolicy(alignment_rows=1)
    assert identity.capacity_of(0) == 0
    assert identity.capacity_of(37) == 37
    aligned = RowCapacityPolicy(alignment_rows=16)
    assert aligned.capacity_of(0) == 0
    assert aligned.capacity_of(1) == 16
    assert aligned.capacity_of(16) == 16
    assert aligned.capacity_of(17) == 32
    with pytest.raises(MdpPlanError):
        aligned.capacity_of(-1)


# ------------------------- frame derivation -------------------------


def test_frame_lengths_and_cu_seqlens_from_grid():
    segments = (_segment(0, (2, 4, 6)), _segment(1, (1, 8, 8), payload_start=48))
    # (2,4,6) -> two frames of 24; (1,8,8) -> one frame of 64.
    assert frame_lengths(segments) == (24, 24, 64)
    assert frame_cu_seqlens(segments) == (0, 24, 48, 112)


def test_empty_producer_layout():
    assert frame_lengths(()) == ()
    assert frame_cu_seqlens(()) == (0,)
    layout = EncoderThdLayout(producer_worker_id=0, segments=())
    assert layout.total_payload_rows == 0
    assert layout.total_output_rows == 0


# ------------------------- chunk splitting -------------------------


def _layout():
    segments = []
    payload = output = 0
    for item_id, grid in enumerate(((1, 4, 4), (1, 8, 8), (2, 4, 4), (1, 4, 4))):
        segment = _segment(item_id, grid, payload_start=payload, output_start=output)
        segments.append(segment)
        payload += segment.payload_rows
        output += segment.output_rows
    return EncoderThdLayout(producer_worker_id=3, segments=tuple(segments))


def test_split_none_returns_single_chunk():
    layout = _layout()
    assert split_encoder_layout(layout, max_payload_rows=None) == (layout,)


def test_split_at_item_boundaries_with_rebased_offsets():
    layout = _layout()  # payload rows: 16, 64, 32, 16
    chunks = split_encoder_layout(layout, max_payload_rows=80)
    assert [len(c.segments) for c in chunks] == [2, 2]
    for chunk in chunks:
        assert chunk.producer_worker_id == 3
        assert chunk.segments[0].payload_row_start == 0
        assert chunk.segments[0].output_row_start == 0
        for prev, cur in zip(chunk.segments, chunk.segments[1:]):
            assert cur.payload_row_start == prev.payload_row_start + prev.payload_rows
            assert cur.output_row_start == prev.output_row_start + prev.output_rows
    # Coverage: chunk contents equal the original segment sequence.
    flat = [s.global_item_id for c in chunks for s in c.segments]
    assert flat == [0, 1, 2, 3]


def test_split_allows_single_oversized_item():
    layout = _layout()
    chunks = split_encoder_layout(layout, max_payload_rows=20)
    # The 64-row item exceeds the cap and gets its own oversized chunk.
    assert [c.total_payload_rows for c in chunks] == [16, 64, 32, 16]


def test_split_rejects_nonpositive_cap():
    with pytest.raises(MdpPlanError):
        split_encoder_layout(_layout(), max_payload_rows=0)


# ------------------------- digest -------------------------


def _entry(
    item_id,
    worker=0,
    order=0,
    endpoint=0,
    grid=(1, 4, 4),
    slice_id=0,
    item_row_start=0,
    item_rows=None,
):
    t, h, w = grid
    output_rows = t * (h // 2) * (w // 2)
    return (
        item_id,
        slice_id,
        worker,
        order,
        endpoint,
        item_row_start,
        output_rows if item_rows is None else item_rows,
        t * h * w,
        output_rows,
        t,
        h,
        w,
    )


def test_digest_is_deterministic_and_16_bytes():
    policy = RowCapacityPolicy()
    a = compute_plan_digest(PLAN_SCHEMA_VERSION, policy, [_entry(0), _entry(1, worker=1)])
    b = compute_plan_digest(PLAN_SCHEMA_VERSION, policy, [_entry(0), _entry(1, worker=1)])
    assert a == b
    assert len(a) == 16


def test_digest_covers_the_minimal_sufficient_set():
    policy = RowCapacityPolicy()
    base = compute_plan_digest(PLAN_SCHEMA_VERSION, policy, [_entry(0)])
    # Same payload_rows, different grid -> different frame boundaries -> the
    # digest must change (design doc 7.4).
    same_rows_other_grid = (0, 0, 0, 0, 0, 0, 4, 16, 4, 1, 2, 8)
    assert compute_plan_digest(
        PLAN_SCHEMA_VERSION, policy, [same_rows_other_grid]
    ) != base
    # Worker assignment changes the digest.
    assert compute_plan_digest(PLAN_SCHEMA_VERSION, policy, [_entry(0, worker=1)]) != base
    # The capacity policy is part of the digest.
    assert (
        compute_plan_digest(PLAN_SCHEMA_VERSION, RowCapacityPolicy(16), [_entry(0)]) != base
    )
    # Schema version is part of the digest.
    assert compute_plan_digest(PLAN_SCHEMA_VERSION + 1, policy, [_entry(0)]) != base
    # Decoder-CP slice identity is part of the digest. Without this, two ranks
    # that derive different slice tables would agree on the digest, pass the
    # consistency check, and then hang in all_to_all_single with mismatched
    # split sizes.
    assert (
        compute_plan_digest(PLAN_SCHEMA_VERSION, policy, [_entry(0, item_row_start=2)])
        != base
    )
    assert (
        compute_plan_digest(PLAN_SCHEMA_VERSION, policy, [_entry(0, item_rows=1)]) != base
    )
    assert (
        compute_plan_digest(PLAN_SCHEMA_VERSION, policy, [_entry(0, slice_id=1)]) != base
    )
    # The CP topology itself is in the header, so a cp mismatch is diagnosed
    # even when every per-slice record happens to coincide.
    assert compute_plan_digest(PLAN_SCHEMA_VERSION, policy, [_entry(0)], cp_size=2) != base
    # Encoder CP likewise. Without it an encoder_cp=1 and an encoder_cp=2 plan
    # over the same descriptors hash identically -- with few descriptors LPT
    # picks worker 0 either way, while worker_ranks(0, 0) resolves to (0,) vs
    # (0, 1).
    assert (
        compute_plan_digest(
            PLAN_SCHEMA_VERSION, policy, [_entry(0)], ranks_per_worker=2
        )
        != base
    )


# ------------------------- batch plan indexes -------------------------


def _plan():
    routes = (
        RouteSlice(
            global_item_id=0,
            producer_worker_id=1,
            endpoint_rank=0,
            owner_worker_id=0,
            item_rows=16,
        ),
        RouteSlice(
            global_item_id=1,
            producer_worker_id=0,
            endpoint_rank=0,
            owner_worker_id=0,
            item_rows=4,
        ),
    )
    encoder_layouts = (
        EncoderThdLayout(producer_worker_id=0, segments=(_segment(1, (1, 4, 4)),)),
        EncoderThdLayout(producer_worker_id=1, segments=(_segment(0, (1, 8, 8)),)),
    )
    layouts = (
        MicrobatchLayout(
            microbatch_id=0,
            text_only=False,
            total_output_rows=20,
            segments=(
                LayoutSegment(global_item_id=0, leaf_row_start=0, output_rows=16),
                LayoutSegment(global_item_id=1, leaf_row_start=16, output_rows=4),
            ),
        ),
        MicrobatchLayout(microbatch_id=1, text_only=True, total_output_rows=0, segments=()),
    )
    return MdpBatchPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        iteration=7,
        outer_dp_rank=0,
        capacity_policy=RowCapacityPolicy(),
        routes=routes,
        layouts=layouts,
        encoder_layouts=encoder_layouts,
        digest=b"\x00" * 16,
    )


def test_plan_indexes_are_dictionaries():
    plan = _plan()
    assert [r.global_item_id for r in plan.routes_for_producer(0)] == [1]
    assert [r.global_item_id for r in plan.routes_for_producer(1)] == [0]
    assert plan.routes_for_producer(9) == ()
    assert len(plan.routes_for_endpoint(0)) == 2
    assert plan.encoder_layout_for_producer(1).segments[0].global_item_id == 0
    empty = plan.encoder_layout_for_producer(5)
    assert empty.segments == () and empty.producer_worker_id == 5
    assert plan.layout_for_microbatch(1).text_only
    assert plan.segment_for_item(0).grid_thw == (1, 8, 8)
    with pytest.raises(MdpPlanError):
        plan.segment_for_item(42)
    with pytest.raises(MdpPlanError):
        plan.layout_for_microbatch(42)


def test_plan_rejects_duplicate_item_assignment():
    duplicated = (
        EncoderThdLayout(producer_worker_id=0, segments=(_segment(0, (1, 4, 4)),)),
        EncoderThdLayout(producer_worker_id=1, segments=(_segment(0, (1, 4, 4)),)),
    )
    with pytest.raises(MdpPlanError, match="exactly once"):
        MdpBatchPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            iteration=0,
            outer_dp_rank=0,
            capacity_policy=RowCapacityPolicy(),
            routes=(),
            layouts=(),
            encoder_layouts=duplicated,
            digest=b"\x00" * 16,
        )
