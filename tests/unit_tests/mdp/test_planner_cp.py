# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Planner tests for decoder context parallelism. Pure compute, no CUDA.

These cover the plan-shaped consequences of the zigzag split that
``test_cp_partition.py`` verifies arithmetically: how many route slices an item
produces, which endpoint each lands on, that a rank's leaf rows come out in
rank-local order, and that a CP rank owning nothing is a normal state.
"""

import pytest

from megatron.core.mdp.cp_partition import split_item
from megatron.core.mdp.errors import MdpPlanError
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankView

# One planning group at tp=1, cp=CP, pp=2: ranks are ordered cp-fastest, so the
# decoder endpoints are the first CP entries. Mirrors what build_rank_map emits.
MERGE = 2


def _view(cp_size, pp=2):
    group = tuple(range(cp_size * pp))
    return MdpRankView(
        global_rank=0,
        outer_dp_rank=0,
        lane_id=0,
        my_worker_id=0,
        endpoint_rank=group[0],
        planning_group_ranks=group,
        worker_ids=tuple(range(len(group))),
        decoder_endpoint_ranks=group[:cp_size],
        my_cp_rank=0,
        my_pp_rank=0,
    )


def _descriptor(
    item_id,
    *,
    offset,
    sample_padded_start,
    sample_padded_len,
    grid=(1, 4, 4),
    mb=0,
    sample=0,
    ordinal=0,
    cost=10,
):
    t, h, w = grid
    return VisionDescriptor(
        global_item_id=item_id,
        sample_id=sample,
        image_ordinal=ordinal,
        owner_dp_lane=0,
        microbatch_id=mb,
        estimated_cost_units=cost,
        payload_rows=t * h * w,
        output_rows=t * (h // MERGE) * (w // MERGE),
        grid_thw=grid,
        owner_worker_id=0,
        sample_padded_start=sample_padded_start,
        sample_padded_len=sample_padded_len,
        decoder_offset_in_sample=offset,
    )


def _planner(cp_size, pp=2):
    return MdpPlanner(
        _view(cp_size, pp),
        locality_slack_permille=10,
        capacity_policy=RowCapacityPolicy(),
    )


def test_cp1_plan_is_byte_identical_to_a_plan_with_no_span_data():
    """The CP=1 path must not depend on the new descriptor columns at all."""
    with_span = _descriptor(0, offset=8, sample_padded_start=0, sample_padded_len=64)
    without_span = VisionDescriptor(
        global_item_id=0,
        sample_id=0,
        image_ordinal=0,
        owner_dp_lane=0,
        microbatch_id=0,
        estimated_cost_units=10,
        payload_rows=16,
        output_rows=4,
        grid_thw=(1, 4, 4),
        owner_worker_id=0,
    )
    a = _planner(1).build_plan(0, [with_span], [0])
    b = _planner(1).build_plan(0, [without_span], [0])
    assert a.digest == b.digest
    assert a.routes == b.routes
    assert len(a.routes) == 1
    assert a.routes[0].slice_id == 0
    assert a.routes[0].item_row_start == 0
    assert a.routes[0].item_rows == 4


def test_an_item_inside_one_chunk_produces_one_slice():
    # cp=2 over a 64-row sample: chunks of 16. Rows [16, 20) sit inside chunk 1.
    descriptor = _descriptor(0, offset=16, sample_padded_start=0, sample_padded_len=64)
    plan = _planner(2).build_plan(0, [descriptor], [0])
    assert len(plan.routes) == 1
    assert plan.routes[0].endpoint_rank == 1  # chunk 1 -> cp_rank 1 -> group[1]
    assert plan.routes[0].item_rows == 4


def test_an_item_spanning_every_chunk_produces_2cp_minus_1_slices():
    # A 4x16 grid is 32 post-merge rows, exactly one 32-row sample at cp=2:
    # chunks of 8 -> ranks 0,1,1,0, and the middle pair fuses.
    descriptor = _descriptor(
        0, offset=0, sample_padded_start=0, sample_padded_len=32, grid=(2, 8, 8)
    )
    assert descriptor.output_rows == 32
    plan = _planner(2).build_plan(0, [descriptor], [0])
    assert len(plan.routes) == 2 * 2 - 1
    assert [r.endpoint_rank for r in plan.routes] == [0, 1, 0]
    assert [(r.item_row_start, r.item_rows) for r in plan.routes] == [
        (0, 8),
        (8, 16),
        (24, 8),
    ]
    # Two disjoint runs of one item on rank 0 -- distinct slice_ids, which is
    # what keeps their bridge buffer keys apart.
    on_rank0 = [r for r in plan.routes if r.endpoint_rank == 0]
    assert len(on_rank0) == 2
    assert len({r.slice_id for r in on_rank0}) == 2


def test_leaf_segments_are_ordered_by_rank_local_row():
    """The leaf must be consumable by masked_scatter over the local mask.

    Two items in one sample, the second earlier in rank-local order than part of
    the first, so a naive (sample_id, image_ordinal) sort would be wrong.
    """
    cp_size = 2
    length = 64
    descriptors = [
        _descriptor(
            0, offset=0, sample_padded_start=0, sample_padded_len=length,
            grid=(1, 8, 8), ordinal=0,
        ),  # 16 rows, offsets [0, 16)
        _descriptor(
            1, offset=48, sample_padded_start=0, sample_padded_len=length,
            grid=(1, 8, 8), ordinal=1,
        ),  # 16 rows, offsets [48, 64)
    ]
    plan = _planner(cp_size).build_plan(0, descriptors, [0])
    for endpoint in (0, 1):
        layout = plan.layout_for_microbatch(0, endpoint)
        # Recompute each segment's true rank-local start and assert the layout
        # is sorted by it, gapless from 0.
        local_rows = []
        for segment in layout.segments:
            descriptor = next(
                d for d in descriptors if d.global_item_id == segment.global_item_id
            )
            interval = split_item(
                offset_in_sample=descriptor.decoder_offset_in_sample,
                output_rows=descriptor.output_rows,
                sample_padded_start=descriptor.sample_padded_start,
                sample_padded_len=descriptor.sample_padded_len,
                cp_size=cp_size,
            )[segment.slice_id]
            local_rows.append(interval.local_row_start)
        assert local_rows == sorted(local_rows)
        assert len(set(local_rows)) == len(local_rows)


def test_a_cp_rank_owning_no_rows_gets_a_text_only_layout():
    """4.8% of (microbatch, cp_rank) pairs are empty; it must not be an error."""
    # cp=4 over a 64-row sample: chunks of 8. Rows [8, 12) are entirely inside
    # chunk 1, so only cp_rank 1 owns anything.
    descriptor = _descriptor(0, offset=8, sample_padded_start=0, sample_padded_len=64)
    plan = _planner(4).build_plan(0, [descriptor], [0])
    owners = {r.endpoint_rank for r in plan.routes}
    assert owners == {1}
    for endpoint in (0, 2, 3):
        layout = plan.layout_for_microbatch(0, endpoint)
        assert layout.text_only
        assert layout.segments == ()
        assert layout.total_output_rows == 0
    assert not plan.layout_for_microbatch(0, 1).text_only


def test_every_endpoint_gets_a_layout_for_every_microbatch():
    descriptor = _descriptor(0, offset=0, sample_padded_start=0, sample_padded_len=32)
    plan = _planner(2).build_plan(0, [descriptor], [0, 1])
    assert len(plan.layouts) == 2 * 2  # (mb0, mb1) x (endpoint 0, endpoint 1)
    for mb_id in (0, 1):
        for endpoint in (0, 1):
            assert plan.layout_for_microbatch(mb_id, endpoint) is not None
    # Ambiguous lookup must raise rather than silently pick cp0.
    with pytest.raises(MdpPlanError, match="endpoint_rank is required"):
        plan.layout_for_microbatch(0)


def test_routes_and_layouts_agree_slice_for_slice():
    descriptors = [
        _descriptor(
            0, offset=0, sample_padded_start=0, sample_padded_len=32, grid=(2, 8, 8)
        ),
        _descriptor(
            1, offset=8, sample_padded_start=32, sample_padded_len=32, sample=1
        ),
    ]
    plan = _planner(2).build_plan(0, descriptors, [0])
    route_slices = {(r.global_item_id, r.slice_id) for r in plan.routes}
    layout_slices = {
        (s.global_item_id, s.slice_id)
        for layout in plan.layouts
        for s in layout.segments
    }
    assert route_slices == layout_slices
    # Row conservation: each item's slices tile its output rows exactly.
    for descriptor in descriptors:
        rows = sum(
            r.item_rows
            for r in plan.routes_for_item(descriptor.global_item_id)
        )
        assert rows == descriptor.output_rows


def test_plans_are_bit_identical_across_builds_at_cp4():
    descriptors = [
        _descriptor(
            i,
            offset=(i * 4) % 48,
            sample_padded_start=0,
            sample_padded_len=64,
            ordinal=i,
            cost=10 + (i * 7) % 5,
        )
        for i in range(4)
    ]
    a = _planner(4).build_plan(3, descriptors, [0])
    b = _planner(4).build_plan(3, descriptors, [0])
    assert a.digest == b.digest
    assert a.routes == b.routes
    assert a.layouts == b.layouts


def test_an_item_straddling_its_sample_is_rejected_as_a_plan_error():
    # 4 rows starting at offset 62 of a 64-row sample runs past its end.
    descriptor = _descriptor(0, offset=62, sample_padded_start=0, sample_padded_len=64)
    with pytest.raises(MdpPlanError, match="cannot be split"):
        _planner(2).build_plan(0, [descriptor], [0])


def test_locality_preference_does_not_collapse_to_worker_zero():
    """At encoder_cp >= cp the old mapping preferred worker 0 for every item.

    EMBEDDING is sourced by a worker's LEAD rank, so a self-edge exists only
    when that lead is the endpoint. Mapping endpoints through
    `index // ranks_per_worker` sent every endpoint to worker 0 once
    ranks_per_worker >= cp -- a constant bias with no locality behind it.
    """
    from megatron.core.mdp.plan import RowCapacityPolicy
    from megatron.core.mdp.planner import MdpPlanner

    # cp=2, pp=2, encoder_cp=2: group of 4, workers {0: ranks 0-1, 1: ranks 2-3},
    # endpoints = group[:2] = ranks 0 and 1. Only rank 0 is a worker lead.
    group = (0, 1, 2, 3)
    view = MdpRankView(
        global_rank=0, outer_dp_rank=0, lane_id=0, my_worker_id=0,
        endpoint_rank=0, planning_group_ranks=group, worker_ids=(0, 1),
        decoder_endpoint_ranks=group[:2], my_cp_rank=0, my_pp_rank=0,
        my_encoder_cp_rank=0,
    )
    planner = MdpPlanner(
        view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy()
    )
    # An item lying entirely in a chunk owned by cp_rank 1 -> endpoint group[1],
    # which is NOT a worker lead, so no self-edge is available.
    only_on_endpoint1 = _descriptor(
        0, offset=16, sample_padded_start=0, sample_padded_len=64
    )
    assert planner._preferred_endpoint_worker(only_on_endpoint1) == -1, (
        "an endpoint that is not a worker lead offers no self-edge; the planner "
        "must express that as no preference rather than defaulting to worker 0"
    )
    # An item on cp_rank 0 -> endpoint group[0] == worker 0's lead: real self-edge.
    on_endpoint0 = _descriptor(
        0, offset=0, sample_padded_start=0, sample_padded_len=64
    )
    assert planner._preferred_endpoint_worker(on_endpoint0) == 0
