# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-compute tests for the deterministic LPT planner. No distributed state,
no CUDA."""

import pytest

from megatron.core.mdp.errors import MdpPlanError
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner, assert_consistent_plan
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankView


def _view(worker_ids=(0, 1), group=(0, 4), endpoint=0):
    return MdpRankView(
        global_rank=0,
        outer_dp_rank=0,
        lane_id=0,
        my_worker_id=0,
        endpoint_rank=endpoint,
        planning_group_ranks=group,
        worker_ids=worker_ids,
    )


def _descriptor(item_id, cost, mb=0, sample=None, ordinal=0, grid=(1, 4, 4), lane=0):
    t, h, w = grid
    return VisionDescriptor(
        global_item_id=item_id,
        sample_id=sample if sample is not None else item_id,
        image_ordinal=ordinal,
        owner_dp_lane=lane,
        microbatch_id=mb,
        estimated_cost_units=cost,
        payload_rows=t * h * w,
        output_rows=t * (h // 2) * (w // 2),
        grid_thw=grid,
    )


def _planner(slack=10, alignment=1, view=None):
    return MdpPlanner(
        view or _view(),
        locality_slack_permille=slack,
        capacity_policy=RowCapacityPolicy(alignment_rows=alignment),
    )


def _assignment(plan):
    return {r.global_item_id: r.producer_worker_id for r in plan.routes}


def test_plans_are_bit_identical_across_builds():
    descriptors = [_descriptor(i, cost=10 + (i * 7) % 5) for i in range(9)]
    a = _planner().build_plan(3, descriptors, [0])
    b = _planner().build_plan(3, descriptors, [0])
    assert a.digest == b.digest
    assert a.routes == b.routes
    assert a.encoder_layouts == b.encoder_layouts


def test_lpt_balances_by_cost():
    # costs 8,7,3,2 over two workers: LPT yields {8,2} and {7,3}.
    descriptors = [
        _descriptor(0, cost=8),
        _descriptor(1, cost=7),
        _descriptor(2, cost=3),
        _descriptor(3, cost=2),
    ]
    assignment = _assignment(_planner(slack=0).build_plan(0, descriptors, [0]))
    loads = {0: 0, 1: 0}
    for item, worker in assignment.items():
        loads[worker] += descriptors[item].estimated_cost_units
    assert sorted(loads.values()) == [10, 10]


def test_locality_prefers_the_endpoint_worker_within_slack():
    # LPT places cost 10 on worker 0 (endpoint) and cost 9 on worker 1. For
    # the cost-100 item the loads (10 vs 9) are near-equal relative to the
    # item: with slack 10 permille (1000*10 <= 1000*9 + 10*100) worker 0 is
    # eligible and preferred for hosting the endpoint; with zero slack only
    # the min-load worker 1 is eligible.
    descriptors = [
        _descriptor(0, cost=30),
        _descriptor(1, cost=29),
        _descriptor(2, cost=25),
    ]
    # LPT order 30 -> worker 0 (endpoint), 29 -> worker 1. For the cost-25
    # item, worker 0 is eligible iff 1000*30 <= 1000*29 + slack*25, i.e.
    # slack >= 40; the endpoint preference then keeps the item local.
    with_slack = _assignment(_planner(slack=40).build_plan(0, descriptors, [0]))
    assert with_slack[2] == 0
    without = _assignment(_planner(slack=0).build_plan(0, descriptors, [0]))
    assert without[2] == 1


def test_endpoint_layout_follows_sample_order_not_lpt_order():
    # LPT visits high-cost items first (item 2), but the leaf layout must be
    # ordered by (sample_id, image_ordinal).
    descriptors = [
        _descriptor(0, cost=1, sample=0),
        _descriptor(1, cost=2, sample=1),
        _descriptor(2, cost=50, sample=2),
    ]
    plan = _planner().build_plan(0, descriptors, [0])
    layout = plan.layout_for_microbatch(0)
    assert [s.global_item_id for s in layout.segments] == [0, 1, 2]
    starts = [s.leaf_row_start for s in layout.segments]
    assert starts == [0, 4, 8]
    assert layout.total_output_rows == 12


def test_text_only_microbatch_gets_empty_layout():
    descriptors = [_descriptor(0, cost=5, mb=1)]
    plan = _planner().build_plan(0, descriptors, [0, 1])
    assert plan.layout_for_microbatch(0).text_only
    assert plan.layout_for_microbatch(0).segments == ()
    assert not plan.layout_for_microbatch(1).text_only


def test_producer_layout_offsets_follow_capacity_policy():
    descriptors = [_descriptor(0, cost=10), _descriptor(1, cost=9)]
    single_worker = _view(worker_ids=(0,), group=(0,))
    plan = _planner(alignment=16, view=single_worker).build_plan(0, descriptors, [0])
    layout = plan.encoder_layout_for_producer(0)
    assert len(layout.segments) == 2
    first, second = layout.segments
    # 16 payload rows align to 16; 4 output rows align to 16.
    assert second.payload_row_start == 16
    assert second.output_row_start == 16


def test_cross_microbatch_packing_in_one_producer():
    # One producer's THD pack may contain items from several decoder
    # microbatches (design doc 8.6).
    descriptors = [_descriptor(0, cost=5, mb=0), _descriptor(1, cost=5, mb=1)]
    single_worker = _view(worker_ids=(0,), group=(0,))
    plan = _planner(view=single_worker).build_plan(0, descriptors, [0, 1])
    layout = plan.encoder_layout_for_producer(0)
    assert {s.microbatch_id for s in layout.segments} == {0, 1}
    # ...but each microbatch layout only consumes its own item.
    assert [s.global_item_id for s in plan.layout_for_microbatch(0).segments] == [0]
    assert [s.global_item_id for s in plan.layout_for_microbatch(1).segments] == [1]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: [d[0], d[0]], "unique"),
        (lambda d: [_descriptor(0, cost=-1)], "non-negative"),
        (
            lambda d: [
                VisionDescriptor(
                    global_item_id=0,
                    sample_id=0,
                    image_ordinal=0,
                    owner_dp_lane=0,
                    microbatch_id=0,
                    estimated_cost_units=1,
                    payload_rows=99,
                    output_rows=4,
                    grid_thw=(1, 4, 4),
                )
            ],
            "t\\*h\\*w",
        ),
        (lambda d: [_descriptor(0, cost=1, mb=5)], "microbatch"),
        (lambda d: [_descriptor(0, cost=1, lane=3)], "outer-DP"),
    ],
)
def test_descriptor_validation(mutate, match):
    descriptors = mutate([_descriptor(0, cost=1)])
    with pytest.raises(MdpPlanError, match=match):
        _planner().build_plan(0, descriptors, [0])


def test_consistency_check_interval_must_be_positive():
    plan = _planner().build_plan(0, [_descriptor(0, cost=1)], [0])
    with pytest.raises(MdpPlanError, match="never be fully disabled"):
        assert_consistent_plan(
            plan, planning_group=None, iteration=0, interval=0
        )
