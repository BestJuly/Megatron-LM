# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-CP slicing in the planner (direct per-CP-rank routing). Pure compute.

Under decoder CP the planner cuts each vision item into the runs of rows each CP
rank owns and routes every run straight to that rank's PP0 TP source. These
tests check the plan against an independent token-level reference: the owner of
every leaf row (``cp_owner_of_position``) and each owner's local scatter order
(ascending decoder position).
"""

import random

import pytest

from megatron.core.mdp.cp_partition import cp_owner_of_position
from megatron.core.mdp.errors import MdpPlanError
from megatron.core.mdp.plan import MdpBatchPlan, RouteSlice, RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner, _validate_plan
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map


def _view(tp, cp, pp=1):
    rank_map = build_rank_map(
        MdpRankSpec(world_size=tp * cp * pp, tp=tp, pp=pp, cp=cp, ep=1, encoder_cp=1)
    )
    return rank_map.view(0)


def _planner(view, cp, mode):
    return MdpPlanner(
        view,
        locality_slack_permille=10,
        capacity_policy=RowCapacityPolicy(),
        shape_key_width=1,
        cp_size=cp,
        cp_partition_mode=mode,
    )


def _pack(rng, cp, mb):
    """One packed microbatch: padded sequences (multiples of 2*cp), image items
    as contiguous slot runs placed anywhere inside their sample."""
    lengths = [2 * cp * rng.randint(2, 8) for _ in range(rng.randint(1, 4))]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    items = []
    for sample, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        cursor = start
        ordinal = 0
        while cursor < end - 1 and rng.random() < 0.7:
            first = rng.randint(cursor, end - 1)
            rows = rng.randint(1, end - first)
            items.append((mb, sample, ordinal, first, rows, start, end - start, cu[-1]))
            cursor = first + rows
            ordinal += 1
    return items


def _descriptors(items):
    out = []
    for item_id, (mb, sample, ordinal, first, rows, s_start, s_len, total) in enumerate(items):
        out.append(
            VisionDescriptor(
                global_item_id=item_id,
                sample_id=sample,
                image_ordinal=ordinal,
                owner_dp_lane=0,
                microbatch_id=mb,
                estimated_cost_units=rows,
                payload_rows=4 * rows,
                output_rows=rows,
                shape_key=(rows,),
                owner_worker_id=0,
                decoder_start=first,
                sample_padded_start=s_start,
                sample_padded_len=s_len,
                microbatch_rows=total,
            )
        )
    return out


@pytest.mark.parametrize("mode", ["zigzag", "contiguous"])
@pytest.mark.parametrize("tp,cp", [(1, 2), (2, 2), (1, 4), (2, 4)])
def test_slices_route_every_row_to_its_owner_in_local_order(mode, tp, cp):
    rng = random.Random(97 * cp + tp + len(mode))
    view = _view(tp, cp)
    owners = view.shard_owners
    assert len(owners) == cp
    for _ in range(20):
        items = _pack(rng, cp, 0) + _pack(rng, cp, 1)
        if not items:
            continue
        descriptors = _descriptors(items)
        plan = _planner(view, cp, mode).build_plan(0, descriptors, [0, 1])
        by_id = {d.global_item_id: d for d in descriptors}

        # Reference: every image slot's owner from token-level ownership.
        expected = {}  # (mb, owner_rank) -> [(position, item, row)]
        for d in descriptors:
            cu = [0]  # contiguous needs only the total; zigzag the sample span
            for row in range(d.output_rows):
                pos = d.decoder_start + row
                if mode == "zigzag":
                    owner = cp_owner_of_position(
                        pos - d.sample_padded_start,
                        cu_seqlens_padded=[0, d.sample_padded_len],
                        total_rows=d.sample_padded_len,
                        cp_size=cp,
                        mode=mode,
                    )
                else:
                    owner = cp_owner_of_position(
                        pos,
                        cu_seqlens_padded=cu,
                        total_rows=d.microbatch_rows,
                        cp_size=cp,
                        mode=mode,
                    )
                expected.setdefault((d.microbatch_id, owners[owner]), []).append(
                    (pos, d.global_item_id, row)
                )

        # Plan: expand each owner's leaf back into (position, item, row).
        routes = {(r.global_item_id, r.slice_id): r for r in plan.routes}
        got = {}
        for layout in plan.layouts:
            if layout.text_only:
                continue
            rows_out = []
            for segment in layout.segments:
                route = routes[(segment.global_item_id, segment.slice_id)]
                assert route.endpoint_rank == layout.endpoint_rank
                d = by_id[segment.global_item_id]
                for k in range(segment.output_rows):
                    row = route.item_row_start + k
                    rows_out.append((d.decoder_start + row, d.global_item_id, row))
            got[(layout.microbatch_id, layout.endpoint_rank)] = rows_out
        assert got == {key: sorted(value) for key, value in expected.items()}
        # Every routed row goes to an owner; every owner's leaf is in local order.
        for rows_out in got.values():
            assert rows_out == sorted(rows_out)
        assert all(r.endpoint_rank in owners for r in plan.routes)


def test_cp1_plan_keeps_the_single_slice_shape():
    rng = random.Random(5)
    view = _view(tp=2, cp=1)
    descriptors = _descriptors(_pack(rng, 1, 0))
    plan = _planner(view, 1, "zigzag").build_plan(0, descriptors, [0])
    assert len(plan.routes) == len(descriptors)
    for route, d in zip(plan.routes, sorted(descriptors, key=lambda x: x.global_item_id)):
        assert route.slice_id == 0 and route.item_row_start == 0
        assert route.item_rows == d.output_rows
        assert route.endpoint_rank == view.endpoint_rank
    (layout,) = plan.layouts
    assert layout.endpoint_rank == view.endpoint_rank
    order = [(d.sample_id, d.image_ordinal) for d in descriptors]
    by_id = {d.global_item_id: d for d in descriptors}
    assert [
        (by_id[s.global_item_id].sample_id, by_id[s.global_item_id].image_ordinal)
        for s in layout.segments
    ] == sorted(order)


def test_rank_without_rows_gets_no_layout_and_text_only_stays_at_endpoint():
    view = _view(tp=1, cp=2)
    # One item entirely inside rank 0's contiguous half; microbatch 1 is text-only.
    d = _descriptors([(0, 0, 0, 1, 3, 0, 16, 16)])
    plan = _planner(view, 2, "contiguous").build_plan(0, d, [0, 1])
    layouts = {(l.microbatch_id, l.endpoint_rank): l for l in plan.layouts}
    assert set(layouts) == {(0, view.shard_owners[0]), (1, view.endpoint_rank)}
    assert layouts[(1, view.endpoint_rank)].text_only
    assert plan.layouts_for_endpoint(view.shard_owners[1]) == ()


def test_split_item_fails_without_its_decoder_span():
    view = _view(tp=1, cp=2)
    (d,) = _descriptors([(0, 0, 0, 2, 4, 0, 16, 16)])
    bad = VisionDescriptor(**{**d.__dict__, "decoder_start": -1})
    with pytest.raises(MdpPlanError, match="decoder span"):
        _planner(view, 2, "zigzag").build_plan(0, [bad], [0])


def test_plan_validation_rejects_gaps_and_overlaps():
    view = _view(tp=1, cp=2)
    descriptors = _descriptors([(0, 0, 0, 6, 4, 0, 16, 16)])  # straddles the CP boundary
    plan = _planner(view, 2, "contiguous").build_plan(0, descriptors, [0])
    assert len(plan.routes) == 2

    def rebuilt(routes):
        return MdpBatchPlan(
            schema_version=plan.schema_version,
            iteration=0,
            outer_dp_rank=0,
            capacity_policy=plan.capacity_policy,
            routes=tuple(routes),
            layouts=plan.layouts,
            encoder_layouts=plan.encoder_layouts,
            digest=plan.digest,
        )

    first, second = plan.routes
    gap = RouteSlice(**{**second.__dict__, "item_row_start": second.item_row_start + 1})
    with pytest.raises(MdpPlanError, match="tile"):
        _validate_plan(rebuilt([first, gap]), view)
    overlap = RouteSlice(**{**second.__dict__, "item_rows": second.item_rows + 1})
    with pytest.raises(MdpPlanError, match="tile"):
        _validate_plan(rebuilt([first, overlap]), view)
    wrong_endpoint = RouteSlice(**{**second.__dict__, "endpoint_rank": first.endpoint_rank})
    with pytest.raises(MdpPlanError, match="endpoint"):
        _validate_plan(rebuilt([first, wrong_endpoint]), view)


def test_digest_tracks_cp_size_and_layout():
    view2 = _view(tp=1, cp=2)
    descriptors = _descriptors([(0, 0, 0, 6, 4, 0, 16, 16)])
    zig = _planner(view2, 2, "zigzag").build_plan(0, descriptors, [0])
    con = _planner(view2, 2, "contiguous").build_plan(0, descriptors, [0])
    assert zig.digest != con.digest
    one = _planner(_view(tp=2, cp=1), 1, "zigzag").build_plan(0, descriptors, [0])
    assert one.digest not in (zig.digest, con.digest)
