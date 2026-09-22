# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The worker-lead gradient buffer must be written in full before it is used.

``runtime.py`` P5 acquires the regroup buffer from ``DirectBufferAllocator``,
which returns ``torch.empty`` -- uninitialised. Non-lead ranks call
``grad_buffer.zero_()``; the LEAD rank does not. Instead it hands out a view per
routed slice and relies on the GRADIENT exchange to write every row:

    for segment in chunk.segments:
        for route in plan.routes_for_item(segment.global_item_id):
            start = segment.output_row_start + route.item_row_start
            grad_dest[key] = grad_buffer[start : start + route.item_rows]
    ...
    chunk_grads.append(grad_buffer[: chunk.total_output_rows])

Any row of ``[0, total_output_rows)`` that no route claims therefore reaches
backward as allocator garbage -- finite, plausible, and different every
iteration. That is exactly the shape of the symptom that prompted this test:
at ``encoder_cp=2`` the grad-norm profile is not reproducible run to run
(max 8.09 then 2.25 on identical configs) while the ``encoder_cp=1`` baseline
is (mean 0.625 then 0.626).

So this pins the invariant the lead branch depends on: for every producer chunk,
the routed slices tile ``[0, chunk.total_output_rows)`` exactly -- no gap and no
overlap. ``encoder_cp`` changes ``num_workers_per_group`` (= ``cp*pp //
encoder_cp``), which changes how items group into chunks, so the parametrisation
covers both settings.
"""

import pytest

from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankView

MERGE = 2


def _view(cp_size, pp, encoder_cp):
    group = tuple(range(cp_size * pp))
    num_workers = len(group) // encoder_cp
    return MdpRankView(
        global_rank=0,
        outer_dp_rank=0,
        lane_id=0,
        my_worker_id=0,
        endpoint_rank=group[0],
        planning_group_ranks=group,
        worker_ids=tuple(range(num_workers)),
        decoder_endpoint_ranks=group[:cp_size],
        my_cp_rank=0,
        my_pp_rank=0,
        my_encoder_cp_rank=0,
    )


def _descriptor(item_id, *, grid, sample, ordinal, offset, span_start, span_len, cost):
    t, h, w = grid
    return VisionDescriptor(
        global_item_id=item_id,
        sample_id=sample,
        image_ordinal=ordinal,
        owner_dp_lane=0,
        microbatch_id=0,
        estimated_cost_units=cost,
        payload_rows=t * h * w,
        output_rows=t * (h // MERGE) * (w // MERGE),
        grid_thw=grid,
        owner_worker_id=0,
        sample_padded_start=span_start,
        sample_padded_len=span_len,
        decoder_offset_in_sample=offset,
    )


def _descriptors(count, cp_size):
    """A batch of differently-shaped items laid end to end in one sample.

    Sizes vary so the LPT assignment is not trivially uniform -- an even split
    could hide a coverage gap that only appears when a chunk mixes item sizes.
    """
    grids = [(1, 4, 4), (1, 8, 8), (2, 4, 4), (1, 4, 8)]
    out = []
    offset = 0
    total = 0
    for index in range(count):
        grid = grids[index % len(grids)]
        rows = grid[0] * (grid[1] // MERGE) * (grid[2] // MERGE)
        total += rows
    # One sample long enough to hold every item, padded to a multiple of 2*cp.
    span = total
    multiple = 2 * cp_size
    if span % multiple:
        span += multiple - (span % multiple)
    for index in range(count):
        grid = grids[index % len(grids)]
        rows = grid[0] * (grid[1] // MERGE) * (grid[2] // MERGE)
        out.append(
            _descriptor(
                index,
                grid=grid,
                sample=0,
                ordinal=index,
                offset=offset,
                span_start=0,
                span_len=span,
                cost=10 + (index * 7) % 5,
            )
        )
        offset += rows
    return out


@pytest.mark.parametrize("cp_size,pp,encoder_cp", [
    (1, 2, 1),
    (1, 2, 2),   # the configuration whose grad norm did not reproduce
    (2, 2, 1),
    (2, 2, 2),
    (1, 4, 2),
    (1, 4, 4),
    (2, 4, 2),
])
@pytest.mark.parametrize("num_items", [1, 2, 3, 5, 8])
def test_routed_slices_tile_every_producer_chunk(cp_size, pp, encoder_cp, num_items):
    view = _view(cp_size, pp, encoder_cp)
    planner = MdpPlanner(
        view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy()
    )
    descriptors = _descriptors(num_items, cp_size)
    plan = planner.build_plan(0, descriptors, [0])

    for worker_id in view.worker_ids:
        layout = plan.encoder_layout_for_producer(worker_id)
        if not layout.segments:
            continue
        # Exactly the expression runtime.py P5 uses to hand out buffer views.
        claimed = []
        for segment in layout.segments:
            for route in plan.routes_for_item(segment.global_item_id):
                start = segment.output_row_start + route.item_row_start
                claimed.extend(range(start, start + route.item_rows))

        rows = layout.total_output_rows
        assert sorted(claimed) == list(range(rows)), (
            f"cp={cp_size} pp={pp} e={encoder_cp} items={num_items} "
            f"worker={worker_id}: the routed slices must tile "
            f"[0, {rows}) exactly. Rows no route claims are never written, so "
            f"the lead rank feeds allocator garbage (torch.empty) into "
            f"backward. missing={sorted(set(range(rows)) - set(claimed))} "
            f"duplicated={sorted({r for r in claimed if claimed.count(r) > 1})}"
        )
