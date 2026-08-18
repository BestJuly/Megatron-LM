# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP planner: deterministic LPT assignment to logical encoder workers.

The plan-building path is pure compute — every group member independently runs
the same integer-only algorithm from byte-identical descriptor input and must
produce a bit-identical plan. Only :func:`assert_consistent_plan` touches
``torch.distributed``, and only when called.
"""

from typing import Sequence

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
)
from megatron.core.mdp.rank_mapping import MdpRankView


class MdpPlanner:
    """Builds the per-iteration batch plan for one planning group."""

    def __init__(
        self,
        rank_view: MdpRankView,
        *,
        locality_slack_permille: int,
        capacity_policy: RowCapacityPolicy,
        pixel_locality: bool = False,
    ) -> None:
        self._rank_view = rank_view
        self._locality_slack_permille = locality_slack_permille
        self._capacity_policy = capacity_policy
        self._pixel_locality = pixel_locality
        # The logical worker hosting the owner endpoint, derived purely from the
        # view: workers partition the group ranks in fixed-width blocks.
        ranks_per_worker = len(rank_view.planning_group_ranks) // len(rank_view.worker_ids)
        self._endpoint_worker_id = (
            rank_view.planning_group_ranks.index(rank_view.endpoint_rank) // ranks_per_worker
        )

    def build_plan(
        self,
        iteration: int,
        descriptors: Sequence,
        microbatch_ids: Sequence[int],
    ) -> MdpBatchPlan:
        """Run deterministic LPT and assemble routes, layouts, and the digest."""
        view = self._rank_view
        self._validate_descriptors(descriptors, microbatch_ids)

        # LPT: (cost descending, item_id ascending); integer comparisons only.
        ordered = sorted(
            descriptors, key=lambda d: (-d.estimated_cost_units, d.global_item_id)
        )
        loads = {worker_id: 0 for worker_id in view.worker_ids}
        assignment = {}  # global_item_id -> worker_id
        producer_items = {worker_id: [] for worker_id in view.worker_ids}
        for descriptor in ordered:
            min_load = min(loads.values())
            slack = self._locality_slack_permille * max(1, descriptor.estimated_cost_units)
            eligible = [
                worker_id
                for worker_id in view.worker_ids
                if 1000 * loads[worker_id] <= 1000 * min_load + slack
            ]
            if self._pixel_locality:
                # Owner-sharded pixels: within the slack window, prefer the
                # item's pixel owner (a self-edge in the PIXEL exchange). This
                # replaces the endpoint preference, whose purpose — keeping
                # pixel traffic local — attaches to the owner once pixels are
                # owner-sharded.
                preferred = descriptor.owner_worker_id
            else:
                preferred = self._endpoint_worker_id
            chosen = min(
                eligible,
                key=lambda worker_id: (
                    0 if worker_id == preferred else 1,
                    loads[worker_id],
                    worker_id,
                ),
            )
            assignment[descriptor.global_item_id] = chosen
            producer_items[chosen].append(descriptor)
            loads[chosen] += descriptor.estimated_cost_units

        # Producer encoder THD layouts in assignment order, offsets cumulative.
        encoder_layouts = []
        order_in_producer = {}  # global_item_id -> index within its producer
        for worker_id in view.worker_ids:
            items = producer_items[worker_id]
            if not items:
                continue
            segments = []
            payload_offset = 0
            output_offset = 0
            for index, descriptor in enumerate(items):
                order_in_producer[descriptor.global_item_id] = index
                segments.append(
                    EncoderThdSegment(
                        global_item_id=descriptor.global_item_id,
                        microbatch_id=descriptor.microbatch_id,
                        sample_id=descriptor.sample_id,
                        image_ordinal=descriptor.image_ordinal,
                        payload_row_start=payload_offset,
                        payload_rows=descriptor.payload_rows,
                        output_row_start=output_offset,
                        output_rows=descriptor.output_rows,
                        grid_thw=descriptor.grid_thw,
                    )
                )
                # Offsets accumulate VALID rows: the encoder consumes a
                # contiguous pack whose frame boundaries derive from grid_thw
                # alone. The capacity policy sizes buffers (bridge and pack
                # tails), never inter-segment gaps.
                payload_offset += descriptor.payload_rows
                output_offset += descriptor.output_rows
            encoder_layouts.append(
                EncoderThdLayout(producer_worker_id=worker_id, segments=tuple(segments))
            )

        # Endpoint microbatch layouts, ordered by (sample_id, image_ordinal) only.
        items_by_microbatch = {mb_id: [] for mb_id in microbatch_ids}
        for descriptor in descriptors:
            items_by_microbatch[descriptor.microbatch_id].append(descriptor)
        layouts = []
        for mb_id in microbatch_ids:
            items = sorted(
                items_by_microbatch[mb_id], key=lambda d: (d.sample_id, d.image_ordinal)
            )
            segments = []
            leaf_offset = 0
            for descriptor in items:
                segments.append(
                    LayoutSegment(
                        global_item_id=descriptor.global_item_id,
                        leaf_row_start=leaf_offset,
                        output_rows=descriptor.output_rows,
                    )
                )
                leaf_offset += descriptor.output_rows
            layouts.append(
                MicrobatchLayout(
                    microbatch_id=mb_id,
                    text_only=not items,
                    total_output_rows=leaf_offset,
                    segments=tuple(segments),
                )
            )

        # Routes: one slice per item in v1. The endpoint is single-valued and
        # owner_worker_id names the owner-sharded PIXEL source.
        routes = tuple(
            RouteSlice(
                global_item_id=descriptor.global_item_id,
                producer_worker_id=assignment[descriptor.global_item_id],
                endpoint_rank=view.endpoint_rank,
                owner_worker_id=descriptor.owner_worker_id,
            )
            for descriptor in sorted(descriptors, key=lambda d: d.global_item_id)
        )

        digest_entries = [
            (
                descriptor.global_item_id,
                assignment[descriptor.global_item_id],
                order_in_producer[descriptor.global_item_id],
                view.endpoint_rank,
                descriptor.payload_rows,
                descriptor.output_rows,
                descriptor.grid_thw[0],
                descriptor.grid_thw[1],
                descriptor.grid_thw[2],
            )
            for descriptor in sorted(descriptors, key=lambda d: d.global_item_id)
        ]
        digest = compute_plan_digest(
            PLAN_SCHEMA_VERSION, self._capacity_policy, digest_entries
        )

        plan = MdpBatchPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            iteration=iteration,
            outer_dp_rank=view.outer_dp_rank,
            capacity_policy=self._capacity_policy,
            routes=routes,
            layouts=tuple(layouts),
            encoder_layouts=tuple(encoder_layouts),
            digest=digest,
        )
        _validate_plan(plan, view)
        return plan

    def _validate_descriptors(self, descriptors: Sequence, microbatch_ids: Sequence[int]) -> None:
        view = self._rank_view
        known_microbatches = set(microbatch_ids)
        if len(known_microbatches) != len(microbatch_ids):
            raise MdpPlanError("MDP: microbatch_ids violates: ids are unique.")
        seen = set()
        for descriptor in descriptors:
            item_id = descriptor.global_item_id
            if item_id in seen:
                raise MdpPlanError(
                    f"MDP: global_item_id={item_id} violates: item ids are unique within "
                    "the planning group."
                )
            seen.add(item_id)
            if descriptor.estimated_cost_units < 0:
                raise MdpPlanError(
                    f"MDP: estimated_cost_units={descriptor.estimated_cost_units} for item "
                    f"{item_id} violates: cost is a non-negative integer."
                )
            t, h, w = descriptor.grid_thw
            if t * h * w != descriptor.payload_rows:
                raise MdpPlanError(
                    f"MDP: payload_rows={descriptor.payload_rows} for item {item_id} "
                    f"violates: payload_rows == t*h*w with grid_thw={descriptor.grid_thw}."
                )
            if descriptor.payload_rows <= 0 or descriptor.output_rows <= 0:
                raise MdpPlanError(
                    f"MDP: item {item_id} violates: payload_rows and output_rows are "
                    "positive."
                )
            if descriptor.microbatch_id not in known_microbatches:
                raise MdpPlanError(
                    f"MDP: microbatch_id={descriptor.microbatch_id} for item {item_id} "
                    f"violates: microbatch is part of this iteration window."
                )
            if descriptor.owner_worker_id not in view.worker_ids:
                raise MdpPlanError(
                    f"MDP: owner_worker_id={descriptor.owner_worker_id} for item "
                    f"{item_id} violates: the pixel owner is a worker of this "
                    f"planning group {view.worker_ids}."
                )
            if descriptor.owner_dp_lane != view.outer_dp_rank:
                raise MdpPlanError(
                    f"MDP: owner_dp_lane={descriptor.owner_dp_lane} for item {item_id} "
                    f"violates: items never cross outer-DP groups "
                    f"(outer_dp_rank={view.outer_dp_rank})."
                )


def _validate_plan(plan: MdpBatchPlan, view: MdpRankView) -> None:
    """Full coverage / no-overlap validation in O(items + routes)."""
    route_items = {route.global_item_id for route in plan.routes}
    layout_items = set()
    for layout in plan.encoder_layouts:
        if layout.producer_worker_id not in view.worker_ids:
            raise MdpPlanError(
                f"MDP: producer_worker_id={layout.producer_worker_id} violates: producer "
                "belongs to this planning group."
            )
        for segment in layout.segments:
            layout_items.add(segment.global_item_id)
    if route_items != layout_items:
        raise MdpPlanError(
            "MDP: plan violates: routes and encoder layouts cover exactly the same items "
            f"(routes-only={sorted(route_items - layout_items)}, "
            f"layouts-only={sorted(layout_items - route_items)})."
        )
    endpoint_items = set()
    for layout in plan.layouts:
        for segment in layout.segments:
            if segment.global_item_id in endpoint_items:
                raise MdpPlanError(
                    f"MDP: global_item_id={segment.global_item_id} violates: one endpoint "
                    "layout entry per item."
                )
            endpoint_items.add(segment.global_item_id)
    if endpoint_items != route_items:
        raise MdpPlanError(
            "MDP: plan violates: endpoint layouts cover exactly the routed items."
        )
    for route in plan.routes:
        if route.endpoint_rank not in view.planning_group_ranks:
            raise MdpPlanError(
                f"MDP: endpoint_rank={route.endpoint_rank} violates: routes never cross an "
                "outer-DP group boundary."
            )


def assert_consistent_plan(
    plan: MdpBatchPlan,
    *,
    planning_group,
    iteration: int,
    interval: int,
    debug_payload_check: bool = False,
) -> None:
    """Cross-rank plan consistency check; called before any bridge collective.

    All-gathers the 16-byte digest inside the planning group when
    ``iteration % interval == 0`` and raises a coordinated :class:`MdpPlanError`
    on any mismatch. ``interval`` can sample but never fully disables the check:
    an undetected plan mismatch degrades from a diagnosable error into a collective hang.
    """
    import torch
    import torch.distributed as dist

    if interval < 1:
        raise MdpPlanError(
            f"MDP: plan_check_interval={interval} violates: interval >= 1; the check "
            "must never be fully disabled."
        )
    if iteration % interval != 0:
        return

    local = torch.tensor(list(plan.digest), dtype=torch.uint8, device="cuda")
    group_size = dist.get_world_size(group=planning_group)
    gathered = [torch.empty_like(local) for _ in range(group_size)]
    dist.all_gather(gathered, local, group=planning_group)
    digests = [bytes(t.tolist()) for t in gathered]
    if any(digest != plan.digest for digest in digests):
        raise MdpPlanError(
            f"MDP: plan digest mismatch at iteration {iteration} in planning group of "
            f"outer_dp_rank={plan.outer_dp_rank}: {[d.hex() for d in digests]}."
        )

    if debug_payload_check:
        payload = _canonical_plan_payload(plan)
        gathered_payloads = [None] * group_size
        dist.all_gather_object(gathered_payloads, payload, group=planning_group)
        if any(other != payload for other in gathered_payloads):
            raise MdpPlanError(
                f"MDP: canonical plan payload mismatch at iteration {iteration} despite "
                "matching metadata; see gathered payloads on rank 0."
            )


def _canonical_plan_payload(plan: MdpBatchPlan):
    """A plain, comparable rendering of the full plan for debug comparison."""
    return (
        plan.schema_version,
        plan.iteration,
        plan.outer_dp_rank,
        plan.capacity_policy.alignment_rows,
        tuple(
            (r.global_item_id, r.producer_worker_id, r.endpoint_rank) for r in plan.routes
        ),
        tuple(
            (
                l.microbatch_id,
                l.text_only,
                l.total_output_rows,
                tuple((s.global_item_id, s.leaf_row_start, s.output_rows) for s in l.segments),
            )
            for l in plan.layouts
        ),
        tuple(
            (
                e.producer_worker_id,
                tuple(
                    (
                        s.global_item_id,
                        s.microbatch_id,
                        s.sample_id,
                        s.image_ordinal,
                        s.payload_row_start,
                        s.payload_rows,
                        s.output_row_start,
                        s.output_rows,
                        s.grid_thw,
                    )
                    for s in e.segments
                ),
            )
            for e in plan.encoder_layouts
        ),
    )
