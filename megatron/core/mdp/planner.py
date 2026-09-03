# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP planner: deterministic LPT assignment to logical encoder workers.

The plan-building path is pure compute — every group member independently runs
the same integer-only algorithm from byte-identical descriptor input and must
produce a bit-identical plan. Only :func:`assert_consistent_plan` touches
``torch.distributed``, and only when called.
"""

from typing import Sequence

from megatron.core.mdp.cp_partition import CpRowInterval, split_item
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError
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
from megatron.core.mdp.rank_mapping import MdpRankView, endpoint_worker_id


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
        # One derivation of the endpoint's worker id, shared with rank_mapping.
        self._endpoint_worker_id = endpoint_worker_id(rank_view)
        # Decoder endpoints, indexed by cp_rank. A hand-built view (tests) may
        # omit them; at CP=1 the descriptor source is the only endpoint.
        self._endpoints = rank_view.decoder_endpoint_ranks or (rank_view.endpoint_rank,)
        self._cp_size = len(self._endpoints)
        self._ranks_per_worker = len(rank_view.planning_group_ranks) // len(
            rank_view.worker_ids
        )
        # Ranks per logical worker IS encoder_cp; the planner needs it both to
        # validate frame alignment and to make the digest encoder_cp-sensitive.
        self._encoder_cp = self._ranks_per_worker

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
                preferred = self._preferred_endpoint_worker(descriptor)
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

        # Decoder-CP split: each item's contiguous decoder run breaks into
        # per-endpoint runs. At CP=1 this is one whole-item run per item, so
        # both the routes and the layouts below reduce to the pre-CP form.
        slices_by_item = {
            descriptor.global_item_id: self._slice_item(descriptor)
            for descriptor in descriptors
        }

        # Endpoint microbatch layouts, one per (microbatch, endpoint). Segments
        # are ordered by the endpoint's rank-LOCAL row, which is the order the
        # decoder's post-CP-split image-token mask sees; at CP=1 that is exactly
        # (sample_id, image_ordinal).
        by_microbatch_endpoint = {
            (mb_id, endpoint): []
            for mb_id in microbatch_ids
            for endpoint in self._endpoints
        }
        for descriptor in descriptors:
            for slice_id, interval in enumerate(slices_by_item[descriptor.global_item_id]):
                endpoint = self._endpoints[interval.cp_rank]
                by_microbatch_endpoint[(descriptor.microbatch_id, endpoint)].append(
                    (interval.local_row_start, descriptor, slice_id, interval)
                )
        layouts = []
        for mb_id in microbatch_ids:
            for endpoint in self._endpoints:
                entries = sorted(
                    by_microbatch_endpoint[(mb_id, endpoint)],
                    key=lambda e: (e[0], e[1].sample_id, e[1].image_ordinal, e[2]),
                )
                segments = []
                leaf_offset = 0
                for _local_row, descriptor, slice_id, interval in entries:
                    segments.append(
                        LayoutSegment(
                            global_item_id=descriptor.global_item_id,
                            leaf_row_start=leaf_offset,
                            output_rows=interval.rows,
                            slice_id=slice_id,
                        )
                    )
                    leaf_offset += interval.rows
                layouts.append(
                    MicrobatchLayout(
                        microbatch_id=mb_id,
                        text_only=not entries,
                        total_output_rows=leaf_offset,
                        segments=tuple(segments),
                        endpoint_rank=endpoint,
                    )
                )

        # Routes: one slice per (item, cp run). owner_worker_id names the
        # owner-sharded PIXEL source, which is per item, not per slice.
        ordered_descriptors = sorted(descriptors, key=lambda d: d.global_item_id)
        routes = tuple(
            RouteSlice(
                global_item_id=descriptor.global_item_id,
                producer_worker_id=assignment[descriptor.global_item_id],
                endpoint_rank=self._endpoints[interval.cp_rank],
                owner_worker_id=descriptor.owner_worker_id,
                slice_id=slice_id,
                item_row_start=interval.item_row_start,
                item_rows=interval.rows,
            )
            for descriptor in ordered_descriptors
            for slice_id, interval in enumerate(slices_by_item[descriptor.global_item_id])
        )

        digest_entries = [
            (
                descriptor.global_item_id,
                slice_id,
                assignment[descriptor.global_item_id],
                order_in_producer[descriptor.global_item_id],
                self._endpoints[interval.cp_rank],
                interval.item_row_start,
                interval.rows,
                descriptor.payload_rows,
                descriptor.output_rows,
                descriptor.grid_thw[0],
                descriptor.grid_thw[1],
                descriptor.grid_thw[2],
            )
            for descriptor in ordered_descriptors
            for slice_id, interval in enumerate(slices_by_item[descriptor.global_item_id])
        ]
        digest = compute_plan_digest(
            PLAN_SCHEMA_VERSION,
            self._capacity_policy,
            digest_entries,
            cp_size=self._cp_size,
            ranks_per_worker=self._encoder_cp,
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

    def _preferred_endpoint_worker(self, descriptor) -> int:
        """The endpoint worker holding most of this item's decoder rows.

        The EMBEDDING edge is producer -> endpoint, so producing an item on the
        worker that already hosts most of its rows turns the biggest slice into
        a self-edge. At CP=1 every item's single endpoint is ``group[0]``, whose
        worker id is 0, so this reduces exactly to the previous fixed
        ``_endpoint_worker_id`` preference. Ties break to the lowest worker id,
        keeping the choice deterministic across ranks.
        """
        if self._cp_size == 1:
            return self._endpoint_worker_id
        rows_by_worker = {}
        for interval in self._slice_item(descriptor):
            endpoint = self._endpoints[interval.cp_rank]
            index = self._rank_view.planning_group_ranks.index(endpoint)
            # EMBEDDING is sourced by the worker's LEAD rank, so a self-edge
            # exists only when that lead IS this endpoint. Mapping the endpoint
            # through `index // ranks_per_worker` answers a different question
            # and, once encoder_cp >= cp, sends EVERY endpoint to worker 0 --
            # a constant preference that biases the LPT load inside the slack
            # window while buying no locality at all.
            if index % self._ranks_per_worker != 0:
                continue  # not any worker's lead: no self-edge is available
            worker = index // self._ranks_per_worker
            rows_by_worker[worker] = rows_by_worker.get(worker, 0) + interval.rows
        if not rows_by_worker:
            # No endpoint of this item is a worker lead. Express that as "no
            # preference" (-1 matches no worker id) rather than defaulting to
            # worker 0, so the choice falls through to pure load balance.
            return -1
        return min(rows_by_worker, key=lambda w: (-rows_by_worker[w], w))

    def _slice_item(self, descriptor) -> tuple:
        """This item's per-endpoint decoder row runs, ascending in item rows.

        Pure integer arithmetic over descriptor fields only, so every planning
        group member derives the identical split; the result is folded into the
        plan digest so a divergence surfaces as a plan mismatch rather than a
        mismatched collective.
        """
        if self._cp_size == 1:
            # Skip the sample-span arithmetic entirely: at CP=1 the span
            # columns may legitimately be absent (a hand-built descriptor, or a
            # capture path that predates them) and the answer is always the
            # whole item.
            return (
                CpRowInterval(
                    cp_rank=0,
                    item_row_start=0,
                    rows=descriptor.output_rows,
                    local_row_start=descriptor.sample_padded_start
                    + descriptor.decoder_offset_in_sample,
                ),
            )
        try:
            return split_item(
                offset_in_sample=descriptor.decoder_offset_in_sample,
                output_rows=descriptor.output_rows,
                sample_padded_start=descriptor.sample_padded_start,
                sample_padded_len=descriptor.sample_padded_len,
                cp_size=self._cp_size,
            )
        except MdpConfigurationError as error:
            raise MdpPlanError(
                f"MDP: item {descriptor.global_item_id} cannot be split across "
                f"cp_size={self._cp_size}: {error}"
            ) from error

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
            if self._encoder_cp > 1 and (h * w) % (2 * self._encoder_cp) != 0:
                raise MdpPlanError(
                    f"MDP: item {item_id} grid_thw={descriptor.grid_thw} violates: "
                    f"each frame's h*w ({h * w}) is divisible by 2 * encoder_cp "
                    f"({2 * self._encoder_cp}). The vision encoder packs one THD "
                    "sub-sequence per frame and has no frame-padding path, so a "
                    "non-conforming grid cannot be context-parallel split. "
                    "Checked here in integer host arithmetic so every member "
                    "reaches the same verdict; otherwise TE aborts mid-iteration "
                    "on whichever image happens to be non-conforming, and a short "
                    "smoke run passes while a long one dies."
                )
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
    endpoint_slices = set()
    for layout in plan.layouts:
        # Leaf rows must tile exactly. At CP=1 leaf_row_start is a cumulative
        # sum and this is trivially true; once the offsets come from the zigzag
        # split an off-by-one produces overlapping views (last writer wins) or a
        # gap (uninitialised rows read as embeddings), and the bridge's
        # per-entry size check cannot see either.
        cursor = 0
        for segment in layout.segments:
            if segment.leaf_row_start != cursor:
                raise MdpPlanError(
                    f"MDP: layout (microbatch={layout.microbatch_id}, "
                    f"endpoint={layout.endpoint_rank}) violates: leaf segments "
                    f"tile [0, total_output_rows) without gap or overlap "
                    f"(expected leaf_row_start={cursor}, got {segment.leaf_row_start})."
                )
            cursor += segment.output_rows
        if cursor != layout.total_output_rows:
            raise MdpPlanError(
                f"MDP: layout (microbatch={layout.microbatch_id}, "
                f"endpoint={layout.endpoint_rank}) violates: "
                f"total_output_rows == sum of segment rows "
                f"({layout.total_output_rows} != {cursor})."
            )
        for segment in layout.segments:
            key = (segment.global_item_id, segment.slice_id)
            if key in endpoint_slices:
                raise MdpPlanError(
                    f"MDP: (global_item_id, slice_id)={key} violates: one endpoint "
                    "layout entry per route slice."
                )
            endpoint_slices.add(key)
    route_slices = {(route.global_item_id, route.slice_id) for route in plan.routes}
    if endpoint_slices != route_slices:
        raise MdpPlanError(
            "MDP: plan violates: endpoint layouts cover exactly the routed slices "
            f"(routes-only={sorted(route_slices - endpoint_slices)}, "
            f"layouts-only={sorted(endpoint_slices - route_slices)})."
        )
    endpoints = set(view.decoder_endpoint_ranks or (view.endpoint_rank,))
    for route in plan.routes:
        if route.endpoint_rank not in view.planning_group_ranks:
            raise MdpPlanError(
                f"MDP: endpoint_rank={route.endpoint_rank} violates: routes never cross an "
                "outer-DP group boundary."
            )
        if route.endpoint_rank not in endpoints:
            raise MdpPlanError(
                f"MDP: endpoint_rank={route.endpoint_rank} violates: routes land on a "
                f"decoder endpoint (pipeline stage 0) of this group {sorted(endpoints)}."
            )
    # Every item's slices must tile its rows exactly once, in order.
    rows_by_item = {}
    for route in sorted(plan.routes, key=lambda r: (r.global_item_id, r.slice_id)):
        expected_start = rows_by_item.get(route.global_item_id, 0)
        if route.item_row_start != expected_start:
            raise MdpPlanError(
                f"MDP: item {route.global_item_id} slice {route.slice_id} violates: "
                f"slices tile the item's rows without gap or overlap "
                f"(expected item_row_start={expected_start}, got {route.item_row_start})."
            )
        rows_by_item[route.global_item_id] = expected_start + route.item_rows
    for item_id, covered in rows_by_item.items():
        segment = plan.segment_for_item(item_id)
        if covered != segment.output_rows:
            raise MdpPlanError(
                f"MDP: item {item_id} violates: its slices cover all "
                f"{segment.output_rows} output rows (covered {covered})."
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
            (
                r.global_item_id,
                r.slice_id,
                r.producer_worker_id,
                r.endpoint_rank,
                r.item_row_start,
                r.item_rows,
            )
            for r in plan.routes
        ),
        tuple(
            (
                l.microbatch_id,
                l.endpoint_rank,
                l.text_only,
                l.total_output_rows,
                tuple(
                    (s.global_item_id, s.slice_id, s.leaf_row_start, s.output_rows)
                    for s in l.segments
                ),
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
