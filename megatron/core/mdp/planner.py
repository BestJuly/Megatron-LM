# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP planner: deterministic LPT assignment to logical encoder workers.

The plan-building path is pure compute — every group member independently runs
the same integer-only algorithm from byte-identical descriptor input and must
produce a bit-identical plan. Only :func:`assert_consistent_plan` touches
``torch.distributed``, and only when called.
"""

from typing import Sequence

from megatron.core.mdp.cp_partition import CP_PARTITION_MODES, cp_item_runs
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
        shape_key_width: int,
        pixel_locality: bool = False,
        cp_size: int = 1,
        cp_partition_mode: str = "zigzag",
    ) -> None:
        if cp_size > 1 and cp_partition_mode not in CP_PARTITION_MODES:
            raise MdpPlanError(
                f"MDP: cp_partition_mode={cp_partition_mode!r} violates: one of "
                f"{CP_PARTITION_MODES}."
            )
        if cp_size > 1 and len(rank_view.shard_owners) != cp_size:
            raise MdpPlanError(
                f"MDP: cp_size={cp_size} violates: one shard owner per CP rank "
                f"(rank view has {rank_view.shard_owners})."
            )
        self._cp_size = cp_size
        self._cp_partition_mode = cp_partition_mode
        self._rank_view = rank_view
        self._locality_slack_permille = locality_slack_permille
        self._capacity_policy = capacity_policy
        self._shape_key_width = shape_key_width
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
                        shape_key=descriptor.shape_key,
                    )
                )
                # Offsets accumulate VALID rows: the encoder consumes a
                # contiguous pack whose attention boundaries the adapter derives
                # from shape_key alone. The capacity policy sizes buffers (bridge
                # and pack tails), never inter-segment gaps.
                payload_offset += descriptor.payload_rows
                output_offset += descriptor.output_rows
            encoder_layouts.append(
                EncoderThdLayout(producer_worker_id=worker_id, segments=tuple(segments))
            )

        # Slices: without decoder CP each item is one slice to the group
        # endpoint. Under decoder CP the item's contiguous decoder span is cut
        # into the runs each CP rank owns (the decoder's own input layout), and
        # each run goes straight to that rank's PP0 TP source -- no rank ever
        # stages another rank's rows.
        cp = self._cp_size
        by_id = sorted(descriptors, key=lambda d: d.global_item_id)
        routes = []
        placed = {}  # (microbatch_id, endpoint_rank) -> [(order_key, item_id, slice_id, rows)]
        for descriptor in by_id:
            if cp == 1:
                runs = [(0, 0, descriptor.output_rows, None)]
            else:
                runs = cp_item_runs(
                    descriptor.decoder_start,
                    descriptor.output_rows,
                    sample_padded_start=descriptor.sample_padded_start,
                    sample_padded_len=descriptor.sample_padded_len,
                    total_rows=descriptor.microbatch_rows,
                    cp_size=cp,
                    mode=self._cp_partition_mode,
                )
            for slice_id, (cp_rank, row_start, rows, position) in enumerate(runs):
                endpoint = view.endpoint_rank if cp == 1 else view.shard_owners[cp_rank]
                routes.append(
                    RouteSlice(
                        global_item_id=descriptor.global_item_id,
                        producer_worker_id=assignment[descriptor.global_item_id],
                        endpoint_rank=endpoint,
                        owner_worker_id=descriptor.owner_worker_id,
                        slice_id=slice_id,
                        item_row_start=row_start,
                        item_rows=rows,
                    )
                )
                order_key = (
                    (descriptor.sample_id, descriptor.image_ordinal) if cp == 1 else (position,)
                )
                placed.setdefault((descriptor.microbatch_id, endpoint), []).append(
                    (order_key, descriptor.global_item_id, slice_id, rows)
                )

        # Endpoint leaf layouts in each endpoint's local decoder order:
        # (sample_id, image_ordinal) without CP, ascending decoder position under
        # it -- never LPT assignment order. A microbatch without items keeps one
        # text-only layout at the group endpoint.
        endpoints = (view.endpoint_rank,) if cp == 1 else tuple(view.shard_owners)
        layouts = []
        for mb_id in microbatch_ids:
            if not any((mb_id, endpoint) in placed for endpoint in endpoints):
                layouts.append(
                    MicrobatchLayout(
                        microbatch_id=mb_id,
                        text_only=True,
                        total_output_rows=0,
                        segments=(),
                        endpoint_rank=view.endpoint_rank,
                    )
                )
                continue
            for endpoint in endpoints:
                entries = sorted(placed.get((mb_id, endpoint), ()))
                if not entries:
                    continue
                segments = []
                leaf_offset = 0
                for _, item_id, slice_id, rows in entries:
                    segments.append(
                        LayoutSegment(
                            global_item_id=item_id,
                            leaf_row_start=leaf_offset,
                            output_rows=rows,
                            slice_id=slice_id,
                        )
                    )
                    leaf_offset += rows
                layouts.append(
                    MicrobatchLayout(
                        microbatch_id=mb_id,
                        text_only=False,
                        total_output_rows=leaf_offset,
                        segments=tuple(segments),
                        endpoint_rank=endpoint,
                    )
                )

        descriptor_by_id = {d.global_item_id: d for d in by_id}
        digest_entries = []
        for route in routes:
            descriptor = descriptor_by_id[route.global_item_id]
            digest_entries.append(
                (
                    route.global_item_id,
                    route.slice_id,
                    route.producer_worker_id,
                    order_in_producer[route.global_item_id],
                    route.endpoint_rank,
                    route.item_row_start,
                    route.item_rows,
                    descriptor.payload_rows,
                    descriptor.output_rows,
                    *(int(value) for value in descriptor.shape_key),
                )
            )
        digest = compute_plan_digest(
            PLAN_SCHEMA_VERSION,
            self._capacity_policy,
            digest_entries,
            shape_key_width=self._shape_key_width,
            cp_size=cp,
            cp_layout_code=CP_PARTITION_MODES.index(self._cp_partition_mode) if cp > 1 else 0,
        )
        routes = tuple(routes)

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
            # Model geometry (how shape_key relates to the row counts) is the
            # adapter's business and was already checked in the window; the
            # planner only enforces the model-independent invariants it needs.
            if len(descriptor.shape_key) != self._shape_key_width:
                raise MdpPlanError(
                    f"MDP: shape_key={descriptor.shape_key} for item {item_id} violates: "
                    f"exactly {self._shape_key_width} slots."
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
            if self._cp_size > 1 and (
                descriptor.decoder_start < 0 or descriptor.microbatch_rows <= 0
            ):
                raise MdpPlanError(
                    f"MDP: item {item_id} violates: under decoder CP every item carries "
                    "its decoder span (one contiguous run of image-token slots and the "
                    f"packed microbatch rows); got decoder_start={descriptor.decoder_start}, "
                    f"microbatch_rows={descriptor.microbatch_rows}."
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
    # Every routed slice appears in exactly one endpoint layout -- the one at
    # its route's endpoint -- and an item's slices tile its rows exactly.
    route_by_slice = {}
    for route in plan.routes:
        key = (route.global_item_id, route.slice_id)
        if key in route_by_slice:
            raise MdpPlanError(f"MDP: slice {key} violates: one route per slice.")
        route_by_slice[key] = route
    placed = set()
    for layout in plan.layouts:
        for segment in layout.segments:
            key = (segment.global_item_id, segment.slice_id)
            if key in placed:
                raise MdpPlanError(
                    f"MDP: slice {key} violates: one endpoint layout entry per slice."
                )
            placed.add(key)
            route = route_by_slice.get(key)
            if route is None:
                raise MdpPlanError(f"MDP: slice {key} violates: laid out slices are routed.")
            if layout.endpoint_rank is not None and layout.endpoint_rank != route.endpoint_rank:
                raise MdpPlanError(
                    f"MDP: slice {key} violates: laid out at its route's endpoint "
                    f"({layout.endpoint_rank} != {route.endpoint_rank})."
                )
    if placed != set(route_by_slice):
        raise MdpPlanError(
            "MDP: plan violates: endpoint layouts cover exactly the routed slices."
        )
    slices_by_item = {}
    for (item_id, _), route in route_by_slice.items():
        slices_by_item.setdefault(item_id, []).append(route)
    for item_id, routes in slices_by_item.items():
        item_rows = plan.segment_for_item(item_id).output_rows
        cursor = 0
        for route in sorted(routes, key=lambda r: r.item_row_start):
            rows = item_rows if route.item_rows is None else route.item_rows
            if route.item_row_start != cursor or rows <= 0:
                raise MdpPlanError(
                    f"MDP: item {item_id} violates: its slices tile its {item_rows} rows "
                    "without gap or overlap."
                )
            cursor += rows
        if cursor != item_rows:
            raise MdpPlanError(
                f"MDP: item {item_id} violates: its slices tile its {item_rows} rows "
                "without gap or overlap."
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
                        s.shape_key,
                    )
                    for s in e.segments
                ),
            )
            for e in plan.encoder_layouts
        ),
    )
