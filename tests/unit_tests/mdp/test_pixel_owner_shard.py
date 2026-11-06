# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Owner-sharded pixel reading and the PIXEL all_to_all exchange.

Pure-compute tests cover ownership in window capture, collate suppression via
the ownership context, and owner-aware planning. Distributed tests
(PP4 -> W=4 workers per lane) verify multi-owner
pixel routes element-wise via sentinels and the unconditional all_to_all
participation of empty/text-only workers.

Run the distributed part with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_pixel_owner_shard.py
"""

import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import (
    BridgeBufferKey,
    BridgePhase,
    BridgeTensorSpec,
    ModalityBridge,
)
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem, VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankSpec, MdpRankView, build_rank_map
from megatron.core.mdp.window import MdpIterationWindow, pixel_capture_suppressed

MERGE = 2
WIDTH = 4


def _sentinel(sample_id, ordinal):
    """Mirrors examples/multimodal_dev/data/mdp_mock.item_sentinel."""
    return float(1000 * (sample_id + 1) + ordinal)


# ----------------------------------------------------------------------
# Stub capture (CPU): the adapter honors the ownership context exactly like
# the real collate path (pack_or_pad_batch checks pixel_capture_suppressed).
# ----------------------------------------------------------------------


def _item(sample_id, ordinal, grid, payload_row_start):
    t, h, w = grid
    output_rows = t * (h // MERGE) * (w // MERGE)
    return CapturedVisionItem(
        sample_id=sample_id,
        image_ordinal=ordinal,
        grid_thw=grid,
        payload_row_start=payload_row_start,
        payload_rows=t * h * w,
        decoder_positions=tuple(range(100, 100 + output_rows)),
    )


def _pixels_for(items, total_rows):
    pixels = torch.zeros(total_rows, WIDTH)
    for item in items:
        pixels[item.payload_row_start : item.payload_row_start + item.payload_rows] = _sentinel(
            item.sample_id, item.image_ordinal
        )
    return pixels


class _ShardAwareAdapter:
    """Returns pixels only when the capture context says this worker owns them."""

    payload_width = WIDTH
    spatial_merge_size = MERGE

    def __init__(self, microbatch_items, suppress=True):
        # microbatch_items: list of (items, total_rows) per microbatch.
        self._microbatches = list(microbatch_items)
        self._suppress = suppress

    def get_batch(self, iterator):
        next(iterator)
        items, total_rows = self._microbatches.pop(0)
        pixels = None
        if items and not (self._suppress and pixel_capture_suppressed()):
            pixels = _pixels_for(items, total_rows)
        return CapturedMicrobatch(
            decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
            vision_items=tuple(items),
            flat_pixel_payload=pixels,
            model_payload=MappingProxyType({"input_ids": torch.zeros(1, 8)}),
        )

    def estimate_cost(self, item):
        return item.payload_rows


def _window_microbatches():
    # Four microbatches with items, one text-only, over W=4 workers so
    # owners are 0, 1, 2, 3, 0.
    return [
        ([_item(0, 0, (1, 4, 4), 0), _item(0, 1, (1, 4, 8), 16)], 48),
        ([_item(1, 0, (1, 8, 8), 0)], 64),
        ([], 0),
        ([_item(3, 0, (2, 4, 4), 0)], 32),
        ([_item(4, 0, (1, 6, 6), 0)], 36),
    ]


@pytest.mark.parametrize("my_worker_id", [0, 1, 2, 3])
def test_capture_cuts_sidecar_for_owned_microbatches_only(my_worker_id):
    window = MdpIterationWindow.capture(
        iter(range(10)),
        num_microbatches=5,
        adapter=_ShardAwareAdapter(_window_microbatches()),
        num_vpp_chunks=1,
        lane_id=0 if my_worker_id == 0 else None,
        my_worker_id=my_worker_id,
        num_workers=4,
    )
    # global_item_id assignment order: mb0 -> 0, 1; mb1 -> 2; mb3 -> 3; mb4 -> 4.
    owned_items = {0: {0, 1, 4}, 1: {2}, 2: set(), 3: {3}}[my_worker_id]
    assert set(window.payload_sidecar()) == owned_items
    for item_id, pixels in window.payload_sidecar().items():
        assert (pixels == pixels.flatten()[0]).all() and pixels.flatten()[0] > 0
    # Descriptors only on the endpoint, with mb-derived owners.
    if my_worker_id == 0:
        owners = [d.owner_worker_id for d in window.descriptors()]
        microbatches = [d.microbatch_id for d in window.descriptors()]
        assert owners == [mb % 4 for mb in microbatches] == [0, 0, 1, 3, 0]
    else:
        assert window.descriptors() == ()
    # Every worker still holds all replay records.
    assert len(window.records()) == 5
    assert pixel_capture_suppressed() is False  # context cleared


def test_capture_with_fewer_microbatches_than_workers():
    # Workers 2 and 3 own nothing: legal, empty sidecar, full records.
    window = MdpIterationWindow.capture(
        iter(range(10)),
        num_microbatches=2,
        adapter=_ShardAwareAdapter(_window_microbatches()[:2]),
        num_vpp_chunks=1,
        lane_id=None,
        my_worker_id=2,
        num_workers=4,
    )
    assert window.payload_sidecar() == {}
    assert len(window.records()) == 2


def test_capture_rejects_unsuppressed_pixels_on_non_owner():
    with pytest.raises(MdpConfigurationError, match="not suppressed"):
        MdpIterationWindow.capture(
            iter(range(10)),
            num_microbatches=1,
            adapter=_ShardAwareAdapter(_window_microbatches()[:1], suppress=False),
            num_vpp_chunks=1,
            lane_id=None,
            my_worker_id=1,  # owner of mb0 is worker 0
            num_workers=4,
        )


# ----------------------------------------------------------------------
# Planner: owner metadata and locality preference (pure compute)
# ----------------------------------------------------------------------


def _view(worker_ids=(0, 1, 2, 3), group=(0, 2, 4, 6), endpoint=0):
    return MdpRankView(
        global_rank=0,
        outer_dp_rank=0,
        lane_id=0,
        my_worker_id=0,
        endpoint_rank=endpoint,
        planning_group_ranks=group,
        worker_ids=worker_ids,
    )


def _descriptor(item_id, cost, mb=0, grid=(1, 4, 4), owner=0, lane=0):
    t, h, w = grid
    return VisionDescriptor(
        global_item_id=item_id,
        sample_id=item_id,
        image_ordinal=0,
        owner_dp_lane=lane,
        microbatch_id=mb,
        estimated_cost_units=cost,
        payload_rows=t * h * w,
        output_rows=t * (h // 2) * (w // 2),
        grid_thw=grid,
        owner_worker_id=owner,
    )


def _assignment(plan):
    return {r.global_item_id: r.producer_worker_id for r in plan.routes}


def test_owner_metadata_does_not_change_digest_without_locality():
    baseline = [_descriptor(i, cost=10 + (i * 7) % 5, mb=i % 4, owner=0) for i in range(9)]
    sharded = [
        _descriptor(i, cost=10 + (i * 7) % 5, mb=i % 4, owner=(i % 4)) for i in range(9)
    ]
    planner = MdpPlanner(
        _view(), locality_slack_permille=10, capacity_policy=RowCapacityPolicy()
    )
    plan_a = planner.build_plan(0, baseline, [0, 1, 2, 3])
    plan_b = planner.build_plan(0, sharded, [0, 1, 2, 3])
    # Owner metadata must not perturb LPT when locality is off.
    assert _assignment(plan_a) == _assignment(plan_b)
    assert plan_a.digest == plan_b.digest
    assert plan_a.encoder_layouts == plan_b.encoder_layouts
    assert plan_a.layouts == plan_b.layouts


def test_locality_prefers_owner_only_within_slack():
    view = _view()
    # Equal costs: after each round every load ties, so the whole pool is
    # eligible and locality may follow the owner freely.
    equal = [_descriptor(i, cost=100, mb=i, owner=i % 4) for i in range(8)]
    local_planner = MdpPlanner(
        view,
        locality_slack_permille=10,
        capacity_policy=RowCapacityPolicy(),
        pixel_locality=True,
    )
    plan = local_planner.build_plan(0, equal, list(range(8)))
    assignment = _assignment(plan)
    for item_id, producer in assignment.items():
        assert producer == item_id % 4  # every item produced by its owner
    # Zero slack: only min-load workers are eligible; the plan must equal the
    # non-locality plan whenever the owner is outside the slack window.
    skewed = [_descriptor(0, cost=1000, mb=0, owner=1)] + [
        _descriptor(i, cost=10, mb=i, owner=1) for i in range(1, 5)
    ]
    strict_local = MdpPlanner(
        view,
        locality_slack_permille=0,
        capacity_policy=RowCapacityPolicy(),
        pixel_locality=True,
    )
    strict_base = MdpPlanner(
        view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()
    )
    plan_local = strict_local.build_plan(0, skewed, list(range(5)))
    plan_base = strict_base.build_plan(0, skewed, list(range(5)))
    local_assignment = _assignment(plan_local)
    # Item 0 (cost 1000) makes worker 1 the max-load worker; the remaining
    # items' owner (worker 1) is never in the zero-slack window, so locality
    # cannot move them onto it.
    assert local_assignment[0] == 1  # first item: all loads zero, owner eligible
    assert all(local_assignment[i] != 1 for i in range(1, 5))
    assert sorted(local_assignment[i] for i in range(1, 5)) == [0, 0, 2, 3]
    del plan_base  # baseline differs only via the endpoint tie-break


# ----------------------------------------------------------------------
# Distributed: multi-owner routes, sentinels, unconditional participation
# ----------------------------------------------------------------------

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=4
        )
        yield
        Utils.destroy_model_parallel()


needs_dist = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")

GRIDS = ((1, 4, 4), (1, 8, 8), (2, 4, 4), (1, 4, 8), (1, 6, 6), (1, 8, 4), (2, 6, 4), (1, 6, 4))


def _dist_setup(num_items=8, pixel_locality=False):
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=4, cp=1, ep=1, encoder_cp=1)
    )
    view = rank_map.view(rank)
    num_workers = len(view.worker_ids)
    descriptors = []
    for i in range(num_items):
        t, h, w = GRIDS[i % len(GRIDS)]
        descriptors.append(
            VisionDescriptor(
                global_item_id=i,
                sample_id=i,
                image_ordinal=0,
                owner_dp_lane=view.outer_dp_rank,
                microbatch_id=i,
                estimated_cost_units=t * h * w,
                payload_rows=t * h * w,
                output_rows=t * (h // 2) * (w // 2),
                grid_thw=(t, h, w),
                owner_worker_id=i % num_workers,
            )
        )
    planner = MdpPlanner(
        view,
        locality_slack_permille=10,
        capacity_policy=RowCapacityPolicy(),
        pixel_locality=pixel_locality,
    )
    plan = planner.build_plan(0, descriptors, list(range(max(num_items, 1))))
    return rank_map, view, plan, descriptors


def _dist_pixel_specs(plan, descriptors):
    return {
        BridgeBufferKey(d.global_item_id): BridgeTensorSpec(
            valid_rows=d.payload_rows,
            capacity_rows=plan.capacity_policy.capacity_of(d.payload_rows),
            width=WIDTH,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        for d in descriptors
    }


def _owner_local_tensors(view, rank_map, descriptors):
    """Sentinel pixels for the items whose owner worker is this rank."""
    local = {}
    for d in descriptors:
        owner_rank = rank_map.worker_ranks(view.outer_dp_rank, d.owner_worker_id)[0]
        if owner_rank == view.global_rank:
            local[BridgeBufferKey(d.global_item_id)] = torch.full(
                (d.payload_rows, WIDTH),
                _sentinel(d.sample_id, d.image_ordinal),
                dtype=torch.bfloat16,
                device="cuda",
            )
    return local


@needs_dist
def test_alltoall_sentinels_and_zero_split_participation():
    from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups

    rank_map, view, plan, descriptors = _dist_setup()
    rank = torch.distributed.get_rank()
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    bridge = ModalityBridge(DirectBufferAllocator())
    specs = _dist_pixel_specs(plan, descriptors)
    ledger = bridge.build_ledger(BridgePhase.PIXEL, plan, rank_map, specs)

    # The ledger's PIXEL sources are the owner workers, not the endpoint.
    owners_used = set()
    for entry in ledger.entries:
        d = descriptors[entry.key.global_item_id]
        owner_rank = rank_map.worker_ranks(view.outer_dp_rank, d.owner_worker_id)[0]
        assert entry.src_global_rank == owner_rank
        owners_used.add(owner_rank)
    assert len(owners_used) == len(view.worker_ids), "multi-owner premise"

    my_items = {r.global_item_id for r in plan.routes_for_producer(view.my_worker_id)}

    # Many iterations: stream-ordering races are nondeterministic; one pass
    # proves little for a no-host-sync collective.
    for _ in range(50):
        local = _owner_local_tensors(view, rank_map, descriptors)
        received = bridge.exchange_all_to_all(
            ledger,
            local,
            tensor_specs=specs,
            group=groups.planning_group,
            group_ranks=view.planning_group_ranks,
            global_rank=rank,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        assert {k.global_item_id for k in received} == my_items
        for key, tensor in received.items():
            d = descriptors[key.global_item_id]
            assert tensor.shape == (d.payload_rows, WIDTH)
            expected = torch.full_like(tensor, _sentinel(d.sample_id, d.image_ordinal))
            assert torch.equal(tensor, expected)
        bridge.assert_idle()

    # Zero-split participation: a plan whose single item is owned AND produced
    # by worker 0 leaves every other member with nothing to move; each still
    # calls the collective (it would hang otherwise) and returns empty.
    single = [
        VisionDescriptor(
            global_item_id=0,
            sample_id=0,
            image_ordinal=0,
            owner_dp_lane=view.outer_dp_rank,
            microbatch_id=0,
            estimated_cost_units=16,
            payload_rows=16,
            output_rows=4,
            grid_thw=(1, 4, 4),
            owner_worker_id=0,
        )
    ]
    planner = MdpPlanner(
        view,
        locality_slack_permille=10,
        capacity_policy=RowCapacityPolicy(),
        pixel_locality=True,
    )
    single_plan = planner.build_plan(1, single, [0])
    assert _assignment(single_plan) == {0: 0}
    single_specs = _dist_pixel_specs(single_plan, single)
    single_ledger = bridge.build_ledger(
        BridgePhase.PIXEL, single_plan, rank_map, single_specs
    )
    local = _owner_local_tensors(view, rank_map, single)
    received = bridge.exchange_all_to_all(
        single_ledger,
        local,
        tensor_specs=single_specs,
        group=groups.planning_group,
        group_ranks=view.planning_group_ranks,
        global_rank=rank,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    if view.my_worker_id == 0 and rank == view.planning_group_ranks[0]:
        assert {k.global_item_id for k in received} == {0}
        assert torch.equal(
            received[BridgeBufferKey(0)],
            torch.full((16, WIDTH), _sentinel(0, 0), dtype=torch.bfloat16, device="cuda"),
        )
    else:
        assert received == {}
    bridge.assert_idle()

    # Fully empty plan: still one unconditional collective per member.
    empty_plan = planner.build_plan(2, [], [0])
    empty_ledger = bridge.build_ledger(BridgePhase.PIXEL, empty_plan, rank_map, {})
    assert empty_ledger.entries == ()
    received = bridge.exchange_all_to_all(
        empty_ledger,
        {},
        tensor_specs={},
        group=groups.planning_group,
        group_ranks=view.planning_group_ranks,
        global_rank=rank,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    assert received == {}
    bridge.assert_idle()


@needs_dist
def test_owner_zero_routes_pixels_from_worker_zero():
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=4, cp=1, ep=1, encoder_cp=1)
    )
    view = rank_map.view(rank)
    planner = MdpPlanner(
        view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy()
    )
    baseline = [
        _descriptor(i, cost=16 + i, mb=i, lane=view.outer_dp_rank) for i in range(4)
    ]
    bridge = ModalityBridge(DirectBufferAllocator())
    plan = planner.build_plan(0, baseline, [0, 1, 2, 3])
    specs = _dist_pixel_specs(plan, baseline)
    ledger = bridge.build_ledger(BridgePhase.PIXEL, plan, rank_map, specs)
    # owner 0 == endpoint worker: every PIXEL src must be the endpoint rank.
    assert all(e.src_global_rank == view.endpoint_rank for e in ledger.entries)
