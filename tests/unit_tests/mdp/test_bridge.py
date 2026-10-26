# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unified modality bridge tests: one ledger and one transport for pixels,
embeddings, and gradients, with a real non-local edge.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_bridge.py
"""

import os

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import (
    BridgeBufferKey,
    BridgePhase,
    BridgeTensorSpec,
    ModalityBridge,
)
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1
pytestmark = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


WIDTH = 8
GRIDS = {0: (1, 4, 4), 1: (1, 8, 8), 2: (2, 4, 4)}


def _sentinel(outer_dp_rank, item_id):
    return float(100 * (outer_dp_rank + 1) + item_id)


def _setup():
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    view = rank_map.view(rank)
    # costs 10, 9, 8 over two workers: item 0 -> worker 0 (endpoint-local),
    # items 1 and 2 -> worker 1 (remote edge, coalesced).
    descriptors = [
        _make_descriptor(0, 10, view.outer_dp_rank),
        _make_descriptor(1, 9, view.outer_dp_rank),
        _make_descriptor(2, 8, view.outer_dp_rank),
    ]
    planner = MdpPlanner(
        view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()
    )
    plan = planner.build_plan(0, descriptors, [0])
    return rank_map, view, plan, descriptors


def _make_descriptor(item_id, cost, lane):
    t, h, w = GRIDS[item_id]
    return VisionDescriptor(
        global_item_id=item_id,
        sample_id=item_id,
        image_ordinal=0,
        owner_dp_lane=lane,
        microbatch_id=0,
        estimated_cost_units=cost,
        payload_rows=t * h * w,
        output_rows=t * (h // 2) * (w // 2),
        grid_thw=GRIDS[item_id],
        owner_worker_id=0,
    )


def _pixel_specs(plan, descriptors):
    return {
        BridgeBufferKey(d.global_item_id): BridgeTensorSpec(
            valid_rows=d.payload_rows,
            capacity_rows=plan.capacity_policy.capacity_of(d.payload_rows),
            width=WIDTH,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        for d in descriptors
    }


def test_pixel_embedding_gradient_share_one_transport():
    rank_map, view, plan, descriptors = _setup()
    rank = torch.distributed.get_rank()
    bridge = ModalityBridge(DirectBufferAllocator())
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())

    def exchange(ledger, local_tensors, specs):
        return bridge.exchange_all_to_all(
            ledger,
            local_tensors,
            tensor_specs=specs,
            group=groups.planning_group,
            group_ranks=view.planning_group_ranks,
            global_rank=rank,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )

    assignment = {r.global_item_id: r.producer_worker_id for r in plan.routes}
    assert assignment == {0: 0, 1: 1, 2: 1}, "test premise: one local, two remote"

    # ---- PIXEL: endpoint -> producers ----
    specs = _pixel_specs(plan, descriptors)
    ledger = bridge.build_ledger(BridgePhase.PIXEL, plan, rank_map, specs)
    # Canonical order and coalesced offsets: the two entries for the remote
    # edge carry cumulative element offsets.
    remote_entries = [
        e for e in ledger.entries if e.src_global_rank != e.dst_global_rank
    ]
    assert [e.key.global_item_id for e in remote_entries] == [1, 2]
    assert remote_entries[0].plan_offset == 0
    assert remote_entries[1].plan_offset == 64 * WIDTH
    assert ledger.remote_bytes == (64 + 32) * WIDTH * 4
    assert ledger.total_bytes == (16 + 64 + 32) * WIDTH * 4

    local_tensors = {}
    if view.lane_id is not None:
        for d in descriptors:
            local_tensors[BridgeBufferKey(d.global_item_id)] = torch.full(
                (d.payload_rows, WIDTH),
                _sentinel(view.outer_dp_rank, d.global_item_id),
                device="cuda",
            )
    received = exchange(ledger, local_tensors, specs)
    my_items = [
        r.global_item_id for r in plan.routes_for_producer(view.my_worker_id)
    ] if view.my_worker_id is not None else []
    expected_keys = {BridgeBufferKey(i) for i in my_items}
    if view.lane_id is None and view.my_worker_id == 0:
        expected_keys = set()  # non-endpoint rank hosting no items
    assert set(received.keys()) == expected_keys
    for key, tensor in received.items():
        d = descriptors[key.global_item_id]
        assert tensor.shape == (d.payload_rows, WIDTH)
        assert (tensor == _sentinel(view.outer_dp_rank, key.global_item_id)).all()
    bridge.assert_idle()

    # ---- EMBEDDING: producers -> endpoint, same transport ----
    emb_specs = {
        BridgeBufferKey(d.global_item_id): BridgeTensorSpec(
            valid_rows=d.output_rows,
            capacity_rows=plan.capacity_policy.capacity_of(d.output_rows),
            width=WIDTH,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        for d in descriptors
    }
    emb_ledger = bridge.build_ledger(BridgePhase.EMBEDDING, plan, rank_map, emb_specs)
    emb_local = {}
    for item_id in my_items:
        d = descriptors[item_id]
        emb_local[BridgeBufferKey(item_id)] = torch.full(
            (d.output_rows, WIDTH),
            -_sentinel(view.outer_dp_rank, item_id),
            device="cuda",
        )
    emb_received = exchange(emb_ledger, emb_local, emb_specs)
    if view.lane_id is not None:
        assert set(emb_received.keys()) == {BridgeBufferKey(i) for i in range(3)}
        for key, tensor in emb_received.items():
            assert (tensor == -_sentinel(view.outer_dp_rank, key.global_item_id)).all()
    else:
        assert emb_received == {}

    # ---- GRADIENT: endpoint -> producers, same shapes as embeddings ----
    grad_ledger = bridge.build_ledger(BridgePhase.GRADIENT, plan, rank_map, emb_specs)
    grad_local = {}
    if view.lane_id is not None:
        for d in descriptors:
            grad_local[BridgeBufferKey(d.global_item_id)] = torch.full(
                (d.output_rows, WIDTH),
                2.0 * _sentinel(view.outer_dp_rank, d.global_item_id),
                device="cuda",
            )
    grad_received = exchange(grad_ledger, grad_local, emb_specs)
    assert set(grad_received.keys()) == expected_keys
    for key, tensor in grad_received.items():
        assert (tensor == 2.0 * _sentinel(view.outer_dp_rank, key.global_item_id)).all()

    stats = bridge.last_stats()
    assert set(stats.keys()) == {"pixel", "embedding", "gradient"}
    assert stats["pixel"].remote_bytes == (64 + 32) * WIDTH * 4
    assert stats["pixel"].elapsed_ms >= 0.0


def test_empty_member_performs_noop_exchange():
    rank_map, view, plan, descriptors = _setup()
    bridge = ModalityBridge(DirectBufferAllocator())
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    # An artificial plan with no routes: every member still calls exchange
    # exactly once and gets an empty mapping back.
    empty_planner = MdpPlanner(
        view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()
    )
    empty_plan = empty_planner.build_plan(1, [], [0])
    ledger = bridge.build_ledger(BridgePhase.PIXEL, empty_plan, rank_map, {})
    assert ledger.entries == ()
    received = bridge.exchange_all_to_all(
        ledger,
        {},
        tensor_specs={},
        group=groups.planning_group,
        group_ranks=view.planning_group_ranks,
        global_rank=torch.distributed.get_rank(),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    assert received == {}
    bridge.assert_idle()
