# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PIXEL routing under encoder CP: every producing rank receives exactly its
zigzag shard of each item, sized by the same formula the runtime uses.

Pure compute: rank map, planner, and ``build_ledger`` -- no process groups.

The hazard this pins: the PIXEL key gained a ``shard_id`` axis and the payload
sizing became per shard in the SAME change. Had only the key been split, every
rank would have received the item's FIRST ``payload_rows`` rows with matching
sizes on both ends of the wire and a silently wrong loss. So the tests below
check the ledger against ``encoder_cp_partition.shard_rows`` row for row, not
just that the entry count looks right.
"""

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import BridgeBufferKey, BridgePhase, BridgeTensorSpec, ModalityBridge
from megatron.core.mdp.encoder_cp_partition import shard_rows
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map

MERGE = 2
WIDTH = 24


def _descriptors(grids):
    out, offset = [], 0
    for index, grid in enumerate(grids):
        t, h, w = grid
        rows = t * (h // MERGE) * (w // MERGE)
        out.append(
            VisionDescriptor(
                global_item_id=index,
                sample_id=0,
                image_ordinal=index,
                owner_dp_lane=0,
                microbatch_id=0,
                estimated_cost_units=t * h * w,
                payload_rows=t * h * w,
                output_rows=rows,
                grid_thw=grid,
                owner_worker_id=0,
                sample_padded_start=0,
                sample_padded_len=1 << 12,
                decoder_offset_in_sample=offset,
            )
        )
        offset += rows
    return out


def _pixel_specs(plan, encoder_cp):
    """The runtime's PIXEL sizing rule, transcribed: per (item, shard), rows/e."""
    specs = {}
    for route in plan.routes:
        if route.slice_id != 0:
            continue
        segment = plan.segment_for_item(route.global_item_id)
        assert segment.payload_rows % encoder_cp == 0
        valid = segment.payload_rows // encoder_cp
        for shard_id in range(encoder_cp):
            specs[BridgeBufferKey(route.global_item_id, 0, shard_id)] = BridgeTensorSpec(
                valid_rows=valid,
                capacity_rows=plan.capacity_policy.capacity_of(valid),
                width=WIDTH,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
    return specs


def _plan_and_ledger(cp, pp, encoder_cp, grids):
    world = cp * pp
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=pp, cp=cp, ep=1, encoder_cp=encoder_cp)
    )
    view = rank_map.view(0)
    planner = MdpPlanner(view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy())
    plan = planner.build_plan(0, _descriptors(grids), [0])
    specs = _pixel_specs(plan, encoder_cp)
    ledger = ModalityBridge(DirectBufferAllocator()).build_ledger(
        BridgePhase.PIXEL, plan, rank_map, specs
    )
    return rank_map, plan, specs, ledger


GRIDS = [(1, 4, 4), (1, 8, 8), (2, 4, 8)]  # 16, 64, 64 patch rows; all % 4 == 0


@pytest.mark.parametrize("cp,pp,encoder_cp", [(1, 2, 2), (2, 2, 2), (1, 4, 2), (1, 4, 4)])
def test_each_producer_rank_receives_exactly_its_shard(cp, pp, encoder_cp):
    rank_map, plan, specs, ledger = _plan_and_ledger(cp, pp, encoder_cp, GRIDS)
    by_item = {}
    for entry in ledger.entries:
        by_item.setdefault(entry.key.global_item_id, []).append(entry)

    for route in plan.routes:
        if route.slice_id != 0:
            continue
        item = route.global_item_id
        segment = plan.segment_for_item(item)
        producer_ranks = rank_map.worker_ranks(plan.outer_dp_rank, route.producer_worker_id)
        owner_ranks = rank_map.worker_ranks(plan.outer_dp_rank, route.owner_worker_id)
        entries = sorted(by_item[item], key=lambda e: e.key.shard_id)

        # One entry per producing rank, keyed by that rank's shard, sourced by
        # the owner's lead.
        assert [e.key.shard_id for e in entries] == list(range(encoder_cp))
        assert [e.dst_global_rank for e in entries] == list(producer_ranks)
        assert {e.src_global_rank for e in entries} == {owner_ranks[0]}

        # Each entry carries exactly the shard's rows -- the partition's answer,
        # not a fraction of the item that merely happens to have the right size.
        t, h, w = segment.grid_thw
        frames = [h * w] * t
        for entry in entries:
            runs = shard_rows(frames, encoder_cp, entry.key.shard_id)
            assert entry.element_count == sum(r.rows for r in runs) * WIDTH
            assert entry.element_count == specs[entry.key].valid_rows * WIDTH

        # The shards tile the item exactly once.
        covered = []
        for shard_id in range(encoder_cp):
            for run in shard_rows(frames, encoder_cp, shard_id):
                covered.extend(range(run.start, run.start + run.rows))
        assert sorted(covered) == list(range(segment.payload_rows))


def test_encoder_cp_one_is_the_old_behaviour():
    """At e=1 there is one entry per item with shard_id 0, sized by payload_rows."""
    _, plan, specs, ledger = _plan_and_ledger(1, 2, 1, GRIDS)
    items = {r.global_item_id for r in plan.routes}
    assert len(ledger.entries) == len(items)
    for entry in ledger.entries:
        assert entry.key.shard_id == 0
        segment = plan.segment_for_item(entry.key.global_item_id)
        assert entry.element_count == segment.payload_rows * WIDTH


def test_split_key_without_split_spec_is_refused():
    """The hazard, made loud: a spec keyed per item cannot serve per-shard keys.

    If the sizing were still per item while the key carried a shard axis,
    build_ledger would not find a spec for (item, shard=1) and must raise
    rather than fall back to the whole-item size.
    """
    from megatron.core.mdp.errors import MdpBridgeError

    rank_map, plan, _, _ = _plan_and_ledger(1, 2, 2, GRIDS)
    item_only_specs = {}
    for route in plan.routes:
        if route.slice_id != 0:
            continue
        segment = plan.segment_for_item(route.global_item_id)
        item_only_specs[BridgeBufferKey(route.global_item_id)] = BridgeTensorSpec(
            valid_rows=segment.payload_rows,
            capacity_rows=plan.capacity_policy.capacity_of(segment.payload_rows),
            width=WIDTH,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
    with pytest.raises(MdpBridgeError, match="tensor spec"):
        ModalityBridge(DirectBufferAllocator()).build_ledger(
            BridgePhase.PIXEL, plan, rank_map, item_only_specs
        )
