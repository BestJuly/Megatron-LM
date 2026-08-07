# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Registered extension hooks must be exercised at non-degenerate values
(design doc 12.1). Two of the four hooks live in their subsystem suites:

* encoder CP: pure-compute rank-map test at encoder_cp=2
  (test_rank_mapping.test_extension_hook_encoder_cp2);
* CUDA graph: allocator zero-reuse assertion
  (test_allocator_storage.test_allocator_reports_zero_reuse).

This file covers the remaining two:

* decoder CP: a bridge exchange where one item is artificially split into two
  route slices (two ledger entries with distinct slice_ids);
* FP8 row capacity: a full training iteration at row_alignment=16, where
  capacity_rows != valid_rows for every buffer.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_extension_hooks.py
"""

import os

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import (
    BridgeBufferKey,
    BridgeLedger,
    BridgeLedgerEntry,
    BridgePhase,
    BridgeTensorSpec,
    ModalityBridge,
)
from megatron.core.mdp.plan import RowCapacityPolicy

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


def test_bridge_supports_two_slices_of_one_item():
    """Decoder-CP hook: the structures and the ledger carry a split item.

    One 24-row item produced on every even rank is delivered to its odd
    neighbour as two slices (rows [0, 16) and [16, 24)), each with its own
    slice_id, coalesced on the same edge. The reassembled halves must equal
    the source rows exactly.

    Deliberately constructed at the ledger layer: in v1 the plan cannot
    represent a split (RouteSlice gains item_row_start/item_rows only when a
    real split lands, API design 6.2), so build_ledger emits one entry per
    route by definition. When decoder CP adds those fields, this test MUST be
    extended to drive the split through MdpBatchPlan and build_ledger and to
    assert the generated ledger before exchanging it.
    """
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    src = rank - (rank % 2)  # even peer
    dst = src + 1
    item_id = src  # unique per pair

    keys = (BridgeBufferKey(item_id, slice_id=0), BridgeBufferKey(item_id, slice_id=1))
    rows = (16, 8)
    specs = {
        key: BridgeTensorSpec(
            valid_rows=row_count,
            capacity_rows=row_count,
            width=WIDTH,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        for key, row_count in zip(keys, rows)
    }
    entries = []
    offset = 0
    for key, row_count in zip(keys, rows):
        entries.append(
            BridgeLedgerEntry(
                phase=BridgePhase.EMBEDDING,
                src_global_rank=src,
                dst_global_rank=dst,
                dtype=torch.float32,
                element_count=row_count * WIDTH,
                plan_offset=offset,
                key=key,
            )
        )
        offset += row_count * WIDTH
    total_bytes = offset * 4
    ledger = BridgeLedger(
        phase=BridgePhase.EMBEDDING,
        entries=tuple(entries),
        total_bytes=total_bytes,
        remote_bytes=total_bytes,
    )

    source_item = torch.arange(24 * WIDTH, dtype=torch.float32, device="cuda").view(
        24, WIDTH
    ) + 1000.0 * src
    local_tensors = {}
    if rank == src:
        local_tensors = {keys[0]: source_item[:16], keys[1]: source_item[16:]}

    bridge = ModalityBridge(DirectBufferAllocator())
    received = bridge.exchange(
        ledger, local_tensors, tensor_specs=specs, global_rank=rank
    )
    if rank == dst:
        assert set(received) == set(keys)
        reassembled = torch.cat([received[keys[0]], received[keys[1]]])
        expected = torch.arange(24 * WIDTH, dtype=torch.float32, device="cuda").view(
            24, WIDTH
        ) + 1000.0 * src
        assert torch.equal(reassembled, expected)
    else:
        assert received == {}
    bridge.assert_idle()


def test_full_iteration_at_row_alignment_16():
    """FP8 hook: the full consistency suite must pass with capacity != valid.

    Reuses the runtime harness with MdpConfig(row_alignment=16); item rows
    (16/64/32 payload, 4/16/8 output) are mostly not multiples of 16, so every
    leaf, pixel pack, and bridge buffer over-allocates while only valid rows
    move and unpack.
    """
    from megatron.core.mdp.config import MdpConfig
    from tests.unit_tests.mdp import test_runtime as harness

    policy = RowCapacityPolicy(alignment_rows=16)
    assert policy.capacity_of(4) == 16  # genuinely non-degenerate

    runtime, view = harness._build_runtime()
    # Rebuild planner and config at alignment 16 on the same domain.
    from megatron.core.mdp.planner import MdpPlanner

    runtime.config = MdpConfig(enable=True, row_alignment=16)
    runtime.planner = MdpPlanner(
        view, locality_slack_permille=10, capacity_policy=policy
    )

    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    harness._drive_decoder(runtime, view, replay, backward=True)
    tokens = torch.tensor(16.0, device="cuda")
    runtime.capture_global_num_tokens(tokens)
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    param = next(runtime.encoder_domain.encoder_ddp.module.parameters())
    assert param.main_grad.abs().sum() > 0
