# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP structured observability (API design section 20).

Bridge stats come from ``ModalityBridge.last_stats()``; the iteration metrics
are assembled by the runtime at ``end_iteration``. Timing fields measure
completed phase latency, not asynchronous launch latency. Lifecycle facts
(unconsumed handles, non-empty storage) are enforced as invariants at the
iteration boundary, not reported as metrics — they are necessarily zero at a
clean boundary and carry no information as time series.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class MdpIterationMetrics:
    """One iteration's structured MDP metrics."""

    iteration: int
    outer_dp_rank: int
    plan_build_ms: float
    encoder_forward_ms: float
    decoder_schedule_ms: float
    encoder_backward_ms: float
    worker_loads: tuple
    empty_workers: int
    bridge_stats: Mapping
    allocator_reuse: Mapping


@contextmanager
def nvtx_phase(name: str):
    """NVTX range for one MDP phase (visible in nsys timelines)."""
    torch.cuda.nvtx.range_push(f"mdp.{name}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def worker_loads_from_plan(plan, num_workers: int) -> tuple:
    """Per-logical-worker payload rows for all ``num_workers`` workers."""
    loads = {
        layout.producer_worker_id: layout.total_payload_rows
        for layout in plan.encoder_layouts
    }
    return tuple(loads.get(worker_id, 0) for worker_id in range(num_workers))
