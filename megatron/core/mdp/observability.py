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
    # The same loads divided by encoder_cp: what each GPU actually encodes.
    # Identical to worker_loads at encoder_cp == 1.
    rank_loads: tuple
    empty_workers: int
    bridge_stats: Mapping
    allocator_reuse: Mapping


@contextmanager
def nvtx_phase(name: str, prefix: str = "mdp"):
    """NVTX range for one phase (visible in nsys timelines).

    Args:
        name: Phase name, e.g. ``p2_encoder_forward``.
        prefix: Namespace prepended with a dot. Defaults to ``mdp`` for the
            phases of this package; the multimodal training path passes ``mm``
            so an MDP-off timeline carries comparable ranges from the same
            helper instead of a second, divergent mechanism.
    """
    torch.cuda.nvtx.range_push(f"{prefix}.{name}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def worker_loads_from_plan(plan, num_workers: int) -> tuple:
    """Payload rows per logical worker for all ``num_workers`` workers.

    These are LOGICAL-worker units. At ``encoder_cp > 1`` a worker is
    ``encoder_cp`` GPUs sharing one chunk, so a load here is ``encoder_cp``
    times what any single GPU encodes -- see :func:`rank_loads_from_worker_loads`
    for the per-GPU view. Reporting only this number at ``encoder_cp > 1``
    overstates per-GPU vision work by exactly that factor, and at
    ``encoder_cp == cp*pp`` there is a single worker per group so the tuple
    carries no balance information at all.
    """
    loads = {
        layout.producer_worker_id: layout.total_payload_rows
        for layout in plan.encoder_layouts
    }
    return tuple(loads.get(worker_id, 0) for worker_id in range(num_workers))


def rank_loads_from_worker_loads(worker_loads: tuple, encoder_cp: int) -> tuple:
    """What each GPU actually encodes: the worker's load split ``encoder_cp`` ways.

    Identity at ``encoder_cp == 1``. The split is even because the zigzag shard
    gives every rank ``sum(frame_lengths) // encoder_cp`` rows exactly.
    """
    if encoder_cp <= 1:
        return worker_loads
    return tuple(load // encoder_cp for load in worker_loads)
