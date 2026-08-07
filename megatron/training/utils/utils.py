# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import logging
import os
from argparse import Namespace
from pathlib import Path

import torch

from megatron.core._rank_utils import safe_get_rank
from megatron.training.config import ProfilingConfig

logger = logging.getLogger(__name__)


def ranked_memory_snapshot_path(snapshot_path: str, rank: int | None = None) -> str:
    """Return a collision-free per-rank CUDA memory snapshot path."""
    rank = safe_get_rank() if rank is None else rank
    path = Path(snapshot_path)
    suffix = path.suffix or ".pickle"
    stem = path.stem if path.suffix else path.name
    return str(path.with_name(f"{stem}_rank-{rank}{suffix}"))


def start_memory_history_recording(profiling: ProfilingConfig | Namespace | None) -> None:
    """Enable the CUDA caching allocator trace so memory snapshots contain history.

    ``torch.cuda.memory._snapshot()`` only includes allocation/free events and
    Python stack context after ``_record_memory_history()`` has been enabled.
    Without this call, dumped snapshots contain only the current live
    allocations — no timeline, no call sites.

    Must be invoked before model construction so every tensor allocation is
    captured. Guarded by ``profile_ranks`` so only ranks that will dump a
    snapshot pay the recording overhead.
    """
    if profiling is None or not getattr(profiling, "record_memory_history", False):
        return
    profile_ranks = getattr(profiling, "profile_ranks", [])
    if len(profile_ranks) != 0:
        if safe_get_rank() not in profile_ranks:
            return

    torch.cuda.memory._record_memory_history(
        True,
        # Retain up to 100k alloc/free events.
        trace_alloc_max_entries=100_000,
        # Record the Python stack at each event — lets memory_viz show call sites.
        trace_alloc_record_context=True,
    )

    def _oom_observer(device: int, alloc: int, device_alloc: int, device_free: int) -> None:
        """Dump a snapshot on OOM so we can inspect what was live at the failure."""
        rank = safe_get_rank()
        base, ext = os.path.splitext(profiling.memory_snapshot_path)
        filename = f"{base}_oom_rank-{rank}{ext}"
        torch.cuda.memory._dump_snapshot(filename)
        # logger.info so the message reaches stderr on any profiled rank, not just rank 0.
        logger.info(f"[OOM] rank {rank} saved memory snapshot to {filename}")

    torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
    snapshot_path = ranked_memory_snapshot_path(profiling.memory_snapshot_path)
    Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Memory history recording enabled on rank %s; snapshot will be written to '%s'.",
        safe_get_rank(),
        snapshot_path,
    )
