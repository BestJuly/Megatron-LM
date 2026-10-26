# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP buffer allocator: the single point through which MDP allocates memory.

Every buffer MDP itself creates goes through the allocator — bridge staging and
receive buffers, packed pixel buffers, encoder ``cu_seqlens`` metadata tensors,
detached embedding leaves, and gradient regroup buffers. Model-internal
activations and the encoder output returned by ``adapter.encode()`` are not
routed through it.

``DirectBufferAllocator`` is the original transparent wrapper over direct
allocation, reporting zero reuse. ``PooledBufferAllocator`` is the pooled
implementation the module was converged for: it recycles blocks at quantized
sizes so the caching allocator sees a bounded set of block sizes instead of one
fresh size per iteration.
"""

from typing import Mapping, Optional, Protocol

import torch
from torch import Tensor


class MdpBufferAllocator(Protocol):
    """Allocator protocol; see module docstring for scope."""

    def acquire(
        self, *, rows: int, width: int, dtype: torch.dtype, device: torch.device, tag: str
    ) -> Tensor:
        """Return a ``[rows, width]`` tensor (``[rows]`` when ``width == 0``)."""
        ...

    def release(self, tensor: Tensor) -> None:
        """Return a buffer to the allocator."""
        ...

    def reuse_stats(self) -> Mapping[str, int]:
        """Per-tag reuse counts; v1 reports zero reuse for every tag."""
        ...

    def reclaim_iteration(self) -> None:
        """Reclaim every buffer still outstanding at an iteration boundary.

        MDP-owned buffers never outlive one iteration (the same invariant
        ``MdpEmbeddingStorage.assert_empty`` enforces for leaves), but not every
        call site releases explicitly -- packed pixel and gradient-regroup
        buffers are dropped by going out of scope. The runtime calls this once
        per iteration so a pooling implementation can recycle those blocks
        instead of leaking them.
        """
        ...


class DirectBufferAllocator:
    """v1 allocator: direct allocation each iteration, zero reuse."""

    def __init__(self) -> None:
        self._acquired_by_tag: dict = {}
        self._reuse_by_tag: dict = {}
        self._outstanding = 0

    def acquire(
        self, *, rows: int, width: int, dtype: torch.dtype, device: torch.device, tag: str
    ) -> Tensor:
        """Allocate directly; ``width == 0`` requests a 1-D ``[rows]`` tensor."""
        self._acquired_by_tag[tag] = self._acquired_by_tag.get(tag, 0) + 1
        self._reuse_by_tag.setdefault(tag, 0)
        self._outstanding += 1
        shape = (rows,) if width == 0 else (rows, width)
        return torch.empty(shape, dtype=dtype, device=device)

    def release(self, tensor: Tensor) -> None:
        """Drop the reference; the caller must not use the tensor afterwards."""
        del tensor
        self._outstanding = max(0, self._outstanding - 1)

    def reuse_stats(self) -> Mapping[str, int]:
        """Zero for every tag in v1; the CUDA-graph hook test asserts this."""
        return dict(self._reuse_by_tag)

    def acquire_stats(self) -> Mapping[str, int]:
        """Per-tag acquire counts, for observability."""
        return dict(self._acquired_by_tag)

    def reclaim_iteration(self) -> None:
        """No-op: this allocator owns nothing between iterations."""
        self._outstanding = 0


# Blocks are rounded up to eight steps per power-of-two octave. A request wastes
# at most 12.5%, and the number of distinct block sizes stays logarithmic in the
# largest request instead of growing with the number of distinct request sizes --
# which is the whole point, since that count is what drives the caching
# allocator's segment growth.
_BUCKET_STEPS_PER_OCTAVE = 8
_MIN_BUCKET_ELEMS = 1024
_DEFAULT_MAX_FREE_BLOCKS = 64


def bucket_elems(count: int) -> int:
    """Round an element count up to the next pool bucket size."""
    if count <= _MIN_BUCKET_ELEMS:
        return _MIN_BUCKET_ELEMS
    # Step by an eighth of the octave *below* count. Using the octave above
    # would make the step too coarse just past a power of two (1025 -> 1280,
    # a 25% overshoot) and break the 12.5% bound.
    octave = 1 << (count.bit_length() - 1)
    step = octave // _BUCKET_STEPS_PER_OCTAVE
    return (count + step - 1) // step * step


class PooledBufferAllocator:
    """Allocator that recycles MDP-owned buffers at quantized block sizes.

    Why this exists. ``RowCapacityPolicy.alignment_rows`` is 1 in production, so
    ``capacity_of(valid) == valid`` and every MDP buffer -- packed pixels, leaves,
    gradient-regroup buffers, and above all the ``all_to_all_single`` send/receive
    buffers, whose length is ``sum(split_sizes)`` -- is requested at a size that
    depends on how many vision items the planner assigned this iteration. That
    count changes every iteration, so ``DirectBufferAllocator`` hands the caching
    allocator a fresh size every time; blocks cannot be reused, segments
    accumulate, and reserved memory settles far above the live set. Measured on
    4xGB300 (A-mdp, PP2/EP2, 16K THD): rank 0 reserved 162.3 GiB against a
    111.3 GiB allocation peak -- 51 GiB of reserved-but-unused memory, versus
    3.7 GiB on the MDP-off baseline running the same shape.

    The fix is to decouple the block size from the request size. Requests are
    served from per-(dtype, device) pools of 1-D blocks rounded up to a bucket,
    and the caller gets an exactly-shaped view into a block. The set of live
    block sizes is then bounded by the bucket ladder rather than by the number of
    distinct iteration shapes.

    Lifetime. Blocks return to the pool on ``release``, or at the latest on
    ``reclaim_iteration`` at the iteration boundary. Reuse is safe without
    per-block stream bookkeeping because every release point is already
    stream-ordered after the work that reads the buffer (``exchange_all_to_all``
    releases only after ``work.wait()``), and reclaimed blocks are not handed out
    again until the next iteration's acquires on the same stream.
    """

    def __init__(self, *, max_free_blocks_per_pool: int = _DEFAULT_MAX_FREE_BLOCKS) -> None:
        self._max_free = max_free_blocks_per_pool
        self._free: dict = {}
        self._in_use: dict = {}
        self._acquired_by_tag: dict = {}
        self._reuse_by_tag: dict = {}

    @staticmethod
    def _storage_key(tensor: Tensor) -> int:
        """Identify the underlying block, independent of how it was sliced."""
        return tensor.untyped_storage().data_ptr()

    def acquire(
        self, *, rows: int, width: int, dtype: torch.dtype, device: torch.device, tag: str
    ) -> Tensor:
        """Return a ``[rows, width]`` view (``[rows]`` when ``width == 0``)."""
        self._acquired_by_tag[tag] = self._acquired_by_tag.get(tag, 0) + 1
        self._reuse_by_tag.setdefault(tag, 0)
        shape = (rows,) if width == 0 else (rows, width)
        count = rows if width == 0 else rows * width
        if count == 0:
            # Empty buffers have no meaningful storage address; never pool them.
            return torch.empty(shape, dtype=dtype, device=device)

        key = (dtype, str(device))
        free = self._free.setdefault(key, [])
        best = None
        for index, block in enumerate(free):
            if block.numel() >= count and (best is None or block.numel() < free[best].numel()):
                best = index
        if best is None:
            block = torch.empty(bucket_elems(count), dtype=dtype, device=device)
        else:
            block = free.pop(best)
            self._reuse_by_tag[tag] += 1
        self._in_use[self._storage_key(block)] = (key, block)
        flat = block[:count]
        return flat if width == 0 else flat.view(rows, width)

    def release(self, tensor: Tensor) -> None:
        """Return a buffer to its pool; unknown tensors are ignored."""
        entry = self._in_use.pop(self._storage_key(tensor), None)
        if entry is None:
            return
        self._recycle(*entry)

    def reclaim_iteration(self) -> None:
        """Return every still-outstanding block to its pool."""
        outstanding = list(self._in_use.values())
        self._in_use.clear()
        for key, block in outstanding:
            self._recycle(key, block)

    def _recycle(self, key, block: Tensor) -> None:
        free = self._free.setdefault(key, [])
        # Past the cap, drop the block and let the caching allocator take it
        # back: the pool is a working set, not an unbounded cache.
        if len(free) < self._max_free:
            free.append(block)

    def reuse_stats(self) -> Mapping[str, int]:
        """Per-tag counts of acquires served from the pool."""
        return dict(self._reuse_by_tag)

    def acquire_stats(self) -> Mapping[str, int]:
        """Per-tag acquire counts, for observability."""
        return dict(self._acquired_by_tag)

    def pool_stats(self) -> Mapping[str, int]:
        """Pooled block count and bytes, for observability."""
        blocks = [block for blocks in self._free.values() for block in blocks]
        return {
            "free_blocks": len(blocks),
            "free_bytes": sum(b.numel() * b.element_size() for b in blocks),
            "in_use_blocks": len(self._in_use),
        }
