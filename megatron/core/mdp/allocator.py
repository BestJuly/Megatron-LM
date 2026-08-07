# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP buffer allocator: the single point through which MDP allocates memory.

Every buffer MDP itself creates goes through the allocator — bridge staging and
receive buffers, packed pixel buffers, encoder ``cu_seqlens`` metadata tensors,
detached embedding leaves, and gradient regroup buffers. Model-internal
activations and the encoder output returned by ``adapter.encode()`` are not
routed through it.

The v1 implementation is a transparent wrapper over direct allocation and
reports zero reuse; converging allocation into one point makes a future
graph-safe pool a single-file change.
"""

from typing import Mapping, Protocol

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
