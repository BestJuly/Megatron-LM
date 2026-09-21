# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NVTX instrumentation for the multimodal_dev training path.

``megatron/core/mdp`` is well covered by ``nvtx_phase``; the native (MDP-off)
path was not, which made an MDP-on/MDP-off timeline comparison guesswork. These
helpers close that gap using the SAME ``nvtx_phase`` helper under the ``mm.``
namespace, so both arms of a comparison come from one mechanism.

Forward ranges are ordinary context managers. Backward ranges need autograd
markers: a context manager entered during forward is long gone by the time the
backward pass runs, so a region's backward is bracketed by two identity
autograd functions that push/pop from the autograd thread instead.
"""

from contextlib import contextmanager

import torch

from megatron.core.mdp.observability import nvtx_phase as _core_nvtx_phase


@contextmanager
def nvtx_phase(name: str):
    """NVTX range for one multimodal training phase (``mm.<name>``)."""
    with _core_nvtx_phase(name, prefix="mm"):
        yield


class _NvtxBackwardOpen(torch.autograd.Function):
    """Identity forward; pushes an NVTX range when backward reaches it.

    Applied to a region's OUTPUT tensor, which is the first node of that region
    the backward pass visits.
    """

    @staticmethod
    def forward(ctx, tensor, name):
        ctx.nvtx_name = name
        return tensor

    @staticmethod
    def backward(ctx, grad):
        torch.cuda.nvtx.range_push(ctx.nvtx_name)
        return grad, None


class _NvtxBackwardClose(torch.autograd.Function):
    """Identity forward; pops the range when backward leaves the region.

    Applied to the EARLIEST differentiable tensor of the region, which is the
    last node of that region the backward pass visits.
    """

    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad):
        torch.cuda.nvtx.range_pop()
        return grad


def backward_range_begin(tensor: torch.Tensor):
    """Start bracketing a region's backward pass at its EARLIEST tensor.

    Returns ``(tensor, active)``. Pass ``active`` to
    :func:`backward_range_end` unchanged -- the decision to instrument is taken
    here, once, and never re-derived downstream. That is what keeps the NVTX
    push/pop balanced: if this tensor is differentiable then so is every tensor
    the region derives from it, so both markers land on the same autograd graph
    or neither does. Under ``torch.no_grad()`` (evaluation) nothing is marked.

    Both markers are pure identities in forward and only touch NVTX in
    backward, so training results are bit-identical either way.
    """
    if not torch.is_grad_enabled() or not tensor.requires_grad:
        return tensor, False
    return _NvtxBackwardClose.apply(tensor), True


def backward_range_end(tensor: torch.Tensor, name: str, active: bool) -> torch.Tensor:
    """Close the bracket at a region's OUTPUT; ``mm.<name>`` spans its backward."""
    if not active:
        return tensor
    return _NvtxBackwardOpen.apply(tensor, f"mm.{name}")
