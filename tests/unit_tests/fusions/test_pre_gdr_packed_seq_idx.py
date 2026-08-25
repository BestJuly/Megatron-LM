# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``_resolve_packed_seq_idx``'s CUDA-graph-safe seq_idx construction.

The map must stay element-wise identical to the ``repeat_interleave``
formulation it replaced, while having a shape fixed by ``total_tokens`` rather
than one read from device data -- otherwise pre-GDR fusion cannot be captured.
"""

import pytest
import torch

from megatron.core.fusions.fused_pre_gated_delta_rule import _resolve_packed_seq_idx

# (cu_seqlens, total_tokens); the last two exercise trailing zero-length
# sequences, which is how padded THD batches present.
LAYOUTS = [
    ([0, 512, 1024], 1024),
    ([0, 100, 3100, 4096], 4096),
    ([0, 64, 128, 192, 4096], 4096),
    ([0, 4096], 4096),
    ([0, 1, 4096], 4096),
    ([0, 512, 1024, 1024, 1024], 1024),
]


def _reference(cu_seqlens, total_tokens):
    """The pre-fix expression: correct, but its length comes from device data."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.repeat_interleave(
        torch.arange(lens.numel(), device=cu_seqlens.device, dtype=torch.int32), lens
    ).unsqueeze(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("cu, total", LAYOUTS)
def test_matches_repeat_interleave_reference(cu, total):
    cu_seqlens = torch.tensor(cu, device="cuda", dtype=torch.int32)
    got = _resolve_packed_seq_idx(cu_seqlens, None, total)
    expected = _reference(cu_seqlens, total)
    assert got.shape == expected.shape == (1, total)
    assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_is_cuda_graph_capturable():
    """The whole point: no device->host dependency during capture."""
    cu_seqlens = torch.tensor([0, 512, 1024], device="cuda", dtype=torch.int32)
    _resolve_packed_seq_idx(cu_seqlens, None, 1024)  # warm up allocators/kernels
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _resolve_packed_seq_idx(cu_seqlens, None, 1024)

    # Replay with a different packing: values must follow the buffer contents.
    cu_seqlens.copy_(torch.tensor([0, 256, 1024], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, _reference(cu_seqlens, 1024))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_passthrough_and_none_paths():
    assert _resolve_packed_seq_idx(None, None, 0) is None

    cu_seqlens = torch.tensor([0, 512, 1024], device="cuda", dtype=torch.int32)
    supplied = torch.zeros(1024, device="cuda", dtype=torch.int32)
    out = _resolve_packed_seq_idx(cu_seqlens, supplied, 1024)
    assert out.shape == (1, 1024)
    assert torch.equal(out[0], supplied)
