# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MTP's precomputed-embedding fill: semantics, pass-through, and CUDA graph capture."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

_LAYER = SimpleNamespace(config=SimpleNamespace(mtp_detach_heads=False))


def _fill(decoder_input, hidden=None):
    if hidden is None:
        hidden = torch.randn(*decoder_input.shape[:2], 3, device=decoder_input.device)
    out, _ = MultiTokenPredictionLayer._get_precomputed_embeddings(_LAYER, decoder_input, hidden)
    return out


def test_zero_rows_take_the_mean_of_the_non_zero_rows():
    x = torch.randn(6, 2, 4, dtype=torch.float64)
    x[1, 0] = 0
    x[4, 1] = 0
    out = _fill(x)
    non_zero = torch.ones(6, 2, dtype=torch.bool)
    non_zero[1, 0] = non_zero[4, 1] = False
    expected = x.clone()
    expected[~non_zero] = x[non_zero].mean(dim=0)
    torch.testing.assert_close(out, expected)


def test_rows_pass_through_unchanged_without_zero_rows():
    x = torch.randn(5, 2, 4, requires_grad=True)
    out = _fill(x)
    assert torch.equal(out, x)
    grad = torch.randn_like(out)
    out.backward(grad)
    assert torch.equal(x.grad, grad)


def test_all_zero_input_stays_zero():
    assert torch.equal(_fill(torch.zeros(3, 1, 4)), torch.zeros(3, 1, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fill_is_cuda_graph_capturable():
    x = torch.randn(8, 2, 4, device="cuda")
    x[3, 1] = 0
    hidden = torch.randn(8, 2, 3, device="cuda")
    _fill(x, hidden)  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = _fill(x, hidden)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, _fill(x))
