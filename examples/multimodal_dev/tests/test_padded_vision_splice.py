# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The padded-leaf vision splice (MDP full-iteration graphs) against masked_scatter.

pytest examples/multimodal_dev/tests/test_padded_vision_splice.py
"""

from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.models.base import MultimodalModel

IMAGE = 7
B, S, D, CAPACITY = 1, 12, 4, 10


def _model(padded):
    return SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=False),
        image_token_id=IMAGE,
        vision_embeddings_padded=padded,
    )


def _splice(padded, input_ids, text, vision):
    return MultimodalModel._scatter_vision_embeddings(_model(padded), input_ids, text, vision)


def _inputs(device="cpu"):
    input_ids = torch.tensor([[1, IMAGE, IMAGE, 2, 3, IMAGE, 4, IMAGE, IMAGE, IMAGE, 5, 6]])
    text = torch.randn(S, B, D, dtype=torch.float64)
    leaf = torch.randn(CAPACITY, D, dtype=torch.float64)
    return input_ids.to(device), text.to(device), leaf.to(device)


def test_padded_splice_matches_masked_scatter_in_values_and_gradients():
    input_ids, text, leaf = _inputs()
    rows = int((input_ids == IMAGE).sum())
    ref_text, ref_vision = text.clone().requires_grad_(), leaf[:rows].clone().requires_grad_()
    pad_text, pad_leaf = text.clone().requires_grad_(), leaf.clone().requires_grad_()
    ref = _splice(False, input_ids, ref_text, ref_vision)
    out = _splice(True, input_ids, pad_text, pad_leaf)
    assert torch.equal(out, ref)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    out.backward(grad)
    assert torch.equal(pad_text.grad, ref_text.grad)
    assert torch.equal(pad_leaf.grad[:rows], ref_vision.grad)
    assert torch.equal(pad_leaf.grad[rows:], torch.zeros_like(pad_leaf.grad[rows:]))


def test_padded_splice_of_a_text_only_microbatch_is_the_identity():
    input_ids, text, leaf = _inputs()
    input_ids = torch.where(input_ids == IMAGE, torch.zeros_like(input_ids), input_ids)
    assert torch.equal(_splice(True, input_ids, text, leaf), text)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_padded_splice_forward_and_backward_are_cuda_graph_capturable():
    input_ids, text, leaf = _inputs("cuda")
    leaf.requires_grad_()
    grad = torch.randn(S, B, D, dtype=torch.float64, device="cuda")

    def step():
        out = _splice(True, input_ids, text, leaf)
        out.backward(grad)
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()  # warm up outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    leaf.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, _splice(False, input_ids, text, leaf.detach()[:6]))
