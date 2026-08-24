# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Encoder forward-handle and encoder THD params tests (CPU only)."""

from contextlib import contextmanager

import pytest
import torch

from megatron.core.mdp.activation import (
    EncoderAllRecomputeHandle,
    EncoderForwardHandle,
    EncoderOutputMetadata,
    build_encoder_packed_seq_params,
)
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.plan import EncoderThdLayout, EncoderThdSegment, split_encoder_layout


def _segment(item_id, grid, payload_start, output_start):
    t, h, w = grid
    return EncoderThdSegment(
        global_item_id=item_id,
        microbatch_id=0,
        sample_id=item_id,
        image_ordinal=0,
        payload_row_start=payload_start,
        payload_rows=t * h * w,
        output_row_start=output_start,
        output_rows=t * (h // 2) * (w // 2),
        grid_thw=grid,
    )


def _layout():
    segments = []
    payload = output = 0
    for item_id, grid in enumerate(((1, 4, 4), (2, 4, 8), (1, 8, 8))):
        segment = _segment(item_id, grid, payload, output)
        segments.append(segment)
        payload += segment.payload_rows
        output += segment.output_rows
    return EncoderThdLayout(producer_worker_id=0, segments=tuple(segments))


def test_encoder_packed_seq_params_from_segments():
    layout = _layout()
    params = build_encoder_packed_seq_params(
        layout.segments, allocator=DirectBufferAllocator(), device=torch.device("cpu")
    )
    assert params.qkv_format == "thd"
    # Frames: (1,4,4)->16; (2,4,8)->32,32; (1,8,8)->64.
    assert params.cu_seqlens_q.tolist() == [0, 16, 48, 80, 144]
    assert params.cu_seqlens_q is params.cu_seqlens_kv
    assert params.max_seqlen_q == 64


def test_empty_segments_yield_zero_params():
    params = build_encoder_packed_seq_params(
        (), allocator=DirectBufferAllocator(), device=torch.device("cpu")
    )
    assert params.cu_seqlens_q.tolist() == [0]
    assert params.max_seqlen_q == 0


def _run_encoder(weight, payload_rows_list, seed=0):
    """A tiny stochastic 'encoder': linear + dropout, chunked."""
    torch.manual_seed(seed)
    outputs = []
    for rows in payload_rows_list:
        x = torch.full((rows, 4), 0.5)
        outputs.append(torch.nn.functional.linear(x, weight).relu())
    return outputs


def test_multi_tensor_backward_matches_unchunked_backward():
    layout = _layout()
    chunks = split_encoder_layout(layout, max_payload_rows=80)
    assert len(chunks) == 2

    # Unchunked reference.
    weight_ref = torch.randn(4, 4, requires_grad=True)
    (full,) = _run_encoder(weight_ref, [layout.total_output_rows])
    grad = torch.randn_like(full)
    full.backward(grad)

    # Chunked path through the handle: one multi-tensor backward.
    weight = weight_ref.detach().clone().requires_grad_(True)
    outputs = _run_encoder(weight, [c.total_output_rows for c in chunks])
    handle = EncoderForwardHandle(
        iteration=0,
        producer_worker_id=0,
        chunk_outputs=tuple(outputs),
        chunk_layouts=chunks,
    )
    detached = handle.detached_outputs()
    assert all(t.grad_fn is None and not t.requires_grad for t in detached)

    split_grads = grad.split([c.total_output_rows for c in chunks])
    handle.backward([g.contiguous() for g in split_grads])
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=1e-6, atol=1e-6)
    handle.release()
    assert handle.consumed


def test_handle_validates_shapes_and_lifecycle():
    layout = _layout()
    weight = torch.randn(4, 4, requires_grad=True)
    (output,) = _run_encoder(weight, [layout.total_output_rows])
    handle = EncoderForwardHandle(
        iteration=0,
        producer_worker_id=0,
        chunk_outputs=(output,),
        chunk_layouts=(layout,),
    )
    with pytest.raises(MdpStateError, match="release only after backward"):
        handle.release()
    with pytest.raises(MdpStateError, match="one gradient per chunk"):
        handle.backward([])
    with pytest.raises(MdpStateError, match="shape and dtype"):
        handle.backward([torch.zeros(1, 4)])
    handle.backward([torch.zeros_like(output)])
    with pytest.raises(MdpStateError, match="exactly one backward"):
        handle.backward([torch.zeros_like(output)])
    handle.release()


def test_handle_rejects_mismatched_layout_rows():
    with pytest.raises(MdpStateError, match="total_output_rows"):
        EncoderForwardHandle(
            iteration=0,
            producer_worker_id=0,
            chunk_outputs=(torch.zeros(5, 4, requires_grad=True) * 1.0,),
            chunk_layouts=(_layout(),),
        )


def test_forward_only_release_needs_no_backward():
    layout = _layout()
    with torch.no_grad():
        weight = torch.randn(4, 4)
        (output,) = _run_encoder(weight, [layout.total_output_rows])
    handle = EncoderForwardHandle(
        iteration=0,
        producer_worker_id=0,
        chunk_outputs=(output,),
        chunk_layouts=(layout,),
    )
    handle.release_forward_only()
    assert handle.consumed


def test_all_recompute_handle_replays_and_releases(monkeypatch):
    layout = _layout()
    chunks = split_encoder_layout(layout, max_payload_rows=80)
    assert len(chunks) == 2
    payloads = tuple(torch.randn(chunk.total_payload_rows, 4) for chunk in chunks)
    grads = tuple(torch.randn(chunk.total_output_rows, 4) for chunk in chunks)

    reference = torch.nn.Linear(4, 4, bias=False)
    replay = torch.nn.Linear(4, 4, bias=False)
    replay.load_state_dict(reference.state_dict())

    def encode(module, pixels, chunk_layout):
        return module(pixels[: chunk_layout.total_output_rows])

    reference_outputs = tuple(
        encode(reference, payload, chunk)
        for payload, chunk in zip(payloads, chunks)
    )
    torch.autograd.backward(reference_outputs, grads)
    with torch.no_grad():
        p2_outputs = tuple(
            encode(replay, payload, chunk)
            for payload, chunk in zip(payloads, chunks)
        )

    @contextmanager
    def _noop_fork_rng():
        yield

    from megatron.core.tensor_parallel import random

    monkeypatch.setattr(random, "_fork_rng", _noop_fork_rng)
    monkeypatch.setattr(random, "_set_all_rng_states", lambda *unused: None)

    handle = EncoderAllRecomputeHandle(
        iteration=0,
        producer_worker_id=0,
        chunk_payloads=payloads,
        chunk_layouts=chunks,
        output_metadata=tuple(
            EncoderOutputMetadata.from_tensor(output) for output in p2_outputs
        ),
        chunk_rng_states=((), ()),
    )
    assert handle.output_dtype(0) == p2_outputs[0].dtype
    with pytest.raises(MdpStateError, match="release only after backward"):
        handle.release()
    handle.backward(grads, encoder=replay, encode=encode)
    assert torch.equal(replay.weight.grad, reference.weight.grad)
    handle.release()
    assert handle.consumed
