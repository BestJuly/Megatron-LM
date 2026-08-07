# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the MDP buffer allocator and embedding storage (CPU only)."""

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.storage import MdpEmbeddingStorage


def test_allocator_shapes_and_tags():
    allocator = DirectBufferAllocator()
    two_d = allocator.acquire(
        rows=5, width=7, dtype=torch.float32, device=torch.device("cpu"), tag="leaf"
    )
    assert two_d.shape == (5, 7)
    one_d = allocator.acquire(
        rows=9, width=0, dtype=torch.int32, device=torch.device("cpu"), tag="cu_seqlens"
    )
    assert one_d.shape == (9,)
    assert allocator.acquire_stats() == {"leaf": 1, "cu_seqlens": 1}


def test_allocator_reports_zero_reuse():
    # Registered extension hook (design doc 12.1, CUDA graph): v1 is direct
    # allocation each iteration and must report zero reuse for every tag.
    allocator = DirectBufferAllocator()
    for _ in range(3):
        buffer = allocator.acquire(
            rows=4, width=4, dtype=torch.float32, device=torch.device("cpu"), tag="pixel"
        )
        allocator.release(buffer)
    stats = allocator.reuse_stats()
    assert stats == {"pixel": 0}
    assert all(count == 0 for count in stats.values())


def _leaf(rows=4, width=8):
    return torch.zeros(rows, width, requires_grad=True)


def test_storage_round_trip_and_pop_grad():
    storage = MdpEmbeddingStorage(DirectBufferAllocator())
    leaf = _leaf()
    storage.put_leaf(0, leaf)
    assert storage.get_leaf(0) is leaf
    assert storage.get_leaf(0) is leaf  # non-destructive
    (leaf.sum()).backward()
    grad = storage.pop_grad(0)
    assert torch.equal(grad, torch.ones_like(leaf))
    assert storage.get_leaf(0) is None
    storage.assert_empty()


def test_storage_text_only_reads_return_none():
    storage = MdpEmbeddingStorage(DirectBufferAllocator())
    assert storage.get_leaf(3) is None
    assert storage.pop_grad(3) is None
    storage.assert_empty()


def test_storage_rejects_duplicates_and_non_leaves():
    storage = MdpEmbeddingStorage(DirectBufferAllocator())
    leaf = _leaf()
    storage.put_leaf(0, leaf)
    with pytest.raises(MdpStateError, match="one leaf per microbatch"):
        storage.put_leaf(0, _leaf())
    with pytest.raises(MdpStateError, match="detached leaves"):
        storage.put_leaf(1, torch.zeros(2, 2))  # requires_grad=False
    with pytest.raises(MdpStateError, match="detached leaves"):
        storage.put_leaf(2, _leaf() * 2)  # graph-connected, not a leaf
    storage.release(0)
    storage.assert_empty()


def test_storage_pop_grad_requires_populated_grad():
    storage = MdpEmbeddingStorage(DirectBufferAllocator())
    storage.put_leaf(0, _leaf())
    with pytest.raises(MdpStateError, match="has .grad"):
        storage.pop_grad(0)


def test_storage_assert_empty_names_leftovers():
    storage = MdpEmbeddingStorage(DirectBufferAllocator())
    storage.put_leaf(4, _leaf())
    with pytest.raises(MdpStateError, match="\\[4\\]"):
        storage.assert_empty()
    storage.release(4)  # evaluation-path release
    storage.assert_empty()
