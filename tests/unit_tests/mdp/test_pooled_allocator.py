# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the pooled MDP buffer allocator (CPU only).

The pool exists to bound the number of distinct block sizes the caching
allocator sees. These tests pin the two properties that makes true -- bucketed
block sizes and actual reuse across iterations -- plus the correctness contracts
the call sites depend on: exact shapes, distinct storage for buffers that are
live at the same time, and the autograd-leaf property MdpEmbeddingStorage
asserts on.
"""

import pytest
import torch

from megatron.core.mdp.allocator import (
    DirectBufferAllocator,
    PooledBufferAllocator,
    bucket_elems,
)
from megatron.core.mdp.storage import MdpEmbeddingStorage

CPU = torch.device("cpu")


def _acquire(allocator, rows, width=0, dtype=torch.float32, tag="t"):
    return allocator.acquire(rows=rows, width=width, dtype=dtype, device=CPU, tag=tag)


# --------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 7, 1024])
def test_small_requests_share_one_bucket(count):
    assert bucket_elems(count) == 1024


@pytest.mark.parametrize("count", [1025, 1500, 4097, 100_003, 1 << 20])
def test_bucket_fits_and_wastes_at_most_one_eighth(count):
    size = bucket_elems(count)
    assert size >= count
    # Eight steps per octave, so the round-up never exceeds 12.5%.
    assert size <= count * 9 // 8 + 1  # 12.5% bound


def test_bucket_ladder_is_small_over_a_wide_range():
    # The point of the pool: many distinct request sizes collapse onto few
    # block sizes. A raw allocator would see 4000 sizes here.
    sizes = {bucket_elems(n) for n in range(1000, 5000)}
    assert len(sizes) <= 24  # measured: 19


# --------------------------------------------------------------------------
# Shapes and reuse
# --------------------------------------------------------------------------


def test_shapes_are_exact_despite_bucketing():
    allocator = PooledBufferAllocator()
    two_d = _acquire(allocator, 5, 7)
    one_d = _acquire(allocator, 9, 0, dtype=torch.int32)
    assert two_d.shape == (5, 7)
    assert one_d.shape == (9,)
    assert allocator.acquire_stats() == {"t": 2}


def test_release_then_acquire_reuses_the_block():
    allocator = PooledBufferAllocator()
    first = _acquire(allocator, 100, 8, tag="leaf")
    ptr = first.untyped_storage().data_ptr()
    allocator.release(first)
    second = _acquire(allocator, 100, 8, tag="leaf")
    assert second.untyped_storage().data_ptr() == ptr
    assert allocator.reuse_stats()["leaf"] == 1


def test_varying_sizes_within_a_bucket_reuse_one_block():
    # The regression this whole change is about: a request size that drifts
    # every iteration must not force a new block every iteration.
    allocator = PooledBufferAllocator()
    pointers = set()
    # 1800..1849 all round up to the same 1920-element block.
    for rows in range(1800, 1850):
        buffer = _acquire(allocator, rows, 0, tag="bridge_a2a_send")
        pointers.add(buffer.untyped_storage().data_ptr())
        allocator.release(buffer)
    assert len(pointers) == 1
    assert allocator.reuse_stats()["bridge_a2a_send"] == 49


def test_drifting_sizes_across_buckets_stay_bounded():
    # Sizes that wander across bucket boundaries still collapse onto the small
    # ladder rather than allocating once per distinct size.
    allocator = PooledBufferAllocator()
    pointers = set()
    for rows in range(1000, 5000):
        buffer = _acquire(allocator, rows, 0, tag="bridge_a2a_send")
        pointers.add(buffer.untyped_storage().data_ptr())
        allocator.release(buffer)
    assert len(pointers) <= 20  # measured: 19 distinct block sizes


def test_a_larger_request_does_not_reuse_a_too_small_block():
    allocator = PooledBufferAllocator()
    small = _acquire(allocator, 1024)
    allocator.release(small)
    big = _acquire(allocator, 1 << 20)
    assert big.numel() == 1 << 20
    assert allocator.reuse_stats()["t"] == 0


def test_pools_do_not_mix_dtypes():
    allocator = PooledBufferAllocator()
    allocator.release(_acquire(allocator, 4096, dtype=torch.float32))
    fresh = _acquire(allocator, 4096, dtype=torch.bfloat16)
    assert fresh.dtype is torch.bfloat16
    assert allocator.reuse_stats()["t"] == 0


def test_simultaneously_live_buffers_get_distinct_storage():
    # P3 assembles one leaf per vision-bearing microbatch and they are all live
    # at once; handing out the same block twice would silently alias them.
    allocator = PooledBufferAllocator()
    live = [_acquire(allocator, 512, 8, tag="leaf") for _ in range(8)]
    pointers = {buffer.untyped_storage().data_ptr() for buffer in live}
    assert len(pointers) == 8


# --------------------------------------------------------------------------
# Iteration lifetime
# --------------------------------------------------------------------------


def test_reclaim_iteration_recovers_unreleased_buffers():
    # packed_pixels and grad_regroup are dropped by scope, never released.
    allocator = PooledBufferAllocator()
    for _ in range(4):
        _acquire(allocator, 4096, 8, tag="grad_regroup")
        allocator.reclaim_iteration()
    assert allocator.pool_stats()["in_use_blocks"] == 0
    # One block was allocated and then reused by each later iteration.
    assert allocator.reuse_stats()["grad_regroup"] == 3


def test_without_reclaim_unreleased_buffers_are_not_reused():
    allocator = PooledBufferAllocator()
    for _ in range(4):
        _acquire(allocator, 4096, 8, tag="grad_regroup")
    assert allocator.reuse_stats()["grad_regroup"] == 0
    assert allocator.pool_stats()["in_use_blocks"] == 4


def test_free_list_is_capped():
    allocator = PooledBufferAllocator(max_free_blocks_per_pool=2)
    live = [_acquire(allocator, 4096, tag="x") for _ in range(5)]
    for buffer in live:
        allocator.release(buffer)
    assert allocator.pool_stats()["free_blocks"] == 2


def test_release_of_an_unknown_tensor_is_ignored():
    allocator = PooledBufferAllocator()
    allocator.release(torch.empty(16))
    assert allocator.pool_stats()["free_blocks"] == 0


def test_zero_row_requests_are_not_pooled():
    # Empty buffers have no meaningful storage address.
    allocator = PooledBufferAllocator()
    empty = _acquire(allocator, 0, 8)
    assert empty.shape == (0, 8)
    allocator.release(empty)
    assert allocator.pool_stats()["free_blocks"] == 0


# --------------------------------------------------------------------------
# Contracts the call sites rely on
# --------------------------------------------------------------------------


def test_acquired_buffer_can_become_an_autograd_leaf():
    # runtime.py hands MdpEmbeddingStorage a slice of the acquired buffer and
    # calls requires_grad_(True) on it; put_leaf asserts is_leaf and no grad_fn.
    allocator = PooledBufferAllocator()
    buffer = _acquire(allocator, 16, 8, tag="leaf")
    leaf = buffer[:10]
    leaf.requires_grad_(True)
    assert leaf.is_leaf and leaf.requires_grad and leaf.grad_fn is None
    MdpEmbeddingStorage(allocator).put_leaf(0, leaf)


def test_storage_release_of_a_slice_returns_the_whole_block():
    # storage.pop_grad releases the *slice* it was handed, not the tensor
    # acquire returned, so the pool must key on the underlying storage.
    allocator = PooledBufferAllocator()
    buffer = _acquire(allocator, 16, 8, tag="leaf")
    ptr = buffer.untyped_storage().data_ptr()
    allocator.release(buffer[:10])
    assert allocator.pool_stats()["free_blocks"] == 1
    assert _acquire(allocator, 16, 8, tag="leaf").untyped_storage().data_ptr() == ptr


def test_reuse_does_not_alias_a_still_live_buffer():
    allocator = PooledBufferAllocator()
    held = _acquire(allocator, 4096, tag="a")
    other = _acquire(allocator, 4096, tag="b")
    assert held.untyped_storage().data_ptr() != other.untyped_storage().data_ptr()


def test_pooled_and_direct_agree_on_shapes_and_dtypes():
    pooled, direct = PooledBufferAllocator(), DirectBufferAllocator()
    for rows, width, dtype in ((5, 7, torch.float32), (9, 0, torch.int32), (0, 4, torch.bfloat16)):
        a = pooled.acquire(rows=rows, width=width, dtype=dtype, device=CPU, tag="t")
        b = direct.acquire(rows=rows, width=width, dtype=dtype, device=CPU, tag="t")
        assert a.shape == b.shape and a.dtype == b.dtype
