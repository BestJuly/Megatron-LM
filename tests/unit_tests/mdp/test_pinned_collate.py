# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``pack_or_pad_batch``'s pinned fast path, checked against the generic one.

``use_pinned`` is what ``--mdp-enable`` selects at TP=1 -- the configuration
production trains in -- and it is a second, independent construction of every
packed field. In particular it reads ``max_seqlen_q`` off a host-side list of
padded lengths instead of off the broadcast ``cu_seqlens_padded``, so the tail
the FP8 alignment adds has to be applied to both. Its whole contract is that
the two paths emit the same bytes, so that is what is asserted here.

Lives here rather than beside the other collation tests in
``examples/multimodal_dev/tests/test_thd_e2e.py`` for the reason given in
test_quantized_alignment.py's docstring -- and an unrun test of a fast path is
how this one stayed uncovered. Only the page locking needs a CUDA host
allocator, and page locking cannot change a value, so the fixture below drops
it; every other statement in the branch executes for real.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as ps
from examples.multimodal_dev import forward_step

# Every tensor pack_or_pad_batch emits on the packed path.
PACKED_FIELDS = (
    "input_ids",
    "labels",
    "loss_mask",
    "padding_mask",
    "pixel_values",
    "image_grid_thw",
)


@pytest.fixture(scope="module", autouse=True)
def _tp_group():
    """``pack_or_pad_batch`` ends in a TP broadcast, so it needs a group even
    at TP=1. gloo over a per-process FileStore keeps it on the CPU and off any
    fixed port, so a torchrun session cannot collide on it."""
    already = torch.distributed.is_initialized()
    if not already:
        store_path = os.path.join(tempfile.gettempdir(), f"mdp_pinned_collate_{os.getpid()}")
        store = torch.distributed.FileStore(store_path, 1)
        torch.distributed.init_process_group(backend="gloo", store=store, world_size=1, rank=0)
    ps.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    ps.destroy_model_parallel()
    if not already:
        torch.distributed.destroy_process_group()


@pytest.fixture
def pin_requests(monkeypatch):
    """Strip page locking -- it needs a CUDA host allocator this suite does not
    have, and as an allocator property it is invisible to every value the
    collation computes -- so the rest of the pinned branch can run.

    Yields the list of stripped requests, the only trace the pinned branch
    leaves anywhere: its contract is that the bytes are identical, so nothing
    in the returned batch can tell the two paths apart.
    """
    seen = []

    def _unpinned_factory(name):
        real = getattr(torch, name)

        def factory(*args, **kwargs):
            if kwargs.pop("pin_memory", False):
                seen.append(name)
            return real(*args, **kwargs)

        return factory

    def pin_memory(self):
        seen.append("pin_memory")
        return self

    for name in ("empty", "zeros"):
        monkeypatch.setattr(torch, name, _unpinned_factory(name))
    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    yield seen


def _pack(lens, *, pinned, monkeypatch, pin_requests):
    """Collate samples of length ``lens`` -- each in the shape
    ``CordV2VLMDataset.__getitem__`` returns -- at ``pad_to_multiple=16``, with
    the pinned fast path on or off. ``mdp_enable`` is the field the branch
    reads, i.e. the switch ``--mdp-enable`` flips."""
    monkeypatch.setattr(
        forward_step,
        "get_args",
        lambda: SimpleNamespace(sequence_parallel=False, mdp_enable=pinned),
    )
    batch = [
        {
            "input_ids": torch.arange(length, dtype=torch.long) + 100 * i,
            "labels": torch.arange(length, dtype=torch.long) + 100 * i + 100,
            "loss_mask": torch.ones(length, dtype=torch.float),
            "pixel_values": torch.full((4, 8), float(100 * i)),
            "image_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long),
        }
        for i, length in enumerate(lens)
    ]
    pin_requests.clear()
    packed = forward_step.pack_or_pad_batch(
        batch, use_packed_sequence=True, pad_to_multiple=16, device="cpu"
    )
    # Guard the comparison below against comparing the generic path with
    # itself, which is what a mis-set switch would silently turn it into.
    assert bool(pin_requests) is pinned
    return packed


@pytest.mark.parametrize(
    "lens, total, max_seqlen",
    # lens=[5, 3]: real total 8, so the last sample's padded region carries an
    # 8-row tail and ends up the longest. lens=[8, 8]: total already a multiple
    # of 16, so there is no tail to apply and the two paths must still agree.
    [((5, 3), 16, 11), ((8, 8), 16, 8)],
    ids=["tail", "already_aligned"],
)
def test_pinned_path_matches_the_generic_one(lens, total, max_seqlen, monkeypatch, pin_requests):
    """Same batch, same ``pad_to_multiple``, both branches."""
    kwargs = dict(monkeypatch=monkeypatch, pin_requests=pin_requests)
    pinned = _pack(lens, pinned=True, **kwargs)
    generic = _pack(lens, pinned=False, **kwargs)
    p, g = pinned["packed_seq_params"], generic["packed_seq_params"]

    # Anchored as well as cross-checked -- two paths agreeing on a wrong tail
    # would otherwise read as a pass.
    assert p.total_tokens == g.total_tokens == total
    assert p.max_seqlen_q == g.max_seqlen_q == max_seqlen
    assert p.cu_seqlens_q.tolist() == g.cu_seqlens_q.tolist()
    assert p.cu_seqlens_q_padded.tolist() == g.cu_seqlens_q_padded.tolist()
    assert p.pad_between_seqs is g.pad_between_seqs is False
    for field in PACKED_FIELDS:
        assert torch.equal(pinned[field], generic[field]), field
