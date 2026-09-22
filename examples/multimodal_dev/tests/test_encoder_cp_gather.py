# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The encoder-CP all-gather and, above all, its backward.

``megatron.core.mdp.encoder_cp_partition`` is covered arithmetically by
``tests/unit_tests/mdp/test_encoder_cp_partition.py``, but nothing exercised
``_GatherChunkAlongSeq`` -- the autograd Function that actually moves the
tensors. That is the one place in the encoder-CP path where a wrong answer is
silent: shapes stay correct, the loss still falls, and only ``1/e`` of the
vision encoder is being trained.

The asymmetry that makes the backward load-bearing: after the gather every rank
of a worker runs the merger on the full chunk, but only the worker's LEAD rank
feeds the bridge, so only the lead receives a real upstream gradient -- the
others are driven with explicitly zeroed buffers (``runtime.py`` zeroes
``grad_buffer`` on non-lead ranks). A "take my own slice of grad_out" backward
would therefore hand every non-lead rank a gradient of exactly zero. The
reduce-scatter sums (real + zeros) first and *then* slices, which is what
routes the lead's gradient back to the blocks that actually computed those rows.

``test_non_lead_ranks_receive_the_leads_gradient`` is the discriminating case:
it fails on the naive implementation and passes on this one.

Run with::

    PYTHONPATH=. torchrun --nproc-per-node 8 -m pytest -q \\
        examples/multimodal_dev/tests/test_encoder_cp_gather.py
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.qwen35_vl.vision_encoder import _GatherChunkAlongSeq
from megatron.core.mdp.encoder_cp_partition import gather_permutation, shard_rows
from tests.unit_tests.test_utilities import Utils

_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(_WORLD < 2, reason="needs a torchrun world of at least 2")

HIDDEN = 8


def _frames(encoder_cp):
    """Two frames, both divisible by ``2 * encoder_cp`` as real grids are."""
    return [16 * encoder_cp, 8 * encoder_cp]


def _full_chunk(frame_lengths, device):
    """The whole chunk, identical on every rank. Distinct value per (row, col)."""
    rows = sum(frame_lengths)
    base = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(HIDDEN, device=device, dtype=torch.float32).unsqueeze(0)
    return base * 100.0 + cols


def _local_shard(full, frame_lengths, encoder_cp, rank):
    """``full`` restricted to ``rank``'s rows, in local buffer order."""
    ordered = sorted(shard_rows(frame_lengths, encoder_cp, rank), key=lambda r: r.local_start)
    return torch.cat([full[r.start : r.start + r.rows] for r in ordered], dim=0)


def _rows_owned(frame_lengths, encoder_cp, rank):
    """Chunk-row indices owned by ``rank``, in local buffer order."""
    ordered = sorted(shard_rows(frame_lengths, encoder_cp, rank), key=lambda r: r.local_start)
    out = []
    for run in ordered:
        out.extend(range(run.start, run.start + run.rows))
    return out


@pytest.fixture
def dist_group():
    Utils.initialize_distributed()
    yield torch.distributed.group.WORLD
    torch.distributed.barrier()


def test_gather_reproduces_the_chunk_exactly(dist_group):
    """Forward: shard on every rank, gather, get the original chunk back."""
    encoder_cp = _WORLD
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    frame_lengths = _frames(encoder_cp)

    full = _full_chunk(frame_lengths, device)
    local = _local_shard(full, frame_lengths, encoder_cp, rank).clone()
    perm = torch.tensor(gather_permutation(frame_lengths, encoder_cp), device=device)

    gathered = _GatherChunkAlongSeq.apply(local, perm, dist_group, encoder_cp)

    assert gathered.shape == full.shape
    assert torch.equal(gathered, full), (
        "the gathered chunk must be bit-identical to the unsharded chunk; "
        f"first differing row {int((gathered != full).any(dim=1).nonzero()[0])}"
    )


def test_non_lead_ranks_receive_the_leads_gradient(dist_group):
    """The production pattern: only the lead rank has a real upstream gradient.

    Every rank must still come out with the gradient for the rows *it* computed.
    A backward that slices its own ``grad_out`` gives non-lead ranks zero here,
    which is the silent failure this test exists to catch.
    """
    encoder_cp = _WORLD
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    frame_lengths = _frames(encoder_cp)
    rows = sum(frame_lengths)

    full = _full_chunk(frame_lengths, device)
    local = _local_shard(full, frame_lengths, encoder_cp, rank).clone().requires_grad_(True)
    perm = torch.tensor(gather_permutation(frame_lengths, encoder_cp), device=device)

    # The lead's real upstream gradient; every other rank is driven with zeros.
    lead_grad = torch.arange(rows * HIDDEN, device=device, dtype=torch.float32).reshape(
        rows, HIDDEN
    ) + 1.0
    upstream = lead_grad if rank == 0 else torch.zeros_like(lead_grad)

    gathered = _GatherChunkAlongSeq.apply(local, perm, dist_group, encoder_cp)
    gathered.backward(upstream)

    owned = _rows_owned(frame_lengths, encoder_cp, rank)
    expected = lead_grad[torch.tensor(owned, device=device)]
    assert torch.equal(local.grad, expected), (
        f"rank {rank} must receive the lead's gradient for the rows it computed"
    )
    # The discriminator: this is exactly what a slice-my-own-grad_out backward
    # would get wrong -- it would be all zeros on every rank but the lead.
    assert local.grad.abs().sum() > 0, (
        f"rank {rank} got a zero gradient; its transformer blocks would never train"
    )


def test_backward_sums_across_ranks(dist_group):
    """Pin the reduce-scatter semantics: contributions add, they do not overwrite.

    Production only ever has one non-zero contributor, so a backward that
    *picked* one rank's gradient instead of summing would pass the test above.
    Driving every rank with a distinct upstream separates the two.
    """
    encoder_cp = _WORLD
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    frame_lengths = _frames(encoder_cp)
    rows = sum(frame_lengths)

    full = _full_chunk(frame_lengths, device)
    local = _local_shard(full, frame_lengths, encoder_cp, rank).clone().requires_grad_(True)
    perm = torch.tensor(gather_permutation(frame_lengths, encoder_cp), device=device)

    # Rank r contributes the constant (r + 1); the sum over ranks is e*(e+1)/2.
    upstream = torch.full((rows, HIDDEN), float(rank + 1), device=device)

    gathered = _GatherChunkAlongSeq.apply(local, perm, dist_group, encoder_cp)
    gathered.backward(upstream)

    total = float(encoder_cp * (encoder_cp + 1) // 2)
    assert torch.equal(local.grad, torch.full_like(local.grad, total)), (
        f"rank {rank}: expected every entry to be the sum {total}, got "
        f"{local.grad.unique().tolist()[:5]}"
    )


def test_gather_is_differentiable_end_to_end(dist_group):
    """Autograd through the Function matches the unsharded reference.

    Builds the same scalar objective two ways -- once from the gathered chunk,
    once from the full chunk as a single leaf -- and compares the gradient each
    rank's rows receive. This checks the Function composes with real autograd
    rather than only behaving under a hand-supplied ``grad_out``.
    """
    encoder_cp = _WORLD
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    frame_lengths = _frames(encoder_cp)

    full = _full_chunk(frame_lengths, device)
    weight = torch.linspace(0.5, 2.0, HIDDEN, device=device)

    # Reference: one leaf, no sharding, same objective.
    reference = full.clone().requires_grad_(True)
    (reference * weight).pow(2).sum().backward()

    local = _local_shard(full, frame_lengths, encoder_cp, rank).clone().requires_grad_(True)
    perm = torch.tensor(gather_permutation(frame_lengths, encoder_cp), device=device)
    gathered = _GatherChunkAlongSeq.apply(local, perm, dist_group, encoder_cp)
    # Every rank computes the objective, so the reduce-scatter sums e identical
    # copies -- divide to recover the single-copy gradient.
    ((gathered * weight).pow(2).sum() / encoder_cp).backward()

    owned = _rows_owned(frame_lengths, encoder_cp, rank)
    expected = reference.grad[torch.tensor(owned, device=device)]
    torch.testing.assert_close(local.grad, expected, rtol=1e-5, atol=1e-5)
