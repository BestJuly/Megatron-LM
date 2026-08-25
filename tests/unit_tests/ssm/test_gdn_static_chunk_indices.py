# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``GatedDeltaNet._static_chunk_indices``, the fixed-shape
``chunk_indices`` table that makes fla's varlen kernels CUDA-graph capturable.

Only the table's *shape* may be a capture-time constant; its values are
recomputed by the replayed kernels from whatever the static ``cu_seqlens``
buffer holds. So the table must

  * agree element-for-element with fla's own host-built table on the real rows,
  * preserve fla's ``row == chunk_offsets[seq] + intra`` invariant, which the
    ``h``/``dh`` reader kernels rely on to index the chunk-state buffer,
  * keep a constant shape across different packings, and
  * park unused slots where every masked load/store no-ops.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.gated_delta_net import GatedDeltaNet

try:
    from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

CHUNK = 64


class _Config:
    def __init__(self, max_num_seqs=8, max_seqlen=4096, cp=1):
        self.thd_max_packed_sequences = max_num_seqs
        self.max_seqlen_per_dp_cp_rank = max_seqlen
        self.context_parallel_size = cp


class _Stub:
    """Minimal stand-in exposing just the pure table builder."""

    _static_chunk_indices = GatedDeltaNet._static_chunk_indices

    def __init__(self, config=None):
        self.config = config if config is not None else _Config()


def _cu_seqlens(lens, max_num_seqs=8, device="cuda"):
    """Statically-shaped cu_seqlens: real lengths then zero-length padding."""
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    cu += [cu[-1]] * (max_num_seqs - len(lens))
    return torch.tensor(cu, device=device, dtype=torch.int32)


def _nt_max(cfg, chunk=CHUNK):
    total_t = cfg.max_seqlen_per_dp_cp_rank * cfg.context_parallel_size
    return (total_t + chunk - 1) // chunk + cfg.thd_max_packed_sequences


LAYOUTS = [
    [4096],
    [2048, 2048],
    [1024, 1024, 1024, 1024],
    [100, 3000, 996],
    [64, 64, 64, 3904],
    [1, 4095],
]


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("lens", LAYOUTS)
def test_matches_fla_prepare_chunk_indices_on_real_rows(lens):
    stub = _Stub()
    cu = _cu_seqlens(lens)

    ours = stub._static_chunk_indices(cu, chunk_size=CHUNK)
    # fla's table is built only over the real (non-padding) sequences.
    real_cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
        device=cu.device,
        dtype=cu.dtype,
    )
    theirs = prepare_chunk_indices(real_cu, CHUNK)

    n_real = theirs.shape[0]
    assert torch.equal(ours[:n_real].to(theirs.dtype), theirs)


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("lens", LAYOUTS)
def test_preserves_chunk_offset_invariant(lens):
    """fla's readers require row == chunk_offsets[seq] + intra."""
    stub = _Stub()
    cu = _cu_seqlens(lens)
    ours = stub._static_chunk_indices(cu, chunk_size=CHUNK).to(torch.int64)

    real_cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
        device=cu.device,
        dtype=torch.int32,
    )
    offsets = prepare_chunk_offsets(real_cu, CHUNK).to(torch.int64)
    n_real = int(offsets[-1].item())

    rows = torch.arange(n_real, device=cu.device, dtype=torch.int64)
    seg, intra = ours[:n_real, 0], ours[:n_real, 1]
    assert torch.equal(offsets[seg] + intra, rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("lens", LAYOUTS)
def test_shape_is_constant_across_packings(lens):
    """The Triton grid is frozen at capture, so NT must not depend on packing."""
    cfg = _Config()
    stub = _Stub(cfg)
    ours = stub._static_chunk_indices(_cu_seqlens(lens), chunk_size=CHUNK)
    assert ours.shape == (_nt_max(cfg), 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("lens", LAYOUTS)
def test_padding_slots_are_inert(lens):
    """Unused slots point at sequence 0 with an out-of-range intra index."""
    cfg = _Config()
    stub = _Stub(cfg)
    cu = _cu_seqlens(lens)
    ours = stub._static_chunk_indices(cu, chunk_size=CHUNK).to(torch.int64)

    n_chunks = sum((n + CHUNK - 1) // CHUNK for n in lens)
    pad = ours[n_chunks:]
    assert torch.all(pad[:, 0] == 0)
    # Beyond any reachable chunk index, so every `o_t < T` mask is False.
    assert torch.all(pad[:, 1] >= _nt_max(cfg))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_returns_none_without_static_thd_bounds():
    """Without the static bounds there is no fixed NT; fall back to fla."""
    for cfg in (_Config(max_num_seqs=None), _Config(max_seqlen=None)):
        assert _Stub(cfg)._static_chunk_indices(_cu_seqlens([1024])) is None
