# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Encoder-CP row partition tests. Pure compute, no torch, no GPU.

The safety property: the rows :mod:`megatron.core.mdp.encoder_cp_partition`
hands a rank must be exactly the rows TransformerEngine expects that rank's
local THD tensor to contain. TE's THD context-parallel split is
``thd_get_partitioned_indices``, the same kernel the decoder side uses, applied
to the vision pack's per-frame ``cu_seqlens`` instead of the decoder's
``cu_seqlens_q_padded`` — so ``_te_partition_indices`` here is the same
transcription pinned to TE's published vectors in ``test_cp_partition.py``.

Frame lengths are drawn from the real grid geometry: every consumer asserts
``h % merge == 0 and w % merge == 0`` with merge = 2, so a frame is
``h*w = 4*mh*mw`` patch rows.
"""

import pytest

from megatron.core.mdp.encoder_cp_partition import (
    RowRun,
    gather_permutation,
    local_rows,
    shard_rows,
    validate_frame,
)
from megatron.core.mdp.errors import MdpConfigurationError


def _te_partition_indices(cu_seqlens, total_tokens, cp_size, cp_rank):
    """Transcription of TE's ``thd_partition_indices_kernel``.

    Pinned to TE's own published expectation vectors below, exactly as in
    ``test_cp_partition.py``.
    """
    for boundary in cu_seqlens:
        assert boundary % (cp_size * 2) == 0
    assert total_tokens % (cp_size * 2) == 0
    cu_s = [boundary // cp_size for boundary in cu_seqlens]
    out = []
    for token_id in range(total_tokens // cp_size):
        seq_id = max(i for i in range(len(cu_s)) if cu_s[i] <= token_id)
        seq_len = cu_s[seq_id + 1] - cu_s[seq_id]
        index = token_id - cu_s[seq_id]
        offset = cp_rank if index < seq_len // 2 else (cp_size - 1) * 2 - cp_rank
        out.append(index + cu_s[seq_id] * cp_size + seq_len // 2 * offset)
    return out


def _cu_seqlens(frame_lengths):
    cu, acc = [0], 0
    for length in frame_lengths:
        acc += length
        cu.append(acc)
    return cu


def _rows_from_runs(runs):
    """Flatten runs into (chunk_row, local_row) pairs in local buffer order."""
    pairs = []
    for run in runs:
        for offset in range(run.rows):
            pairs.append((run.start + offset, run.local_start + offset))
    pairs.sort(key=lambda p: p[1])
    return [chunk_row for chunk_row, _ in pairs]


# Frame-length sets shaped like real grids: h*w = 4*mh*mw with mh, mw in [2,16].
_CASES = [
    ([16], 1),
    ([16], 2),
    ([16, 16], 2),
    ([64], 2),
    ([36, 100, 64], 2),        # 4*3*3, 4*5*5, 4*4*4
    ([144, 400], 2),
    ([1024], 2),
    ([64, 256], 4),            # both divisible by 8
    ([256, 1024, 64], 4),
    ([256], 8),
]


def test_transcription_matches_te_published_vectors():
    # transformer_engine tests/pytorch/attention/test_cp_utils.py::
    # test_thd_get_partitioned_indices_matches_dual_chunk_expected_indices
    assert _te_partition_indices([0, 8, 16], 16, 2, 0) == [0, 1, 6, 7, 8, 9, 14, 15]
    assert _te_partition_indices([0, 8, 16], 16, 2, 1) == [2, 3, 4, 5, 10, 11, 12, 13]


@pytest.mark.parametrize("frames,encoder_cp", _CASES)
def test_shard_rows_matches_te_partition(frames, encoder_cp):
    """Each rank's rows, in local order, are exactly TE's partition for it."""
    cu = _cu_seqlens(frames)
    total = cu[-1]
    for rank in range(encoder_cp):
        expected = _te_partition_indices(cu, total, encoder_cp, rank)
        produced = _rows_from_runs(shard_rows(frames, encoder_cp, rank))
        assert produced == expected, (
            f"frames={frames} e={encoder_cp} rank={rank}: "
            f"first divergence at {next(i for i, (a, b) in enumerate(zip(produced, expected)) if a != b)}"
        )


@pytest.mark.parametrize("frames,encoder_cp", _CASES)
def test_shards_tile_the_chunk_exactly(frames, encoder_cp):
    total = sum(frames)
    seen = []
    for rank in range(encoder_cp):
        runs = shard_rows(frames, encoder_cp, rank)
        assert sum(r.rows for r in runs) == local_rows(frames, encoder_cp)
        # local_start is a gapless 0-based cover of this rank's buffer
        cursor = 0
        for run in sorted(runs, key=lambda r: r.local_start):
            assert run.local_start == cursor
            cursor += run.rows
        seen.extend(_rows_from_runs(runs))
    assert sorted(seen) == list(range(total)), "shards must tile the chunk exactly once"


@pytest.mark.parametrize("frames,encoder_cp", _CASES)
def test_gather_permutation_restores_chunk_order(frames, encoder_cp):
    """index_select(gathered, perm) must reproduce the original chunk order."""
    per_rank = local_rows(frames, encoder_cp)
    # Simulate the all-gather: rank-major concatenation of each rank's rows.
    gathered = []
    for rank in range(encoder_cp):
        gathered.extend(_rows_from_runs(shard_rows(frames, encoder_cp, rank)))
    assert len(gathered) == per_rank * encoder_cp
    perm = gather_permutation(frames, encoder_cp)
    restored = [gathered[perm[i]] for i in range(sum(frames))]
    assert restored == list(range(sum(frames)))


def test_encoder_cp1_is_the_identity():
    """Wiring this in must not perturb the existing encoder_cp=1 path.

    The single rank owns everything, and because each frame's two halves and
    then successive frames are adjacent in chunk space they all merge into one
    run — the coarsest correct decomposition, and byte-for-byte the whole chunk.
    """
    frames = [36, 100]
    assert shard_rows(frames, 1, 0) == (RowRun(start=0, rows=136, local_start=0),)
    assert gather_permutation(frames, 1) == tuple(range(136))
    assert local_rows(frames, 1) == 136


def test_a_rank_gets_two_runs_per_frame_that_do_not_merge():
    # e=2 over one 16-row frame: chunks of 4 -> ranks 0,1,1,0.
    runs = shard_rows([16], 2, 0)
    assert runs == (RowRun(start=0, rows=4, local_start=0),
                    RowRun(start=12, rows=4, local_start=4))
    # rank 1's two chunks are adjacent in chunk space and therefore merge
    assert shard_rows([16], 2, 1) == (RowRun(start=4, rows=8, local_start=0),)


def test_frame_not_divisible_by_two_encoder_cp_is_rejected():
    # 4*3*5 = 60 rows: fine at e=2 (60 % 4 == 0), impossible at e=4 (60 % 8 == 4).
    validate_frame(60, 2)
    with pytest.raises(MdpConfigurationError, match="2 \\* encoder_cp"):
        validate_frame(60, 4)


def test_real_grid_geometry_is_always_splittable_at_encoder_cp_2():
    """h and w are multiples of merge=2, so h*w = 4*mh*mw is a multiple of 4."""
    for mh in range(2, 17):
        for mw in range(2, 17):
            validate_frame(4 * mh * mw, 2)


def test_rank_out_of_range_is_rejected():
    with pytest.raises(MdpConfigurationError, match="0 <= rank < encoder_cp"):
        shard_rows([16], 2, 2)


def test_grid_cache_encoder_cp_indices_match_the_uncached_arithmetic():
    """The cache must return what the per-call construction returned.

    Serving these from the grid cache was a performance change (rebuilding them
    walked the whole vision pack in Python on every forward and again on every
    whole-encoder replay). A cache that returns something subtly different --
    wrong rank's shard, stale permutation for a repeated grid combination --
    would corrupt the encoder silently, so pin the equality rather than trust
    the key.
    """
    import torch

    from examples.multimodal_dev.models.qwen35_vl.vision_pos_cache import GridCache

    device = torch.device("cpu")
    cache = GridCache()

    for grids in (((1, 4, 4),), ((2, 4, 8), (1, 8, 8)), ((1, 8, 8), (1, 4, 4), (3, 4, 4))):
        frames = [h * w for t, h, w in grids for _ in range(t)]
        for encoder_cp in (1, 2):
            for rank in range(encoder_cp):
                expected_rows = []
                for run in shard_rows(frames, encoder_cp, rank):
                    expected_rows.extend(range(run.start, run.start + run.rows))
                # gather_permutation returns a tuple; Tensor.tolist() a list.
                expected_perm = list(gather_permutation(frames, encoder_cp))

                shard, perm = cache.encoder_cp_indices(grids, encoder_cp, rank, device)
                assert shard.tolist() == expected_rows
                assert perm.tolist() == expected_perm

                # A second call takes the cached branch and must still agree.
                shard2, perm2 = cache.encoder_cp_indices(grids, encoder_cp, rank, device)
                assert torch.equal(shard, shard2)
                assert torch.equal(perm, perm2)

    # Distinct (grids, cp, rank) tuples must not collide onto one entry.
    assert len(cache._cp_idx) == 3 * (1 + 2)
