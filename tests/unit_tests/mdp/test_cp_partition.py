# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder CP row-partition tests. Pure compute, no torch, no GPU.

The safety property is that :mod:`megatron.core.mdp.cp_partition` inverts
exactly the map TransformerEngine applies when the decoder slices its THD
inputs. ``_te_partition_indices`` below is a line-by-line transcription of
``thd_partition_indices_kernel`` / ``thd_partition_src_index``
(``transformer_engine/common/fused_attn/context_parallel.cu``); it is pinned to
TE's own published expectation vectors in
``test_transcription_matches_te_published_vectors`` so the transcription cannot
silently drift from the kernel it stands in for.
"""

import pytest

from megatron.core.mdp.cp_partition import (
    CpRowInterval,
    owner_of_row,
    split_item,
    validate_sample_span,
)
from megatron.core.mdp.errors import MdpConfigurationError


def _te_partition_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
    """Transcription of TE's ``thd_partition_indices_kernel``.

    Returns, for one CP rank, the global row index of each of its local rows.
    """
    for boundary in cu_seqlens_padded:
        assert boundary % (cp_size * 2) == 0
    assert total_tokens % (cp_size * 2) == 0
    cu_s = [boundary // cp_size for boundary in cu_seqlens_padded]
    out = []
    for token_id in range(total_tokens // cp_size):
        seq_id = max(i for i in range(len(cu_s)) if cu_s[i] <= token_id)
        seq_len = cu_s[seq_id + 1] - cu_s[seq_id]
        index = token_id - cu_s[seq_id]
        offset = cp_rank if index < seq_len // 2 else (cp_size - 1) * 2 - cp_rank
        out.append(index + cu_s[seq_id] * cp_size + seq_len // 2 * offset)
    return out


# Padded cu_seqlens vectors x CP sizes that keep every sample divisible by 2*cp.
_CASES = [
    ([0, 8, 16], 1),
    ([0, 8, 16], 2),
    ([0, 16], 4),
    ([0, 16, 32, 48], 4),
    ([0, 4, 12, 24], 2),
    ([0, 24, 40, 88], 2),
    ([0, 32, 96], 4),
    ([0, 48, 96, 144, 192], 8),
    ([0, 96, 192], 8),
]


def test_transcription_matches_te_published_vectors():
    # transformer_engine tests/pytorch/attention/test_cp_utils.py::
    # test_thd_get_partitioned_indices_matches_dual_chunk_expected_indices
    assert _te_partition_indices([0, 8, 16], 16, 2, 0) == [0, 1, 6, 7, 8, 9, 14, 15]
    assert _te_partition_indices([0, 8, 16], 16, 2, 1) == [2, 3, 4, 5, 10, 11, 12, 13]


@pytest.mark.parametrize("cu_padded,cp_size", _CASES)
def test_owner_of_row_inverts_the_te_forward_map(cu_padded, cp_size):
    total = cu_padded[-1]
    for cp_rank in range(cp_size):
        forward = _te_partition_indices(cu_padded, total, cp_size, cp_rank)
        for local_row, global_row in enumerate(forward):
            sample = max(i for i in range(len(cu_padded) - 1) if cu_padded[i] <= global_row)
            start = cu_padded[sample]
            length = cu_padded[sample + 1] - start
            owner, local_offset = owner_of_row(global_row - start, length, cp_size)
            assert owner == cp_rank
            assert start // cp_size + local_offset == local_row


@pytest.mark.parametrize("cu_padded,cp_size", _CASES)
def test_split_item_covers_every_row_exactly_once_at_the_right_local_row(cu_padded, cp_size):
    """Every item span, at every offset and length, reassembles exactly.

    Sweeping all spans is the point: the interesting cases are the ones that
    straddle a chunk boundary and the ones that reach into both halves, and a
    hand-picked span set would miss them.
    """
    total = cu_padded[-1]
    local_to_global = {
        cp_rank: _te_partition_indices(cu_padded, total, cp_size, cp_rank)
        for cp_rank in range(cp_size)
    }
    for sample in range(len(cu_padded) - 1):
        start = cu_padded[sample]
        length = cu_padded[sample + 1] - start
        for offset in range(length):
            for rows in range(1, length - offset + 1):
                intervals = split_item(
                    offset_in_sample=offset,
                    output_rows=rows,
                    sample_padded_start=start,
                    sample_padded_len=length,
                    cp_size=cp_size,
                )
                assert sum(i.rows for i in intervals) == rows
                # Contiguous, ascending, gapless cover of the item's rows.
                cursor = 0
                for interval in intervals:
                    assert interval.item_row_start == cursor
                    cursor += interval.rows
                assert cursor == rows
                # Coarsest legal decomposition: no two adjacent runs share a
                # rank and are contiguous in that rank's local sequence.
                for left, right in zip(intervals, intervals[1:]):
                    assert not (
                        left.cp_rank == right.cp_rank
                        and left.local_row_start + left.rows == right.local_row_start
                    )
                # Every row lands where the decoder will actually look for it.
                for interval in intervals:
                    for k in range(interval.rows):
                        global_row = start + offset + interval.item_row_start + k
                        local_row = interval.local_row_start + k
                        assert local_to_global[interval.cp_rank][local_row] == global_row


def test_cp1_is_the_identity():
    """One interval, global row == local row: the CP=1 plan must not change.

    The item still crosses the sample's midpoint chunk boundary, so this also
    covers the merge of the ``cp-1``/``cp`` chunk pair.
    """
    intervals = split_item(
        offset_in_sample=3,
        output_rows=5,
        sample_padded_start=16,
        sample_padded_len=8,
        cp_size=1,
    )
    assert intervals == (
        CpRowInterval(cp_rank=0, item_row_start=0, rows=5, local_row_start=19),
    )


def test_a_full_sample_span_splits_into_the_coarsest_legal_runs():
    # cp=2 over a length-8 sample: chunks of 2 -> ranks 0,1,1,0. The middle two
    # chunks are one rank and locally contiguous, so they fuse; rank 0 keeps two
    # disjoint runs, which is exactly why a route needs a slice_id.
    intervals = split_item(
        offset_in_sample=0,
        output_rows=8,
        sample_padded_start=0,
        sample_padded_len=8,
        cp_size=2,
    )
    assert intervals == (
        CpRowInterval(cp_rank=0, item_row_start=0, rows=2, local_row_start=0),
        CpRowInterval(cp_rank=1, item_row_start=2, rows=4, local_row_start=0),
        CpRowInterval(cp_rank=0, item_row_start=6, rows=2, local_row_start=2),
    )


def test_an_item_inside_one_chunk_is_never_split():
    # cp=4 over a length-16 sample: chunk size 2, chunks -> ranks 0,1,2,3,3,2,1,0.
    inside = split_item(
        offset_in_sample=8,
        output_rows=2,
        sample_padded_start=0,
        sample_padded_len=16,
        cp_size=4,
    )
    assert len(inside) == 1 and inside[0].cp_rank == 3
    # Rows 9..10 straddle chunks 4 and 5, which are different ranks.
    straddling = split_item(
        offset_in_sample=9,
        output_rows=2,
        sample_padded_start=0,
        sample_padded_len=16,
        cp_size=4,
    )
    assert [i.cp_rank for i in straddling] == [3, 2]


def test_empty_item_yields_no_intervals():
    assert (
        split_item(
            offset_in_sample=0,
            output_rows=0,
            sample_padded_start=0,
            sample_padded_len=8,
            cp_size=2,
        )
        == ()
    )


def test_span_not_divisible_by_two_cp_is_rejected():
    with pytest.raises(MdpConfigurationError, match="2 \\* cp_size"):
        validate_sample_span(6, 4)


def test_item_crossing_its_sample_boundary_is_rejected():
    with pytest.raises(MdpConfigurationError, match="contained in its sample"):
        split_item(
            offset_in_sample=6,
            output_rows=4,
            sample_padded_start=0,
            sample_padded_len=8,
            cp_size=2,
        )


def test_unaligned_sample_start_is_rejected():
    with pytest.raises(MdpConfigurationError, match="divisible by cp_size"):
        split_item(
            offset_in_sample=0,
            output_rows=2,
            sample_padded_start=3,
            sample_padded_len=8,
            cp_size=2,
        )
