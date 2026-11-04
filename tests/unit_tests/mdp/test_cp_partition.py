# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-CP shard rows (megatron.core.mdp.cp_partition), pure host arithmetic.

The reference is an independent token-level slicing of the packed buffer, written
the way the decoder slices it (TE's zigzag ``thd_get_partitioned_indices`` per
sequence; ``get_cp_slice_for_thd``'s contiguous row ranges). A shard must be the
vision rows whose image-token slots that slicing gives the rank, in the rank's
local token order.
"""

import random

import pytest

from megatron.core.mdp.cp_partition import cp_owner_of_position, cp_shard_rows
from megatron.core.mdp.errors import MdpStateError


def _local_positions(cu, total, cp, rank, mode):
    if mode == "contiguous":
        local = total // cp
        return list(range(rank * local, (rank + 1) * local))
    out = []
    for start, end in zip(cu[:-1], cu[1:]):
        chunk = (end - start) // (2 * cp)
        for c in (rank, 2 * cp - 1 - rank):
            out.extend(range(start + c * chunk, start + (c + 1) * chunk))
    return out


def _reference_shards(positions, cu, total, cp, mode):
    row_of = {pos: row for row, pos in enumerate(positions)}
    return [
        [row_of[p] for p in _local_positions(cu, total, cp, r, mode) if p in row_of]
        for r in range(cp)
    ]


def _random_pack(rng, cp, num_seqs, align):
    lengths = [align * rng.randint(1, 6) for _ in range(num_seqs)]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    total = cu[-1]
    # Image-token slots: random runs inside the sequences, possibly straddling
    # chunk boundaries and both halves of a sequence.
    positions = sorted(rng.sample(range(total), k=rng.randint(0, total // 2)))
    return positions, cu, total


@pytest.mark.parametrize("mode", ["zigzag", "contiguous"])
@pytest.mark.parametrize("cp", [1, 2, 4])
def test_shards_match_token_level_slicing(mode, cp):
    rng = random.Random(1000 * cp + len(mode))
    for _ in range(50):
        positions, cu, total = _random_pack(rng, cp, rng.randint(1, 4), 2 * cp)
        shards = cp_shard_rows(
            positions, cu_seqlens_padded=cu, total_rows=total, cp_size=cp, mode=mode
        )
        assert shards == _reference_shards(positions, cu, total, cp, mode)
        flat = sorted(row for shard in shards for row in shard)
        assert flat == list(range(len(positions)))  # disjoint and complete
        for shard in shards:
            assert shard == sorted(shard)  # local order is ascending position


def test_block_straddling_both_halves_of_a_sample_zigzag_cp2():
    # One 16-row sample at CP2: chunks of 4 rows; rank 0 owns [0,4)+[12,16),
    # rank 1 owns [4,12). An image block over rows 2..13 splits three ways.
    positions = list(range(2, 14))
    shards = cp_shard_rows(
        positions, cu_seqlens_padded=[0, 16], total_rows=16, cp_size=2, mode="zigzag"
    )
    assert [positions[i] for i in shards[0]] == [2, 3, 12, 13]
    assert [positions[i] for i in shards[1]] == list(range(4, 12))


def test_contiguous_ignores_sequence_boundaries():
    # Contiguous slices the whole pack, so a rank may own the tail of one sample
    # and the head of the next.
    shards = cp_shard_rows(
        [5, 6, 7, 8], cu_seqlens_padded=[0, 6, 12], total_rows=12, cp_size=2, mode="contiguous"
    )
    assert shards == [[0], [1, 2, 3]]


def test_empty_shard_is_a_normal_state():
    shards = cp_shard_rows(
        [0, 1], cu_seqlens_padded=[0, 8], total_rows=8, cp_size=2, mode="contiguous"
    )
    assert shards == [[0, 1], []]
    assert cp_shard_rows([], cu_seqlens_padded=[0, 8], total_rows=8, cp_size=4, mode="zigzag") == [
        [],
        [],
        [],
        [],
    ]


def test_owner_of_position_zigzag_cp4():
    cu = [0, 8, 24]
    owners = [
        cp_owner_of_position(p, cu_seqlens_padded=cu, total_rows=24, cp_size=4, mode="zigzag")
        for p in range(24)
    ]
    # 8 rows / 8 chunks -> 1-row chunks: 0,1,2,3,3,2,1,0; then 2-row chunks.
    assert owners[:8] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert owners[8:] == [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            dict(
                positions=[3, 2], cu_seqlens_padded=[0, 8], total_rows=8, cp_size=2, mode="zigzag"
            ),
            "ascending",
        ),
        (
            dict(positions=[1], cu_seqlens_padded=[0, 6], total_rows=6, cp_size=2, mode="zigzag"),
            "divisible by 2 \\* cp_size",
        ),
        (
            dict(
                positions=[1], cu_seqlens_padded=[0, 7], total_rows=7, cp_size=2, mode="contiguous"
            ),
            "divisible by cp_size",
        ),
        (
            dict(
                positions=[9], cu_seqlens_padded=[0, 8], total_rows=8, cp_size=2, mode="contiguous"
            ),
            "position",
        ),
        (
            dict(positions=[1], cu_seqlens_padded=[0, 8], total_rows=8, cp_size=2, mode="bshd"),
            "cp_partition_mode",
        ),
    ],
)
def test_invalid_inputs_fail_loudly(kwargs, match):
    with pytest.raises(MdpStateError, match=match):
        cp_shard_rows(**kwargs)


@pytest.mark.parametrize("mode", ["zigzag", "contiguous"])
@pytest.mark.parametrize("cp", [1, 2, 4])
def test_item_runs_tile_the_item_by_owner(mode, cp):
    from megatron.core.mdp.cp_partition import cp_item_runs

    rng = random.Random(31 * cp + len(mode))
    for _ in range(200):
        positions, cu, total = _random_pack(rng, cp, rng.randint(1, 4), 2 * cp)
        seq = rng.randrange(len(cu) - 1)
        start, end = cu[seq], cu[seq + 1]
        first = rng.randrange(start, end)
        rows = rng.randint(1, end - first)
        runs = cp_item_runs(
            first,
            rows,
            sample_padded_start=start,
            sample_padded_len=end - start,
            total_rows=total,
            cp_size=cp,
            mode=mode,
        )
        covered = []
        for owner, row_start, run_rows, position in runs:
            assert position == first + row_start
            for k in range(run_rows):
                pos = first + row_start + k
                assert owner == cp_owner_of_position(
                    pos, cu_seqlens_padded=cu, total_rows=total, cp_size=cp, mode=mode
                ) or cp == 1
                covered.append(row_start + k)
        assert covered == list(range(rows))  # contiguous runs, ascending, tiling


def test_item_runs_reject_a_span_outside_its_sequence():
    from megatron.core.mdp.cp_partition import cp_item_runs

    with pytest.raises(MdpStateError, match="inside"):
        cp_item_runs(14, 4, sample_padded_start=0, sample_padded_len=16, total_rows=16,
                      cp_size=2, mode="zigzag")
    with pytest.raises(MdpStateError, match="inside"):
        cp_item_runs(14, 4, sample_padded_start=0, sample_padded_len=16, total_rows=16,
                      cp_size=2, mode="contiguous")
