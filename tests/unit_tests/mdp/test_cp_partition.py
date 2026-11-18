# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-CP item runs (megatron.core.mdp.cp_partition), pure host arithmetic.

The reference (``cp_reference``) is an independent token-level slicing of the
packed buffer, written the way the decoder slices it. An item's runs must give
every CP rank exactly the item rows whose image-token slots that slicing gives
the rank, in the rank's local token order.
"""

import random

import pytest

from megatron.core.mdp.cp_partition import CpRun, cp_item_runs
from megatron.core.mdp.errors import MdpPlanError
from tests.unit_tests.mdp.cp_reference import local_positions


def _random_item(rng, cp):
    """A packed buffer (sequences padded to 2*cp) and one item span inside a sample."""
    lengths = [2 * cp * rng.randint(1, 6) for _ in range(rng.randint(1, 4))]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    seq = rng.randrange(len(lengths))
    first = rng.randrange(cu[seq], cu[seq + 1])
    rows = rng.randint(1, cu[seq + 1] - first)
    return cu, seq, first, rows


@pytest.mark.parametrize("mode", ["zigzag", "contiguous"])
@pytest.mark.parametrize("cp", [1, 2, 4])
def test_runs_match_token_level_slicing(mode, cp):
    rng = random.Random(1000 * cp + len(mode))
    for _ in range(200):
        cu, seq, first, rows = _random_item(rng, cp)
        total = cu[-1]
        runs = cp_item_runs(
            first,
            rows,
            sample_padded_start=cu[seq],
            sample_padded_len=cu[seq + 1] - cu[seq],
            total_rows=total,
            cp_size=cp,
            mode=mode,
        )
        # The runs tile the item in ascending row order ...
        cursor = 0
        for run in runs:
            assert run.item_row_start == cursor and run.item_rows > 0
            cursor += run.item_rows
        assert cursor == rows
        # ... and give every rank its rows in that rank's local token order.
        got = {}
        for run in runs:
            got.setdefault(run.cp_rank, []).extend(
                range(run.item_row_start, run.item_row_start + run.item_rows)
            )
        expected = {}
        for rank in range(cp):
            owned = [
                position - first
                for position in local_positions(cu, total, cp, rank, mode)
                if first <= position < first + rows
            ]
            if owned:
                expected[rank] = owned
        assert got == expected


def _runs(first, rows, cu, seq, cp, mode):
    return cp_item_runs(
        first,
        rows,
        sample_padded_start=cu[seq],
        sample_padded_len=cu[seq + 1] - cu[seq],
        total_rows=cu[-1],
        cp_size=cp,
        mode=mode,
    )


def test_block_straddling_both_halves_of_a_sample_zigzag_cp2():
    # One 16-row sample at CP2: chunks of 4 rows; rank 0 owns [0,4)+[12,16),
    # rank 1 owns [4,12). An image block over rows 2..13 splits three ways.
    assert _runs(2, 12, [0, 16], 0, 2, "zigzag") == [
        CpRun(cp_rank=0, item_row_start=0, item_rows=2),
        CpRun(cp_rank=1, item_row_start=2, item_rows=8),
        CpRun(cp_rank=0, item_row_start=10, item_rows=2),
    ]


def test_contiguous_ignores_sequence_boundaries():
    # Contiguous slices the whole pack: the CP boundary (row 6 of 12) falls
    # inside the second sample [4, 12), and an item over rows 5..7 straddles it.
    assert _runs(5, 3, [0, 4, 12], 1, 2, "contiguous") == [CpRun(0, 0, 1), CpRun(1, 1, 2)]


def test_item_inside_one_rank_is_one_run():
    assert _runs(1, 3, [0, 16], 0, 2, "contiguous") == [CpRun(0, 0, 3)]
    assert _runs(5, 2, [0, 16], 0, 2, "zigzag") == [CpRun(1, 0, 2)]


def test_middle_zigzag_chunks_share_one_bridge_slice():
    # CP4, 64-row sample: chunks 3 and 4 (rows [24, 40)) both belong to rank 3.
    assert _runs(24, 16, [0, 64], 0, 4, "zigzag") == [CpRun(3, 0, 16)]


def test_cp1_is_the_whole_item_and_reads_no_span():
    assert cp_item_runs(
        0, 7, sample_padded_start=0, sample_padded_len=0, total_rows=0, cp_size=1, mode="zigzag"
    ) == [CpRun(0, 0, 7)]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            dict(first=1, rows=1, cu=[0, 6], seq=0, cp=2, mode="zigzag"),
            "divisible by 2 \\* cp_size",
        ),
        (dict(first=1, rows=1, cu=[0, 7], seq=0, cp=2, mode="contiguous"), "divisible by cp_size"),
        (dict(first=14, rows=4, cu=[0, 16], seq=0, cp=2, mode="zigzag"), "inside"),
        (dict(first=14, rows=4, cu=[0, 16], seq=0, cp=2, mode="contiguous"), "inside"),
        (dict(first=1, rows=0, cu=[0, 8], seq=0, cp=2, mode="zigzag"), "rows > 0"),
        (dict(first=1, rows=1, cu=[0, 8], seq=0, cp=2, mode="bshd"), "cp_partition_mode"),
        (dict(first=1, rows=1, cu=[0, 8], seq=0, cp=0, mode="zigzag"), "cp_size >= 1"),
    ],
)
def test_invalid_inputs_fail_loudly(kwargs, match):
    with pytest.raises(MdpPlanError, match=match):
        _runs(**kwargs)
