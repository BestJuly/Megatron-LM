# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Which decoder context-parallel rank owns each vision-embedding row.

Under decoder CP the packed THD sequence is split across the CP ranks of the
first pipeline stage before the decoder runs, and every CP rank scatters only
the vision rows whose image-token slot it owns. The MDP planner cuts each vision
item into the runs of rows each rank owns (:func:`cp_item_runs`) and routes
every run straight to its owner; :func:`cp_shard_rows` gives the same answer per
leaf row and serves as the row-level reference. Pure host integer arithmetic.

The split mirrors the decoder input layout exactly:

* ``zigzag`` -- per packed sequence of padded length ``L`` (a multiple of
  ``2 * cp``), rank ``r`` owns chunks ``r`` and ``2 * cp - 1 - r`` of size
  ``L / (2 * cp)``; this is TransformerEngine's ``thd_get_partitioned_indices``.
* ``contiguous`` -- rank ``r`` owns rows ``[r * T / cp, (r + 1) * T / cp)`` of the
  whole packed buffer of ``T`` rows (``get_cp_slice_for_thd``).

In both layouts a rank's local token order is ascending global position (a
rank's first zigzag chunk precedes its second one inside every sequence), and a
leaf is ordered by ascending image-token position (``masked_scatter`` order), so
each shard is the ascending subsequence of leaf rows the rank owns.
"""

from bisect import bisect_right
from typing import List, Sequence

from megatron.core.mdp.errors import MdpStateError

CP_PARTITION_MODES = ("zigzag", "contiguous")


def cp_owner_of_position(
    position: int, *, cu_seqlens_padded: Sequence[int], total_rows: int, cp_size: int, mode: str
) -> int:
    """CP rank that owns packed row ``position``."""
    if not 0 <= position < total_rows:
        raise MdpStateError(
            f"MDP: decoder position {position} violates: 0 <= position < {total_rows}."
        )
    if mode == "contiguous":
        return position // (total_rows // cp_size)
    if mode != "zigzag":
        raise MdpStateError(
            f"MDP: cp_partition_mode={mode!r} violates: one of {CP_PARTITION_MODES}."
        )
    seq = bisect_right(cu_seqlens_padded, position) - 1
    start = cu_seqlens_padded[seq]
    length = cu_seqlens_padded[seq + 1] - start
    chunk = (position - start) // (length // (2 * cp_size))
    return chunk if chunk < cp_size else 2 * cp_size - 1 - chunk


def cp_shard_rows(
    positions: Sequence[int],
    *,
    cu_seqlens_padded: Sequence[int],
    total_rows: int,
    cp_size: int,
    mode: str,
) -> List[List[int]]:
    """Per-CP-rank leaf-row indices, each in that rank's local scatter order.

    Args:
        positions: Global packed position of every leaf row, in leaf-row order.
        cu_seqlens_padded: Host copy of the microbatch's padded (global, pre-CP)
            sequence boundaries.
        total_rows: Global packed row count ``T`` (``cu_seqlens_padded[-1]``).
        cp_size: Decoder context-parallel size.
        mode: The decoder's ``cp_partition_mode``.

    Returns:
        ``cp_size`` lists; list ``r`` holds the leaf rows rank ``r`` scatters,
        ascending. Together they partition ``range(len(positions))``; an empty
        list is a normal state (a rank whose slice holds no image token).
    """
    if cp_size < 1:
        raise MdpStateError(f"MDP: cp_size={cp_size} violates: cp_size >= 1.")
    if mode == "contiguous" and total_rows % cp_size != 0:
        raise MdpStateError(
            f"MDP: packed rows {total_rows} violate: divisible by cp_size={cp_size} under "
            "the contiguous CP layout."
        )
    if mode == "zigzag" and cp_size > 1:
        for start, end in zip(cu_seqlens_padded[:-1], cu_seqlens_padded[1:]):
            if (end - start) % (2 * cp_size) != 0:
                raise MdpStateError(
                    f"MDP: padded sequence length {end - start} violates: divisible by "
                    f"2 * cp_size = {2 * cp_size} under the zigzag CP layout."
                )
    shards: List[List[int]] = [[] for _ in range(cp_size)]
    previous = -1
    for row, position in enumerate(positions):
        if position <= previous:
            raise MdpStateError(
                "MDP: leaf rows violate: ordered by strictly ascending decoder position "
                f"(row {row} at {position} after {previous})."
            )
        previous = position
        owner = (
            0
            if cp_size == 1
            else cp_owner_of_position(
                position,
                cu_seqlens_padded=cu_seqlens_padded,
                total_rows=total_rows,
                cp_size=cp_size,
                mode=mode,
            )
        )
        shards[owner].append(row)
    return shards


def cp_item_runs(
    decoder_start: int,
    rows: int,
    *,
    sample_padded_start: int,
    sample_padded_len: int,
    total_rows: int,
    cp_size: int,
    mode: str,
) -> List[tuple]:
    """Cut one vision item's decoder span into per-CP-rank runs.

    An item's image-token slots are one contiguous position range
    ``[decoder_start, decoder_start + rows)`` inside its sample. Every CP rank
    owns a union of position intervals (``zigzag``: chunks ``r`` and
    ``2 * cp - 1 - r`` of each padded sequence; ``contiguous``: one interval of
    the whole pack), so the item splits into contiguous runs of its rows, each
    owned by one rank.

    Returns ``(cp_rank, item_row_start, item_rows, decoder_position)`` tuples in
    ascending ``decoder_position`` -- which is also each rank's local order -- so
    a rank's leaf is its runs in this order. ``cp_size == 1`` yields the whole
    item as one run on rank 0.
    """
    if rows <= 0:
        raise MdpStateError(f"MDP: item rows {rows} violate: rows > 0.")
    end = decoder_start + rows
    if cp_size == 1:
        return [(0, 0, rows, decoder_start)]
    if mode == "contiguous":
        if total_rows <= 0 or total_rows % cp_size != 0:
            raise MdpStateError(
                f"MDP: packed rows {total_rows} violate: positive and divisible by "
                f"cp_size={cp_size} under the contiguous CP layout."
            )
        if decoder_start < 0 or end > total_rows:
            raise MdpStateError(
                f"MDP: item span [{decoder_start}, {end}) violates: inside the pack of "
                f"{total_rows} rows."
            )
        local = total_rows // cp_size
        bounds = [(r, r * local, (r + 1) * local) for r in range(cp_size)]
    elif mode == "zigzag":
        chunk = sample_padded_len // (2 * cp_size) if sample_padded_len > 0 else 0
        if chunk == 0 or sample_padded_len % (2 * cp_size) != 0:
            raise MdpStateError(
                f"MDP: padded sequence length {sample_padded_len} violates: positive and "
                f"divisible by 2 * cp_size = {2 * cp_size} under the zigzag CP layout."
            )
        if decoder_start < sample_padded_start or end > sample_padded_start + sample_padded_len:
            raise MdpStateError(
                f"MDP: item span [{decoder_start}, {end}) violates: inside its padded "
                f"sequence [{sample_padded_start}, {sample_padded_start + sample_padded_len})."
            )
        bounds = []
        for c in range(2 * cp_size):
            owner = c if c < cp_size else 2 * cp_size - 1 - c
            lo = sample_padded_start + c * chunk
            bounds.append((owner, lo, lo + chunk))
    else:
        raise MdpStateError(
            f"MDP: cp_partition_mode={mode!r} violates: one of {CP_PARTITION_MODES}."
        )
    runs = []
    for owner, lo, hi in bounds:
        a, b = max(lo, decoder_start), min(hi, end)
        if a < b:
            runs.append((owner, a - decoder_start, b - a, a))
    runs.sort(key=lambda run: run[3])
    if sum(run[2] for run in runs) != rows:
        raise MdpStateError(
            f"MDP: item span [{decoder_start}, {end}) violates: CP runs tile the item "
            f"({runs})."
        )
    return runs
