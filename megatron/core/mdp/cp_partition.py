# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Which decoder context-parallel rank owns each vision-embedding row.

Under decoder CP the packed THD sequence is split across the CP ranks of the
first pipeline stage before the decoder runs, and every CP rank scatters only
the vision rows whose image-token slot it owns. The MDP planner cuts each vision
item into the runs of rows each rank owns (:func:`cp_item_runs`) and routes
every run straight to its owner. Pure host integer arithmetic.

The split mirrors the decoder input layout exactly:

* ``zigzag`` -- per packed sequence of padded length ``L`` (a multiple of
  ``2 * cp``), rank ``r`` owns chunks ``r`` and ``2 * cp - 1 - r`` of size
  ``L / (2 * cp)``; this is TransformerEngine's ``thd_get_partitioned_indices``.
* ``contiguous`` -- rank ``r`` owns rows ``[r * T / cp, (r + 1) * T / cp)`` of the
  whole packed buffer of ``T`` rows (``get_cp_slice_for_thd``).

In both layouts a rank's local token order is ascending global position (a
rank's first zigzag chunk precedes its second one inside every sequence), so a
rank's leaf is its runs in ascending position.
"""

from typing import List, NamedTuple

from megatron.core.mdp.errors import MdpPlanError

CP_PARTITION_MODES = ("zigzag", "contiguous")


class CpRun(NamedTuple):
    """Consecutive rows of one vision item that one decoder CP rank owns."""

    cp_rank: int
    item_row_start: int
    item_rows: int


def cp_item_runs(
    decoder_start: int,
    rows: int,
    *,
    sample_padded_start: int,
    sample_padded_len: int,
    total_rows: int,
    cp_size: int,
    mode: str,
) -> List[CpRun]:
    """Cut one vision item's decoder span into per-CP-rank runs.

    An item's image-token slots are one contiguous position range
    ``[decoder_start, decoder_start + rows)`` inside its sample. Every CP rank
    owns a union of position intervals (``zigzag``: chunks ``r`` and
    ``2 * cp - 1 - r`` of each padded sequence; ``contiguous``: one interval of
    the whole pack), so the item splits into contiguous runs of its rows, each
    owned by one rank.

    The runs tile the item in ascending row order, which is ascending decoder
    position and therefore each rank's local order. ``cp_size == 1`` yields the
    whole item as one run on rank 0 and reads no span.
    """
    if cp_size < 1 or mode not in CP_PARTITION_MODES:
        raise MdpPlanError(
            f"MDP: CP partition ({cp_size=}, {mode=}) violates: cp_size >= 1 and "
            f"cp_partition_mode one of {CP_PARTITION_MODES}."
        )
    if rows <= 0:
        raise MdpPlanError(f"MDP: item rows {rows} violate: rows > 0.")
    if cp_size == 1:
        return [CpRun(0, 0, rows)]
    end = decoder_start + rows
    sample_end = sample_padded_start + sample_padded_len
    if (
        sample_padded_start < 0
        or sample_padded_len <= 0
        or sample_end > total_rows
        or decoder_start < sample_padded_start
        or end > sample_end
    ):
        raise MdpPlanError(
            f"MDP: item span [{decoder_start}, {end}) violates: inside its padded sequence "
            f"[{sample_padded_start}, {sample_end}) and the pack of {total_rows} rows."
        )
    # The owned intervals as (cp_rank, lo, hi), tiling in ascending position.
    if mode == "contiguous":
        if total_rows % cp_size != 0:
            raise MdpPlanError(
                f"MDP: packed rows {total_rows} violate: divisible by cp_size={cp_size} "
                "under the contiguous CP layout."
            )
        local = total_rows // cp_size
        bounds = [(r, r * local, (r + 1) * local) for r in range(cp_size)]
    else:  # zigzag
        if sample_padded_len % (2 * cp_size) != 0:
            raise MdpPlanError(
                f"MDP: padded sequence length {sample_padded_len} violates: divisible by "
                f"2 * cp_size = {2 * cp_size} under the zigzag CP layout."
            )
        chunk = sample_padded_len // (2 * cp_size)
        bounds = [
            (
                c if c < cp_size else 2 * cp_size - 1 - c,
                sample_padded_start + c * chunk,
                sample_padded_start + (c + 1) * chunk,
            )
            for c in range(2 * cp_size)
        ]
    runs: List[CpRun] = []
    for cp_rank, lo, hi in bounds:
        lo, hi = max(lo, decoder_start), min(hi, end)
        if lo >= hi:
            continue
        if runs and runs[-1].cp_rank == cp_rank:
            # The two middle zigzag chunks share an owner and are adjacent in its
            # local sequence: keep one run (one bridge slice) for both.
            runs[-1] = runs[-1]._replace(item_rows=runs[-1].item_rows + hi - lo)
        else:
            runs.append(CpRun(cp_rank, lo - decoder_start, hi - lo))
    return runs
