# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Encoder context-parallel row partition for MDP.

Pure-compute module: integer arithmetic only, no ``torch``, no
``torch.distributed``, no device access.

Under ``--mdp-encoder-cp e`` one logical worker spans ``e`` physical ranks and
its vision chunk is zigzag-sharded across them, matching TransformerEngine's own
THD context-parallel chunking. This module answers two questions:

- which payload rows of the chunk does rank ``r`` own (:func:`shard_rows`);
- how are the ``e`` shards stitched back into chunk order after the all-gather
  that precedes the patch merger (:func:`gather_permutation`).

Why not reuse :mod:`megatron.core.mdp.cp_partition`. That module inverts the
*decoder's* split, whose unit is a padded sample from ``cu_seqlens_q_padded``
and whose preconditions are sample-level. The vision pack has no padding at all
(``_build_packed_seq_params`` emits only ``cu_seqlens``), its sub-sequences are
per temporal frame, and the question here is "which rows do I compute", not
"which rank consumes this row". Sharing code would mean sharing preconditions
that do not hold.

The geometry, per frame of length ``L``:

    C = L // (2 * e)                      chunk size
    rank r owns chunk r and chunk 2*e-1-r, in that order

which is exactly what TE's ``get_seq_chunk_ids_for_reordering_before_attn``
produces, and it is the only layout TE implements — ``contiguous`` is not an
option here the way it is for the decoder.
"""

from dataclasses import dataclass
from typing import Sequence

from megatron.core.mdp.errors import MdpConfigurationError


@dataclass(frozen=True)
class RowRun:
    """One contiguous run of chunk rows owned by a rank.

    ``start``/``rows`` index the producer chunk's payload rows (the same space
    as ``EncoderThdSegment.payload_row_start``). ``local_start`` is where the
    run lands in that rank's compacted local buffer.
    """

    start: int
    rows: int
    local_start: int


def validate_frame(length: int, encoder_cp: int) -> int:
    """Chunk size ``L // (2 * e)``; rejects a frame CP cannot split evenly.

    TE asserts the same divisibility before dispatch. Unlike the decoder, whose
    collator pads every sample to the required multiple, the vision pack has no
    padding path — so this is a real constraint on the data, checked at plan
    time rather than discovered inside TE mid-iteration.
    """
    if encoder_cp < 1:
        raise MdpConfigurationError(
            f"MDP: encoder_cp={encoder_cp} violates: encoder_cp >= 1."
        )
    if length <= 0:
        raise MdpConfigurationError(f"MDP: frame length={length} violates: length > 0.")
    if length % (2 * encoder_cp) != 0:
        raise MdpConfigurationError(
            f"MDP: vision frame length={length} violates: divisible by "
            f"2 * encoder_cp = {2 * encoder_cp}. Vision frames are h*w patch rows "
            "and the encoder has no frame-padding path, so a non-conforming grid "
            "cannot be split; reduce --mdp-encoder-cp or constrain the grids."
        )
    return length // (2 * encoder_cp)


def shard_rows(
    frame_lengths: Sequence[int], encoder_cp: int, encoder_cp_rank: int
) -> tuple:
    """The chunk rows owned by ``encoder_cp_rank``, in local buffer order.

    ``frame_lengths`` are the chunk's per-frame sub-sequence lengths in payload
    rows, i.e. ``megatron.core.mdp.plan.frame_lengths(layout.segments)``.

    At ``encoder_cp == 1`` this returns one run covering the whole chunk, with
    ``local_start == start`` — the identity, so wiring it in cannot perturb the
    existing path.
    """
    if not 0 <= encoder_cp_rank < encoder_cp:
        raise MdpConfigurationError(
            f"MDP: encoder_cp_rank={encoder_cp_rank} violates: "
            f"0 <= rank < encoder_cp ({encoder_cp})."
        )
    runs = []
    base = 0
    local = 0
    for length in frame_lengths:
        chunk = validate_frame(length, encoder_cp)
        # rank r takes chunk r then chunk 2e-1-r; at e == 1 those are the two
        # halves of the frame and they merge into one run.
        for index in (encoder_cp_rank, 2 * encoder_cp - 1 - encoder_cp_rank):
            start = base + index * chunk
            if runs and runs[-1].start + runs[-1].rows == start:
                previous = runs.pop()
                runs.append(
                    RowRun(
                        start=previous.start,
                        rows=previous.rows + chunk,
                        local_start=previous.local_start,
                    )
                )
            else:
                runs.append(RowRun(start=start, rows=chunk, local_start=local))
            local += chunk
        base += length
    return tuple(runs)


def local_rows(frame_lengths: Sequence[int], encoder_cp: int) -> int:
    """Rows each rank holds locally: ``sum(frame_lengths) // encoder_cp``."""
    total = 0
    for length in frame_lengths:
        validate_frame(length, encoder_cp)
        total += length
    return total // encoder_cp


def gather_permutation(frame_lengths: Sequence[int], encoder_cp: int) -> tuple:
    """Un-zigzag index for the all-gathered block output.

    The all-gather over the encoder-CP group yields rank-major order
    ``[rank0 rows | rank1 rows | ...]``. This returns, for each row of the
    reassembled chunk, its position in that gathered buffer — i.e.
    ``gathered[perm[i]]`` is chunk row ``i``, so ``index_select(0, perm)``
    restores chunk order before the patch merger.

    Identity at ``encoder_cp == 1``.
    """
    per_rank = local_rows(frame_lengths, encoder_cp)
    perm = [0] * sum(frame_lengths)
    for rank in range(encoder_cp):
        for run in shard_rows(frame_lengths, encoder_cp, rank):
            for offset in range(run.rows):
                perm[run.start + offset] = rank * per_rank + run.local_start + offset
    return tuple(perm)
