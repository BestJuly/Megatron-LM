# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder context-parallel row partition for MDP.

Pure-compute module: integer arithmetic only, no ``torch``, no
``torch.distributed``, no device access. It answers one question — given a row
of the *global* (pre-CP-slice) decoder THD sequence, which context-parallel rank
holds it and at which rank-local row — and derives from that the sub-intervals a
vision item breaks into once the decoder is context-parallel.

This mirrors TransformerEngine's ``thd_get_partitioned_indices``
(``transformer_engine/common/fused_attn/context_parallel.cu``,
``thd_partition_src_index``), which is what
``MultimodalModel._cp_split_for_forward`` uses to slice the decoder inputs.
The two must never diverge: ``tests/unit_tests/mdp/test_cp_partition.py``
brute-force checks this implementation against a transcription of that kernel,
and pins the transcription to TE's own published expectation vectors.

The partition is per-sample "zigzag": a sample's *padded* length ``L`` is cut
into ``2 * cp_size`` equal chunks of ``C = L // (2 * cp_size)``, and rank ``r``
takes chunk ``r`` followed by chunk ``2 * cp_size - 1 - r``. So one sample's
rank-local layout is ``[chunk r][chunk 2*cp-1-r]``, and samples keep their order.
"""

from dataclasses import dataclass

from megatron.core.mdp.errors import MdpConfigurationError


@dataclass(frozen=True)
class CpRowInterval:
    """One contiguous run of an item's rows that lands on a single CP rank.

    ``item_row_start``/``rows`` index the item's own post-merge output rows —
    the encoder-side coordinate space, which is also the EMBEDDING/GRADIENT
    payload space. ``local_row_start`` is where the run begins in the
    *rank-local* decoder sequence of ``cp_rank``; the run is contiguous there
    too, which is what lets one bridge slice carry it.
    """

    cp_rank: int
    item_row_start: int
    rows: int
    local_row_start: int


def validate_sample_span(sample_padded_len: int, cp_size: int) -> int:
    """Chunk size ``L // (2 * cp_size)``; rejects a span CP cannot split evenly.

    TE's kernel asserts the same divisibility. The collator guarantees it by
    padding every packed sample to a multiple of ``2 * cp_size`` (and of
    ``tp_size * cp_size * 2`` under sequence parallelism); see
    ``pack_or_pad_batch``'s ``divisible_by`` and
    :func:`megatron.core.mdp.config.thd_row_alignment`.
    """
    if cp_size < 1:
        raise MdpConfigurationError(f"MDP: cp_size={cp_size} violates: cp_size >= 1.")
    if sample_padded_len <= 0:
        raise MdpConfigurationError(
            f"MDP: sample_padded_len={sample_padded_len} violates: sample_padded_len > 0."
        )
    if sample_padded_len % (2 * cp_size) != 0:
        raise MdpConfigurationError(
            f"MDP: sample_padded_len={sample_padded_len} violates: divisible by "
            f"2 * cp_size = {2 * cp_size}. The collator must pad every packed sample "
            "to that alignment before the decoder is context-parallel."
        )
    return sample_padded_len // (2 * cp_size)


def owner_of_row(offset_in_sample: int, sample_padded_len: int, cp_size: int) -> tuple:
    """``(cp_rank, local_offset_in_sample)`` for one row of one sample.

    ``offset_in_sample`` is the row's offset from the sample's padded start in
    the global sequence; ``local_offset_in_sample`` is its offset from that
    sample's start in the rank-local sequence. Inverse of TE's forward map.
    """
    chunk_size = validate_sample_span(sample_padded_len, cp_size)
    if not 0 <= offset_in_sample < sample_padded_len:
        raise MdpConfigurationError(
            f"MDP: offset_in_sample={offset_in_sample} violates: "
            f"0 <= offset < sample_padded_len ({sample_padded_len})."
        )
    chunk = offset_in_sample // chunk_size
    within = offset_in_sample % chunk_size
    if chunk < cp_size:  # first half: chunk c belongs to rank c
        return chunk, within
    return 2 * cp_size - 1 - chunk, chunk_size + within


def split_item(
    *,
    offset_in_sample: int,
    output_rows: int,
    sample_padded_start: int,
    sample_padded_len: int,
    cp_size: int,
) -> tuple:
    """Break one vision item's contiguous decoder run into per-CP-rank runs.

    The collator validates that an item's image-token slots are contiguous, so
    the item owns global rows ``[start, start + output_rows)`` inside one
    sample. Chunk boundaries cut that run; runs that stay on one rank *and* stay
    contiguous in that rank's local sequence are merged back together, so the
    result is the coarsest legal decomposition. That merge is what makes
    ``cp_size == 1`` return exactly one interval whose ``local_row_start`` is
    the global row — the identity, so the CP=1 plan is bit-identical to the
    pre-CP one — and it also fuses the ``cp-1``/``cp`` chunk pair, the only
    adjacent chunks that share a rank at any ``cp_size``.

    A rank can still receive **two** disjoint runs of one item (chunks ``r`` and
    ``2*cp-1-r`` when the item spans both), which is why a route needs a
    ``slice_id``. Runs are returned in ascending ``item_row_start`` order.
    """
    chunk_size = validate_sample_span(sample_padded_len, cp_size)
    if output_rows < 0:
        raise MdpConfigurationError(
            f"MDP: output_rows={output_rows} violates: output_rows >= 0."
        )
    if output_rows == 0:
        return ()
    if offset_in_sample < 0 or offset_in_sample + output_rows > sample_padded_len:
        raise MdpConfigurationError(
            f"MDP: item span [{offset_in_sample}, {offset_in_sample + output_rows}) "
            f"violates: contained in its sample's padded span [0, {sample_padded_len}). "
            "An image-token block must never straddle a sample boundary."
        )
    if sample_padded_start % cp_size != 0:
        raise MdpConfigurationError(
            f"MDP: sample_padded_start={sample_padded_start} violates: divisible by "
            f"cp_size={cp_size}. Every preceding sample is padded to a multiple of "
            "2 * cp_size, so a non-divisible start means the cu_seqlens_padded vector "
            "was not built by the MDP collator."
        )

    local_sample_start = sample_padded_start // cp_size
    intervals = []
    cursor = offset_in_sample
    end = offset_in_sample + output_rows
    while cursor < end:
        run_end = min(end, (cursor // chunk_size + 1) * chunk_size)
        cp_rank, local_offset = owner_of_row(cursor, sample_padded_len, cp_size)
        local_row_start = local_sample_start + local_offset
        previous = intervals[-1] if intervals else None
        if (
            previous is not None
            and previous.cp_rank == cp_rank
            and previous.local_row_start + previous.rows == local_row_start
        ):
            intervals[-1] = CpRowInterval(
                cp_rank=cp_rank,
                item_row_start=previous.item_row_start,
                rows=previous.rows + (run_end - cursor),
                local_row_start=previous.local_row_start,
            )
        else:
            intervals.append(
                CpRowInterval(
                    cp_rank=cp_rank,
                    item_row_start=cursor - offset_in_sample,
                    rows=run_end - cursor,
                    local_row_start=local_row_start,
                )
            )
        cursor = run_end
    return tuple(intervals)
