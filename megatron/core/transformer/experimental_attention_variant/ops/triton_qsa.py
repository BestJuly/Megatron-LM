# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sparse-attention metadata and Triton backward kernels of the QSA sparse-GQA attention.

The attention entry point and the forward and dq kernels (CuTe DSL) live in
:mod:`cute_qsa_attn`; the indexer's block scoring and exact top-k live in
:mod:`triton_qsa_select`.

Design (SGLang-prefill style, replacing the earlier fla-NSA adaptation):

1. **Query-tile shared-superset gather**: the sparse attention kernels process
   ``BQ`` consecutive queries per program against the UNION of their selected
   blocks (plus per-query membership bits), so each gathered K/V tile is reused
   across the whole query tile instead of being re-fetched per query — the
   gather-traffic roofline that made the per-query kernels lose to dense flash.
   The union / membership / CSR structures are precomputed on the device
   (:func:`build_qsa_supersets`; sort-free Triton builders in :mod:`triton_qsa_select`).
2. **Backward**: ``dk``/``dv`` have two deterministic implementations (sequential
   accumulation, no atomics):

   * ``"tile"`` (default, :func:`qsa_bwd_dkv_tile_kernel`): one program per KV TILE
     of ``KB`` consecutive selection blocks (``KB * 4`` KV rows, a full dot tile)
     iterating the UNION of the queries that selected any block of the tile, with a
     per-entry block bitmask, over a flattened (query, group-head) column space (no
     power-of-two head padding). Every gathered q/dO row is reused by ``KB * 4``
     KV rows instead of 4. Optionally the column range of a tile is split over
     ``NSPLIT`` programs whose fp32 partials are summed in a fixed order.
   * ``"block"`` (:func:`qsa_bwd_dkv_kernel`, the original kernel): one program per
     4-token KV block (padded to a 16-row dot tile) iterating the queries that
     selected that block.

All varlen (packed THD) handling is folded into per-block token ``base``/``end``
arrays and the membership bits: the kernels never see ``cu_seqlens``. Selection
blocks may be partial (the trailing ragged block of a document) or fully in the
future (empty token range) — both are handled by the ``base``/``end`` masks,
which is how the query's own block reproduces the unconditional ragged tail.

The GQA group size ``G = HQ // H`` may be any value >= 1 (rows are padded to the
next power of two in-tile); no host-side head padding is required.
"""

import os
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

BS = 4  # tokens per selection block (qsa_indexer_compress_ratio)
RS = 16  # padded KV rows per dkv program (>= 16 for tl.dot; BS rows are valid)


# ---------------------------------------------------------------------------
# 1. Sparse-attention metadata: query-tile supersets and the KV-tile query CSR
# ---------------------------------------------------------------------------


class QSASupersets(NamedTuple):
    """Query-tile shared-superset structures (all device tensors).

    For every tile of ``BQ`` consecutive queries, the UNION of the tile's selected
    blocks is stored as a CSR of "entries"; each entry carries the block's token
    ``base``/``end`` and a ``[BQ]`` membership byte-vector saying which queries of
    the tile actually selected it. The inverse CSR groups entries by
    (batch, block) for the deterministic dkv backward.
    """

    tile_offsets: torch.Tensor  # [b * n_tiles + 1] int64 — CSR into the entry arrays
    entry_base: torch.Tensor  # [E] int32 — first token of the entry's block
    entry_end: torch.Tensor  # [E] int32 — one past the last valid token
    membership: torch.Tensor  # [E, BQ] uint8
    n_tiles: int
    num_entries: int


def build_qsa_supersets_torch(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    block_bases: torch.Tensor,  # [num_blocks] token base per block
    block_ends: torch.Tensor,  # [num_blocks]
    num_blocks: int,
    tile_queries: int,
) -> QSASupersets:
    """Build per-query-tile block unions, membership bits and the dkv CSR."""
    b, s, S = block_indices.shape
    device = block_indices.device
    BQ = tile_queries
    n_tiles = (s + BQ - 1) // BQ
    s_pad = n_tiles * BQ

    slot_valid = torch.arange(S, device=device) < block_counts.unsqueeze(-1)
    ids = torch.where(
        slot_valid, block_indices.long(), torch.full_like(block_indices.long(), num_blocks)
    )
    if s_pad != s:
        ids = torch.cat(
            [ids, torch.full((b, s_pad - s, S), num_blocks, dtype=ids.dtype, device=device)], 1
        )
        slot_valid = torch.cat(
            [slot_valid, torch.zeros(b, s_pad - s, S, dtype=torch.bool, device=device)], 1
        )
    n_flat = b * n_tiles
    tiles = ids.view(n_flat, BQ * S)
    sorted_ids, _ = torch.sort(tiles, dim=-1)
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    uniq = first & (sorted_ids < num_blocks)
    counts_per_tile = uniq.sum(-1)  # [n_flat]
    tile_offsets = torch.zeros(n_flat + 1, dtype=torch.long, device=device)
    tile_offsets[1:] = counts_per_tile.cumsum(0)
    num_entries = int(tile_offsets[-1])  # single host sync (allocation sizes)
    u_max = int(counts_per_tile.max()) if num_entries > 0 else 1

    entry_block = sorted_ids[uniq]  # row-major flatten => ascending within each tile
    # padded per-tile unions for one batched searchsorted
    padded = torch.full((n_flat, u_max), num_blocks, dtype=sorted_ids.dtype, device=device)
    rank = uniq.long().cumsum(-1) - 1
    row_idx = torch.arange(n_flat, device=device).unsqueeze(-1).expand_as(sorted_ids)
    padded[row_idx[uniq], rank[uniq]] = entry_block

    # membership: global entry index of every valid (query, slot)
    upos = torch.searchsorted(padded, tiles)  # [n_flat, BQ * S]
    gpos = tile_offsets[:-1].unsqueeze(-1) + upos
    valid_flat = slot_valid.view(n_flat, BQ * S)
    q_local = (torch.arange(BQ * S, device=device) // S).unsqueeze(0).expand(n_flat, -1)
    membership = torch.zeros(max(num_entries, 1), BQ, dtype=torch.uint8, device=device)
    membership[gpos[valid_flat], q_local[valid_flat]] = 1

    entry_base = block_bases[entry_block].to(torch.int32)
    entry_end = block_ends[entry_block].to(torch.int32)

    return QSASupersets(
        tile_offsets=tile_offsets,
        entry_base=entry_base,
        entry_end=entry_end,
        membership=membership,
        n_tiles=n_tiles,
        num_entries=num_entries,
    )


def build_qsa_supersets_static_torch(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    block_bases: torch.Tensor,  # [num_blocks] token base per block
    block_ends: torch.Tensor,  # [num_blocks]
    num_blocks: int,
    tile_queries: int,
) -> QSASupersets:
    """Fixed-shape, host-sync-free :func:`build_qsa_supersets` (CUDA-graph capturable).

    The CSR is identical to the dynamic builder's -- same ``tile_offsets``, and the same
    entries in the same order at ``[0, num_entries)`` -- but every buffer is sized by a
    shape-only upper bound instead of the data-dependent entry count: each tile owns at
    most ``u_cap = min(BQ * S, num_blocks)`` unique blocks, so ``E_cap = n_tiles * u_cap``.
    Compaction uses device-computed scatter positions (invalid writes go to a dump slot
    that is sliced off) instead of boolean-mask indexing. Entries past a tile's count are
    never read by the kernels, which loop over ``[tile_offsets[i], tile_offsets[i + 1])``.
    """
    b, s, S = block_indices.shape
    device = block_indices.device
    BQ = tile_queries
    n_tiles = (s + BQ - 1) // BQ
    s_pad = n_tiles * BQ

    slot_valid = torch.arange(S, device=device) < block_counts.unsqueeze(-1)
    ids = torch.where(
        slot_valid, block_indices.long(), torch.full_like(block_indices.long(), num_blocks)
    )
    if s_pad != s:
        ids = torch.cat(
            [ids, torch.full((b, s_pad - s, S), num_blocks, dtype=ids.dtype, device=device)], 1
        )
        slot_valid = torch.cat(
            [slot_valid, torch.zeros(b, s_pad - s, S, dtype=torch.bool, device=device)], 1
        )
    n_flat = b * n_tiles
    width = BQ * S
    u_cap = max(1, min(width, num_blocks))
    e_cap = n_flat * u_cap
    tiles = ids.view(n_flat, width)
    sorted_ids, _ = torch.sort(tiles, dim=-1)
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    uniq = first & (sorted_ids < num_blocks)
    counts_per_tile = uniq.sum(-1)  # [n_flat]
    tile_offsets = torch.zeros(n_flat + 1, dtype=torch.long, device=device)
    tile_offsets[1:] = counts_per_tile.cumsum(0)

    rank = uniq.long().cumsum(-1) - 1
    # Per-tile padded unions (row width u_cap) for the batched searchsorted; column u_cap
    # is a dump slot for the non-unique positions.
    padded = torch.full((n_flat, u_cap + 1), num_blocks, dtype=sorted_ids.dtype, device=device)
    padded.scatter_(1, torch.where(uniq, rank, u_cap), sorted_ids)
    padded = padded[:, :u_cap].contiguous()
    # Global CSR entries: tile i's unique blocks land at tile_offsets[i] + rank (exactly the
    # row-major compaction of the dynamic builder); position e_cap is the dump slot.
    entry_pos = torch.where(uniq, tile_offsets[:-1].unsqueeze(-1) + rank, e_cap)
    entry_block = torch.full((e_cap + 1,), num_blocks, dtype=sorted_ids.dtype, device=device)
    entry_block.scatter_(0, entry_pos.view(-1), sorted_ids.view(-1))
    entry_block = entry_block[:e_cap]

    upos = torch.searchsorted(padded, tiles)  # [n_flat, BQ * S]
    gpos = tile_offsets[:-1].unsqueeze(-1) + upos
    valid_flat = slot_valid.view(n_flat, width)
    q_local = (torch.arange(width, device=device) // S).unsqueeze(0).expand(n_flat, -1)
    membership = torch.zeros(e_cap + 1, BQ, dtype=torch.uint8, device=device)
    membership.index_put_(
        (torch.where(valid_flat, gpos, e_cap), q_local),
        torch.ones((), dtype=torch.uint8, device=device),
    )
    membership = membership[:e_cap]

    # The sentinel id (num_blocks) of the unused trailing entries maps to an empty range.
    pad = torch.zeros(1, dtype=block_bases.dtype, device=device)
    entry_base = torch.cat([block_bases, pad])[entry_block].to(torch.int32)
    entry_end = torch.cat([block_ends, pad])[entry_block].to(torch.int32)

    return QSASupersets(
        tile_offsets=tile_offsets,
        entry_base=entry_base,
        entry_end=entry_end,
        membership=membership,
        n_tiles=n_tiles,
        num_entries=e_cap,
    )


def build_qsa_query_csr_static(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    num_blocks: int,
):
    """Fixed-shape, host-sync-free :func:`build_qsa_query_csr` (CUDA-graph capturable).

    Invalid slots get the sentinel key ``b * num_blocks``, which sorts after every real
    key; a stable sort of the full ``[b * s * S]`` key array therefore orders the valid
    prefix exactly as the dynamic builder's stable sort of the valid subset does. The
    offsets come from a ``searchsorted`` over the sorted keys instead of ``bincount``
    (which reads the maximum key on the host).
    """
    b, s, S = block_indices.shape
    device = block_indices.device
    slot_valid = torch.arange(S, device=device) < block_counts.unsqueeze(-1)
    batch_ix = torch.arange(b, device=device).view(b, 1, 1).expand(b, s, S)
    t_ix = torch.arange(s, device=device).view(1, s, 1).expand(b, s, S)
    sentinel = b * num_blocks
    key = torch.where(slot_valid, batch_ix * num_blocks + block_indices.long(), sentinel).reshape(
        -1
    )
    perm = torch.argsort(key, stable=True)
    qsel_ids = t_ix.reshape(-1)[perm].to(torch.int32)
    qsel_offsets = torch.searchsorted(
        key[perm], torch.arange(sentinel + 1, device=device, dtype=key.dtype)
    )
    return qsel_ids, qsel_offsets


def build_qsa_query_csr(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    num_blocks: int,
):
    """Per-(batch, block) CSR of the query positions that selected it (dkv backward).

    Queries are stably ordered within each block, so the dkv accumulation order is
    deterministic.
    """
    b, s, S = block_indices.shape
    device = block_indices.device
    slot_valid = torch.arange(S, device=device) < block_counts.unsqueeze(-1)
    batch_ix = (
        torch.arange(b, device=device).view(b, 1, 1).expand(b, s, S)[slot_valid]
    )
    t_ix = torch.arange(s, device=device).view(1, s, 1).expand(b, s, S)[slot_valid]
    blk = block_indices.long()[slot_valid]
    key = batch_ix * num_blocks + blk
    perm = torch.argsort(key, stable=True)  # stable => ascending t within each block
    qsel_ids = t_ix[perm].to(torch.int32)
    counts = torch.bincount(key, minlength=b * num_blocks)
    qsel_offsets = torch.zeros(b * num_blocks + 1, dtype=torch.long, device=device)
    qsel_offsets[1:] = counts.cumsum(0)
    return qsel_ids, qsel_offsets


class QSAKVTileCSR(NamedTuple):
    """Per-(batch, KV tile) CSR of the queries that selected any block of the tile.

    A KV tile is ``tile_blocks`` consecutive selection blocks of the ``block_bases``
    space. Entries of one tile are ordered by ascending query row, so the dkv
    accumulation order is deterministic and independent of the slot order.
    """

    tile_offsets: torch.Tensor  # [b * n_tiles + 1] int64 — CSR into the entry arrays
    entry_query: torch.Tensor  # [E] int32 — query ROW of the entry
    entry_bits: torch.Tensor  # [E] int32 — bit r set iff the query selected block r of the tile
    tile_blocks: int
    n_tiles: int


def build_qsa_kv_tile_csr_torch(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    num_blocks: int,
    tile_blocks: int,
) -> QSAKVTileCSR:
    """Build the KV-tile query CSR for :func:`qsa_bwd_dkv_tile_kernel`.

    Host-sync free and fixed-shape (CUDA-graph capturable): the entry arrays are sized
    by the shape-only bound ``b * s * S``; the ``n`` unique (tile, query) entries occupy
    ``[0, n)`` and the kernel only reads ``[tile_offsets[i], tile_offsets[i + 1])``.
    A query selects a block at most once, so the per-entry bitmask is the integer sum of
    distinct powers of two (an exact, order-independent reduction).
    """
    assert 1 <= tile_blocks <= 30, "the block bitmask is an int32"
    b, s, S = block_indices.shape
    device = block_indices.device
    n_tiles = max(1, (num_blocks + tile_blocks - 1) // tile_blocks)
    slot_valid = torch.arange(S, device=device) < block_counts.unsqueeze(-1)
    blk = torch.where(slot_valid, block_indices.long(), 0)
    batch_ix = torch.arange(b, device=device).view(b, 1, 1)
    t_ix = torch.arange(s, device=device).view(1, s, 1)
    sentinel = b * n_tiles * s
    key = torch.where(
        slot_valid, (batch_ix * n_tiles + blk // tile_blocks) * s + t_ix, sentinel
    ).reshape(-1)
    bits = torch.where(
        slot_valid, torch.bitwise_left_shift(torch.ones_like(blk), blk % tile_blocks), 0
    )
    n = key.numel()
    sorted_key, perm = torch.sort(key)
    sorted_bits = bits.reshape(-1)[perm]
    valid = sorted_key < sentinel
    first = valid.clone()
    first[1:] &= sorted_key[1:] != sorted_key[:-1]
    cum_first = torch.zeros(n + 1, dtype=torch.long, device=device)
    cum_first[1:] = first.long().cumsum(0)
    entry_ix = cum_first[1:] - 1  # entry of every sorted element (valid ones)
    entry_query = torch.zeros(n + 1, dtype=torch.int32, device=device)
    entry_query.scatter_(
        0, torch.where(first, entry_ix, n), (sorted_key % s).to(torch.int32)
    )
    entry_bits = torch.zeros(n + 1, dtype=torch.long, device=device)
    entry_bits.scatter_add_(0, torch.where(valid, entry_ix, n), sorted_bits)
    bounds = torch.arange(b * n_tiles + 1, device=device, dtype=sorted_key.dtype) * s
    tile_offsets = cum_first[torch.searchsorted(sorted_key, bounds)]
    return QSAKVTileCSR(
        tile_offsets=tile_offsets,
        entry_query=entry_query[:n],
        entry_bits=entry_bits[:n].to(torch.int32),
        tile_blocks=tile_blocks,
        n_tiles=n_tiles,
    )


# Sort-free (K3) builders: dense bitmask + count + compaction Triton kernels in
# ops/triton_qsa_select.py, bit-identical to the sort-based ``*_torch`` builders above
# (which remain the CPU path and the reference).


def _use_fused_meta(t: torch.Tensor) -> bool:
    return t.is_cuda


def build_qsa_supersets_static(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    block_bases: torch.Tensor,  # [num_blocks] token base per block
    block_ends: torch.Tensor,  # [num_blocks]
    num_blocks: int,
    tile_queries: int,
) -> QSASupersets:
    """Fixed-shape, host-sync-free supersets (CUDA-graph capturable); see
    :func:`build_qsa_supersets_static_torch` for the contract."""
    if not _use_fused_meta(block_indices):
        return build_qsa_supersets_static_torch(
            block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
        )
    from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
        qsa_build_supersets_fused,
    )

    tile_offsets, entry_base, entry_end, membership, n_tiles, e_cap = qsa_build_supersets_fused(
        block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
    )
    return QSASupersets(
        tile_offsets=tile_offsets,
        entry_base=entry_base,
        entry_end=entry_end,
        membership=membership,
        n_tiles=n_tiles,
        num_entries=e_cap,
    )


def build_qsa_supersets(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    block_bases: torch.Tensor,  # [num_blocks] token base per block
    block_ends: torch.Tensor,  # [num_blocks]
    num_blocks: int,
    tile_queries: int,
) -> QSASupersets:
    """Per-query-tile block unions, membership bits (exact-size buffers; one host sync)."""
    if not _use_fused_meta(block_indices):
        return build_qsa_supersets_torch(
            block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
        )
    ss = build_qsa_supersets_static(
        block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
    )
    num_entries = int(ss.tile_offsets[-1])  # single host sync (allocation sizes)
    return QSASupersets(
        tile_offsets=ss.tile_offsets,
        entry_base=ss.entry_base[:num_entries],
        entry_end=ss.entry_end[:num_entries],
        membership=ss.membership[: max(num_entries, 1)],
        n_tiles=ss.n_tiles,
        num_entries=num_entries,
    )


def build_qsa_kv_tile_csr(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    num_blocks: int,
    tile_blocks: int,
) -> "QSAKVTileCSR":
    """KV-tile query CSR for :func:`qsa_bwd_dkv_tile_kernel` (host-sync free, fixed
    shape); see :func:`build_qsa_kv_tile_csr_torch` for the contract."""
    if not _use_fused_meta(block_indices):
        return build_qsa_kv_tile_csr_torch(block_indices, block_counts, num_blocks, tile_blocks)
    from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
        qsa_build_kv_tile_csr_fused,
    )

    tile_offsets, entry_query, entry_bits, n_tiles = qsa_build_kv_tile_csr_fused(
        block_indices, block_counts, num_blocks, tile_blocks
    )
    return QSAKVTileCSR(
        tile_offsets=tile_offsets,
        entry_query=entry_query,
        entry_bits=entry_bits,
        tile_blocks=tile_blocks,
        n_tiles=n_tiles,
    )


# ---------------------------------------------------------------------------
# 2. Backward: dk/dv (no atomics)
# ---------------------------------------------------------------------------


def _prune_dkv_bq(configs, nargs, **kwargs):
    # The dkv gather stacks BQD queries x GP group heads into a [BQD*GP]-wide tile.
    # tl.dot needs the tile at least 16 wide, and past ~1024 columns the Triton
    # compiler fails outright (which autotune cannot catch), so keep 16 <= BQD*GP
    # <= 256. At least one config always survives (BQD=16 covers GP <= 16; BQD=1
    # covers GP >= 16).
    gp = {**(nargs or {}), **kwargs}.get('GP', 1)
    kept = [c for c in configs if 16 <= c.kwargs['BQD'] * gp <= 256]
    return kept or [c for c in configs if c.kwargs['BQD'] == 1]


@triton.autotune(
    configs=[
        triton.Config({'BQD': bqd}, num_warps=w, num_stages=st)
        for bqd in [1, 2, 4, 8, 16]
        for w in [4, 8]
        for st in [1, 2]
    ],
    key=['BK', 'BV', 'GP'],
    prune_configs_by={'early_config_prune': _prune_dkv_bq},
)
@triton.jit
def qsa_bwd_dkv_kernel(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    q_pos,  # [TQ] global position of each query row
    qsel_ids,  # [Nsel] int32 — query ROWS grouped by (batch, block)
    qsel_offsets,  # [b * NB + 1] int64
    block_bases,
    block_ends,
    scale,
    TQ,
    TK,
    NB,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BSZ: tl.constexpr,
    RSZ: tl.constexpr,
    BQD: tl.constexpr,
):
    i_bj = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = i_bj // NB
    j = i_bj % NB

    CQ: tl.constexpr = BQD * GP  # query-column tile width
    base = tl.load(block_bases + j).to(tl.int64)
    tend = tl.load(block_ends + j).to(tl.int64)
    o_rs = tl.arange(0, RSZ)
    toks = base + o_rs
    m_tok = (o_rs < BSZ) & (toks < tend)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_d = o_d < K
    m_v = o_v < V

    kv_ptr = (i_b * TK + toks) * H + i_h
    b_k = tl.load(
        k + kv_ptr[:, None] * K + o_d[None, :], mask=m_tok[:, None] & m_d[None, :], other=0.0
    )
    b_v = tl.load(
        v + kv_ptr[:, None] * V + o_v[None, :], mask=m_tok[:, None] & m_v[None, :], other=0.0
    )

    e0 = tl.load(qsel_offsets + i_bj)
    e1 = tl.load(qsel_offsets + i_bj + 1)

    b_dk = tl.zeros([RSZ, BK], dtype=tl.float32)
    b_dv = tl.zeros([RSZ, BV], dtype=tl.float32)
    o_c = tl.arange(0, CQ)
    q_slot = o_c // GP
    g = o_c % GP
    head = i_h * G + g
    for es in range(e0, e1, BQD):
        o_e = es + q_slot
        m_e = o_e < e1
        o_t = tl.load(qsel_ids + o_e, mask=m_e, other=0).to(tl.int64)
        o_pt = tl.load(q_pos + o_t, mask=m_e, other=0).to(tl.int64)
        m_col = m_e & (g < G)

        rowptr = (i_b * TQ + o_t) * HQ + head
        b_q = tl.load(
            q + rowptr[:, None] * K + o_d[None, :], mask=m_col[:, None] & m_d[None, :], other=0.0
        )
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(
            do + rowptr[:, None] * V + o_v[None, :], mask=m_col[:, None] & m_v[None, :], other=0.0
        )
        b_lse = tl.load(lse + rowptr, mask=m_col, other=0.0)
        b_delta = tl.load(delta + rowptr, mask=m_col, other=0.0)

        # [RSZ, BK] @ [BK, CQ] -> [RSZ, CQ]
        b_s = tl.dot(b_k, tl.trans(b_q))
        m_all = m_tok[:, None] & m_col[None, :] & (o_pt[None, :] >= toks[:, None])
        # b_q here keeps the plain `scale` factor (it doubles as the dk weight
        # below), so convert to the log2 domain at the exponent instead.
        b_p = tl.where(m_all, tl.exp2(b_s * 1.4426950408889634 - b_lse[None, :]), 0.0)

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(
        dk + kv_ptr[:, None] * K + o_d[None, :],
        b_dk.to(dk.dtype.element_ty),
        mask=m_tok[:, None] & m_d[None, :],
    )
    tl.store(
        dv + kv_ptr[:, None] * V + o_v[None, :],
        b_dv.to(dv.dtype.element_ty),
        mask=m_tok[:, None] & m_v[None, :],
    )


@triton.jit
def qsa_bwd_dkv_tile_kernel(
    q,  # [B, TQ, HQ, K]
    k,  # [B, TK, H, K]
    v,  # [B, TK, H, V]
    lse,  # [B, TQ, HQ] fp32 (log2 domain)
    delta,  # [B, TQ, HQ] fp32
    do,  # [B, TQ, HQ, V]
    dk,  # NSPLIT == 1: [B, TK, H, K]; else fp32 partials [NSPLIT, B, TK, H, BK]
    dv,  # NSPLIT == 1: [B, TK, H, V]; else fp32 partials [NSPLIT, B, TK, H, BV]
    q_pos,  # [TQ] global position of each query row
    entry_query,  # [E] int32 — query ROWS grouped by (batch, KV tile), ascending
    entry_bits,  # [E] int32 — per-entry bitmask of the tile's blocks the query selected
    tile_offsets,  # [B * NT + 1] int64
    block_bases,
    block_ends,
    scale,
    TQ,
    TK,
    NB,
    NT,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BSZ: tl.constexpr,
    KB: tl.constexpr,
    CQ: tl.constexpr,
    NSPLIT: tl.constexpr,
    TRANS: tl.constexpr,
):
    """dk/dv for ``KB`` consecutive selection blocks (``RT = KB * BSZ`` KV rows).

    Columns are the flattened (entry, group head) pairs ``c = entry * G + g`` of the
    tile's query CSR, processed ``CQ`` at a time in ascending order (deterministic).
    ``TRANS`` computes the transposed products (``[CQ, RT]`` scores, ``[BK, RT]``
    accumulators) so the long dimension is the MMA M dimension.
    """
    i_bt = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_s = tl.program_id(2).to(tl.int64)
    i_b = i_bt // NT
    tile = i_bt % NT

    RT: tl.constexpr = KB * BSZ
    o_r = tl.arange(0, RT)
    r_blk = o_r // BSZ
    j = tile * KB + r_blk
    m_blk = j < NB
    base = tl.load(block_bases + j, mask=m_blk, other=0).to(tl.int64)
    tend = tl.load(block_ends + j, mask=m_blk, other=0).to(tl.int64)
    toks = base + (o_r % BSZ)
    m_tok = m_blk & (toks < tend)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_d = o_d < K
    m_v = o_v < V

    kv_ptr = (i_b * TK + toks) * H + i_h
    b_k = tl.load(
        k + kv_ptr[:, None] * K + o_d[None, :], mask=m_tok[:, None] & m_d[None, :], other=0.0
    )
    b_v = tl.load(
        v + kv_ptr[:, None] * V + o_v[None, :], mask=m_tok[:, None] & m_v[None, :], other=0.0
    )

    e0 = tl.load(tile_offsets + i_bt)
    e1 = tl.load(tile_offsets + i_bt + 1)
    c0 = e0 * G
    c1 = e1 * G
    if NSPLIT > 1:
        # Contiguous, CQ-aligned column chunk per split (shape-only split count).
        chunk = tl.cdiv(tl.cdiv(c1 - c0, NSPLIT), CQ) * CQ
        cs0 = c0 + i_s * chunk
        cs1 = tl.minimum(cs0 + chunk, c1)
    else:
        cs0 = c0
        cs1 = c1

    if TRANS:
        b_dk = tl.zeros([BK, RT], dtype=tl.float32)
        b_dv = tl.zeros([BV, RT], dtype=tl.float32)
    else:
        b_dk = tl.zeros([RT, BK], dtype=tl.float32)
        b_dv = tl.zeros([RT, BV], dtype=tl.float32)
    o_c = tl.arange(0, CQ)
    for cs in range(cs0, cs1, CQ):
        c = cs + o_c
        m_c = c < cs1
        e = c // G
        g = c % G
        o_t = tl.load(entry_query + e, mask=m_c, other=0).to(tl.int64)
        bits = tl.load(entry_bits + e, mask=m_c, other=0)
        o_pt = tl.load(q_pos + o_t, mask=m_c, other=0).to(tl.int64)
        rowptr = (i_b * TQ + o_t) * HQ + i_h * G + g
        b_q = tl.load(
            q + rowptr[:, None] * K + o_d[None, :], mask=m_c[:, None] & m_d[None, :], other=0.0
        )
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(
            do + rowptr[:, None] * V + o_v[None, :], mask=m_c[:, None] & m_v[None, :], other=0.0
        )
        b_lse = tl.load(lse + rowptr, mask=m_c, other=0.0)
        b_delta = tl.load(delta + rowptr, mask=m_c, other=0.0)
        if TRANS:
            # [CQ, RT] orientation
            mem = ((bits[:, None] >> r_blk[None, :]) & 1) != 0
            m_all = mem & m_c[:, None] & m_tok[None, :] & (o_pt[:, None] >= toks[None, :])
            b_s = tl.dot(b_q, tl.trans(b_k))
            b_p = tl.where(m_all, tl.exp2(b_s * 1.4426950408889634 - b_lse[:, None]), 0.0)
            b_dv += tl.dot(tl.trans(b_do), b_p.to(b_do.dtype))
            b_dp = tl.dot(b_do, tl.trans(b_v))
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dk += tl.dot(tl.trans(b_q), b_ds.to(b_q.dtype))
        else:
            # [RT, CQ] orientation (same algebra as qsa_bwd_dkv_kernel)
            mem = ((bits[None, :] >> r_blk[:, None]) & 1) != 0
            m_all = mem & m_c[None, :] & m_tok[:, None] & (o_pt[None, :] >= toks[:, None])
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = tl.where(m_all, tl.exp2(b_s * 1.4426950408889634 - b_lse[None, :]), 0.0)
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[None, :])
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    if TRANS:
        b_dk = tl.trans(b_dk)
        b_dv = tl.trans(b_dv)
    if NSPLIT > 1:
        # fp32 partials, padded to BK/BV columns; summed over splits on the host side.
        p_ptr = i_s * (tl.num_programs(0) // NT) * TK * H + kv_ptr
        tl.store(dk + p_ptr[:, None] * BK + o_d[None, :], b_dk, mask=m_tok[:, None])
        tl.store(dv + p_ptr[:, None] * BV + o_v[None, :], b_dv, mask=m_tok[:, None])
    else:
        tl.store(
            dk + kv_ptr[:, None] * K + o_d[None, :],
            b_dk.to(dk.dtype.element_ty),
            mask=m_tok[:, None] & m_d[None, :],
        )
        tl.store(
            dv + kv_ptr[:, None] * V + o_v[None, :],
            b_dv.to(dv.dtype.element_ty),
            mask=m_tok[:, None] & m_v[None, :],
        )


# ---------------------------------------------------------------------------
# 3. Host wrappers
# ---------------------------------------------------------------------------


def _pow2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def _launch_dims(K: int, V: int, G: int):
    BK = max(16, _pow2(K))
    BV = max(16, _pow2(V))
    assert BK <= 256 and BV <= 256, "QSA kernels support head dims up to 256"
    GP = _pow2(G)
    return BK, BV, GP


class QSADkvTileConfig(NamedTuple):
    """Launch configuration of :func:`qsa_bwd_dkv_tile_kernel` (shape-only, never data).

    The default was the fastest in a GB300 sweep at the Qwen3.8 proxy shape (THD 4096,
    24 query / 2 KV heads, head dim 256, budget 2048): 1.15-1.22 ms vs 9.2 ms for the
    per-block kernel. Larger KV tiles amortize each gathered q/dO row over more KV rows;
    the column split balances the early tiles, which almost every query selects.
    """

    tile_blocks: int = 16  # selection blocks per KV tile (RT = 4 * tile_blocks KV rows)
    cq: int = 64  # (query, group head) columns per iteration
    num_warps: int = 8
    num_stages: int = 2
    trans: bool = False
    nsplit: int = 0  # column-range splits per tile; 0 = derive from the grid size


# Implementation of the dk/dv backward: "tile" (default) or "block" (original kernel).
# The environment overrides exist for A/B measurements only.
QSA_DKV_IMPL = os.environ.get("MEGATRON_QSA_DKV_IMPL", "tile")
QSA_DKV_TILE_CONFIG = QSADkvTileConfig()


def _parse_dkv_tile_config(spec: str) -> QSADkvTileConfig:
    """Parse ``"kb=4,cq=64,w=4,st=2,trans=0,nsplit=0"`` (unspecified fields keep defaults)."""
    names = {
        "kb": "tile_blocks",
        "cq": "cq",
        "w": "num_warps",
        "st": "num_stages",
        "trans": "trans",
        "nsplit": "nsplit",
    }
    kw = {}
    for item in filter(None, (x.strip() for x in spec.split(","))):
        key, val = item.split("=")
        field = names[key.strip()]
        kw[field] = bool(int(val)) if field == "trans" else int(val)
    return QSA_DKV_TILE_CONFIG._replace(**kw)


if os.environ.get("MEGATRON_QSA_DKV_TILE_CFG"):
    QSA_DKV_TILE_CONFIG = _parse_dkv_tile_config(os.environ["MEGATRON_QSA_DKV_TILE_CFG"])

# (dtype, BK, BV, G, config, nsplit) -> config that compiled within the shared-memory budget
_DKV_TILE_RESOLVED = {}
# same key -> last compiled kernel (register / spill / shared-memory introspection)
_DKV_TILE_LAST_KERNEL = {}


def _dkv_tile_nsplit(cfg: QSADkvTileConfig, n_programs: int, device) -> int:
    if cfg.nsplit > 0:
        return cfg.nsplit
    # Shape-only: split a tile's column range until the grid fills ~4 waves (the per-tile
    # work is very uneven: early KV tiles are selected by almost every later query).
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    nsplit = 1
    while nsplit < 8 and n_programs * nsplit < 4 * sms:
        nsplit *= 2
    return nsplit


def qsa_bwd_dkv_tile(
    q,
    k,
    v,
    lse,
    delta,
    do,
    q_pos,
    csr: QSAKVTileCSR,
    block_bases,
    block_ends,
    scale,
    config: Optional[QSADkvTileConfig] = None,
):
    """dk, dv via :func:`qsa_bwd_dkv_tile_kernel` (deterministic; see the module header)."""
    cfg = QSA_DKV_TILE_CONFIG if config is None else config
    assert cfg.tile_blocks == csr.tile_blocks, "CSR built for a different KV tile size"
    assert cfg.tile_blocks & (cfg.tile_blocks - 1) == 0, "tile_blocks must be a power of two"
    B, TQ, HQ, K = q.shape
    TK, H, V = k.shape[1], k.shape[2], v.shape[-1]
    G = HQ // H
    BK, BV, _ = _launch_dims(K, V, G)
    NB = block_bases.shape[0]
    NT = csr.n_tiles
    nsplit = _dkv_tile_nsplit(cfg, B * NT * H, q.device)
    key = (q.dtype, BK, BV, G, cfg, nsplit)
    run_cfg = _DKV_TILE_RESOLVED.get(key, cfg)
    while True:
        if nsplit > 1:
            dk_out = torch.zeros(nsplit, B, TK, H, BK, dtype=torch.float32, device=q.device)
            dv_out = torch.zeros(nsplit, B, TK, H, BV, dtype=torch.float32, device=q.device)
        else:
            dk_out = torch.zeros_like(k)
            dv_out = torch.zeros_like(v)
        try:
            _DKV_TILE_LAST_KERNEL[key] = qsa_bwd_dkv_tile_kernel[(B * NT, H, nsplit)](
                q=q,
                k=k,
                v=v,
                lse=lse,
                delta=delta,
                do=do,
                dk=dk_out,
                dv=dv_out,
                q_pos=q_pos,
                entry_query=csr.entry_query,
                entry_bits=csr.entry_bits,
                tile_offsets=csr.tile_offsets,
                block_bases=block_bases,
                block_ends=block_ends,
                scale=scale,
                TQ=TQ,
                TK=TK,
                NB=NB,
                NT=NT,
                H=H,
                HQ=HQ,
                G=G,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                BSZ=BS,
                KB=run_cfg.tile_blocks,
                CQ=run_cfg.cq,
                NSPLIT=nsplit,
                TRANS=run_cfg.trans,
                num_warps=run_cfg.num_warps,
                num_stages=run_cfg.num_stages,
            )
            break
        except OutOfResources:
            # Shape-only fallback (e.g. fp32 inputs at head dim 256): halve the column
            # tile, then drop pipelining. Resolved once per key, before any capture.
            if run_cfg.cq > 16:
                run_cfg = run_cfg._replace(cq=run_cfg.cq // 2)
            elif run_cfg.num_stages > 1:
                run_cfg = run_cfg._replace(num_stages=1)
            else:
                raise
    _DKV_TILE_RESOLVED[key] = run_cfg
    if nsplit > 1:
        # Fixed-order (deterministic) reduction of the split partials.
        dk_out = dk_out.sum(0)[..., :K].to(k.dtype)
        dv_out = dv_out.sum(0)[..., :V].to(v.dtype)
    return dk_out, dv_out


def qsa_bwd_dkv_block(q, k, v, lse, delta, do, q_pos, query_csr, block_bases, block_ends, scale):
    """dk, dv via the original per-block :func:`qsa_bwd_dkv_kernel`."""
    B, TQ, HQ, K = q.shape
    TK, H, V = k.shape[1], k.shape[2], v.shape[-1]
    G = HQ // H
    BK, BV, GP = _launch_dims(K, V, G)
    NB = block_bases.shape[0]
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    qsel_ids, qsel_offsets = query_csr
    qsa_bwd_dkv_kernel[(B * NB, H)](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        q_pos=q_pos,
        qsel_ids=qsel_ids,
        qsel_offsets=qsel_offsets,
        block_bases=block_bases,
        block_ends=block_ends,
        scale=scale,
        TQ=TQ,
        TK=TK,
        NB=NB,
        H=H,
        HQ=HQ,
        G=G,
        GP=GP,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BSZ=BS,
        RSZ=RS,
    )
    return dk, dv
