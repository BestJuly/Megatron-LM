# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Self-contained Triton kernels for the QSA sparse-GQA attention path.

Design (SGLang-prefill style, replacing the earlier fla-NSA adaptation):

1. **Index-score kernel** (:func:`qsa_indexer_keys`): fused block scoring
   (``relu(q @ k).sum(heads) / sqrt(d)`` in fp32), causal-visibility masking and
   int64 sort-key encoding in one kernel — no ``[s, chunk, heads]`` intermediate is
   ever materialized. The keys feed the deterministic int64 ``torch.topk`` merge
   (valid keys are unique integers, so selection is deterministic by construction).
2. **Query-tile shared-superset gather**: the sparse attention kernels process
   ``BQ`` consecutive queries per program against the UNION of their selected
   blocks (plus per-query membership bits), so each gathered K/V tile is reused
   across the whole query tile instead of being re-fetched per query — the
   gather-traffic roofline that made the per-query kernels lose to dense flash.
   The union / membership / CSR structures are precomputed on the host
   (:func:`build_qsa_supersets`).
3. **Backward**: ``dq`` mirrors the forward per query tile. ``dk``/``dv`` have two
   deterministic implementations (sequential accumulation, no atomics):

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
CB = 16  # union entries gathered per attention-kernel iteration (CB * BS = 64 tokens)
RS = 16  # padded KV rows per dkv program (>= 16 for tl.dot; BS rows are valid)
ID_BITS = 21  # block-id field width in the int64 sort keys (supports 2M blocks)


# ---------------------------------------------------------------------------
# 1. Index-score kernel: fused scoring + visibility + int64 key encoding
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [4, 8]], key=['BM', 'BN', 'DI']
)
@triton.jit
def qsa_indexer_keys_kernel(
    q,  # [s, NH, DI] indexer queries (post norm/RoPE)
    bk,  # [n_blocks, DI] pooled block keys
    keys,  # [s, cb] int64 output
    m_t,  # [s] number of visible complete blocks per query (exclusive upper block id)
    m_lo,  # [s] first visible block id per query (read only when HAS_LO)
    j0,  # first block id of this chunk
    cb,  # chunk width (number of blocks scored)
    s,
    scale,  # 1 / sqrt(DI)
    max_id_code,  # (1 << IDB) - 1
    NH: tl.constexpr,
    DI: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IDB: tl.constexpr,
    HAS_LO: tl.constexpr,
):
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    o_d = tl.arange(0, DI)
    m_m = o_m < s
    m_n = o_n < cb

    # [DI, BN] block-key tile for this chunk
    b_k = tl.load(
        bk + (j0 + o_n)[None, :] * DI + o_d[:, None], mask=m_n[None, :], other=0.0
    ).to(tl.float32)

    b_acc = tl.zeros([BM, BN], dtype=tl.float32)
    for h in tl.static_range(NH):
        b_q = tl.load(
            q + (o_m[:, None] * NH + h) * DI + o_d[None, :], mask=m_m[:, None], other=0.0
        ).to(tl.float32)
        # fp32 IEEE dot: keeps scores reproducible and tie behavior stable
        b_acc += tl.maximum(tl.dot(b_q, b_k, input_precision="ieee"), 0.0)

    score = b_acc * scale
    # relu scores are >= 0, so the fp32 bit pattern is order-isomorphic to the value
    bits = score.to(tl.int32, bitcast=True).to(tl.int64)
    key = ((bits + 1) << IDB) | (max_id_code - (j0 + o_n))[None, :]
    visible = (j0 + o_n)[None, :] < tl.load(m_t + o_m, mask=m_m, other=0)[:, None]
    if HAS_LO:
        # Packed documents share one block-key buffer: a query sees only its own
        # document's blocks, i.e. ids in [m_lo, m_t).
        visible &= (j0 + o_n)[None, :] >= tl.load(m_lo + o_m, mask=m_m, other=0)[:, None]
    key = tl.where(visible & m_n[None, :] & m_m[:, None], key, 0)
    tl.store(keys + o_m[:, None] * cb + o_n[None, :], key, mask=m_m[:, None] & m_n[None, :])


def qsa_indexer_keys(
    q: torch.Tensor,  # [s, NH, DI]
    block_keys: torch.Tensor,  # [n_blocks, DI]
    m_t: torch.Tensor,  # [s]
    j0: int,
    cb: int,
    keys_out: torch.Tensor,  # [s, cb] int64 (reused buffer)
    m_lo: Optional[torch.Tensor] = None,  # [s] first visible block id, default 0
) -> torch.Tensor:
    """Fill ``keys_out`` with int64 (score, block_id) sort keys for blocks [j0, j0+cb).

    Block ``j`` is visible to query ``t`` iff ``m_lo[t] <= j < m_t[t]``. ``m_lo`` lets
    packed documents share one block-key buffer with global block ids; whether it is
    present is a launch-time constant, never a data-dependent one.
    """
    s, NH, DI = q.shape
    BM, BN = 64, 64
    grid = (triton.cdiv(s, BM), triton.cdiv(cb, BN))
    qsa_indexer_keys_kernel[grid](
        q=q,
        bk=block_keys,
        keys=keys_out,
        m_t=m_t,
        m_lo=m_t if m_lo is None else m_lo,
        j0=j0,
        cb=cb,
        s=s,
        scale=DI**-0.5,
        max_id_code=(1 << ID_BITS) - 1,
        NH=NH,
        DI=DI,
        BM=BM,
        BN=BN,
        IDB=ID_BITS,
        HAS_LO=m_lo is not None,
    )
    return keys_out


# ---------------------------------------------------------------------------
# 2. Host-side superset precompute
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


def build_qsa_supersets(
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


def build_qsa_supersets_static(
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


def build_qsa_kv_tile_csr(
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


# ---------------------------------------------------------------------------
# 3. Sparse GQA forward
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=st) for w in [4, 8] for st in [1, 2, 3]
    ],
    key=['BK', 'BV', 'GP', 'BQ'],
)
@triton.jit
def qsa_fwd_kernel(
    q,  # [B, TQ, HQ, K]
    k,  # [B, TK, H, K]
    v,  # [B, TK, H, V]
    o,  # [B, TQ, HQ, V]
    lse,  # [B, TQ, HQ] fp32
    q_pos,  # [TQ] global token position of each query row (arange when TQ == TK)
    tile_offsets,
    entry_base,
    entry_end,
    membership,
    scale,
    TQ,
    TK,
    n_tiles,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    BQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BSZ: tl.constexpr,
    CBZ: tl.constexpr,
):
    i_tile = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1)
    i_h = tl.program_id(2).to(tl.int64)
    i_b = i_tile // n_tiles
    tile = i_tile % n_tiles

    R: tl.constexpr = BQ * GP
    CBT: tl.constexpr = CBZ * BSZ
    o_r = tl.arange(0, R)
    q_loc = o_r // GP
    g = o_r % GP
    o_t = tile * BQ + q_loc  # query ROW within the (possibly CP-local) sample
    m_row = (o_t < TQ) & (g < G)
    # global position of each query row (for causal masking against K tokens)
    b_pos = tl.load(q_pos + o_t, mask=o_t < TQ, other=0).to(tl.int64)
    head = i_h * G + g
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_d = o_d < K
    m_v = o_v < V

    p_q = q + ((i_b * TQ + o_t) * HQ + head)[:, None] * K + o_d[None, :]
    b_q = tl.load(p_q, mask=m_row[:, None] & m_d[None, :], other=0.0)
    # log2-domain online softmax: fold scale * log2(e) into q so exp/log become
    # exp2/log2 (the LSE is stored in the log2 domain; only these kernels read it).
    b_q = (b_q * (scale * 1.4426950408889634)).to(b_q.dtype)

    e0 = tl.load(tile_offsets + i_tile)
    e1 = tl.load(tile_offsets + i_tile + 1)

    b_o = tl.zeros([R, BV], dtype=tl.float32)
    b_m = tl.full([R], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([R], dtype=tl.float32)
    for es in range(e0, e1, CBZ):
        o_e = es + tl.arange(0, CBZ)
        m_e = o_e < e1
        base = tl.load(entry_base + o_e, mask=m_e, other=0).to(tl.int64)
        tend = tl.load(entry_end + o_e, mask=m_e, other=0).to(tl.int64)
        offs2 = base[:, None] + tl.arange(0, BSZ)[None, :]
        val2 = (offs2 < tend[:, None]) & m_e[:, None]
        offs = tl.reshape(offs2, [CBT])
        m_tok = tl.reshape(val2, [CBT])

        p_k = k + ((i_b * TK + offs) * H + i_h)[None, :] * K + o_d[:, None]
        b_k = tl.load(p_k, mask=m_tok[None, :] & m_d[:, None], other=0.0)
        p_v = v + ((i_b * TK + offs) * H + i_h)[:, None] * V + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tok[:, None] & m_v[None, :], other=0.0)

        # [R, CBT] scores
        b_s = tl.dot(b_q, b_k)
        # membership: [BQ, CB] -> broadcast to [R, CBT]
        mem = tl.load(
            membership + o_e[None, :] * BQ + tl.arange(0, BQ)[:, None],
            mask=m_e[None, :],
            other=0,
        )
        m_mem = (
            tl.reshape(
                tl.broadcast_to(tl.reshape(mem, [BQ, 1, CBZ, 1]), [BQ, GP, CBZ, BSZ]), [R, CBT]
            )
            != 0
        )
        causal = offs[None, :] <= b_pos[:, None]
        m_all = m_tok[None, :] & m_mem & causal & m_row[:, None]
        b_s = tl.where(m_all, b_s, float('-inf'))

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_r = tl.where(b_m == float('-inf'), 0.0, tl.exp2(b_m - b_m_new))
        b_p = tl.where(b_s == float('-inf'), 0.0, tl.exp2(b_s - b_m_new[:, None]))
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = b_m_new

    b_o = b_o / tl.maximum(b_acc, 1e-10)[:, None]
    b_lse = tl.where(b_m == float('-inf'), 0.0, b_m + tl.log2(tl.maximum(b_acc, 1e-10)))
    p_o = o + ((i_b * TQ + o_t) * HQ + head)[:, None] * V + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_row[:, None] & m_v[None, :])
    if i_v == 0:
        p_lse = lse + (i_b * TQ + o_t) * HQ + head
        tl.store(p_lse, b_lse, mask=m_row)


# ---------------------------------------------------------------------------
# 4. Backward: dq (per query tile) and dkv (per KV block, no atomics)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=st) for w in [4, 8] for st in [1, 2]
    ],
    key=['BK', 'BV', 'GP', 'BQ'],
)
@triton.jit
def qsa_bwd_dq_kernel(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    q_pos,
    tile_offsets,
    entry_base,
    entry_end,
    membership,
    scale,
    TQ,
    TK,
    n_tiles,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    BQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BSZ: tl.constexpr,
    CBZ: tl.constexpr,
):
    i_tile = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = i_tile // n_tiles
    tile = i_tile % n_tiles

    R: tl.constexpr = BQ * GP
    CBT: tl.constexpr = CBZ * BSZ
    o_r = tl.arange(0, R)
    q_loc = o_r // GP
    g = o_r % GP
    o_t = tile * BQ + q_loc
    m_row = (o_t < TQ) & (g < G)
    b_pos = tl.load(q_pos + o_t, mask=o_t < TQ, other=0).to(tl.int64)
    head = i_h * G + g
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_d = o_d < K
    m_v = o_v < V

    rowptr = (i_b * TQ + o_t) * HQ + head
    b_q = tl.load(
        q + rowptr[:, None] * K + o_d[None, :], mask=m_row[:, None] & m_d[None, :], other=0.0
    )
    # scores in the log2 domain, matching the forward's log2-domain LSE
    b_q = (b_q * (scale * 1.4426950408889634)).to(b_q.dtype)
    b_do = tl.load(
        do + rowptr[:, None] * V + o_v[None, :], mask=m_row[:, None] & m_v[None, :], other=0.0
    )
    b_lse = tl.load(lse + rowptr, mask=m_row, other=0.0)
    b_delta = tl.load(delta + rowptr, mask=m_row, other=0.0)

    e0 = tl.load(tile_offsets + i_tile)
    e1 = tl.load(tile_offsets + i_tile + 1)

    b_dq = tl.zeros([R, BK], dtype=tl.float32)
    for es in range(e0, e1, CBZ):
        o_e = es + tl.arange(0, CBZ)
        m_e = o_e < e1
        base = tl.load(entry_base + o_e, mask=m_e, other=0).to(tl.int64)
        tend = tl.load(entry_end + o_e, mask=m_e, other=0).to(tl.int64)
        offs2 = base[:, None] + tl.arange(0, BSZ)[None, :]
        val2 = (offs2 < tend[:, None]) & m_e[:, None]
        offs = tl.reshape(offs2, [CBT])
        m_tok = tl.reshape(val2, [CBT])

        b_k = tl.load(
            k + ((i_b * TK + offs) * H + i_h)[None, :] * K + o_d[:, None],
            mask=m_tok[None, :] & m_d[:, None],
            other=0.0,
        )
        b_v = tl.load(
            v + ((i_b * TK + offs) * H + i_h)[None, :] * V + o_v[:, None],
            mask=m_tok[None, :] & m_v[:, None],
            other=0.0,
        )

        b_s = tl.dot(b_q, b_k)
        mem = tl.load(
            membership + o_e[None, :] * BQ + tl.arange(0, BQ)[:, None],
            mask=m_e[None, :],
            other=0,
        )
        m_mem = (
            tl.reshape(
                tl.broadcast_to(tl.reshape(mem, [BQ, 1, CBZ, 1]), [BQ, GP, CBZ, BSZ]), [R, CBT]
            )
            != 0
        )
        causal = offs[None, :] <= b_pos[:, None]
        m_all = m_tok[None, :] & m_mem & causal & m_row[:, None]

        b_p = tl.where(m_all, tl.exp2(b_s - b_lse[:, None]), 0.0)
        # [R, BV] @ [BV, CBT] -> [R, CBT]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(
        dq + rowptr[:, None] * K + o_d[None, :],
        b_dq.to(dq.dtype.element_ty),
        mask=m_row[:, None] & m_d[None, :],
    )


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
# 5. Host wrappers and autograd
# ---------------------------------------------------------------------------


def _pow2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def qsa_choose_tile_queries(num_q_heads: int, num_kv_heads: int) -> int:
    """Queries per tile so the row dim BQ * next_pow2(G) is a >=16 dot tile."""
    gp = _pow2(num_q_heads // num_kv_heads)
    return max(4, 16 // gp)


def _launch_dims(K: int, V: int, G: int):
    BK = max(16, _pow2(K))
    BV = max(16, _pow2(V))
    assert BK <= 256 and BV <= 256, "QSA kernels support head dims up to 256"
    GP = _pow2(G)
    return BK, BV, GP


class QSADkvTileConfig(NamedTuple):
    """Launch configuration of :func:`qsa_bwd_dkv_tile_kernel` (shape-only, never data)."""

    tile_blocks: int = 4  # selection blocks per KV tile (RT = 4 * tile_blocks KV rows)
    cq: int = 64  # (query, group head) columns per iteration
    num_warps: int = 4
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


def _dkv_tile_nsplit(cfg: QSADkvTileConfig, n_programs: int, device) -> int:
    if cfg.nsplit > 0:
        return cfg.nsplit
    # Shape-only: split a tile's column range until the grid fills ~2 waves.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    nsplit = 1
    while nsplit < 8 and n_programs * nsplit < 2 * sms:
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
            qsa_bwd_dkv_tile_kernel[(B * NT, H, nsplit)](
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


class QSASparseAttnFunction(torch.autograd.Function):
    """Autograd over the query-tile shared-superset sparse GQA kernels."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        q_pos,
        supersets,
        query_csr,
        block_bases,
        block_ends,
        scale,
        tile_queries,
        dkv_tile_config=None,
    ):
        B, TQ, HQ, K = q.shape
        TK, H, V = k.shape[1], k.shape[2], v.shape[-1]
        G = HQ // H
        BK, BV, GP = _launch_dims(K, V, G)
        NV = triton.cdiv(V, BV)
        assert NV == 1

        o = torch.empty(B, TQ, HQ, V, dtype=v.dtype, device=q.device)
        lse = torch.empty(B, TQ, HQ, dtype=torch.float32, device=q.device)
        n_tiles_total = B * supersets.n_tiles
        qsa_fwd_kernel[(n_tiles_total, NV, H)](
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            q_pos=q_pos,
            tile_offsets=supersets.tile_offsets,
            entry_base=supersets.entry_base,
            entry_end=supersets.entry_end,
            membership=supersets.membership,
            scale=scale,
            TQ=TQ,
            TK=TK,
            n_tiles=supersets.n_tiles,
            H=H,
            HQ=HQ,
            G=G,
            GP=GP,
            BQ=tile_queries,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            BSZ=BS,
            CBZ=CB,
        )
        ctx.save_for_backward(q, k, v, o, lse, q_pos, block_bases, block_ends)
        ctx.supersets = supersets
        ctx.query_csr = query_csr
        ctx.dkv_tile_config = dkv_tile_config
        ctx.scale = scale
        ctx.tile_queries = tile_queries
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, q_pos, block_bases, block_ends = ctx.saved_tensors
        ss: QSASupersets = ctx.supersets
        B, TQ, HQ, K = q.shape
        TK, H, V = k.shape[1], k.shape[2], v.shape[-1]
        G = HQ // H
        BK, BV, GP = _launch_dims(K, V, G)
        BQ = ctx.tile_queries

        do = do.contiguous()
        # delta = rowwise sum(o * do) over the value dim, fp32
        delta = (o.float() * do.float()).sum(-1)

        dq = torch.empty_like(q)
        n_tiles_total = B * ss.n_tiles
        qsa_bwd_dq_kernel[(n_tiles_total, H)](
            q=q,
            k=k,
            v=v,
            lse=lse,
            delta=delta,
            do=do,
            dq=dq,
            q_pos=q_pos,
            tile_offsets=ss.tile_offsets,
            entry_base=ss.entry_base,
            entry_end=ss.entry_end,
            membership=ss.membership,
            scale=ctx.scale,
            TQ=TQ,
            TK=TK,
            n_tiles=ss.n_tiles,
            H=H,
            HQ=HQ,
            G=G,
            GP=GP,
            BQ=BQ,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            BSZ=BS,
            CBZ=CB,
        )

        if isinstance(ctx.query_csr, QSAKVTileCSR):
            dk, dv = qsa_bwd_dkv_tile(
                q,
                k,
                v,
                lse,
                delta,
                do,
                q_pos,
                ctx.query_csr,
                block_bases,
                block_ends,
                ctx.scale,
                ctx.dkv_tile_config,
            )
        else:
            dk, dv = qsa_bwd_dkv_block(
                q, k, v, lse, delta, do, q_pos, ctx.query_csr, block_bases, block_ends, ctx.scale
            )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def qsa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    block_bases: torch.Tensor,
    block_ends: torch.Tensor,
    scale: Optional[float] = None,
    tile_queries: Optional[int] = None,
    q_positions: Optional[torch.Tensor] = None,
    static_shapes: bool = False,
    dkv_impl: Optional[str] = None,
    dkv_tile_config: Optional[QSADkvTileConfig] = None,
) -> torch.Tensor:
    """Sparse GQA attention over per-query selected 4-token blocks.

    Args:
        q: ``[B, TQ, HQ, K]`` (no head padding required; any ``HQ/H >= 1``).
        k/v: ``[B, TK, H, K/V]``. ``TK`` may exceed ``TQ`` under allgather context
            parallelism (local queries against the globally gathered K/V).
        block_indices: ``[B, TQ, S]`` GLOBAL block ids in the ``block_bases`` space
            (document-local ids must be offset by the caller under packed THD).
            Slots beyond ``block_counts`` are ignored.
        block_counts: ``[B, TQ]`` valid slots per query.
        block_bases / block_ends: ``[num_blocks]`` first token and one-past-last
            valid token of every block (encodes ragged document tails and
            fully-future own blocks; the kernels never see cu_seqlens).
        scale: attention scale, default ``K ** -0.5``.
        tile_queries: queries per shared-superset tile (default derived from G).
        q_positions: ``[TQ]`` global token position of each query row for causal
            masking. Defaults to ``arange(TQ)``; REQUIRED when ``TQ != TK``.
        static_shapes: build the superset / query CSR structures at shape-only upper
            bounds without host synchronization (CUDA-graph capturable). Results are
            bit-identical to the default builders.
        dkv_impl: ``"tile"`` or ``"block"`` dk/dv backward (default
            :data:`QSA_DKV_IMPL`). ``"tile"`` builds its query CSR without host
            synchronization in both modes. Both are deterministic.
        dkv_tile_config: launch configuration of the ``"tile"`` backward (default
            :data:`QSA_DKV_TILE_CONFIG`).
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    HQ, H = q.shape[2], k.shape[2]
    if tile_queries is None:
        tile_queries = qsa_choose_tile_queries(HQ, H)
    if q_positions is None:
        assert q.shape[1] == k.shape[1], (
            "q_positions is required when the query rows are a CP shard (TQ != TK)."
        )
        q_positions = torch.arange(q.shape[1], device=q.device, dtype=torch.int64)
    num_blocks = block_bases.shape[0]
    build_supersets = build_qsa_supersets_static if static_shapes else build_qsa_supersets
    build_query_csr = build_qsa_query_csr_static if static_shapes else build_qsa_query_csr
    supersets = build_supersets(
        block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
    )
    needs_grad = torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad
    )
    dkv_impl = QSA_DKV_IMPL if dkv_impl is None else dkv_impl
    assert dkv_impl in ("tile", "block"), f"unknown QSA dkv implementation {dkv_impl!r}"
    if dkv_tile_config is None:
        dkv_tile_config = QSA_DKV_TILE_CONFIG
    query_csr = None
    if needs_grad and dkv_impl == "tile":
        query_csr = build_qsa_kv_tile_csr(
            block_indices, block_counts, num_blocks, dkv_tile_config.tile_blocks
        )
    elif needs_grad:
        query_csr = build_query_csr(block_indices, block_counts, num_blocks)
    return QSASparseAttnFunction.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        q_positions.contiguous(),
        supersets,
        query_csr,
        block_bases.contiguous(),
        block_ends.contiguous(),
        scale,
        tile_queries,
        dkv_tile_config,
    )
