# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA selection helpers: fast exact indexer top-k."""

import functools

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Indexer top-k (K4): scores + exact (score desc, block id asc) top-k without the
# chunked int64 ``cat`` + ``torch.topk``.
# ---------------------------------------------------------------------------
#
# Stage 1 (_qsa_select_score_kernel): the opt2 scoring arithmetic (per head an fp32 IEEE
# dot, relu, sequential head sum, * 1/sqrt(d)) written as fp32 scores at row-relative
# columns ``c = j - m_lo``; tiles no row of the tile can see are skipped (causal skip).
# Stage 2 selects per row, exactly as the int64-key torch.topk does: the k largest
# (score bits, -id) keys among the visible window.
#   * "triton": an in-register 4 x 8-bit radix select of the k-th score (tl.histogram),
#   * "cudnn":  cuDNN FE ``DSA.indexer_top_k_wrapper`` (radix, arbitrary ties) only to
#               find the k-th score value;
#   then the same compaction kernel takes every key above the threshold plus the
#   smallest-id ties at the threshold (the unique-key tie-break), and sorts the k keys
#   descending. The resulting keys are bit-identical to the opt2 path's ``run_keys``.


ID_BITS = 21
_MAX_ID_CODE = (1 << ID_BITS) - 1


@triton.autotune(configs=[triton.Config({}, num_warps=w) for w in [4, 8]], key=['BM', 'BN', 'DI'])
@triton.jit
def _qsa_select_score_kernel(
    q,  # [s, NH, DI]
    bk,  # [n_blocks, DI]
    out,  # [s, ld] fp32 scores at row-relative columns
    m_t,  # [s] exclusive upper visible block id
    m_lo,  # [s] first visible block id (read only when HAS_LO)
    s,
    nb,
    ld,
    scale,
    NH: tl.constexpr,
    DI: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_LO: tl.constexpr,
):
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    o_d = tl.arange(0, DI)
    m_m = o_m < s
    mt = tl.load(m_t + o_m, mask=m_m, other=0)
    if HAS_LO:
        lo = tl.load(m_lo + o_m, mask=m_m, other=0)
    else:
        lo = tl.zeros_like(mt)
    # causal skip: no row of this tile sees any block of this column tile
    if i_n * BN >= tl.max(mt):
        return
    if HAS_LO:
        if (i_n + 1) * BN <= tl.min(tl.where(m_m, lo, nb)):
            return
    m_n = o_n < nb
    b_k = tl.load(bk + o_n[None, :] * DI + o_d[:, None], mask=m_n[None, :], other=0.0).to(
        tl.float32
    )
    b_acc = tl.zeros([BM, BN], dtype=tl.float32)
    for h in tl.static_range(NH):
        b_q = tl.load(
            q + (o_m[:, None] * NH + h) * DI + o_d[None, :], mask=m_m[:, None], other=0.0
        ).to(tl.float32)
        # same arithmetic as the opt2 qsa_indexer_keys_kernel (bit-identical scores)
        b_acc += tl.maximum(tl.dot(b_q, b_k, input_precision="ieee"), 0.0)
    score = b_acc * scale
    vis = (o_n[None, :] < mt[:, None]) & (o_n[None, :] >= lo[:, None]) & m_m[:, None]
    col = o_n[None, :] - lo[:, None]
    tl.store(out + o_m[:, None].to(tl.int64) * ld + col, score, mask=vis)


@triton.jit
def _qsa_select_topk_kernel(
    scores,  # [rows, ld] fp32 (row-relative columns)
    thr,  # [rows] int32 k-th score bits (read only when GIVEN_THR)
    keys_out,  # [rows, KP] int64
    m_t,
    m_lo,
    k,
    ld,
    max_id_code,
    TILE: tl.constexpr,
    KP: tl.constexpr,
    IDB: tl.constexpr,
    HAS_LO: tl.constexpr,
    GIVEN_THR: tl.constexpr,
):
    """One row: exact top-k keys over the visible window, streamed in TILE columns."""
    row = tl.program_id(0)
    mt = tl.load(m_t + row).to(tl.int32)
    if HAS_LO:
        lo = tl.load(m_lo + row).to(tl.int32)
    else:
        lo = mt * 0
    n_vis = mt - lo
    srow = scores + row.to(tl.int64) * ld
    ct = tl.arange(0, TILE)
    zero = mt * 0
    if GIVEN_THR:
        T = tl.load(thr + row)
        T = tl.where(n_vis > k, T, -1)
        gt = zero
        n_eq = zero
        for c0 in range(0, n_vis, TILE):
            c = c0 + ct
            valid = c < n_vis
            x = tl.load(srow + c, mask=valid, other=0.0)
            bits = tl.where(valid, x.to(tl.int32, bitcast=True), -1)
            gt += tl.sum((bits > T).to(tl.int32), axis=0)
            n_eq += tl.sum(((bits == T) & valid).to(tl.int32), axis=0)
        kk = k - gt
    else:
        T = zero - 1
        kk = k + zero
        n_eq = zero
        if n_vis > k:
            prefix = zero
            bins = tl.arange(0, 256)
            for p in tl.static_range(4):
                shift = 24 - 8 * p
                hist = tl.zeros([256], dtype=tl.int32)
                for c0 in range(0, n_vis, TILE):
                    c = c0 + ct
                    valid = c < n_vis
                    x = tl.load(srow + c, mask=valid, other=0.0)
                    bits = tl.where(valid, x.to(tl.int32, bitcast=True), -1)
                    if p == 0:
                        m = valid
                    else:
                        m = valid & ((bits >> (shift + 8)) == (prefix >> (shift + 8)))
                    hist += tl.histogram((bits >> shift) & 255, 256, mask=m)
                cge = tl.cumsum(hist, 0, reverse=True)
                dstar = tl.max(tl.where(cge >= kk, bins, -1), axis=0)
                kk = kk - tl.sum(tl.where(bins > dstar, hist, 0), axis=0)
                prefix = prefix | (dstar << shift)
                if p == 3:
                    n_eq = tl.sum(tl.where(bins == dstar, hist, 0), axis=0)
            T = prefix
    # Every key above T, plus the smallest-id ties at T up to k (the unique-key
    # tie-break), compacted with a per-tile scan, then sorted descending.
    base = keys_out + row.to(tl.int64) * KP
    run_eq = zero
    run_pos = zero
    partial_ties = n_eq > kk
    for c0 in range(0, n_vis, TILE):
        c = c0 + ct
        valid = c < n_vis
        x = tl.load(srow + c, mask=valid, other=0.0)
        # relu scores are >= +0: the int32 bit pattern is order-isomorphic to the value
        bits = tl.where(valid, x.to(tl.int32, bitcast=True), -1)
        if partial_ties:
            eq = (bits == T) & valid
            eq_i = eq.to(tl.int32)
            rank_eq = run_eq + tl.cumsum(eq_i, 0) - eq_i
            sel = valid & ((bits > T) | (eq & (rank_eq < kk)))
            run_eq += tl.sum(eq_i, axis=0)
        else:
            # every tie at T is taken (the common case): no rank scan needed
            sel = valid & (bits >= T)
        sel_i = sel.to(tl.int32)
        pos = run_pos + tl.cumsum(sel_i, 0) - sel_i
        key = ((bits.to(tl.int64) + 1) << IDB) | (max_id_code - (lo + c)).to(tl.int64)
        tl.store(base + pos, key, mask=sel)
        run_pos += tl.sum(sel_i, axis=0)
    o_k = tl.arange(0, KP)
    tl.store(base + o_k, tl.zeros([KP], dtype=tl.int64), mask=o_k >= run_pos)
    tl.debug_barrier()
    kv = tl.load(base + o_k)
    kv = tl.sort(kv, descending=True)
    tl.debug_barrier()
    tl.store(base + o_k, kv)


def _pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


_TOPK_TILE = 2048  # columns per streamed step of the selection kernel
_CHUNK_ELEMS = 1 << 27  # score buffer cap: 128M fp32 = 512 MiB
_CUDNN_MAX_ROWS = 32768  # FE: larger fused top-k calls are unsafe (see notes/02)


def _row_chunk(s: int, nb: int, impl: str) -> int:
    rc = max(64, min(s, _CHUNK_ELEMS // max(nb, 1)))
    if impl == "cudnn":
        rc = min(rc, _CUDNN_MAX_ROWS)
    return rc


_AUTO_CUDNN_MIN_BLOCKS = 4096  # GB200 H2H: Triton radix wins at 1K blocks, cuDNN at 8K/16K


@functools.lru_cache(maxsize=None)
def _has_cudnn_topk() -> bool:
    try:
        from cudnn import DSA  # noqa: F401

        return hasattr(DSA, "indexer_top_k_wrapper")
    except Exception:  # pragma: no cover - depends on the container
        return False


def _auto_impl(nb: int) -> str:
    """Shape-only dispatch (never data): the measured faster exact top-k per width."""
    if nb >= _AUTO_CUDNN_MIN_BLOCKS and _has_cudnn_topk():
        return "cudnn"
    return "triton"


def _cudnn_kth_bits(buf, n_vis, k):
    """k-th largest score bits per row via cuDNN FE radix top-k (rows with n_vis <= k
    are ignored by the caller). ``buf``/``n_vis`` have a fixed row count, so the FE
    wrapper cache (keyed on n_rows / num_cols) never recompiles in steady state."""
    from cudnn import DSA  # lazy: the Triton path works without cudnn

    out = DSA.indexer_top_k_wrapper(buf, n_vis, k, 1, True)
    vals = out["values"].float()
    vals = torch.where(out["indices"] >= 0, vals, float("inf"))
    return vals.amin(dim=-1).view(torch.int32).contiguous()


def qsa_select_topk_keys(
    q: torch.Tensor,  # [s, NH, DI] (one batch row)
    block_keys: torch.Tensor,  # [n_blocks, DI]
    m_t: torch.Tensor,  # [s]
    k: int,
    m_lo=None,  # [s] or None
    impl: str = "auto",
) -> torch.Tensor:
    """Top-k int64 keys ``[s, k]`` (descending), bit-identical to the opt2 chunked
    ``torch.topk`` merge (kept as the test oracle in ``qsa_reference.py``). ``impl``:
    ``"triton"`` radix select, ``"cudnn"`` FE k-th value, or ``"auto"`` (by static width)."""
    assert impl in ("triton", "cudnn", "auto"), impl
    s, NH, DI = q.shape
    nb = block_keys.shape[0]
    if impl == "auto":
        impl = _auto_impl(nb)
    assert nb < (1 << ID_BITS)
    device = q.device
    KP = _pow2(max(k, 1))
    NBP = _pow2(max(nb, 1))
    keys = torch.empty(s, KP, dtype=torch.int64, device=device)
    if s == 0:
        return keys[:, :k]
    rc = _row_chunk(s, nb, impl)
    buf = torch.empty(rc, nb, dtype=torch.float32, device=device)
    has_lo = m_lo is not None
    m_t = m_t.contiguous()
    m_lo_t = m_lo.contiguous() if has_lo else m_t
    if impl == "cudnn":
        nvis_all = (m_t - m_lo_t if has_lo else m_t).to(torch.int32)
        nvis_buf = torch.zeros(rc, dtype=torch.int32, device=device)
    TILE = min(_TOPK_TILE, NBP)
    nw = 8 if TILE >= 2048 else 4
    for r0 in range(0, s, rc):
        r1 = min(r0 + rc, s)
        n = r1 - r0
        grid = (triton.cdiv(n, 64), triton.cdiv(nb, 64))
        _qsa_select_score_kernel[grid](
            q[r0:r1],
            block_keys,
            buf,
            m_t[r0:r1],
            m_lo_t[r0:r1],
            n,
            nb,
            nb,
            DI**-0.5,
            NH=NH,
            DI=DI,
            BM=64,
            BN=64,
            HAS_LO=has_lo,
        )
        if impl == "cudnn":
            nvis_buf[:n].copy_(nvis_all[r0:r1])
            if n < rc:
                nvis_buf[n:].zero_()
            thr = _cudnn_kth_bits(buf, nvis_buf, k)
        else:
            thr = m_t  # unused
        _qsa_select_topk_kernel[(n,)](
            buf,
            thr,
            keys[r0:r1],
            m_t[r0:r1],
            m_lo_t[r0:r1],
            k,
            nb,
            _MAX_ID_CODE,
            TILE=TILE,
            KP=KP,
            IDB=ID_BITS,
            HAS_LO=has_lo,
            GIVEN_THR=impl == "cudnn",
            num_warps=nw,
        )
    return keys[:, :k]
