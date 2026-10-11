# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused Triton teacher for the QSA indexer distillation loss.

The teacher of the QSA sparse-KL loss (``QSASelfAttention._compute_indexer_loss``) is the
main attention distribution of every query over the tokens of its *selected* blocks:
per head, ``softmax(q . k * scale)`` over the selected tokens, summed over all heads, then
max-pooled to blocks. The eager reference materializes the selected keys
(``[b, chunk, budget, groups, head_dim]``) and the logits, which makes it data-movement
bound. This kernel computes the same quantity without materializing either:

* one program per (query row, batch); the query's heads of one KV group form a
  ``[HP, D]`` tile (``HP`` = heads per group padded to a power of two >= 16);
* the selected blocks are visited in tiles of ``BLK`` blocks (``BLK * R`` tokens); the
  keys of a tile are gathered straight from the ``[b, s, groups, D]`` key tensor;
* pass 1 runs an online max / sum-exp per head, pass 2 recomputes the logits and
  accumulates the head-summed probabilities per token; groups are summed into a per-row
  fp32 token-mass buffer;
* with ``pool=True`` (no tensor parallelism) a last sweep max-pools the token mass to
  blocks and zeroes unpicked slots, which is exactly the pre-normalization target.
  With ``pool=False`` the token mass is returned so the caller can all-reduce it over
  tensor-parallel ranks before pooling (the max-pool does not commute with the sum).

Numerics follow the reference: bf16 operands, fp32 accumulation, the logit is rounded to
bf16 (the reference's einsum output dtype) before the fp32 scale and softmax (fp32 inputs
use an IEEE fp32 dot and no rounding). The remaining differences are fp32 summation order
and the hardware exp.

Graph safety: no host synchronization; the per-row loop bound (index of the last picked
slot + 1) is read on the device. All ``constexpr`` arguments are model constants (head
counts, head dim, block size, tile width), never data-dependent values.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

try:  # accurate exp (diagnostic option); the default path uses tl.exp
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    libdevice = None


@triton.jit
def _exp(x, EXACT_EXP: tl.constexpr):
    if EXACT_EXP:
        return libdevice.exp(x)
    else:
        return tl.exp(x)


@triton.jit
def _teacher_logits(q, kt, scale, IEEE: tl.constexpr, SWAP: tl.constexpr):
    """``[HP, BN]`` fp32 logits of the head tile ``q`` against the gathered key tile."""
    if IEEE:
        if SWAP:
            x = tl.trans(tl.dot(kt, tl.trans(q), input_precision="ieee"))
        else:
            x = tl.dot(q, tl.trans(kt), input_precision="ieee")
        x = x * scale
    else:
        # Reference rounding: the matmul output is in the input dtype (bf16), promoted to
        # fp32 before the scale.
        if SWAP:
            x = tl.trans(tl.dot(kt, tl.trans(q)))
        else:
            x = tl.dot(q, tl.trans(kt))
        x = x.to(q.dtype).to(tl.float32) * scale
    return x


@triton.jit
def _qsa_indexer_teacher_kernel(
    Q,  # [b, s, NG * HPG, D] bf16 / fp16 / fp32 (last dim contiguous)
    K,  # [b, s_kv, NG, D] same dtype (last dim contiguous)
    ORDER,  # [b, s, n_sel] int64 block ids (document-local)
    PICKED,  # [b, s, n_sel] uint8
    NHI,  # [b, s] int32: last picked slot + 1
    TOKOFF,  # [s] int64 token offset of the row's document (read when HAS_TOKOFF)
    ACC,  # [b, s, n_sel * R] fp32 token mass (output when not POOL, scratch otherwise)
    OUT,  # [b, s, n_sel] fp32 pooled target (written when POOL)
    s,
    sq_b,
    sq_s,
    sq_h,
    sk_b,
    sk_s,
    sk_g,
    so_b,
    so_s,
    sp_b,
    sp_s,
    sa_b,
    sa_s,
    sout_b,
    sout_s,
    scale,
    HPG: tl.constexpr,
    HP: tl.constexpr,
    NG: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BLK: tl.constexpr,
    HAS_TOKOFF: tl.constexpr,
    POOL: tl.constexpr,
    IEEE: tl.constexpr,
    SWAP: tl.constexpr,
    EXACT_EXP: tl.constexpr,
):
    t = tl.program_id(0)
    bi = tl.program_id(1).to(tl.int64)
    BN: tl.constexpr = BLK * R
    o_h = tl.arange(0, HP)
    m_h = o_h < HPG
    o_d = tl.arange(0, D)
    o_n = tl.arange(0, BN)
    o_blk = o_n // R
    o_r = o_n % R
    t64 = t.to(tl.int64)

    n_hi = tl.load(NHI + bi * s + t64)
    if HAS_TOKOFF:
        tok0 = tl.load(TOKOFF + t64)
    else:
        tok0 = 0
    order_row = ORDER + bi * so_b + t64 * so_s
    picked_row = PICKED + bi * sp_b + t64 * sp_s
    acc_row = ACC + bi * sa_b + t64 * sa_s
    k_base = K + bi * sk_b

    for g in tl.static_range(NG):
        q = tl.load(
            Q + bi * sq_b + t64 * sq_s + (g * HPG + o_h)[:, None] * sq_h + o_d[None, :],
            mask=m_h[:, None],
            other=0.0,
        )
        # Pass 1: online max / sum-exp per head over the selected tokens.
        m_i = tl.full([HP], float("-inf"), tl.float32)
        l_i = tl.zeros([HP], dtype=tl.float32)
        for j0 in range(0, n_hi, BLK):
            jj = j0 + o_blk
            mj = jj < n_hi
            blk = tl.load(order_row + jj, mask=mj, other=0)
            pk = (tl.load(picked_row + jj, mask=mj, other=0) != 0) & mj
            tok = blk * R + o_r + tok0
            kt = tl.load(
                k_base + tok[:, None] * sk_s + g * sk_g + o_d[None, :],
                mask=pk[:, None],
                other=0.0,
            )
            x = _teacher_logits(q, kt, scale, IEEE, SWAP)
            x = tl.where(pk[None, :], x, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(x, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            l_i = l_i * _exp(m_i - m_safe, EXACT_EXP) + tl.sum(
                _exp(x - m_safe[:, None], EXACT_EXP), 1
            )
            m_i = m_new
        m_fin = tl.where(m_i == float("-inf"), 0.0, m_i)
        l_fin = tl.maximum(l_i, 1e-10)
        # Pass 2: probabilities, summed over this group's heads, accumulated per token.
        for j0 in range(0, n_hi, BLK):
            jj = j0 + o_blk
            mj = jj < n_hi
            blk = tl.load(order_row + jj, mask=mj, other=0)
            pk = (tl.load(picked_row + jj, mask=mj, other=0) != 0) & mj
            tok = blk * R + o_r + tok0
            kt = tl.load(
                k_base + tok[:, None] * sk_s + g * sk_g + o_d[None, :],
                mask=pk[:, None],
                other=0.0,
            )
            x = _teacher_logits(q, kt, scale, IEEE, SWAP)
            x = tl.where(pk[None, :], x, float("-inf"))
            p = _exp(x - m_fin[:, None], EXACT_EXP) / l_fin[:, None]
            p = tl.where(m_h[:, None] & pk[None, :], p, 0.0)
            mass = tl.sum(p, 0)  # [BN]
            ptr = acc_row + j0 * R + o_n
            if g > 0:
                mass += tl.load(ptr, mask=mj, other=0.0)
            tl.store(ptr, mass, mask=mj)
        # Make this group's token mass visible to every thread of the program.
        tl.debug_barrier()

    if POOL:
        o_b = tl.arange(0, BLK)
        out_row = OUT + bi * sout_b + t64 * sout_s
        for j0 in range(0, n_hi, BLK):
            mj = (j0 + o_n // R) < n_hi
            mass = tl.load(acc_row + j0 * R + o_n, mask=mj, other=0.0)
            pooled = tl.max(tl.reshape(mass, (BLK, R)), 1)
            jb = j0 + o_b
            mb = jb < n_hi
            pkb = tl.load(picked_row + jb, mask=mb, other=0) != 0
            pooled = tl.where(pkb, pooled, 0.0)
            tl.store(out_row + jb, pooled, mask=mb)


def _next_pow2_at_least_16(n: int) -> int:
    p = 16
    while p < n:
        p *= 2
    return p


def qsa_indexer_teacher(
    q: torch.Tensor,
    k: torch.Tensor,
    order: torch.Tensor,
    picked: torch.Tensor,
    compress_ratio: int,
    scale: float,
    token_offsets: Optional[torch.Tensor] = None,
    pool: bool = True,
    block_tile: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    swap_dot: bool = False,
    exact_exp: bool = False,
) -> torch.Tensor:
    """Head-summed teacher attention mass over each query's selected tokens.

    Args:
        q: ``[b, s, num_heads, D]`` teacher queries (post qk-norm and RoPE), bf16/fp16,
            last dim contiguous (any other strides).
        k: ``[b, s_kv, num_groups, D]`` teacher keys, same dtype, last dim contiguous.
            Token ids are ``order * compress_ratio + j (+ token_offsets[row])``.
        order: ``[b, s, n_sel]`` selected block ids (int64).
        picked: ``[b, s, n_sel]`` bool validity of each slot.
        compress_ratio: tokens per block (R).
        scale: softmax scale applied to the bf16-rounded logits.
        token_offsets: optional ``[s]`` per-row token offset (packed THD).
        pool: return ``[b, s, n_sel]`` block-max-pooled mass with unpicked slots zeroed
            (no tensor parallelism); else ``[b, s, n_sel, R]`` token mass (to be summed
            over TP ranks before pooling). Neither is L1-normalized.
        block_tile / num_warps / num_stages: launch configuration (selected blocks per
            tile, i.e. ``block_tile * R`` gathered tokens). The defaults were the fastest
            of a sweep at the Qwen3.8 proxy shape (GB300).
        swap_dot / exact_exp: diagnostic numerics options (keys-major dot orientation;
            libdevice exp instead of the hardware exp).

    Returns an fp32 tensor; runs without host synchronization (CUDA-graph capturable).
    """
    assert q.dim() == 4 and k.dim() == 4 and order.dim() == 3
    b, s, n_heads, d = q.shape
    n_groups = k.shape[2]
    assert k.shape[0] == b and k.shape[3] == d
    assert n_heads % n_groups == 0
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "head dim must be contiguous"
    assert q.dtype == k.dtype and q.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert d >= 16 and (d & (d - 1)) == 0, "head dim must be a power of two >= 16"
    n_sel = order.shape[-1]
    r = int(compress_ratio)
    assert (r & (r - 1)) == 0, "compress ratio must be a power of two"
    hpg = n_heads // n_groups
    device = q.device

    order = order.contiguous()
    picked_u8 = picked.contiguous().view(torch.uint8)
    slot = torch.arange(1, n_sel + 1, device=device, dtype=torch.int32)
    n_hi = torch.where(picked, slot, 0).amax(dim=-1).to(torch.int32).contiguous()  # [b, s]

    if pool:
        out = torch.zeros(b, s, n_sel, dtype=torch.float32, device=device)
        acc = torch.empty(b, s, n_sel * r, dtype=torch.float32, device=device)
    else:
        acc = torch.zeros(b, s, n_sel * r, dtype=torch.float32, device=device)
        out = acc
    has_tokoff = token_offsets is not None
    tokoff = token_offsets.to(torch.int64).contiguous() if has_tokoff else n_hi

    if s > 0 and b > 0 and n_sel > 0:
        _qsa_indexer_teacher_kernel[(s, b)](
            q,
            k,
            order,
            picked_u8,
            n_hi,
            tokoff,
            acc,
            out,
            s,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            order.stride(0),
            order.stride(1),
            picked_u8.stride(0),
            picked_u8.stride(1),
            acc.stride(0),
            acc.stride(1),
            out.stride(0),
            out.stride(1),
            float(scale),
            HPG=hpg,
            HP=_next_pow2_at_least_16(hpg),
            NG=n_groups,
            D=d,
            R=r,
            BLK=block_tile,
            HAS_TOKOFF=has_tokoff,
            POOL=pool,
            IEEE=q.dtype == torch.float32,
            SWAP=swap_dot,
            EXACT_EXP=exact_exp,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if pool:
        return out
    return acc.view(b, s, n_sel, r)
