# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QSA sparse GQA attention on Blackwell (SM100): host wrappers and autograd.

:func:`qsa_sparse_attention` is the attention entry point of :class:`QSASelfAttention`. It
builds the sparse-attention metadata of :mod:`triton_qsa` (query-tile supersets, KV-tile query
CSR) and runs the forward and dq as the CuTe DSL kernels of :mod:`cute_qsa_attn_kernels`. dk/dv
still run the Triton kernel of :mod:`triton_qsa`, fed with the log2-domain LSE and the ``delta``
that the dq kernel emits.

Compilation: one compiled kernel per (kernel, dtype, head dim, GQA group, tile) key; every
sequence-dependent size (``B``, ``TQ``, ``TK``, the entry count) is a runtime argument, so the
static-shape proxy never recompiles in steady state. Run one eager step before CUDA-graph
capture so that compilation happens outside the capture. ``cutlass`` is imported on first use.

Supported inputs (validated once when a QSA layer is built, see ``qsa.py``): an SM100-class GPU,
bf16 / fp16, head dim 128 / 256, and a GQA group of at most 128 query heads per KV head.
"""

from typing import Optional

import torch

from megatron.core.transformer.experimental_attention_variant.ops import triton_qsa as tq

# Gathered KV rows per forward step (16 union entries of 4-token blocks).
FWD_N_BLOCK = 128

_COMPILED = {}


def _cutlass_dtype(dtype):
    import cutlass

    return {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}[dtype]


def cute_tile_queries(num_q_heads: int, num_kv_heads: int) -> int:
    """Queries per forward/dq tile: ``BQ * G`` flattened rows fill the 128-row MMA tile."""
    return max(1, 128 // (num_q_heads // num_kv_heads))


def _fake(dtype, ndim: int, last: Optional[int] = None, div: int = 1, align: int = 4):
    import cutlass.cute as cute

    shape = tuple(
        (last if (i == ndim - 1 and last is not None) else cute.sym_int()) for i in range(ndim)
    )
    stride = tuple(cute.sym_int64(divisibility=div) if i != ndim - 1 else 1 for i in range(ndim))
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=align)


def _compiled_fwd(dtype: torch.dtype, D: int, G: int, BQ: int):
    key = ("fwd", dtype, D, G, BQ)
    fn = _COMPILED.get(key)
    if fn is None:
        import cutlass
        import cutlass.cute as cute

        from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_attn_kernels import (  # pylint: disable=line-too-long
            QSAFwdSm100,
        )

        cdt = _cutlass_dtype(dtype)
        kernel = QSAFwdSm100(D, G, BQ, n_block=FWD_N_BLOCK)
        args = [
            _fake(cdt, 4, D, 8, 16),  # q
            _fake(cdt, 4, D, 8, 16),  # k
            _fake(cdt, 4, D, 8, 16),  # v
            _fake(cdt, 4, D, 8, 16),  # o
            _fake(cutlass.Float32, 3),  # lse
            _fake(cutlass.Int64, 1, align=8),  # q_pos
            _fake(cutlass.Int64, 1, align=8),  # tile_offsets
            _fake(cutlass.Int32, 1),  # entry_base
            _fake(cutlass.Int32, 1),  # entry_end
            _fake(cutlass.Uint8, 2, BQ, align=1),  # membership [E, BQ]
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        ]
        fn = cute.compile(kernel, *args, options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


def qsa_cute_fwd(q, k, v, q_pos, ss: tq.QSASupersets, scale: float, tile_queries: int):
    """Sparse GQA forward -> ``(o, lse)``; ``lse`` is in the log2 domain."""
    B, TQ, HQ, D = q.shape
    G = HQ // k.shape[2]
    o = torch.empty(B, TQ, HQ, D, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, TQ, HQ, dtype=torch.float32, device=q.device)
    fn = _compiled_fwd(q.dtype, D, G, tile_queries)
    membership = ss.membership
    if membership.shape[0] == 0:
        membership = torch.zeros(1, tile_queries, dtype=torch.uint8, device=q.device)
    fn(
        q,
        k,
        v,
        o,
        lse,
        q_pos,
        ss.tile_offsets,
        ss.entry_base,
        ss.entry_end,
        membership,
        float(scale) * 1.4426950408889634,
        ss.n_tiles,
    )
    return o, lse


def _compiled_dq(dtype: torch.dtype, D: int, G: int, BQ: int):
    key = ("dq", dtype, D, G, BQ)
    fn = _COMPILED.get(key)
    if fn is None:
        import cutlass
        import cutlass.cute as cute

        from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_attn_kernels import (  # pylint: disable=line-too-long
            QSADqSm100,
        )

        cdt = _cutlass_dtype(dtype)
        kernel = QSADqSm100(D, G, BQ)
        bf = lambda: _fake(cdt, 4, D, 8, 16)  # noqa: E731
        args = [
            bf(),  # q
            bf(),  # k
            bf(),  # v
            bf(),  # o
            bf(),  # do
            _fake(cutlass.Float32, 3),  # lse
            _fake(cutlass.Float32, 3),  # delta (out)
            bf(),  # dq (out)
            _fake(cutlass.Int64, 1, align=8),  # q_pos
            _fake(cutlass.Int64, 1, align=8),  # tile_offsets
            _fake(cutlass.Int32, 1),  # entry_base
            _fake(cutlass.Int32, 1),  # entry_end
            _fake(cutlass.Uint8, 2, BQ, align=1),  # membership
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        ]
        fn = cute.compile(kernel, *args, options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


def qsa_cute_dq(q, k, v, o, do, lse, q_pos, ss: tq.QSASupersets, scale: float, tile_queries: int):
    """dq and the fused ``delta = rowsum(o * do)`` (fp32) -> ``(dq, delta)``."""
    B, TQ, HQ, D = q.shape
    G = HQ // k.shape[2]
    dq = torch.empty_like(q)
    delta = torch.empty(B, TQ, HQ, dtype=torch.float32, device=q.device)
    fn = _compiled_dq(q.dtype, D, G, tile_queries)
    membership = ss.membership
    if membership.shape[0] == 0:
        membership = torch.zeros(1, tile_queries, dtype=torch.uint8, device=q.device)
    fn(q, k, v, o, do, lse, delta, dq, q_pos, ss.tile_offsets, ss.entry_base, ss.entry_end,
       membership, float(scale), ss.n_tiles)
    return dq, delta


class QSASparseAttnFunction(torch.autograd.Function):
    """Autograd over the sparse GQA kernels (forward, dq: CuTe DSL; dk/dv: Triton)."""

    @staticmethod
    def forward(ctx, q, k, v, q_pos, ss, kv_csr, block_bases, block_ends, scale, bq):
        o, lse = qsa_cute_fwd(q, k, v, q_pos, ss, scale, bq)
        ctx.save_for_backward(q, k, v, o, lse, q_pos, block_bases, block_ends)
        ctx.ss = ss
        ctx.kv_csr = kv_csr
        ctx.scale = scale
        ctx.bq = bq
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, q_pos, block_bases, block_ends = ctx.saved_tensors
        do = do.contiguous()
        dq, delta = qsa_cute_dq(q, k, v, o, do, lse, q_pos, ctx.ss, ctx.scale, ctx.bq)
        dk, dv = tq.qsa_bwd_dkv_tile(
            q, k, v, lse, delta, do, q_pos, ctx.kv_csr, block_bases, block_ends, ctx.scale
        )
        return dq, dk, dv, None, None, None, None, None, None, None


# The sort-free superset builder packs a tile's query bits into an int32 (tile_queries <= 30).
_FUSED_SUPERSET_MAX_BQ = 30


def _superset_builder(static_shapes: bool, tile_queries: int):
    """Superset builder for ``tile_queries`` queries per tile. ``BQ = 128 // G`` exceeds the
    sort-free builder's limit for G <= 4; those use the (bit-identical) sort-based builder."""
    if tile_queries > _FUSED_SUPERSET_MAX_BQ:
        if static_shapes:
            return tq.build_qsa_supersets_static_torch
        return tq.build_qsa_supersets_torch
    return tq.build_qsa_supersets_static if static_shapes else tq.build_qsa_supersets


def qsa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    block_bases: torch.Tensor,
    block_ends: torch.Tensor,
    scale: Optional[float] = None,
    q_positions: Optional[torch.Tensor] = None,
    static_shapes: bool = False,
) -> torch.Tensor:
    """Sparse GQA attention over per-query selected 4-token blocks.

    Args:
        q: ``[B, TQ, HQ, D]`` (no head padding required; any ``HQ/H >= 1``).
        k/v: ``[B, TK, H, D]``. ``TK`` may exceed ``TQ`` under allgather context
            parallelism (local queries against the globally gathered K/V).
        block_indices: ``[B, TQ, S]`` GLOBAL block ids in the ``block_bases`` space
            (document-local ids must be offset by the caller under packed THD).
            Slots beyond ``block_counts`` are ignored.
        block_counts: ``[B, TQ]`` valid slots per query.
        block_bases / block_ends: ``[num_blocks]`` first token and one-past-last
            valid token of every block (encodes ragged document tails and
            fully-future own blocks; the kernels never see cu_seqlens).
        scale: attention scale, default ``D ** -0.5``.
        q_positions: ``[TQ]`` global token position of each query row for causal
            masking. Defaults to ``arange(TQ)``; REQUIRED when ``TQ != TK``.
        static_shapes: build the superset / CSR structures at shape-only upper bounds
            without host synchronization (CUDA-graph capturable). Results are
            bit-identical to the default builders.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if q_positions is None:
        assert q.shape[1] == k.shape[1], (
            "q_positions is required when the query rows are a CP shard (TQ != TK)."
        )
        q_positions = torch.arange(q.shape[1], device=q.device, dtype=torch.int64)
    HQ, H = q.shape[2], k.shape[2]
    num_blocks = block_bases.shape[0]
    bq = cute_tile_queries(HQ, H)
    ss = _superset_builder(static_shapes, bq)(
        block_indices, block_counts, block_bases, block_ends, num_blocks, bq
    )
    needs_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    kv_csr = None
    if needs_grad:
        kv_csr = tq.build_qsa_kv_tile_csr(
            block_indices, block_counts, num_blocks, tq.QSA_DKV_TILE_CONFIG.tile_blocks
        )
    return QSASparseAttnFunction.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        q_positions.contiguous(),
        ss,
        kv_csr,
        block_bases.contiguous(),
        block_ends.contiguous(),
        scale,
        bq,
    )
