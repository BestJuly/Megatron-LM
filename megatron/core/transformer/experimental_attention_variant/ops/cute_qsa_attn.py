# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA sparse GQA attention on Blackwell (SM100): host wrappers.

:func:`qsa_cute_fwd` launches the forward CuTe DSL kernel of :mod:`cute_qsa_attn_kernels` over
the query-tile supersets of :mod:`triton_qsa`. The LSE is kept in the log2 domain.

Compilation: one compiled kernel per (kernel, dtype, head dim, GQA group, tile) key; every
sequence-dependent size (``B``, ``TQ``, ``TK``, the entry count) is a runtime argument, so the
static-shape proxy never recompiles in steady state. Run one eager step before CUDA-graph
capture so that compilation happens outside the capture. ``cutlass`` is imported on first use.

Supported inputs: an SM100-class GPU, bf16 / fp16, head dim 128 / 256, and a GQA group of at
most 128 query heads per KV head.
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
