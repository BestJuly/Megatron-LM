# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Triton kernels for the dense half of the Engram module.

The reference path (``Engram._dense_reference``) is a chain of about forty small PyTorch ops
over ``[tokens, num_streams * hidden]`` tensors: three group RMSNorms evaluated in fp32, the
key/query score, the sigmoid gate, the gated value, a channels-first permute for the
depthwise causal convolution, SiLU, the permute back, and the residual add. Most of its time
is strided copies and fp32 intermediates, not arithmetic.

This module computes everything after the two projections in two forward kernels and four
backward kernels:

forward
    ``_gate_fwd_kernel``   one program per token: key/query norms, score, gate, and the
                           conv-norm statistics of the gated value (saved as ``[T, NS]``
                           scalars; the value itself is never materialized).
    ``_conv_fwd_kernel``   2-D tile: recompute the normed value at the ``kernel_size``
                           causal taps, depthwise convolution, SiLU, residual add.
backward
    ``_conv_bwd_kernel``   2-D tile: SiLU backward (stored as the convolution's output
                           gradient) and per-tile partial convolution-weight gradients.
    ``_cnorm_bwd_kernel``  per (token block, stream): convolution input gradient, conv-norm
                           backward, residual gradient, gate gradient, partial gamma grads.
    ``_value_bwd_kernel``  2-D tile: gradient of the shared value (sum over streams).
    ``_kq_bwd_kernel``     per (token block, stream): score/gate scalar chain, key and query
                           norm backward, partial gamma grads.

Numerics follow the reference op by op: every intermediate the reference holds in the
activation dtype is rounded to it here as well (normed key/query/value, the key*query
product, the score chain, the gate, the convolution output, SiLU, the products feeding the
reductions), and every reduction accumulates in fp32. What remains different is summation
order and the last ulp of transcendental functions.

Partial parameter gradients are written per program and reduced with ``torch.sum``, so the
result is bitwise repeatable. No kernel reads a value on the host and every specialization
is fixed by the model configuration (hidden size, stream count, kernel size, dilation), so
the path is CUDA-graph capturable and never recompiles when the token count changes.

Only the unsharded layout is handled: no context parallelism and no sequence-parallel halo.
The caller falls back to the reference path otherwise.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only without Triton
    HAVE_TRITON = False


# Tile sizes. They depend only on the model configuration, never on the data.
_CONV_BLOCK_T = 32
_CONV_BLOCK_C = 128
_ROW_BLOCK = 8


if HAVE_TRITON:

    @triton.jit
    def _rnd(x, DT: tl.constexpr):
        """Round an fp32 value through the activation dtype, as the reference stores it."""
        return x.to(DT).to(tl.float32)

    @triton.jit(do_not_specialize=["T"])
    def _gate_fwd_kernel(
        k_ptr,
        h_ptr,
        sv_ptr,
        gk_ptr,
        gq_ptr,
        gate_ptr,
        score_ptr,
        rk_ptr,
        rq_ptr,
        rv_ptr,
        T,
        eps,
        sqrt_h,
        NS: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ZC: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        DT = k_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        offs = tl.arange(0, BLOCK_H)
        m = offs < H
        sv = tl.load(sv_ptr + t * H + offs, mask=m, other=0.0).to(tl.float32)
        for s in tl.static_range(NS):
            base = t * C + s * H
            k = tl.load(k_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            q = tl.load(h_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            gk = tl.load(gk_ptr + s * H + offs, mask=m, other=0.0).to(tl.float32)
            gq = tl.load(gq_ptr + s * H + offs, mask=m, other=0.0).to(tl.float32)
            if ZC:
                gk = gk + 1.0
                gq = gq + 1.0
            rk = tl.rsqrt(tl.sum(k * k, axis=0) / H + eps)
            rq = tl.rsqrt(tl.sum(q * q, axis=0) / H + eps)
            kn = _rnd((k * rk) * gk, DT)
            qn = _rnd((q * rq) * gq, DT)
            prod = _rnd(kn * qn, DT)
            prod = tl.where(m, prod, 0.0)
            score = _rnd(tl.sum(prod, axis=0), DT)
            score = _rnd(tl.div_rn(score, sqrt_h), DT)
            mag = _rnd(tl.maximum(tl.abs(score), 1e-6), DT)
            root = _rnd(tl.sqrt_rn(mag), DT)
            sign = tl.where(score > 0, 1.0, tl.where(score < 0, -1.0, 0.0))
            signed = _rnd(root * sign, DT)
            gate = _rnd(1.0 / (1.0 + tl.exp(-signed)), DT)
            value = _rnd(gate * sv, DT)
            rv = tl.rsqrt(tl.sum(value * value, axis=0) / H + eps)
            tl.store(gate_ptr + t * NS + s, gate)
            tl.store(score_ptr + t * NS + s, score)
            tl.store(rk_ptr + t * NS + s, rk)
            tl.store(rq_ptr + t * NS + s, rq)
            tl.store(rv_ptr + t * NS + s, rv)

    @triton.jit
    def _normed_value(
        sv_ptr,
        gate_ptr,
        rv_ptr,
        rows,
        valid,
        stream,
        hc,
        gamma,
        NS: tl.constexpr,
        H: tl.constexpr,
        DT: tl.constexpr,
    ):
        """Return (value, conv-normed value) at ``rows``; zeros where ``valid`` is false."""
        sv = tl.load(sv_ptr + rows[:, None] * H + hc[None, :], mask=valid, other=0.0)
        gate = tl.load(gate_ptr + rows[:, None] * NS + stream[None, :], mask=valid, other=0.0)
        rv = tl.load(rv_ptr + rows[:, None] * NS + stream[None, :], mask=valid, other=0.0)
        value = _rnd(gate * sv.to(tl.float32), DT)
        normed = _rnd((value * rv) * gamma[None, :], DT)
        return value, normed

    @triton.jit(do_not_specialize=["T"])
    def _conv_fwd_kernel(
        sv_ptr,
        gate_ptr,
        rv_ptr,
        gc_ptr,
        w_ptr,
        out_ptr,
        T,
        B,
        NS: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        DIL: tl.constexpr,
        ZC: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        DT = out_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        rows = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        rmask = rows < T
        cmask = cols < C
        stream = cols // H
        hc = cols % H
        gamma = tl.load(gc_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        if ZC:
            gamma = gamma + 1.0
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        value_t = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(K):
            src = rows - ((K - 1 - j) * DIL) * B
            valid = (rmask & (src >= 0))[:, None] & cmask[None, :]
            value, normed = _normed_value(
                sv_ptr, gate_ptr, rv_ptr, src, valid, stream, hc, gamma, NS, H, DT
            )
            w = tl.load(w_ptr + cols * K + j, mask=cmask, other=0.0).to(tl.float32)
            acc += w[None, :] * normed
            if j == K - 1:
                value_t = value
        conv = _rnd(acc, DT)
        silu = _rnd(conv / (1.0 + tl.exp(-conv)), DT)
        out = value_t + silu
        omask = rmask[:, None] & cmask[None, :]
        tl.store(out_ptr + rows[:, None] * C + cols[None, :], out.to(DT), mask=omask)

    @triton.jit(do_not_specialize=["T"])
    def _conv_bwd_kernel(
        go_ptr,
        sv_ptr,
        gate_ptr,
        rv_ptr,
        gc_ptr,
        w_ptr,
        gconv_ptr,
        dw_ptr,
        T,
        B,
        NS: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        DIL: tl.constexpr,
        ZC: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        DT = gconv_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        pid_t = tl.program_id(0)
        rows = pid_t.to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        rmask = rows < T
        cmask = cols < C
        omask = rmask[:, None] & cmask[None, :]
        stream = cols // H
        hc = cols % H
        gamma = tl.load(gc_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        if ZC:
            gamma = gamma + 1.0
        # Recompute the convolution output; keep the normed taps for the weight gradient.
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(K):
            src = rows - ((K - 1 - j) * DIL) * B
            valid = (rmask & (src >= 0))[:, None] & cmask[None, :]
            _, normed = _normed_value(
                sv_ptr, gate_ptr, rv_ptr, src, valid, stream, hc, gamma, NS, H, DT
            )
            w = tl.load(w_ptr + cols * K + j, mask=cmask, other=0.0).to(tl.float32)
            acc += w[None, :] * normed
        conv = _rnd(acc, DT)
        sig = 1.0 / (1.0 + tl.exp(-conv))
        go = tl.load(go_ptr + rows[:, None] * C + cols[None, :], mask=omask, other=0.0)
        gconv = _rnd(go.to(tl.float32) * sig * (1.0 + conv * (1.0 - sig)), DT)
        tl.store(gconv_ptr + rows[:, None] * C + cols[None, :], gconv.to(DT), mask=omask)
        # Weight gradient partial for this token tile: sum_t gconv[t] * normed[t - shift_j].
        for j in tl.static_range(K):
            src = rows - ((K - 1 - j) * DIL) * B
            valid = (rmask & (src >= 0))[:, None] & cmask[None, :]
            _, normed = _normed_value(
                sv_ptr, gate_ptr, rv_ptr, src, valid, stream, hc, gamma, NS, H, DT
            )
            part = tl.sum(gconv * normed, axis=0)
            tl.store(dw_ptr + (pid_t.to(tl.int64) * K + j) * C + cols, part, mask=cmask)

    @triton.jit(do_not_specialize=["T"])
    def _cnorm_bwd_kernel(
        go_ptr,
        gconv_ptr,
        sv_ptr,
        gate_ptr,
        rv_ptr,
        gc_ptr,
        w_ptr,
        gval_ptr,
        ggate_ptr,
        dgc_ptr,
        T,
        B,
        NS: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        DIL: tl.constexpr,
        ZC: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        DT = gval_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        pid = tl.program_id(0)
        s = tl.program_id(1)
        offs = tl.arange(0, BLOCK_H)
        m = offs < H
        cols = s * H + offs
        gamma = tl.load(gc_ptr + cols, mask=m, other=0.0).to(tl.float32)
        if ZC:
            gamma = gamma + 1.0
        dgamma = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i in range(ROWS):
            row = pid.to(tl.int64) * ROWS + i
            live = row < T
            lm = m & live
            # Transposed depthwise convolution: the output at row + shift_j used this row as
            # its tap j.
            gnormed = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for j in tl.static_range(K):
                dst = row + ((K - 1 - j) * DIL) * B
                w = tl.load(w_ptr + cols * K + j, mask=m, other=0.0).to(tl.float32)
                g = tl.load(gconv_ptr + dst * C + cols, mask=lm & (dst < T), other=0.0)
                gnormed += w * g.to(tl.float32)
            gnormed = _rnd(gnormed, DT)
            sv = tl.load(sv_ptr + row * H + offs, mask=lm, other=0.0).to(tl.float32)
            gate = tl.load(gate_ptr + row * NS + s, mask=live, other=0.0)
            rv = tl.load(rv_ptr + row * NS + s, mask=live, other=0.0)
            value = _rnd(gate * sv, DT)
            ghat = gnormed * gamma
            dot = tl.sum(ghat * value, axis=0)
            gfrom_norm = _rnd(rv * ghat - (rv * rv * rv) * value * (dot / H), DT)
            go = tl.load(go_ptr + row * C + cols, mask=lm, other=0.0).to(tl.float32)
            gval = _rnd(go + gfrom_norm, DT)
            tl.store(gval_ptr + row * C + cols, gval.to(DT), mask=lm)
            prod = tl.where(m, _rnd(gval * sv, DT), 0.0)
            ggate = _rnd(tl.sum(prod, axis=0), DT)
            tl.store(ggate_ptr + row * NS + s, ggate, mask=live)
            dgamma += gnormed * (value * rv)
        tl.store(dgc_ptr + pid.to(tl.int64) * C + cols, dgamma, mask=m)

    @triton.jit(do_not_specialize=["T"])
    def _value_bwd_kernel(
        gval_ptr,
        gate_ptr,
        gsv_ptr,
        T,
        NS: tl.constexpr,
        H: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        DT = gsv_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        rows = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        hc = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows < T)[:, None] & (hc < H)[None, :]
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for s in tl.static_range(NS):
            gval = tl.load(gval_ptr + rows[:, None] * C + s * H + hc[None, :], mask=mask, other=0.0)
            gate = tl.load(gate_ptr + rows * NS + s, mask=rows < T, other=0.0)
            acc += _rnd(gval.to(tl.float32) * gate[:, None], DT)
        tl.store(gsv_ptr + rows[:, None] * H + hc[None, :], acc.to(DT), mask=mask)

    @triton.jit(do_not_specialize=["T"])
    def _kq_bwd_kernel(
        k_ptr,
        h_ptr,
        gk_ptr,
        gq_ptr,
        gate_ptr,
        score_ptr,
        ggate_ptr,
        rk_ptr,
        rq_ptr,
        dk_ptr,
        dh_ptr,
        dgk_ptr,
        dgq_ptr,
        T,
        sqrt_h,
        NS: tl.constexpr,
        H: tl.constexpr,
        ZC: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        DT = dk_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        pid = tl.program_id(0)
        s = tl.program_id(1)
        offs = tl.arange(0, BLOCK_H)
        m = offs < H
        cols = s * H + offs
        gk = tl.load(gk_ptr + cols, mask=m, other=0.0).to(tl.float32)
        gq = tl.load(gq_ptr + cols, mask=m, other=0.0).to(tl.float32)
        if ZC:
            gk = gk + 1.0
            gq = gq + 1.0
        dgk = tl.zeros((BLOCK_H,), dtype=tl.float32)
        dgq = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i in range(ROWS):
            row = pid.to(tl.int64) * ROWS + i
            live = row < T
            lm = m & live
            # Scalar chain: gate = sigmoid(sign(score) * sqrt(max(|score|, 1e-6))).
            gate = tl.load(gate_ptr + row * NS + s, mask=live, other=0.0)
            score = tl.load(score_ptr + row * NS + s, mask=live, other=0.0)
            ggate = tl.load(ggate_ptr + row * NS + s, mask=live, other=0.0)
            gsigned = _rnd(ggate * (1.0 - gate) * gate, DT)
            sign = tl.where(score > 0, 1.0, tl.where(score < 0, -1.0, 0.0))
            mag = tl.abs(score)
            root = _rnd(tl.sqrt_rn(_rnd(tl.maximum(mag, 1e-6), DT)), DT)
            groot = _rnd(gsigned * sign, DT)
            gmag = _rnd(tl.div_rn(groot, 2.0 * root), DT)
            gmag = tl.where(mag >= 1e-6, gmag, 0.0)
            gscore = _rnd(gmag * sign, DT)
            gsum = _rnd(tl.div_rn(gscore, sqrt_h), DT)

            rk = tl.load(rk_ptr + row * NS + s, mask=live, other=0.0)
            rq = tl.load(rq_ptr + row * NS + s, mask=live, other=0.0)
            k = tl.load(k_ptr + row * C + cols, mask=lm, other=0.0).to(tl.float32)
            q = tl.load(h_ptr + row * C + cols, mask=lm, other=0.0).to(tl.float32)
            khat = k * rk
            qhat = q * rq
            kn = _rnd(khat * gk, DT)
            qn = _rnd(qhat * gq, DT)
            gkn = _rnd(gsum * qn, DT)
            gqn = _rnd(gsum * kn, DT)
            ghk = gkn * gk
            ghq = gqn * gq
            dotk = tl.sum(ghk * k, axis=0)
            dotq = tl.sum(ghq * q, axis=0)
            dk = rk * ghk - (rk * rk * rk) * k * (dotk / H)
            dh = rq * ghq - (rq * rq * rq) * q * (dotq / H)
            tl.store(dk_ptr + row * C + cols, dk.to(DT), mask=lm)
            tl.store(dh_ptr + row * C + cols, dh.to(DT), mask=lm)
            dgk += gkn * khat
            dgq += gqn * qhat
        tl.store(dgk_ptr + pid.to(tl.int64) * C + cols, dgk, mask=m)
        tl.store(dgq_ptr + pid.to(tl.int64) * C + cols, dgq, mask=m)


def _num_warps(block: int) -> int:
    return 8 if block >= 2048 else 4


class FusedEngramDense(torch.autograd.Function):
    """Engram gate, conv-norm, causal depthwise convolution, SiLU, and residual in Triton.

    Inputs are ``[S, B, ...]`` tensors in the activation dtype: ``key`` is the key
    projection output and ``hidden`` the Engram input (both ``[S, B, NS * H]``), ``value``
    the shared value projection (``[S, B, H]``). Returns ``value * gate + conv`` with the
    reference path's layout ``[S, B, NS * H]``.
    """

    @staticmethod
    def forward(
        ctx,
        key: Tensor,
        hidden: Tensor,
        value: Tensor,
        key_gamma: Tensor,
        query_gamma: Tensor,
        conv_gamma: Tensor,
        conv_weight: Tensor,
        num_streams: int,
        eps: float,
        zero_centered: bool,
        dilation: int,
    ) -> Tensor:
        seq, batch = hidden.shape[0], hidden.shape[1]
        hidden_size = value.shape[-1]
        channels = num_streams * hidden_size
        tokens = seq * batch
        kernel_size = conv_weight.shape[-1]
        key = key.contiguous()
        hidden = hidden.contiguous()
        value = value.contiguous()
        conv_weight = conv_weight.contiguous()
        stats = torch.empty((5, tokens, num_streams), device=hidden.device, dtype=torch.float32)
        gate, score, rk, rq, rv = stats.unbind(0)
        block_h = triton.next_power_of_2(hidden_size)
        sqrt_h = float(hidden_size) ** 0.5
        if tokens > 0:
            _gate_fwd_kernel[(tokens,)](
                key,
                hidden,
                value,
                key_gamma,
                query_gamma,
                gate,
                score,
                rk,
                rq,
                rv,
                tokens,
                eps,
                sqrt_h,
                NS=num_streams,
                H=hidden_size,
                BLOCK_H=block_h,
                ZC=zero_centered,
                num_warps=_num_warps(block_h),
            )
        output = torch.empty_like(hidden)
        grid = (triton.cdiv(tokens, _CONV_BLOCK_T), triton.cdiv(channels, _CONV_BLOCK_C))
        if tokens > 0:
            _conv_fwd_kernel[grid](
                value,
                gate,
                rv,
                conv_gamma,
                conv_weight,
                output,
                tokens,
                batch,
                NS=num_streams,
                H=hidden_size,
                K=kernel_size,
                DIL=dilation,
                ZC=zero_centered,
                BLOCK_T=_CONV_BLOCK_T,
                BLOCK_C=_CONV_BLOCK_C,
                num_warps=4,
            )
        ctx.save_for_backward(
            key, hidden, value, key_gamma, query_gamma, conv_gamma, conv_weight, stats
        )
        ctx.config = (num_streams, zero_centered, dilation, seq, batch)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (key, hidden, value, key_gamma, query_gamma, conv_gamma, conv_weight, stats) = (
            ctx.saved_tensors
        )
        num_streams, zero_centered, dilation, seq, batch = ctx.config
        gate, score, rk, rq, rv = stats.unbind(0)
        hidden_size = value.shape[-1]
        channels = num_streams * hidden_size
        tokens = seq * batch
        kernel_size = conv_weight.shape[-1]
        grad_output = grad_output.contiguous()
        device, dtype = hidden.device, hidden.dtype
        block_h = triton.next_power_of_2(hidden_size)
        sqrt_h = float(hidden_size) ** 0.5

        conv_tiles = triton.cdiv(tokens, _CONV_BLOCK_T)
        row_blocks = triton.cdiv(tokens, _ROW_BLOCK)
        grad_conv = torch.empty_like(grad_output)
        # Every element of the partial buffers is written by exactly one program.
        dw_partial = torch.empty(
            (conv_tiles, kernel_size, channels), device=device, dtype=torch.float32
        )
        grad_value_streams = torch.empty_like(grad_output)
        grad_gate = torch.empty((tokens, num_streams), device=device, dtype=torch.float32)
        gamma_partial = torch.empty((3, row_blocks, channels), device=device, dtype=torch.float32)
        dgc_partial, dgk_partial, dgq_partial = gamma_partial.unbind(0)
        grad_value = torch.empty_like(value)
        grad_key = torch.empty_like(key)
        grad_hidden = torch.empty_like(hidden)

        if tokens > 0:
            _conv_bwd_kernel[(conv_tiles, triton.cdiv(channels, _CONV_BLOCK_C))](
                grad_output,
                value,
                gate,
                rv,
                conv_gamma,
                conv_weight,
                grad_conv,
                dw_partial,
                tokens,
                batch,
                NS=num_streams,
                H=hidden_size,
                K=kernel_size,
                DIL=dilation,
                ZC=zero_centered,
                BLOCK_T=_CONV_BLOCK_T,
                BLOCK_C=_CONV_BLOCK_C,
                num_warps=4,
            )
            _cnorm_bwd_kernel[(row_blocks, num_streams)](
                grad_output,
                grad_conv,
                value,
                gate,
                rv,
                conv_gamma,
                conv_weight,
                grad_value_streams,
                grad_gate,
                dgc_partial,
                tokens,
                batch,
                NS=num_streams,
                H=hidden_size,
                K=kernel_size,
                DIL=dilation,
                ZC=zero_centered,
                ROWS=_ROW_BLOCK,
                BLOCK_H=block_h,
                num_warps=_num_warps(block_h),
            )
            _value_bwd_kernel[
                (triton.cdiv(tokens, _CONV_BLOCK_T), triton.cdiv(hidden_size, _CONV_BLOCK_C))
            ](
                grad_value_streams,
                gate,
                grad_value,
                tokens,
                NS=num_streams,
                H=hidden_size,
                BLOCK_T=_CONV_BLOCK_T,
                BLOCK_C=_CONV_BLOCK_C,
                num_warps=4,
            )
            _kq_bwd_kernel[(row_blocks, num_streams)](
                key,
                hidden,
                key_gamma,
                query_gamma,
                gate,
                score,
                grad_gate,
                rk,
                rq,
                grad_key,
                grad_hidden,
                dgk_partial,
                dgq_partial,
                tokens,
                sqrt_h,
                NS=num_streams,
                H=hidden_size,
                ZC=zero_centered,
                ROWS=_ROW_BLOCK,
                BLOCK_H=block_h,
                num_warps=_num_warps(block_h),
            )

        grad_conv_weight = (
            dw_partial.sum(0).t().reshape(conv_weight.shape).to(conv_weight.dtype)
        )
        gamma_grads = gamma_partial.sum(1)
        return (
            grad_key,
            grad_hidden,
            grad_value,
            gamma_grads[1].to(key_gamma.dtype),
            gamma_grads[2].to(query_gamma.dtype),
            gamma_grads[0].to(conv_gamma.dtype),
            grad_conv_weight,
            None,
            None,
            None,
            None,
        )


def fused_engram_dense(
    key: Tensor,
    hidden: Tensor,
    value: Tensor,
    key_gamma: Tensor,
    query_gamma: Tensor,
    conv_gamma: Tensor,
    conv_weight: Tensor,
    num_streams: int,
    eps: float,
    zero_centered: bool,
    dilation: int,
) -> Tensor:
    """Functional entry point for :class:`FusedEngramDense`."""
    return FusedEngramDense.apply(
        key,
        hidden,
        value,
        key_gamma,
        query_gamma,
        conv_gamma,
        conv_weight,
        num_streams,
        eps,
        zero_centered,
        dilation,
    )
