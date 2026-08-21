# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the tiled SBHD (non-THD) mRoPE Triton kernel.

The kernel maps one program to a ``BLOCK_T`` x ``BLOCK_H`` position/head tile of
one batch element and walks the heads inside the program, instead of giving
every ``(position, batch, head)`` triple its own program moving a single
``head_dim`` row. The arithmetic is untouched, so the property checked here is
that the tiled kernel is **bit identical** to the pre-optimization kernel, a
verbatim copy of which is kept below so the comparison stays meaningful after
the production kernel is edited again.

Forward/backward parity against an unfused PyTorch mRoPE already lives in
``test_fused_mrope.py``; this file only guards the scheduling change.
``rotary_interleaved=True`` is not covered: the launcher rejects it before the
kernel is reached, so that ``tl.constexpr`` branch is unreachable in
production.
"""

import pytest
import torch

from megatron.core.fusions.fused_mrope import (
    HAVE_TRITON,
    _launch_fused_mrope,
    _mrope_axis,
    _smallest_power_of_2_at_least,
    _validate_mrope_inputs,
)

if HAVE_TRITON:
    import triton
    import triton.language as tl


pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()),
    reason="Triton fused mRoPE requires Triton and a CUDA device",
)


if HAVE_TRITON:

    @triton.jit
    def _legacy_fused_mrope_kernel(
        T,
        FREQS,
        OUT,
        t_s_seq,
        t_s_batch,
        t_s_head,
        t_s_dim,
        f_s_axis,
        f_s_batch,
        f_s_seq,
        f_s_dim,
        o_s_seq,
        o_s_batch,
        o_s_head,
        o_s_dim,
        HALF_ROTARY_DIM: tl.constexpr,
        PASS_DIM: tl.constexpr,
        SEC_T: tl.constexpr,
        SEC_H: tl.constexpr,
        SEC_W: tl.constexpr,
        INTERLEAVED_MROPE: tl.constexpr,
        ROTARY_INTERLEAVED: tl.constexpr,
        INVERSE: tl.constexpr,
        FP32_COMPUTE: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        BLOCK_PASS: tl.constexpr,
    ):
        """Verbatim copy of the pre-optimization SBHD kernel, kept as reference."""
        seq_idx = tl.program_id(0)
        batch_idx = tl.program_id(1)
        head_idx = tl.program_id(2)

        k = tl.arange(0, BLOCK_HALF)
        mask = k < HALF_ROTARY_DIM

        axis = _mrope_axis(k, SEC_T, SEC_H, SEC_W, INTERLEAVED_MROPE)

        compute_ty = tl.float32 if FP32_COMPUTE else OUT.dtype.element_ty

        freqs_offset = axis * f_s_axis + batch_idx * f_s_batch + seq_idx * f_s_seq + k * f_s_dim
        freqs = tl.load(FREQS + freqs_offset, mask=mask, other=0.0)
        cos_v = tl.cos(freqs).to(compute_ty)
        sin_v = tl.sin(freqs).to(compute_ty)
        if INVERSE:
            sin_v = -sin_v

        t_base = T + seq_idx * t_s_seq + batch_idx * t_s_batch + head_idx * t_s_head
        out_base = OUT + seq_idx * o_s_seq + batch_idx * o_s_batch + head_idx * o_s_head

        if ROTARY_INTERLEAVED:
            lo_offset = (2 * k) * t_s_dim
            hi_offset = (2 * k + 1) * t_s_dim
            out_lo_offset = (2 * k) * o_s_dim
            out_hi_offset = (2 * k + 1) * o_s_dim
        else:
            lo_offset = k * t_s_dim
            hi_offset = (k + HALF_ROTARY_DIM) * t_s_dim
            out_lo_offset = k * o_s_dim
            out_hi_offset = (k + HALF_ROTARY_DIM) * o_s_dim

        t_lo = tl.load(t_base + lo_offset, mask=mask, other=0.0).to(compute_ty)
        t_hi = tl.load(t_base + hi_offset, mask=mask, other=0.0).to(compute_ty)

        lo_cos = (t_lo * cos_v).to(compute_ty)
        hi_sin = (t_hi * sin_v).to(compute_ty)
        hi_cos = (t_hi * cos_v).to(compute_ty)
        lo_sin = (t_lo * sin_v).to(compute_ty)

        out_lo = (lo_cos - hi_sin).to(OUT.dtype.element_ty)
        out_hi = (hi_cos + lo_sin).to(OUT.dtype.element_ty)

        tl.store(out_base + out_lo_offset, out_lo, mask=mask)
        tl.store(out_base + out_hi_offset, out_hi, mask=mask)

        if PASS_DIM > 0:
            pass_idx = tl.arange(0, BLOCK_PASS)
            pass_mask = pass_idx < PASS_DIM
            src_dim = 2 * HALF_ROTARY_DIM + pass_idx
            pass_values = tl.load(t_base + src_dim * t_s_dim, mask=pass_mask, other=0.0)
            tl.store(out_base + src_dim * o_s_dim, pass_values, mask=pass_mask)


def _legacy_launch_fused_mrope(
    t, freqs, launch_metadata, interleaved_mrope, rotary_interleaved, inverse, fp32_compute
):
    """Verbatim copy of the pre-optimization SBHD launcher."""
    seq, batch, heads, head_dim, half_rotary_dim, sec_t, sec_h, sec_w = launch_metadata
    out = torch.empty_like(t)

    block_half = _smallest_power_of_2_at_least(half_rotary_dim)
    pass_dim = head_dim - (2 * half_rotary_dim)
    block_pass = _smallest_power_of_2_at_least(max(pass_dim, 1))

    _legacy_fused_mrope_kernel[(seq, batch, heads)](
        t,
        freqs,
        out,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        t.stride(3),
        freqs.stride(0),
        freqs.stride(1),
        freqs.stride(2),
        freqs.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        HALF_ROTARY_DIM=half_rotary_dim,
        PASS_DIM=pass_dim,
        SEC_T=sec_t,
        SEC_H=sec_h,
        SEC_W=sec_w,
        INTERLEAVED_MROPE=interleaved_mrope,
        ROTARY_INTERLEAVED=rotary_interleaved,
        INVERSE=inverse,
        FP32_COMPUTE=fp32_compute,
        BLOCK_HALF=block_half,
        BLOCK_PASS=block_pass,
        num_warps=4,
    )
    return out


def _mrope_section_for(half_rotary_dim, interleaved_mrope):
    if interleaved_mrope:
        return [(half_rotary_dim + 2) // 3, (half_rotary_dim + 1) // 3, half_rotary_dim // 3]
    sec_t = half_rotary_dim // 2
    sec_h = half_rotary_dim // 4
    return [sec_t, sec_h, half_rotary_dim - sec_t - sec_h]


def _build_case(seq, batch, heads, head_dim, half_rotary_dim, dtype, interleaved_mrope, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t = torch.randn(
        seq, batch, heads, head_dim, dtype=torch.float32, device="cuda", generator=generator
    ).to(dtype)
    freqs = torch.randn(
        3, batch, seq, half_rotary_dim, dtype=torch.float32, device="cuda", generator=generator
    )
    return t, freqs, _mrope_section_for(half_rotary_dim, interleaved_mrope)


def _run_legacy_and_tiled(
    t, freqs, mrope_section, interleaved_mrope, rotary_interleaved, inverse, fp32_compute
):
    metadata = _validate_mrope_inputs(t, freqs, mrope_section, interleaved_mrope)
    legacy = _legacy_launch_fused_mrope(
        t, freqs, metadata, interleaved_mrope, rotary_interleaved, inverse, fp32_compute
    )
    tiled = _launch_fused_mrope(
        t,
        freqs,
        mrope_section,
        interleaved_mrope,
        rotary_interleaved,
        inverse,
        fp32_compute=fp32_compute,
    )
    return legacy, tiled


# The vision-encoder shape (interleaved mRoPE, fp32 compute) and the language
# shape (section mRoPE, non-zero pass-through dim) both reach this kernel from
# the generic RoPE dispatch in rope_utils.py.
_SHAPES = [
    pytest.param(16, 128, 36, True, True, id="vision-h16-d128-half36-interleaved"),
    pytest.param(8, 256, 32, False, False, id="language-h8-d256-half32-section"),
    pytest.param(1, 64, 32, False, False, id="single-head-no-pass-dim"),
]


@pytest.mark.parametrize("heads,head_dim,half_rotary_dim,interleaved_mrope,fp32_compute", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inverse", [False, True])
def test_sbhd_kernel_matches_legacy_bitwise(
    heads, head_dim, half_rotary_dim, interleaved_mrope, fp32_compute, dtype, inverse
):
    """The tile restructure is a pure scheduling change, so outputs are identical."""
    case = _build_case(1024, 2, heads, head_dim, half_rotary_dim, dtype, interleaved_mrope)
    legacy, tiled = _run_legacy_and_tiled(
        *case,
        interleaved_mrope=interleaved_mrope,
        rotary_interleaved=False,
        inverse=inverse,
        fp32_compute=fp32_compute,
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


@pytest.mark.parametrize("seq", [1, 3, 17, 31, 33, 255, 4097])
def test_sbhd_kernel_sequence_lengths_not_multiple_of_tile(seq):
    """Sequence lengths that do not divide the tile height must still be exact."""
    case = _build_case(seq, 2, 4, 64, 16, torch.bfloat16, False, seed=seq)
    legacy, tiled = _run_legacy_and_tiled(
        *case,
        interleaved_mrope=False,
        rotary_interleaved=False,
        inverse=False,
        fp32_compute=False,
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [1, 3, 5, 6, 12])
def test_sbhd_kernel_head_counts_that_are_not_powers_of_two(heads):
    """The head group must divide the head count, or trailing heads go unwritten."""
    case = _build_case(512, 2, heads, 64, 16, torch.bfloat16, False, seed=heads)
    legacy, tiled = _run_legacy_and_tiled(
        *case,
        interleaved_mrope=False,
        rotary_interleaved=False,
        inverse=False,
        fp32_compute=False,
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2, 5, 8])
def test_sbhd_kernel_batch_sizes(batch):
    """The batch is its own grid dimension; every element must be written."""
    case = _build_case(300, batch, 8, 128, 36, torch.bfloat16, True, seed=batch)
    legacy, tiled = _run_legacy_and_tiled(
        *case,
        interleaved_mrope=True,
        rotary_interleaved=False,
        inverse=False,
        fp32_compute=True,
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


def test_sbhd_kernel_handles_zero_sequence():
    """An empty batch must not launch a kernel or read out of bounds."""
    t = torch.empty(0, 2, 4, 64, dtype=torch.bfloat16, device="cuda")
    freqs = torch.empty(3, 2, 0, 16, dtype=torch.float32, device="cuda")
    out = _launch_fused_mrope(t, freqs, _mrope_section_for(16, False), False, False, False)
    assert out.shape == t.shape
