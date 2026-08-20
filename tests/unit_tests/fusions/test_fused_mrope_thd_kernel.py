# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the tiled THD mRoPE Triton kernel.

The kernel maps one program to a ``BLOCK_T`` x ``BLOCK_H`` token/head tile and
walks the heads inside the program, and it resolves the packed sub-sequence
owning a token by bisecting ``cu_seqlens`` instead of scanning it. Neither
change touches the arithmetic, so two properties are checked here:

* the tiled kernel is **bit identical** to the pre-optimization kernel, a
  verbatim copy of which is kept below so the comparison stays meaningful after
  the production kernel is edited again;
* forward and backward still match an unfused PyTorch mRoPE, including under
  context parallelism.

The context-parallel case additionally runs on two ranks. Launch it with::

    torchrun --nproc_per_node=2 -m pytest \
        tests/unit_tests/fusions/test_fused_mrope_thd_kernel.py
"""

import os

import pytest
import torch

from megatron.core.fusions.fused_mrope import (
    HAVE_TRITON,
    _launch_fused_mrope_thd,
    _mrope_axis,
    _smallest_power_of_2_at_least,
    _validate_mrope_thd_inputs,
    fused_apply_mrope_thd,
    mrope_freqs_to_rotary_emb,
)

if HAVE_TRITON:
    import triton
    import triton.language as tl


pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()),
    reason="Triton fused THD mRoPE requires Triton and a CUDA device",
)


if HAVE_TRITON:

    @triton.jit
    def _legacy_fused_mrope_thd_kernel(
        T,
        CU_SEQLENS,
        FREQS,
        OUT,
        t_s_token,
        t_s_head,
        t_s_dim,
        cu_s_idx,
        f_s_axis,
        f_s_seq,
        f_s_dim,
        o_s_token,
        o_s_head,
        o_s_dim,
        NUM_SEQS,
        HALF_ROTARY_DIM: tl.constexpr,
        PASS_DIM: tl.constexpr,
        SEC_T: tl.constexpr,
        SEC_H: tl.constexpr,
        SEC_W: tl.constexpr,
        INTERLEAVED_MROPE: tl.constexpr,
        ROTARY_INTERLEAVED: tl.constexpr,
        INVERSE: tl.constexpr,
        CP_SIZE: tl.constexpr,
        CP_RANK: tl.constexpr,
        FP32_COMPUTE: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
        BLOCK_PASS: tl.constexpr,
    ):
        """Verbatim copy of the pre-optimization THD kernel, kept as reference."""
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        freq_seq_idx = token_idx
        seq_i = 0
        while seq_i < NUM_SEQS:
            global_start = tl.load(CU_SEQLENS + seq_i * cu_s_idx)
            global_end = tl.load(CU_SEQLENS + (seq_i + 1) * cu_s_idx)
            local_start = global_start // CP_SIZE
            local_end = global_end // CP_SIZE
            in_seq = (token_idx >= local_start) & (token_idx < local_end)
            local_offset = token_idx - local_start

            if CP_SIZE > 1:
                local_seq_len = local_end - local_start
                first_cp_seg = (local_seq_len + 1) // 2
                second_cp_seg = local_seq_len // 2
                first_freq_idx = global_start + CP_RANK * first_cp_seg + local_offset
                second_freq_idx = (
                    global_end - (CP_RANK + 1) * second_cp_seg + (local_offset - first_cp_seg)
                )
                seq_freq_idx = tl.where(
                    local_offset < first_cp_seg, first_freq_idx, second_freq_idx
                )
            else:
                seq_freq_idx = global_start + local_offset

            freq_seq_idx = tl.where(in_seq, seq_freq_idx, freq_seq_idx)
            seq_i += 1

        k = tl.arange(0, BLOCK_HALF)
        mask = k < HALF_ROTARY_DIM
        axis = _mrope_axis(k, SEC_T, SEC_H, SEC_W, INTERLEAVED_MROPE)

        compute_ty = tl.float32 if FP32_COMPUTE else OUT.dtype.element_ty

        freqs_offset = axis * f_s_axis + freq_seq_idx * f_s_seq + k * f_s_dim
        freqs = tl.load(FREQS + freqs_offset, mask=mask, other=0.0)
        cos_v = tl.cos(freqs).to(compute_ty)
        sin_v = tl.sin(freqs).to(compute_ty)
        if INVERSE:
            sin_v = -sin_v

        t_base = T + token_idx * t_s_token + head_idx * t_s_head
        out_base = OUT + token_idx * o_s_token + head_idx * o_s_head

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


def _legacy_launch_fused_mrope_thd(
    t,
    cu_seqlens,
    freqs,
    launch_metadata,
    interleaved_mrope,
    inverse,
    cp_size,
    cp_rank,
    fp32_compute,
):
    """Verbatim copy of the pre-optimization THD launcher."""
    tokens, heads, head_dim, half_rotary_dim, sec_t, sec_h, sec_w = launch_metadata
    out = torch.empty_like(t)

    block_half = _smallest_power_of_2_at_least(half_rotary_dim)
    pass_dim = head_dim - (2 * half_rotary_dim)
    block_pass = _smallest_power_of_2_at_least(max(pass_dim, 1))
    num_seqs = cu_seqlens.numel() - 1

    _legacy_fused_mrope_thd_kernel[(tokens, heads)](
        t,
        cu_seqlens,
        freqs,
        out,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        cu_seqlens.stride(0),
        freqs.stride(0),
        freqs.stride(2),
        freqs.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_seqs,
        HALF_ROTARY_DIM=half_rotary_dim,
        PASS_DIM=pass_dim,
        SEC_T=sec_t,
        SEC_H=sec_h,
        SEC_W=sec_w,
        INTERLEAVED_MROPE=interleaved_mrope,
        ROTARY_INTERLEAVED=False,
        INVERSE=inverse,
        CP_SIZE=cp_size,
        CP_RANK=cp_rank,
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


def _build_case(
    seqlens,
    heads,
    head_dim,
    half_rotary_dim,
    dtype,
    interleaved_mrope,
    cp_size=1,
    padding_tokens=0,
    seed=0,
):
    """Build a packed THD mRoPE case.

    ``padding_tokens`` appends tokens past the end of ``cu_seqlens`` so the
    "token belongs to no sub-sequence" fallback is exercised.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cu = [0]
    for length in seqlens:
        cu.append(cu[-1] + length)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")
    tokens = cu[-1] // cp_size + padding_tokens

    t = torch.randn(
        tokens, heads, head_dim, dtype=torch.float32, device="cuda", generator=generator
    ).to(dtype)
    freqs = torch.randn(
        3,
        1,
        tokens * cp_size,
        half_rotary_dim,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    return t, cu_seqlens, freqs, _mrope_section_for(half_rotary_dim, interleaved_mrope)


def _run_legacy_and_tiled(
    t, cu_seqlens, freqs, mrope_section, interleaved_mrope, cp_size, cp_rank, inverse, fp32_compute
):
    metadata = _validate_mrope_thd_inputs(
        t, cu_seqlens, freqs, mrope_section, interleaved_mrope, cp_size
    )
    legacy = _legacy_launch_fused_mrope_thd(
        t, cu_seqlens, freqs, metadata, interleaved_mrope, inverse, cp_size, cp_rank, fp32_compute
    )
    tiled = _launch_fused_mrope_thd(
        t,
        cu_seqlens,
        freqs,
        metadata,
        interleaved_mrope,
        False,
        inverse,
        cp_size=cp_size,
        cp_rank=cp_rank,
        fp32_compute=fp32_compute,
    )
    return legacy, tiled


# Vision-encoder shape (interleaved mRoPE, fp32 compute) and language shape
# (section mRoPE, non-zero pass-through dim) both occur in the Qwen3.5-VL path.
_SHAPES = [
    pytest.param(16, 128, 36, True, True, id="vision-h16-d128-half36-interleaved"),
    pytest.param(8, 256, 32, False, False, id="language-h8-d256-half32-section"),
    pytest.param(1, 64, 32, False, False, id="single-head-no-pass-dim"),
]


@pytest.mark.parametrize("heads,head_dim,half_rotary_dim,interleaved_mrope,fp32_compute", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inverse", [False, True])
def test_thd_kernel_matches_legacy_bitwise(
    heads, head_dim, half_rotary_dim, interleaved_mrope, fp32_compute, dtype, inverse
):
    """The tile restructure is a pure scheduling change, so outputs are identical."""
    # Sequence lengths deliberately avoid powers of two and include a length-1 run.
    case = _build_case(
        [37, 1, 512, 91, 200, 13, 777], heads, head_dim, half_rotary_dim, dtype, interleaved_mrope
    )
    legacy, tiled = _run_legacy_and_tiled(
        *case,
        interleaved_mrope=interleaved_mrope,
        cp_size=1,
        cp_rank=0,
        inverse=inverse,
        fp32_compute=fp32_compute,
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


@pytest.mark.parametrize("num_seqs", [1, 2, 3, 7, 8, 9, 31, 32, 33, 64, 65, 127])
def test_thd_kernel_bisection_matches_linear_scan(num_seqs):
    """The bisection must resolve the same sub-sequence as the old linear scan.

    ``num_seqs`` is swept across powers of two and their neighbours because the
    bisection depth is ``ceil(log2(num_seqs + 1))``; off-by-one errors in the
    depth show up exactly at those boundaries.
    """
    generator = torch.Generator().manual_seed(num_seqs)
    seqlens = torch.randint(1, 40, (num_seqs,), generator=generator).tolist()
    case = _build_case(seqlens, 4, 64, 16, torch.bfloat16, False, seed=num_seqs)
    legacy, tiled = _run_legacy_and_tiled(
        *case, interleaved_mrope=False, cp_size=1, cp_rank=0, inverse=False, fp32_compute=False
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


def test_thd_kernel_handles_empty_subsequences():
    """Zero-length entries in ``cu_seqlens`` create duplicate split points.

    A bisection that stops at the first match lands on an empty entry; the
    kernel must still select the sub-sequence that contains the token.
    """
    case = _build_case([10, 0, 0, 25, 0, 7], 4, 64, 16, torch.bfloat16, False)
    legacy, tiled = _run_legacy_and_tiled(
        *case, interleaved_mrope=False, cp_size=1, cp_rank=0, inverse=False, fp32_compute=False
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


@pytest.mark.parametrize("padding_tokens", [1, 5, 63])
def test_thd_kernel_handles_trailing_padding_tokens(padding_tokens):
    """Tokens past the last ``cu_seqlens`` entry keep their own frequency row."""
    case = _build_case(
        [64, 33, 128], 4, 64, 16, torch.bfloat16, False, padding_tokens=padding_tokens
    )
    legacy, tiled = _run_legacy_and_tiled(
        *case, interleaved_mrope=False, cp_size=1, cp_rank=0, inverse=False, fp32_compute=False
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


@pytest.mark.parametrize("cp_size,cp_rank", [(2, 0), (2, 1), (4, 0), (4, 2), (4, 3)])
def test_thd_kernel_matches_legacy_under_context_parallel(cp_size, cp_rank):
    """The context-parallel token mapping is unchanged by the restructure."""
    seqlens = [length * cp_size for length in (9, 3, 20, 1)]
    case = _build_case(
        seqlens, 8, 128, 36, torch.bfloat16, True, cp_size=cp_size, seed=cp_size * 10 + cp_rank
    )
    legacy, tiled = _run_legacy_and_tiled(
        *case,
        interleaved_mrope=True,
        cp_size=cp_size,
        cp_rank=cp_rank,
        inverse=False,
        fp32_compute=True,
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


@pytest.mark.parametrize("tokens_tail", [1, 3, 5, 11])
def test_thd_kernel_token_count_not_multiple_of_tile(tokens_tail):
    """Token counts that do not divide the tile height must still be exact."""
    case = _build_case([64, tokens_tail], 4, 64, 16, torch.bfloat16, False, seed=tokens_tail)
    legacy, tiled = _run_legacy_and_tiled(
        *case, interleaved_mrope=False, cp_size=1, cp_rank=0, inverse=False, fp32_compute=False
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


def test_thd_kernel_handles_zero_tokens():
    """An empty packed batch must not launch a kernel or read out of bounds."""
    t = torch.empty(0, 4, 64, dtype=torch.bfloat16, device="cuda")
    cu_seqlens = torch.zeros(1, dtype=torch.int32, device="cuda")
    freqs = torch.empty(3, 1, 0, 16, dtype=torch.float32, device="cuda")
    out = fused_apply_mrope_thd(t, cu_seqlens, freqs, _mrope_section_for(16, False))
    assert out.shape == t.shape


@pytest.mark.parametrize("heads", [1, 3, 5, 6, 12])
def test_thd_kernel_head_counts_that_are_not_powers_of_two(heads):
    """The head group must divide the head count, or trailing heads go unwritten.

    Only power-of-two group sizes are considered, so an odd head count has to
    fall back to keeping every head in one program.
    """
    case = _build_case([96, 40], heads, 64, 16, torch.bfloat16, False, seed=heads)
    legacy, tiled = _run_legacy_and_tiled(
        *case, interleaved_mrope=False, cp_size=1, cp_rank=0, inverse=False, fp32_compute=False
    )
    torch.testing.assert_close(tiled, legacy, rtol=0, atol=0)


def _unfused_mrope_thd(t, freqs, mrope_section, interleaved_mrope):
    """Reference mRoPE over a fully packed THD tensor, in fp32.

    ``t`` covers exactly the tokens described by ``freqs``, so token ``i`` uses
    frequency row ``i``. This mirrors the fused kernel's contract for the
    non-context-parallel case with no trailing padding.
    """
    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, interleaved_mrope)
    # [seq, batch=1, 1, rotary_dim] -> [seq, 1, rotary_dim]
    emb = emb.squeeze(1)
    rotary_dim = emb.shape[-1]

    rot = t[..., :rotary_dim].float()
    lo, hi = rot.chunk(2, dim=-1)
    rotated = torch.cat((-hi, lo), dim=-1)
    out = rot * emb.cos() + rotated * emb.sin()
    return torch.cat((out.to(t.dtype), t[..., rotary_dim:]), dim=-1)


def _tolerances(dtype):
    """Tolerances used by the other MCore fused mRoPE parity tests."""
    if dtype == torch.bfloat16:
        return dict(rtol=2.0e-2, atol=5.0e-2)
    return dict(rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_thd_kernel_matches_unfused_forward_backward(dtype, interleaved_mrope):
    """Forward and backward match an unfused PyTorch mRoPE reference."""
    t, cu_seqlens, freqs, mrope_section = _build_case(
        [48, 17, 96], 8, 128, 36, dtype, interleaved_mrope, seed=7
    )
    t_fused = t.detach().clone().requires_grad_(True)
    t_ref = t.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(t)

    fused = fused_apply_mrope_thd(
        t_fused,
        cu_seqlens,
        freqs,
        mrope_section,
        interleaved_mrope=interleaved_mrope,
        fp32_compute=True,
    )
    reference = _unfused_mrope_thd(t_ref, freqs, mrope_section, interleaved_mrope)

    fused.backward(grad_out)
    reference.backward(grad_out)

    tols = _tolerances(dtype)
    torch.testing.assert_close(fused, reference, **tols)
    torch.testing.assert_close(t_fused.grad, t_ref.grad, **tols)


def _context_parallel_shard(tensor, cu_seqlens_cpu, cp_size, cp_rank):
    """Split a globally packed tensor the way THD context parallelism does.

    Each sub-sequence is cut into ``2 * cp_size`` equal chunks; rank ``r`` owns
    chunk ``r`` and chunk ``2 * cp_size - 1 - r`` so the causal load stays
    balanced.
    """
    chunks = []
    for start, end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
        chunk = (end - start) // (2 * cp_size)
        for index in (cp_rank, 2 * cp_size - 1 - cp_rank):
            chunks.append(tensor[start + index * chunk : start + (index + 1) * chunk])
    return torch.cat(chunks, dim=0)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="context-parallel THD mRoPE check needs torchrun --nproc_per_node=2",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_thd_kernel_context_parallel_two_ranks(dtype):
    """Each CP rank rotates its shard exactly as the unsharded reference does.

    Both ranks build the same global packed batch, then each one runs the fused
    kernel on its own shard with its own ``cp_rank``. The result must equal the
    corresponding shard of the unfused reference computed over the whole batch,
    for the forward pass and for the gradient.
    """
    from megatron.core import parallel_state
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank = torch.distributed.get_rank(cp_group)

        # Sub-sequence lengths are multiples of 2 * cp_size so the balanced split
        # is exact, which is what the packed multimodal data loader guarantees.
        seqlens = [64, 32, 96]
        heads, head_dim, half_rotary_dim = 8, 128, 36
        interleaved_mrope = True
        t_global, cu_seqlens, freqs, mrope_section = _build_case(
            seqlens, heads, head_dim, half_rotary_dim, dtype, interleaved_mrope, seed=99
        )
        cu_seqlens_cpu = cu_seqlens.tolist()
        grad_global = torch.randn(
            t_global.shape,
            dtype=torch.float32,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(1234),
        ).to(dtype)

        t_ref = t_global.detach().clone().requires_grad_(True)
        reference = _unfused_mrope_thd(t_ref, freqs, mrope_section, interleaved_mrope)
        reference.backward(grad_global)

        t_local = (
            _context_parallel_shard(t_global, cu_seqlens_cpu, cp_size, cp_rank)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        local = fused_apply_mrope_thd(
            t_local,
            cu_seqlens,
            freqs,
            mrope_section,
            interleaved_mrope=interleaved_mrope,
            cp_size=cp_size,
            cp_rank=cp_rank,
            fp32_compute=True,
        )
        local.backward(_context_parallel_shard(grad_global, cu_seqlens_cpu, cp_size, cp_rank))

        tols = _tolerances(dtype)
        torch.testing.assert_close(
            local, _context_parallel_shard(reference, cu_seqlens_cpu, cp_size, cp_rank), **tols
        )
        torch.testing.assert_close(
            t_local.grad,
            _context_parallel_shard(t_ref.grad, cu_seqlens_cpu, cp_size, cp_rank),
            **tols,
        )
        # Fail the whole job, not just one rank, if either shard mismatched.
        torch.distributed.barrier(group=cp_group)
    finally:
        Utils.destroy_model_parallel()
