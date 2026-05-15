# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_mrope import (
    fused_apply_mrope,
    is_fused_mrope_available,
    mrope_freqs_to_rotary_emb,
)
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.transformer.transformer_config import TransformerConfig


class FakeCPGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


def _dtype_tols(dtype):
    if dtype == torch.bfloat16:
        return dict(rtol=2.0e-2, atol=5.0e-2)
    if dtype == torch.float16:
        return dict(rtol=3.0e-3, atol=1.0e-2)
    return dict(rtol=1.0e-6, atol=1.0e-6)


def _make_inputs(dtype=torch.bfloat16, requires_grad=False):
    seq = 32
    batch = 2
    heads = 3
    head_dim = 20
    rotary_dim = 16
    mrope_section = [2, 3, 3]

    generator = torch.Generator(device="cuda").manual_seed(1234)
    t = torch.randn(
        seq,
        batch,
        heads,
        head_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
        requires_grad=requires_grad,
    )
    freqs = torch.randn(
        3, batch, seq, rotary_dim // 2, dtype=torch.float32, device="cuda", generator=generator
    )
    return t, freqs, mrope_section


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_fused_mrope_matches_unfused_forward_backward(interleaved_mrope):
    t_ref, freqs, mrope_section = _make_inputs(requires_grad=True)
    t_fused = t_ref.detach().clone().requires_grad_(True)

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )
    ref = _apply_rotary_pos_emb_bshd(t_ref, emb, rotary_interleaved=False)
    out = fused_apply_mrope(
        t_fused, freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )

    tols = _dtype_tols(t_ref.dtype)
    torch.testing.assert_close(ref.float(), out.float(), **tols)

    grad = torch.randn_like(ref)
    ref.backward(grad)
    out.backward(grad)
    torch.testing.assert_close(t_ref.grad.float(), t_fused.grad.float(), **tols)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_apply_rotary_pos_emb_dispatches_raw_mrope(interleaved_mrope):
    t, freqs, mrope_section = _make_inputs()
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
        mrope_interleaved=interleaved_mrope,
    )

    out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup())

    emb = mrope_freqs_to_rotary_emb(
        freqs, mrope_section, interleaved_mrope=interleaved_mrope, rotary_interleaved=False
    )
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))
