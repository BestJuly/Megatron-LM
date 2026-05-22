# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions.fused_mrope import (
    fused_apply_mrope,
    get_fused_mrope_unavailable_reason,
    is_fused_mrope_available,
    mrope_freqs_to_rotary_emb,
)
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rope_utils import (
    _ROPE_FUSION_FALLBACK_WARNINGS,
    _apply_rotary_pos_emb_bshd,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


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


def _make_inputs(
    dtype=torch.bfloat16,
    requires_grad=False,
    head_dim=20,
    rotary_dim=16,
    mrope_section=None,
    interleaved_mrope=False,
):
    seq = 32
    batch = 2
    heads = 3
    if mrope_section is None:
        mrope_section = [3, 3, 2] if interleaved_mrope else [2, 3, 3]

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


def _make_position_ids(seq, batch):
    base = torch.arange(seq, device="cuda", dtype=torch.long)
    batch_offsets = torch.arange(batch, device="cuda", dtype=torch.long)
    return (
        torch.stack((base, base * 2 + 3, base * 3 + 5), dim=0)[:, None, :]
        + batch_offsets[None, :, None]
    ).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
@pytest.mark.parametrize("head_dim", [16, 20])
def test_fused_mrope_matches_unfused_forward_backward(interleaved_mrope, head_dim):
    t_ref, freqs, mrope_section = _make_inputs(
        requires_grad=True, head_dim=head_dim, interleaved_mrope=interleaved_mrope
    )
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
@pytest.mark.parametrize(
    "fallback_kwargs, warning_match",
    [
        ({"mscale": 1.25}, "mscale=1.25 is not supported by Triton fused mRoPE"),
        ({"inverse": True}, "inverse RoPE is not supported by Triton fused mRoPE"),
    ],
)
def test_apply_rotary_pos_emb_raw_mrope_fallbacks_match_unfused(
    fallback_kwargs, warning_match
):
    t, freqs, mrope_section = _make_inputs()
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
    )

    _ROPE_FUSION_FALLBACK_WARNINGS.clear()
    with pytest.warns(UserWarning, match=warning_match):
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup(), **fallback_kwargs)

    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False, **fallback_kwargs)
    torch.testing.assert_close(ref.float(), out.float(), **_dtype_tols(t.dtype))

    with warnings.catch_warnings(record=True) as repeated_warnings:
        warnings.simplefilter("always")
        out_again = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup(), **fallback_kwargs)
    assert not repeated_warnings
    torch.testing.assert_close(ref.float(), out_again.float(), **_dtype_tols(t.dtype))


def test_interleaved_mrope_rejects_inconsistent_sections():
    freqs = torch.randn(3, 2, 8, 8, dtype=torch.float32)

    with pytest.raises(AssertionError, match="interleaved mRoPE"):
        mrope_freqs_to_rotary_emb(freqs, [2, 3, 3], interleaved_mrope=True)


def test_raw_mrope_cpu_falls_back_to_unfused():
    t = torch.randn(8, 1, 3, 20, dtype=torch.float32)
    freqs = torch.randn(3, 1, 8, 8, dtype=torch.float32)
    mrope_section = [2, 3, 3]
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
    )

    _ROPE_FUSION_FALLBACK_WARNINGS.clear()
    assert "CUDA tensors" in get_fused_mrope_unavailable_reason(t, freqs)
    with pytest.warns(UserWarning, match="CUDA tensors.*Using unfused implementation"):
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup())

    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)
    torch.testing.assert_close(ref, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
def test_raw_mrope_unsupported_dtype_falls_back_to_unfused():
    t, freqs, mrope_section = _make_inputs(dtype=torch.float64)
    config = TransformerConfig(
        num_attention_heads=t.shape[2],
        num_layers=1,
        apply_rope_fusion=True,
        mrope_section=mrope_section,
    )

    _ROPE_FUSION_FALLBACK_WARNINGS.clear()
    assert "dtype" in get_fused_mrope_unavailable_reason(t, freqs)
    with pytest.warns(UserWarning, match="dtype.*Using unfused implementation"):
        out = apply_rotary_pos_emb(t, freqs, config, cp_group=FakeCPGroup())

    emb = mrope_freqs_to_rotary_emb(freqs, mrope_section, rotary_interleaved=False)
    ref = _apply_rotary_pos_emb_bshd(t, emb, rotary_interleaved=False)
    torch.testing.assert_close(ref, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_apply_rotary_pos_emb_dispatches_raw_mrope(interleaved_mrope):
    t, freqs, mrope_section = _make_inputs(interleaved_mrope=interleaved_mrope)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not is_fused_mrope_available(), reason="Triton fused mRoPE not available")
@pytest.mark.skipif(Utils.world_size < 2, reason="CP test requires at least 2 distributed ranks")
@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_raw_mrope_fusion_matches_unfused_with_context_parallel(interleaved_mrope):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        seq = 32
        batch = 2
        heads = 3
        head_dim = 20
        rotary_dim = 16
        mrope_section = [3, 3, 2] if interleaved_mrope else [2, 3, 3]
        position_ids = _make_position_ids(seq, batch)

        rope = MultimodalRotaryEmbedding(
            head_dim,
            rotary_percent=rotary_dim / head_dim,
            cp_group=cp_group,
            interleaved_mrope=interleaved_mrope,
        )
        raw_freqs = rope(
            position_ids,
            mrope_section,
            cp_group=cp_group,
            return_raw_freqs=True,
        )
        materialized_emb = rope(position_ids, mrope_section, cp_group=cp_group)
        raw_freqs_emb = mrope_freqs_to_rotary_emb(
            raw_freqs,
            mrope_section,
            interleaved_mrope=interleaved_mrope,
            rotary_interleaved=False,
        )
        torch.testing.assert_close(raw_freqs_emb, materialized_emb)

        local_seq = seq // cp_group.size()
        assert raw_freqs.shape == (3, batch, local_seq, rotary_dim // 2)
        assert materialized_emb.shape == (local_seq, batch, 1, rotary_dim)

        generator = torch.Generator(device="cuda").manual_seed(4321)
        t_ref = torch.randn(
            local_seq,
            batch,
            heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
            requires_grad=True,
        )
        t_fused = t_ref.detach().clone().requires_grad_(True)

        config = TransformerConfig(
            num_attention_heads=heads,
            num_layers=1,
            context_parallel_size=cp_group.size(),
            apply_rope_fusion=True,
            mrope_section=mrope_section,
            mrope_interleaved=interleaved_mrope,
        )

        ref = _apply_rotary_pos_emb_bshd(t_ref, materialized_emb, rotary_interleaved=False)
        out = apply_rotary_pos_emb(t_fused, raw_freqs, config, cp_group=cp_group)
        tols = _dtype_tols(t_ref.dtype)
        torch.testing.assert_close(ref.float(), out.float(), **tols)

        grad = torch.randn_like(ref)
        ref.backward(grad)
        out.backward(grad)
        torch.testing.assert_close(t_ref.grad.float(), t_fused.grad.float(), **tols)
    finally:
        Utils.destroy_model_parallel()
