import torch

from megatron.core.ssm import gated_delta_net as gdn


def _make_fixed_inputs(batch=2, seq_len=64, heads=4, dim=128):
    q = torch.empty(batch, seq_len, heads, dim, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.zeros(batch, seq_len, heads, dtype=torch.float32)
    beta = torch.ones(batch, seq_len, heads, dtype=torch.float32)
    return q, k, v, g, beta


def test_flashinfer_fixed_length_cu_seqlens_are_cached_int32():
    q, k, v, g, beta = _make_fixed_inputs()

    *_, cu_first, assume_valid_first = gdn._flashinfer_prepare_flat_inputs(q, k, v, g, beta, None)
    *_, cu_second, assume_valid_second = gdn._flashinfer_prepare_flat_inputs(q, k, v, g, beta, None)

    assert cu_first.dtype == torch.int32
    assert torch.equal(cu_first, torch.tensor([0, 64, 128], dtype=torch.int32))
    assert cu_first is cu_second
    assert assume_valid_first is True
    assert assume_valid_second is True


def test_flashinfer_varlen_cu_seqlens_stay_strictly_validated():
    q, k, v, g, beta = _make_fixed_inputs()
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int64)

    *_, cu_out, assume_valid = gdn._flashinfer_prepare_flat_inputs(q, k, v, g, beta, cu_seqlens)

    assert cu_out.dtype == torch.int64
    assert torch.equal(cu_out, cu_seqlens)
    assert assume_valid is False
