# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch


def test_target_te_auto_upgrades_padded_thd_to_padding_causal(monkeypatch):
    """The target MCore wrapper owns the b436 padding-mask upgrade for THD."""
    from megatron.core.extensions import transformer_engine as mcore_te
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.enums import AttnMaskType

    captured = {}

    def fake_te_forward(_self, query, _key, _value, _attention_mask, **kwargs):
        captured.update(kwargs)
        return query

    monkeypatch.setattr(mcore_te.te.pytorch.DotProductAttention, "forward", fake_te_forward)
    attention = object.__new__(mcore_te.TEDotProductAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        window_size=None, qk_clip=False, log_max_attention_logit=False
    )
    attention.te_forward_mask_type = True
    attention.qkv_format = "sbhd"
    attention.num_splits = None
    attention.kept_packed_seq_params = {
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "max_seqlen_q",
        "max_seqlen_kv",
        "qkv_format",
    }
    physical = torch.tensor([0, 64, 128], dtype=torch.int32)
    packed = PackedSeqParams(
        cu_seqlens_q=physical,
        cu_seqlens_kv=physical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=64,
        max_seqlen_kv=64,
        qkv_format="thd",
        total_tokens=128,
    )
    query = torch.zeros(128, 1, 1)

    output = mcore_te.TEDotProductAttention.forward(
        attention, query, query, query, None, AttnMaskType.causal, packed_seq_params=packed
    )

    assert output is query
    assert captured["attn_mask_type"] == "padding_causal"
    assert captured["qkv_format"] == "thd"
    torch.testing.assert_close(captured["cu_seqlens_q"], physical)
    torch.testing.assert_close(captured["cu_seqlens_q_padded"], physical)
