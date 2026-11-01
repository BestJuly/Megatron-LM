# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PackedSeqParams across the MDP full-iteration decoder graph's static buffers.

pytest examples/multimodal_dev/tests/test_static_graph_packed_seq_params.py
"""

import dataclasses

import pytest
import torch

from examples.multimodal_dev import forward_step
from megatron.core.packed_seq_params import PackedSeqParams


@pytest.fixture(autouse=True)
def _fresh_constants():
    forward_step.set_mdp_static_graph_model(None)
    yield
    forward_step.set_mdp_static_graph_model(None)


def _params(lengths, total_tokens=16, max_seqlen=8):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        total_tokens=total_tokens,
    )


def _tensors(params):
    return {
        f.name: getattr(params, f.name)
        for f in dataclasses.fields(params)
        if isinstance(getattr(params, f.name), torch.Tensor)
    }


def test_every_tensor_field_becomes_a_static_input():
    params = _params([8, 8])
    assert params.seq_idx is not None  # derived by __post_init__
    tensors, template = forward_step._split_static_packed_seq_params(params)
    assert set(tensors) == set(_tensors(params))
    assert "seq_idx" in tensors
    assert not _tensors(template)
    assert template.total_tokens == 16 and template.max_seqlen_q == 8


def test_rebuilt_params_read_only_the_static_buffers():
    tensors, template = forward_step._split_static_packed_seq_params(_params([8, 8]))
    static = {name: t.clone() for name, t in tensors.items()}
    rebuilt = forward_step._static_graph_packed_seq_params(
        {"packed_seq_params": template, "packed_seq_tensors": static}
    )
    for name, t in _tensors(rebuilt).items():
        # The same objects: __post_init__ must not rerun and derive a fresh seq_idx.
        assert t is static[name], name
    assert set(_tensors(rebuilt)) == set(static)


def test_a_changed_baked_in_field_raises():
    forward_step._split_static_packed_seq_params(_params([8, 8]))
    forward_step._split_static_packed_seq_params(_params([4, 12]))  # tensors may change
    with pytest.raises(RuntimeError, match="max_seqlen"):
        forward_step._split_static_packed_seq_params(_params([4, 12], max_seqlen=12))
