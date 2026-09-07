# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.training.arguments import parse_args, validate_args


def test_te_cross_entropy_loss_fusion_warns_in_model_parallel_config():
    with pytest.warns(UserWarning, match="known stability issues"):
        config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'te'


def test_native_cross_entropy_loss_fusion_is_allowed():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'native'


def test_invalid_thd_tail_padding_policy_is_rejected_during_config_initialization():
    with pytest.raises(ValueError, match="thd_tail_padding_policy must be"):
        ModelParallelConfig(thd_tail_padding_policy="bogus")


def _static_packing_config(**overrides):
    kwargs = dict(
        thd_static_packing=True,
        max_seqlen_per_dp_cp_rank=4096,
        pad_packed_seq_alignment="max",
    )
    kwargs.update(overrides)
    return ModelParallelConfig(**kwargs)


def test_static_packing_rejects_extend_last_at_every_cp_size():
    # Not CP-gated: 'extend_last' leaves the valid cu_seqlens at the real token
    # count while the tensors are padded to the static target, so TE returns a
    # shorter attention output than its input even at CP=1.
    for cp_size in (1, 2):
        with pytest.raises(ValueError, match="thd_static_packing requires"):
            _static_packing_config(
                thd_tail_padding_policy="extend_last", context_parallel_size=cp_size
            )


def test_static_packing_accepts_the_default_and_explicit_dummy_tail():
    for policy in (None, "append_dummy_seq"):
        config = _static_packing_config(thd_tail_padding_policy=policy)
        assert config.thd_static_packing


def test_te_cross_entropy_loss_fusion_is_disabled_by_training_args(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['test_model_parallel_config.py'])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    # Let validate_args derive a global batch size that is valid for the
    # active data-parallel size in distributed unit-test jobs.
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 1024
    args.cross_entropy_loss_fusion = True
    args.cross_entropy_fusion_impl = 'te'

    with pytest.raises(AssertionError, match="Transformer Engine cross entropy loss fusion"):
        validate_args(args)
