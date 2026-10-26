# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-compute tests for the MDP encoder DDP policy."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.encoder import build_encoder_ddp_config
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.optimizer import MdpChainedOptimizer, build_encoder_optimizer_config
from megatron.core.optimizer import OptimizerConfig


def test_decoder_overlap_is_not_inherited_by_encoder():
    decoder_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        align_param_gather=True,
        grad_reduce_in_fp32=True,
        bucket_size=1234,
    )

    encoder_config = build_encoder_ddp_config(decoder_config)

    assert encoder_config is not decoder_config
    assert decoder_config.overlap_grad_reduce
    assert decoder_config.overlap_param_gather
    assert decoder_config.align_param_gather
    assert not encoder_config.overlap_grad_reduce
    assert not encoder_config.overlap_param_gather
    assert not encoder_config.align_param_gather
    assert encoder_config.use_distributed_optimizer
    assert encoder_config.grad_reduce_in_fp32
    assert encoder_config.bucket_size == 1234


def test_encoder_does_not_inherit_mxfp8_buffers():
    decoder = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        fp8_param_gather=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
    )
    encoder = build_encoder_ddp_config(decoder)
    assert not encoder.fp8_param_gather
    assert not encoder.reuse_grad_buf_for_mxfp8_param_ag
    assert decoder.fp8_param_gather
    assert decoder.reuse_grad_buf_for_mxfp8_param_ag


def test_encoder_optimizer_uses_bf16_update_path():
    decoder = OptimizerConfig(
        bf16=True,
        fp8_recipe="mxfp8",
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_param_gather=True,
    )
    encoder = build_encoder_optimizer_config(decoder)
    assert encoder.fp8_recipe is None
    assert not encoder.reuse_grad_buf_for_mxfp8_param_ag
    assert not encoder.overlap_param_gather
    assert encoder.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not decoder.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert decoder.reuse_grad_buf_for_mxfp8_param_ag
    assert encoder.lr == decoder.lr
    assert encoder.clip_grad == decoder.clip_grad


def test_composite_rejects_unrelated_encoder_optimizer_changes():
    decoder = OptimizerConfig(bf16=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=True)
    encoder = replace(build_encoder_optimizer_config(decoder), lr=0.123)
    members = [SimpleNamespace(config=config, model_chunks=[]) for config in (decoder, encoder)]
    with pytest.raises(MdpConfigurationError, match="hyperparameters"):
        MdpChainedOptimizer(members, encoder_member_index=1)
