# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-compute tests for the MDP encoder DDP policy."""

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.encoder import build_encoder_ddp_config


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
