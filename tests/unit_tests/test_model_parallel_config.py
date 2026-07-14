# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.model_parallel_config import ModelParallelConfig


def test_native_cross_entropy_loss_fusion_is_allowed():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'native'
