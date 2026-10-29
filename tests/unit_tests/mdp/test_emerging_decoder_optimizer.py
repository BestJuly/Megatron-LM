# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDP with an emerging decoder optimizer (Muon, layer-wise distributed).

``--optimizer muon`` shards the decoder layer-wise (``use_layer_wise_distributed_optimizer``)
instead of with the byte-level DistributedOptimizer, and builds the decoder optimizer as
``ChainedOptimizer([LayerWiseDistributedOptimizer, DistOpt(dense Adam), DistOpt(expert Adam)])``.
The encoder keeps its own Adam ZeRO-1 domain over WORLD. These checks are host-only.
"""

import pytest

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.config import MdpConfig, validate_mdp_config
from megatron.core.mdp.encoder import build_encoder_ddp_config
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.optimizer import _flatten, build_encoder_optimizer_config
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from tests.unit_tests.mdp.test_config import _options


def _muon_config(**overrides):
    kwargs = dict(
        optimizer="muon",
        use_layer_wise_distributed_optimizer=True,
        # The perf recipes' Muon options: Adam rejects muon_tp_mode='layer_sharded'.
        muon_tp_mode="layer_sharded",
        muon_ns_batch_size=32,
        muon_split_qkv=False,
        use_distributed_optimizer=False,
        lr=3e-4,
        min_lr=3e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
    )
    kwargs.update(overrides)
    return OptimizerConfig(**kwargs)


def test_encoder_optimizer_is_adam_zero1_under_a_muon_decoder():
    decoder = _muon_config()
    encoder = build_encoder_optimizer_config(decoder)
    assert encoder.optimizer == "adam"
    assert encoder.use_distributed_optimizer
    assert not encoder.use_layer_wise_distributed_optimizer
    assert encoder.muon_tp_mode == OptimizerConfig().muon_tp_mode
    for field in ("lr", "min_lr", "weight_decay", "adam_beta1", "adam_beta2", "bf16"):
        assert getattr(encoder, field) == getattr(decoder, field), field
    # The decoder config itself is untouched.
    assert decoder.optimizer == "muon" and decoder.use_layer_wise_distributed_optimizer


def test_encoder_optimizer_config_is_the_decoders_under_adam():
    decoder = OptimizerConfig(optimizer="adam", use_distributed_optimizer=True, lr=1e-4)
    assert build_encoder_optimizer_config(decoder) is decoder


def test_encoder_ddp_is_always_distributed():
    decoder_ddp = DistributedDataParallelConfig(
        use_distributed_optimizer=False, overlap_grad_reduce=True
    )
    encoder_ddp = build_encoder_ddp_config(decoder_ddp)
    assert encoder_ddp.use_distributed_optimizer
    assert not encoder_ddp.overlap_grad_reduce


def test_layer_wise_member_is_not_flattened():
    """A LayerWiseDistributedOptimizer keeps its step / norm / checkpoint overrides only as
    one member; flattening it into its raw optimizers would drop the parameter all-gather."""
    layer_wise = object.__new__(LayerWiseDistributedOptimizer)
    dense, expert = object(), object()
    chain = object.__new__(ChainedOptimizer)
    chain.chained_optimizers = [layer_wise, dense, expert]
    assert list(_flatten(chain)) == [layer_wise, dense, expert]


def test_gate_accepts_the_layer_wise_distributed_optimizer():
    validate_mdp_config(
        MdpConfig(enable=True),
        _options(use_distributed_optimizer=False, use_layer_wise_distributed_optimizer=True),
    )


def test_gate_rejects_replicated_optimizer_state():
    with pytest.raises(MdpConfigurationError, match="use_distributed_optimizer"):
        validate_mdp_config(
            MdpConfig(enable=True),
            _options(use_distributed_optimizer=False, use_layer_wise_distributed_optimizer=False),
        )
