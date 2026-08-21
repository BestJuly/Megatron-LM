# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Composite-optimizer checkpoint contract.

Most of these need no GPU and no process group: they pin the keying and
pairing rules that let one checkpoint hold two sharding domains::

    pytest -q tests/unit_tests/mdp/test_optimizer_checkpoint.py

The last test is the end-to-end proof and needs a 4-rank world; it reproduces
the PP2/DP2 shape where the decoder's PP-stage-0 DistributedOptimizer and the
WORLD-sharded encoder both compute ``data_parallel_group_idx == 0``::

    torchrun --nproc_per_node=4 -m pytest -q \\
        tests/unit_tests/mdp/test_optimizer_checkpoint.py
"""

import os

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.mdp.errors import MdpCheckpointError, MdpConfigurationError
from megatron.core.mdp.optimizer import (
    ENCODER_MEMBER_KEY,
    MdpChainedOptimizer,
    build_mdp_composite_optimizer,
)

_DP_RESHARDABLE_METADATA = {
    "distrib_optim_sharding_type": "dp_reshardable",
    "chained_optim_avoid_prefix": True,
}


class _FakeInnerOptimizer:
    def __init__(self, step):
        # A non-empty 'params' list is what _synchronize_steps filters on.
        self.param_groups = [{"params": [object()], "step": step}]


class _FakeMember:
    """Minimal stand-in for a DistributedOptimizer member."""

    config = None
    is_stub_optimizer = False

    def __init__(self, name, step=0):
        self.name = name
        self.optimizer = _FakeInnerOptimizer(step)
        self.model_chunks = []
        self.loaded_state = None

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, **kwargs):
        # Every member collides on dp_group_idx_0 without a prefix: that is
        # exactly the situation the fixed encoder key has to survive.
        return {
            "param_state": ShardedObject(
                "optimizer.distributed.dp_group_idx_0.param_state", None, (1,), (0,)
            )
        }

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


def _flat_build(decoder_members):
    """Build the composite from an explicit member list."""
    decoder = [_FakeMember(f"decoder_{i}") for i in range(decoder_members)]
    encoder = _FakeMember("encoder")
    composite = MdpChainedOptimizer(decoder + [encoder], encoder_member_index=len(decoder))
    return composite, decoder, encoder


def test_encoder_key_does_not_depend_on_the_decoder_member_count():
    """The bug this replaces: ``chained_{idx}`` for the encoder makes its index
    the decoder member count, and a PP stage with no expert parameters
    contributes one member instead of two — so the same logical tensor would be
    written under different keys on different ranks."""
    one_member, _, _ = _flat_build(1)
    two_members, _, _ = _flat_build(2)
    assert one_member._member_keys() == ["chained_0", ENCODER_MEMBER_KEY]
    assert two_members._member_keys() == ["chained_0", "chained_1", ENCODER_MEMBER_KEY]


def test_sharded_state_dict_prefixes_the_encoder_with_the_fixed_key():
    composite, _, _ = _flat_build(2)
    sharded = composite.sharded_state_dict({}, metadata=dict(_DP_RESHARDABLE_METADATA))

    assert set(sharded) == {"chained_0", "chained_1", ENCODER_MEMBER_KEY}
    keys = {name: member["param_state"].key for name, member in sharded.items()}
    assert keys["chained_0"] == "chained_0.optimizer.distributed.dp_group_idx_0.param_state"
    assert keys[ENCODER_MEMBER_KEY] == (
        f"{ENCODER_MEMBER_KEY}.optimizer.distributed.dp_group_idx_0.param_state"
    )
    # All three collided on dp_group_idx_0 before prefixing.
    assert len(set(keys.values())) == 3


def test_encoder_is_prefixed_even_for_fully_reshardable_formats():
    """The prefix is the isolation between domains, not a format detail."""
    composite, _, _ = _flat_build(1)
    sharded = composite.sharded_state_dict(
        {},
        metadata={
            "distrib_optim_sharding_type": "fully_reshardable",
            "chained_optim_avoid_prefix": True,
        },
    )
    assert sharded["chained_0"]["param_state"].key.startswith("optimizer.distributed")
    assert sharded[ENCODER_MEMBER_KEY]["param_state"].key.startswith(f"{ENCODER_MEMBER_KEY}.")


def test_load_state_dict_pairs_members_by_key_not_position():
    """``ChainedOptimizer.load_state_dict`` zips members against
    ``sorted(state_dict.items())``. With the encoder keyed separately that
    ordering is no longer guaranteed to be the member ordering."""
    composite, decoder, encoder = _flat_build(2)
    state = {
        ENCODER_MEMBER_KEY: {"who": "encoder"},
        "chained_1": {"who": "decoder_1"},
        "chained_0": {"who": "decoder_0"},
    }
    composite.load_state_dict(state)
    assert decoder[0].loaded_state == {"who": "decoder_0"}
    assert decoder[1].loaded_state == {"who": "decoder_1"}
    assert encoder.loaded_state == {"who": "encoder"}


def test_load_state_dict_rejects_a_checkpoint_without_the_encoder_key():
    composite, _, _ = _flat_build(1)
    with pytest.raises(MdpCheckpointError, match=ENCODER_MEMBER_KEY):
        composite.load_state_dict({"chained_0": {}, "chained_1": {}})


def test_out_of_range_encoder_index_is_rejected():
    with pytest.raises(MdpCheckpointError, match="out of range"):
        MdpChainedOptimizer([_FakeMember("a")], encoder_member_index=1)


def test_builder_tags_the_encoder_member():
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    decoder = ChainedOptimizer([_FakeMember("dense"), _FakeMember("expert")])
    composite = build_mdp_composite_optimizer(decoder, _FakeMember("encoder"))
    assert composite._member_keys() == ["chained_0", "chained_1", ENCODER_MEMBER_KEY]


def test_builder_rejects_a_multi_member_encoder_side():
    """One fixed key can only isolate one encoder member."""
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    encoder = ChainedOptimizer([_FakeMember("enc_a"), _FakeMember("enc_b")])
    with pytest.raises(MdpConfigurationError, match="one fixed key"):
        build_mdp_composite_optimizer(_FakeMember("decoder"), encoder)


def test_untagged_composite_keeps_the_inherited_behavior():
    """A composite built by hand (no encoder declared) must not change."""
    composite = MdpChainedOptimizer([_FakeMember("a"), _FakeMember("b")])
    sharded = composite.sharded_state_dict({}, metadata=dict(_DP_RESHARDABLE_METADATA))
    assert set(sharded) == {0, 1}


# ----------------------------------------------------------------------------
# End-to-end: two sharding domains through one torch_dist checkpoint.
# ----------------------------------------------------------------------------

_NEEDS_FOUR_RANKS = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4, reason="needs `torchrun --nproc_per_node=4`"
)

_SHARDED_METADATA = {
    "distrib_optim_sharding_type": "dp_reshardable",
    "chained_optim_avoid_prefix": True,
    "singleton_local_shards": False,
}


def _group_containing(rank_lists):
    """Create every group (all ranks must), return the one holding this rank."""
    rank = torch.distributed.get_rank()
    mine = None
    for ranks in rank_lists:
        group = torch.distributed.new_group(ranks=list(ranks))
        if rank in ranks:
            mine = group
    return mine


def _pg_collection(dp_group, mp_group):
    from megatron.core.process_groups_config import ProcessGroupCollection

    singleton = _group_containing([[r] for r in range(torch.distributed.get_world_size())])
    pgs = ProcessGroupCollection()
    pgs.dp = dp_group
    pgs.dp_cp = dp_group
    pgs.intra_dp_cp = dp_group
    pgs.intra_dist_opt = dp_group
    pgs.tp = singleton
    pgs.pp = singleton
    pgs.ep = singleton
    pgs.mp = mp_group
    pgs.expt_dp = None
    pgs.tp_ep_pp = None
    pgs.inter_dist_opt = None
    return pgs


class _Tiny(torch.nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        torch.manual_seed(seed)
        self.proj = torch.nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.proj(x)


def _domain(dp_group, mp_group, seed):
    """One DDP + DistributedOptimizer pair over the given sharding domain."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer.transformer_config import TransformerConfig

    model_config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        bf16=True,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    pgs = _pg_collection(dp_group, mp_group)
    ddp = DistributedDataParallel(
        config=model_config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False, overlap_param_gather=False
        ),
        module=_Tiny(model_config, seed).cuda().to(torch.bfloat16),
        pg_collection=pgs,
    )
    optimizer = get_megatron_optimizer(
        config=OptimizerConfig(
            optimizer="adam",
            lr=1e-2,
            use_distributed_optimizer=True,
            clip_grad=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
        ),
        model_chunks=[ddp],
        pg_collection=pgs,
        use_gloo_process_groups=False,
    )
    return ddp, optimizer


def _build_composite(seed_offset):
    """Decoder-like member over a DP pair, encoder-like member over WORLD.

    ``mp`` for the decoder is the cross-pair group, so its
    ``data_parallel_group_idx`` is its pair index — exactly like a PP rank. The
    encoder passes ``mp=None`` (as ``build_encoder_pg_collection`` does), which
    makes its index 0 and collides with decoder pair 0.
    """
    decoder_dp = _group_containing([[0, 1], [2, 3]])
    decoder_mp = _group_containing([[0, 2], [1, 3]])
    decoder_ddp, decoder_opt = _domain(decoder_dp, decoder_mp, seed=1 + seed_offset)
    encoder_ddp, encoder_opt = _domain(torch.distributed.group.WORLD, None, seed=2 + seed_offset)
    composite = build_mdp_composite_optimizer(decoder_opt, encoder_opt)
    return decoder_ddp, encoder_ddp, composite


def _take_a_step(ddps, composite):
    for ddp in ddps:
        ddp.zero_grad_buffer()
        out = ddp(torch.ones(4, 16, device="cuda", dtype=torch.bfloat16))
        out.float().sum().backward()
        ddp.finish_grad_sync()
    success, _grad_norm, _ = composite.step()
    assert success


def _moments(composite):
    """Every member's Adam moments, in member order."""
    moments = []
    for member in composite.chained_optimizers:
        for state in member.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    moments.append(state[key].detach().clone())
    return moments


@_NEEDS_FOUR_RANKS
def test_both_domains_optimizer_state_round_trips(tmp_path_factory):
    from megatron.core import dist_checkpointing
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        decoder_ddp, encoder_ddp, source = _build_composite(seed_offset=0)
        _take_a_step([decoder_ddp, encoder_ddp], source)
        saved_moments = _moments(source)
        assert saved_moments, "no Adam moments were produced; the step did not run"
        assert any(float(m.abs().sum()) > 0 for m in saved_moments)

        directory = [str(tmp_path_factory.mktemp("mdp_optim_ckpt"))]
        torch.distributed.broadcast_object_list(directory, src=0)

        # Save: the fixed encoder key is what keeps the WORLD-sharded encoder
        # state from overwriting the PP-stage-0 decoder state.
        dist_checkpointing.save(
            source.sharded_state_dict({}, metadata=dict(_SHARDED_METADATA)), directory[0]
        )
        torch.distributed.barrier()

        # Load into an independently initialized composite.
        _, _, target = _build_composite(seed_offset=100)
        skeleton = target.sharded_state_dict({}, is_loading=True, metadata=dict(_SHARDED_METADATA))
        target.load_state_dict(dist_checkpointing.load(skeleton, directory[0]))

        restored = _moments(target)
        assert len(restored) == len(saved_moments)
        for saved, loaded in zip(saved_moments, restored):
            assert torch.equal(saved, loaded)
    finally:
        Utils.destroy_model_parallel()
