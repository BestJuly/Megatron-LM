# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint facade tests: torch_dist round trip of the encoder state with
WORLD replica metadata.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_checkpoint.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.mdp.checkpoint import (
    ENCODER_STATE_KEY,
    add_encoder_state,
    assert_supported_checkpoint_config,
    load_encoder_state,
)
from megatron.core.mdp.errors import MdpCheckpointError

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


def test_exact_resume_flags_are_accepted():
    """Optimizer, LR-scheduler and RNG state round-trip, so the native
    `--no-*-optim`/`--no-*-rng` flags must no longer be demanded."""
    exact_resume = SimpleNamespace(
        save="/tmp/x",
        load="/tmp/x",
        no_save_optim=False,
        no_save_rng=False,
        no_load_optim=False,
        no_load_rng=False,
        ckpt_fully_parallel_save=False,
        ckpt_fully_parallel_load=False,
    )
    assert_supported_checkpoint_config(exact_resume)
    no_ckpt = SimpleNamespace(save=None, load=None)
    assert_supported_checkpoint_config(no_ckpt)


def test_fully_parallel_modes_are_rejected():
    # Megatron defaults ckpt_fully_parallel_save=True: it must be rejected
    # when saving (the fully-parallel path shards over one DP-CP group for
    # every child, which is wrong for the encoder's WORLD replica domain).
    fully_parallel_save = SimpleNamespace(save="/tmp/x", load=None, ckpt_fully_parallel_save=True)
    with pytest.raises(MdpCheckpointError, match="fully-parallel-save"):
        assert_supported_checkpoint_config(fully_parallel_save)
    fully_parallel_load = SimpleNamespace(save=None, load="/tmp/x", ckpt_fully_parallel_load=True)
    with pytest.raises(MdpCheckpointError, match="fully-parallel-load"):
        assert_supported_checkpoint_config(fully_parallel_load)


def test_unsupported_checkpoint_execution_modes_rejected():
    # Design doc section 12: asynchronous, non-persistent, and constant-
    # structure caching modes must fail at startup when a save/load is
    # requested; checkpoint-free runs are unaffected.
    base = dict(save="/tmp/x", load=None, ckpt_fully_parallel_save=False)
    for field, match in (
        ("async_save", "async-save"),
        ("ckpt_assume_constant_structure", "constant-structure"),
    ):
        args = SimpleNamespace(**base)
        setattr(args, field, True)
        with pytest.raises(MdpCheckpointError, match=match):
            assert_supported_checkpoint_config(args)
    args = SimpleNamespace(**base, non_persistent_ckpt_type="global")
    with pytest.raises(MdpCheckpointError, match="non-persistent"):
        assert_supported_checkpoint_config(args)
    # The same flags are ignored when no checkpoint is requested.
    quiet = SimpleNamespace(
        save=None, load=None, async_save=True, ckpt_assume_constant_structure=True
    )
    assert_supported_checkpoint_config(quiet)


def test_add_encoder_state_rejects_duplicates():
    class _FakeDdp:
        def sharded_state_dict(self, prefix="", metadata=None):
            return {"marker": prefix}

    state = add_encoder_state({}, _FakeDdp())
    assert state[ENCODER_STATE_KEY] == {"marker": "vision_model."}
    with pytest.raises(MdpCheckpointError, match="exactly once"):
        add_encoder_state(state, _FakeDdp())


def test_load_encoder_state_strips_both_prefix_levels():
    class _FakeDdp:
        def __init__(self):
            self.loaded = None
            self.strict = None

        def load_state_dict(self, state_dict, strict=True):
            self.loaded = state_dict
            self.strict = strict

    encoder = _FakeDdp()
    load_encoder_state(
        {ENCODER_STATE_KEY: {"vision_model.module.proj.weight": "w"}}, encoder, strict=False
    )
    assert encoder.loaded == {"proj.weight": "w"}
    assert encoder.strict is False

    with pytest.raises(MdpCheckpointError, match=ENCODER_STATE_KEY):
        load_encoder_state({"model": {}}, _FakeDdp())
    with pytest.raises(MdpCheckpointError, match="drifted apart"):
        load_encoder_state({ENCODER_STATE_KEY: {"proj.weight": "w"}}, _FakeDdp())


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_encoder_state_round_trips_strictly(tmp_path_factory):
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
    from megatron.core.mdp.encoder import build_encoder_pg_collection
    from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
    from megatron.core.transformer.transformer_config import TransformerConfig

    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)

    class _Enc(torch.nn.Module):
        def __init__(self, config, seed):
            super().__init__()
            self.config = config
            torch.manual_seed(seed)
            self.proj = torch.nn.Linear(8, 8, bias=False)
            self.head = torch.nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.head(self.proj(x))

    def _build(seed):
        model_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            calculate_per_token_loss=True,
            use_cpu_initialization=True,
        )
        return DistributedDataParallel(
            config=model_config,
            ddp_config=DistributedDataParallelConfig(
                use_distributed_optimizer=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            ),
            module=_Enc(model_config, seed).cuda(),
            pg_collection=encoder_pgs,
        )

    source = _build(seed=7)
    target = _build(seed=99)
    probe = torch.full((2, 8), 0.25, device="cuda")
    with torch.no_grad():
        source_out = source(probe).clone()
        assert not torch.equal(source(probe), target(probe))

    # Every rank must agree on the directory; rank 0 broadcasts its tmp dir.
    if torch.distributed.get_rank() == 0:
        directory = str(tmp_path_factory.mktemp("mdp_ckpt"))
    else:
        directory = None
    holder = [directory]
    torch.distributed.broadcast_object_list(holder, src=0)
    directory = holder[0]

    state = add_encoder_state({}, source)
    dist_checkpointing.save(state[ENCODER_STATE_KEY], directory)
    torch.distributed.barrier()

    load_skeleton = add_encoder_state({}, target)
    loaded = dist_checkpointing.load(load_skeleton[ENCODER_STATE_KEY], directory)
    load_encoder_state({ENCODER_STATE_KEY: loaded}, target, strict=True)

    with torch.no_grad():
        for source_param, target_param in zip(
            source.module.parameters(), target.module.parameters()
        ):
            assert torch.equal(source_param, target_param)
        assert torch.equal(target(probe), source_out)
