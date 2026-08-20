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
        def __init__(self, keys=("proj.weight",)):
            self.loaded = None
            self.strict = None
            self._keys = keys

        def state_dict(self):
            # `load_encoder_state` compares the checkpoint's keys against the
            # module's own before delegating, so the double must expose them.
            return {key: None for key in self._keys}

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


class _RealDdp(torch.nn.Module):
    """Stand-in with the same ``load_state_dict`` contract as the encoder DDP.

    ``_BaseDataParallel.load_state_dict`` forwards to the wrapped module and
    returns ``None``, so a caller cannot learn which keys were missing from its
    return value -- which is why the guard below checks the keys up front.
    """

    def __init__(self):
        super().__init__()
        self.module = torch.nn.Linear(4, 4, bias=False)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


def test_load_encoder_state_rejects_missing_keys_even_when_not_strict():
    """A relaxed load must not leave the encoder randomly initialized.

    ``strict`` reaches :func:`load_encoder_state` from ``load_checkpoint``'s own
    parameter, and ``torch.nn.Module.load_state_dict(strict=False)`` reports
    missing keys instead of raising. The encoder is fully replicated and is
    always written whole, so a key that is absent here means the state did not
    round-trip -- there is no "empty stage" case to tolerate, unlike a decoder
    chunk. Silently skipping it would resume training from the random
    initialization, which is exactly the failure ``load_encoder_state`` exists
    to prevent.
    """
    encoder = _RealDdp()
    complete = {
        "vision_model.module." + key: value for key, value in encoder.module.state_dict().items()
    }
    load_encoder_state({ENCODER_STATE_KEY: complete}, encoder, strict=False)

    with pytest.raises(MdpCheckpointError, match="missing"):
        load_encoder_state({ENCODER_STATE_KEY: {}}, _RealDdp(), strict=False)
    with pytest.raises(MdpCheckpointError, match="missing"):
        load_encoder_state({ENCODER_STATE_KEY: {}}, _RealDdp(), strict=True)


class _ExtraStateLinear(torch.nn.Linear):
    """A layer with TransformerEngine's extra-state contract.

    Overriding ``get_extra_state``/``set_extra_state`` is what makes
    ``torch.nn.Module`` publish an ``_extra_state`` key in ``state_dict()`` and,
    under ``strict=True``, demand it back on load -- the same contract TE's
    layers carry.
    """

    def get_extra_state(self):
        return {"fp8_meta": None}

    def set_extra_state(self, state):
        pass


def test_load_encoder_state_tolerates_absent_te_extra_state():
    """TE's ``_extra_state`` entries hold no weights and may be absent.

    A real TE encoder publishes ``..._extra_state`` keys in ``state_dict()``
    that the sharded state dict does not always carry, so the weight check must
    exempt them -- and so must the delegated load, which would otherwise reject
    the very same keys one line later. ``megatron.training.checkpointing``'s
    ``load_model_state_dict`` gives every decoder chunk that tolerance already;
    the encoder must not be stricter than the decoder it trains beside.

    The double deliberately does *not* override ``load_state_dict``:
    ``_BaseDataParallel.load_state_dict`` forwards ``strict`` verbatim, so a
    double that dropped it would hide the mismatch this test exists to pin down.
    """

    class _ExtraStateDdp(_RealDdp):
        def __init__(self):
            super().__init__()
            self.module = _ExtraStateLinear(4, 4, bias=False)

    encoder = _ExtraStateDdp()
    assert "_extra_state" in encoder.state_dict()
    weights = {"vision_model.module.weight": torch.full((4, 4), 3.0)}

    load_encoder_state({ENCODER_STATE_KEY: weights}, encoder, strict=True)
    assert torch.equal(encoder.module.weight, torch.full((4, 4), 3.0))

    # The weight guard still fires for the same module: exempting extra state
    # must not have relaxed the check that keeps the encoder off its random
    # initialization.
    with pytest.raises(MdpCheckpointError, match="missing"):
        load_encoder_state({ENCODER_STATE_KEY: {}}, _ExtraStateDdp(), strict=True)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_encoder_state_round_trips_strictly(tmp_path_factory):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.mdp.encoder import build_encoder_pg_collection
    from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
    from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
    from megatron.core.transformer.transformer_config import TransformerConfig

    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1))
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
