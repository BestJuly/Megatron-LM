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
from torch.distributed.checkpoint.api import CheckpointException

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.mdp.checkpoint import (
    ENCODER_STATE_KEY,
    add_encoder_state,
    assert_supported_checkpoint_config,
    assert_zero_pad_vision_ffn_resume,
    encoder_sharded_state_dict,
    load_encoder_state,
)
from megatron.core.mdp.encoder import zero_pad_vision_mlp_channels
from megatron.core.mdp.errors import MdpCheckpointError
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule

# The encoder model/config harness lives next to the tests that own it; this
# file builds the same tiny encoder, so it reuses rather than restates it.
from tests.unit_tests.mdp.test_encoder_domain import (
    _assert_padding_is_zero,
    _perturb_padding_channels,
    _tiny_config,
    _TinyMLPEncoder,
)

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1

if _DISTRIBUTED:
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from tests.unit_tests.mdp.test_encoder_domain import _build_encoder_pgs
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture(scope="module")
    def encoder_pgs(_init_parallel):
        """The encoder's process groups for tp=1, pp=2, encoder_cp=1. Installed
        once per module: ``dist.new_group`` is collective, so every rank has to
        reach it the same number of times."""
        return _build_encoder_pgs()

    def _encoder_ddp(encoder_class, encoder_pgs, *, seed, **config_overrides):
        """The DDP wrapper build_encoder_domain() puts around the encoder, over
        this file's tiny config at the given seed."""
        torch.manual_seed(seed)
        config = _tiny_config(**config_overrides)
        return DistributedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_distributed_optimizer=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            ),
            module=encoder_class(config).cuda(),
            pg_collection=encoder_pgs,
        )

    def _vision_mlp_ddp(encoder_pgs, ffn_hidden_size, *, seed):
        """A DDP-wrapped real (ungated, biased) vision MLP at the given FFN
        width -- the shape zero_pad_vision_mlp_channels() pads."""
        return _encoder_ddp(
            _TinyVisionEncoder,
            encoder_pgs,
            seed=seed,
            ffn_hidden_size=ffn_hidden_size,
            gated_linear_unit=False,
            add_bias_linear=True,
        )

    def _shared_checkpoint_dir(tmp_path_factory, name):
        """A directory every rank agrees on; rank 0 broadcasts its tmp dir."""
        holder = [str(tmp_path_factory.mktemp(name)) if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(holder, src=0)
        return holder[0]



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


@pytest.mark.parametrize(
    "run_pads,checkpoint_args,rejected",
    [
        # A run that widened the FFN *without* the flag trained the alignment
        # channels as ordinary weights. Resuming it with the flag would load
        # them verbatim.
        (True, dict(mdp_zero_pad_vision_ffn=False, encoder_ffn_hidden_size=4320), True),
        # Official vision FFN width: allow_shape_mismatch zero-fills the padding.
        (True, dict(mdp_zero_pad_vision_ffn=False, encoder_ffn_hidden_size=None), False),
        # Written by another zero-padding run at the same width: the padding
        # channels are already zero.
        (True, dict(mdp_zero_pad_vision_ffn=True, encoder_ffn_hidden_size=4320), False),
        # ...but at a different padded width both sides are shape-mismatch
        # tolerant, so DCP would silently truncate the wider tensors.
        (True, dict(mdp_zero_pad_vision_ffn=True, encoder_ffn_hidden_size=4352), True),
        # Not padding this run: there is no invariant to protect.
        (False, dict(mdp_zero_pad_vision_ffn=False, encoder_ffn_hidden_size=4320), False),
        # Args that predate the flags (or no args at all) can only have been
        # written at the official width -- the widened FFN never existed outside
        # these flags -- so they are accepted, not rejected on a missing attribute.
        (True, dict(), False),
        (True, dict(tensor_model_parallel_size=1), False),
    ],
    ids=[
        "trained_padding",
        "official_width",
        "another_padding_run",
        "another_padding_run_different_width",
        "not_padding_this_run",
        "no_args",
        "args_predate_the_flags",
    ],
)
def test_zero_pad_vision_ffn_resume_rejects_a_trained_padding_checkpoint(
    run_pads, checkpoint_args, rejected
):
    """The zero-padding invariant has to be re-checked against the checkpoint's
    own args, not just re-established at build time. The guard keys on the
    persisted --encoder-ffn-hidden-size / --mdp-zero-pad-vision-ffn."""
    args = SimpleNamespace(mdp_zero_pad_vision_ffn=run_pads, encoder_ffn_hidden_size=4320)
    checkpoint_args = SimpleNamespace(**checkpoint_args)
    if not rejected:
        assert_zero_pad_vision_ffn_resume(args, checkpoint_args)
        return
    with pytest.raises(
        MdpCheckpointError, match="padding channels must be zero|widths must be equal"
    ):
        assert_zero_pad_vision_ffn_resume(args, checkpoint_args)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_encoder_state_round_trips_strictly(tmp_path_factory, encoder_pgs):
    class _Enc(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = torch.nn.Linear(8, 8, bias=False)
            self.head = torch.nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.head(self.proj(x))

    source = _encoder_ddp(_Enc, encoder_pgs, seed=7)
    target = _encoder_ddp(_Enc, encoder_pgs, seed=99)
    probe = torch.full((2, 8), 0.25, device="cuda")
    with torch.no_grad():
        source_out = source(probe).clone()
        assert not torch.equal(source(probe), target(probe))

    directory = _shared_checkpoint_dir(tmp_path_factory, "mdp_ckpt")
    state = add_encoder_state({}, source)
    dist_checkpointing.save(state[ENCODER_STATE_KEY], directory)
    torch.distributed.barrier()

    load_skeleton = add_encoder_state({}, target)[ENCODER_STATE_KEY]
    loaded = dist_checkpointing.load(load_skeleton, directory)
    load_encoder_state({ENCODER_STATE_KEY: loaded}, target, strict=True)

    with torch.no_grad():
        for source_param, target_param in zip(
            source.module.parameters(), target.module.parameters()
        ):
            assert torch.equal(source_param, target_param)
        assert torch.equal(target(probe), source_out)


# ----------------- zero_pad_vision_ffn: checkpoint load paths -----------------


class _PatchMergerStandIn(MegatronModule):
    """The patch merger's shape: ``linear_fc1`` merge_dim -> merge_dim and
    ``linear_fc2`` merge_dim -> out_hidden_size, both biased -- the same child
    names as an MLP on a module that is not one. ``Qwen35VLPatchMerger``
    (examples/multimodal_dev/models/qwen35_vl/vision_encoder.py) is never
    zero-padded, so its tensors must keep strict global shape validation. Its
    ``TENorm`` is left out -- it needs Transformer Engine and the marking never
    looks past ``linear_fc1``/``linear_fc2``."""

    def __init__(self, config, merge_dim, out_hidden_size):
        super().__init__(config=config)
        self.linear_fc1 = ColumnParallelLinear(
            merge_dim,
            merge_dim,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
        )
        self.linear_fc2 = RowParallelLinear(
            merge_dim,
            out_hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
        )


class _TinyVisionEncoder(_TinyMLPEncoder):
    """A real vision MLP next to a patch-merger stand-in, so the round trips
    below exercise the shape-mismatch scoping on the real module path: only the
    MLP's three tensors may be marked, never the merger's same-named ones."""

    def __init__(self, config):
        super().__init__(config)
        # spatial_merge_size ** 2, and deliberately != ffn_hidden_size so the
        # merger's tensors could never be mistaken for padded MLP ones.
        self.merger = _PatchMergerStandIn(
            config, merge_dim=config.hidden_size * 2**2, out_hidden_size=config.hidden_size
        )


#: The only tensors --mdp-zero-pad-vision-ffn may load shape-mismatch-tolerantly.
_PADDED_VISION_MLP_KEYS = {
    "vision_model.module.mlp.linear_fc1.weight",
    "vision_model.module.mlp.linear_fc1.bias",
    "vision_model.module.mlp.linear_fc2.weight",
}


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
@pytest.mark.parametrize(
    "source_is_padded,target_is_padded",
    [(False, True), (True, True), (True, False)],
    ids=["official_to_padded", "padded_to_padded", "padded_to_official"],
)
def test_zero_padded_vision_ffn_checkpoint_directions(
    tmp_path_factory, encoder_pgs, source_is_padded, target_is_padded
):
    """Every save/load direction across the --mdp-zero-pad-vision-ffn width.

    Loading into a padded model must leave the real channels holding the
    checkpoint's weights and the padding channels at exactly zero (not leftover
    random init) -- the invariant zero_pad_vision_mlp_channels() establishes for
    training-from-scratch, now also true after a load. ``official_to_padded`` is
    the unpadded release checkpoint the flag exists for; ``padded_to_padded`` is
    an ordinary resume, since production passes the run's ``zero_pad_vision_ffn``
    value on save too, so the padded state dict is written with the
    shape-mismatch marking already active.

    ``padded_to_official`` -- a padded model written back out for an
    official-architecture consumer -- is not implemented, and fails loudly: the
    padded global shape is what reaches the checkpoint, and a consumer built at
    the official width does not pad, so it has no allow_shape_mismatch marking to
    fall back on and trips ``MCoreLoadPlanner._validate_global_shapes`` instead
    of silently loading a truncated FFN. Implementing that export path
    (truncating the guaranteed-zero padding channels) means updating this case on
    purpose.
    """
    real_ffn, padded_ffn = 6, 8

    # The saving run: either the "official" unpadded architecture or a padded
    # one built the way build_encoder_domain() builds it.
    source = _vision_mlp_ddp(encoder_pgs, padded_ffn if source_is_padded else real_ffn, seed=7)
    if source_is_padded:
        zero_pad_vision_mlp_channels(source.module, real_ffn_hidden_size=real_ffn)
    with torch.no_grad():
        # MCore initializes biases to zero, which would make the bias assertions
        # below pass without anything being transferred.
        source.module.mlp.linear_fc1.bias.data[:real_ffn].normal_()
    real_fc1 = source.module.mlp.linear_fc1.weight.data[:real_ffn, :].clone()
    real_fc1_bias = source.module.mlp.linear_fc1.bias.data[:real_ffn].clone()
    real_fc2 = source.module.mlp.linear_fc2.weight.data[:, :real_ffn].clone()

    target = _vision_mlp_ddp(encoder_pgs, padded_ffn if target_is_padded else real_ffn, seed=99)
    if target_is_padded:
        # Built the way build_encoder_domain() builds it: construct, zero-pad
        # once...
        zero_pad_vision_mlp_channels(target.module, real_ffn_hidden_size=real_ffn)
        # ...then put non-zero values back into the padding channels. Production
        # never reaches a load in this state, but left at their construction-time
        # zeros the padding assertions below would hold on the setup alone: they
        # would stay green even if the load stopped zero-filling and only copied
        # the checkpoint's overlapping (real) prefix. Same reason for the bias.
        _perturb_padding_channels(target.module.mlp, real_ffn)
        # Sanity: before loading, target's real channels do NOT already match
        # source (different seeds) -- the load below is what must make them match.
        assert not torch.equal(target.module.mlp.linear_fc1.weight.data[:real_ffn, :], real_fc1)

    directory = _shared_checkpoint_dir(tmp_path_factory, "mdp_ckpt_vision_ffn")
    save_state = encoder_sharded_state_dict(source, vision_ffn_may_be_padded=source_is_padded)
    dist_checkpointing.save(save_state, directory)
    torch.distributed.barrier()

    load_skeleton = encoder_sharded_state_dict(target, vision_ffn_may_be_padded=target_is_padded)
    # Scoping, on the real module path: the marking matches parameters by
    # identity through the DDP wrapper, so exactly the vision MLP's three
    # tensors are shape-mismatch tolerant and the merger's same-named
    # linear_fc1/linear_fc2 are not (marking by key suffix would catch them).
    marked = {
        key
        for key, value in load_skeleton.items()
        if isinstance(value, ShardedTensor) and value.allow_shape_mismatch
    }
    assert any(key.startswith("vision_model.module.merger.") for key in load_skeleton)
    assert marked == (_PADDED_VISION_MLP_KEYS if target_is_padded else set())
    if not target_is_padded:
        # Two exception types, because DCP wraps differently depending on the
        # world. _validate_global_shapes always raises MCore's
        # CheckpointingException, but under a real distributed world
        # _DistWrapper.reduce_scatter catches it and re-raises torch's
        # CheckpointException (torch/distributed/checkpoint/utils.py), so a
        # single-type assertion passes on one rank and fails on eight. Matching
        # on the message works for both: CheckpointException.__str__ renders
        # each wrapped failure with traceback.format_exception_only, so the
        # inner "Global shape mismatch" text survives. Note torch's class
        # derives from BaseException, not Exception, so it has to be named.
        with pytest.raises(
            (CheckpointingException, CheckpointException), match="Global shape mismatch"
        ):
            dist_checkpointing.load(load_skeleton, directory)
        return

    loaded = dist_checkpointing.load(load_skeleton, directory)
    load_encoder_state({ENCODER_STATE_KEY: loaded}, target, strict=True)

    with torch.no_grad():
        # Real channels: exactly the checkpoint's weights.
        assert torch.equal(target.module.mlp.linear_fc1.weight.data[:real_ffn, :], real_fc1)
        assert torch.equal(target.module.mlp.linear_fc1.bias.data[:real_ffn], real_fc1_bias)
        assert torch.equal(target.module.mlp.linear_fc2.weight.data[:, :real_ffn], real_fc2)
        # Padding channels: exactly zero, not the non-zero values the target
        # carried into the load. The fc1 bias is load-bearing here -- a non-zero
        # one would make the padding activations non-zero and start feeding them
        # gradient.
        _assert_padding_is_zero(target.module.mlp, real_ffn)
