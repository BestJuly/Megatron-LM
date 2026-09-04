# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP checkpoint facade: synchronous global torch_dist save/load.

Logical keys: ``language_model.*`` stays with the decoder chunks (PP/VPP
shards, decoder DP-CP replica metadata, produced by the native checkpoint
path); ``vision_model.*`` comes from the encoder DDP with **encoder WORLD**
replica metadata — one logical copy replicated on every rank. Plans, leaves,
forward handles, autograd graphs, and communication handles are never
persisted.

Optimizer, LR-scheduler, and RNG state round-trip through the native paths, so
a resume is exact rather than weight-only; the composite optimizer keeps the
two sharding domains apart with a fixed encoder key (see
:mod:`megatron.core.mdp.optimizer`). What remains rejected are the *execution
modes* that cannot work here: the fully-parallel save/load wrappers (they
reshard every child over a single DP-CP group, which is wrong for the encoder's
WORLD domain), asynchronous and non-persistent saves, and constant-structure
caching (MDP rebuilds its plan-derived structures every iteration).

``--mdp-zero-pad-vision-ffn`` adds one asymmetry to the weight direction: a
padded model loads an official (unpadded) vision FFN checkpoint and its own
padded checkpoints round-trip, but writing a padded model back out at the
official width is not implemented (see
``_mark_vision_ffn_padding_shape_mismatch``). An official checkpoint carries
no MDP optimizer state, so that load is a weights-only start
(``--no-load-optim``/``--finetune``); the missing optimizer key fails loudly
otherwise.
"""

from typing import Mapping

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.mdp.errors import MdpCheckpointError

#: The state-dict key the encoder state travels under. It sits next to the
#: native ``model``/``model<N>`` keys so the sharded save/load skeleton is
#: symmetric between save and load.
ENCODER_STATE_KEY = "mdp_vision_model"

#: The logical prefix the encoder weights are published under, matching the
#: keys a native (non-MDP) multimodal checkpoint carries.
ENCODER_STATE_PREFIX = "vision_model."

#: ``DistributedDataParallel`` contributes one ``module.`` level below the
#: logical prefix, and ``DDP.load_state_dict`` delegates straight to that
#: child — so both levels come off before the state is handed back.
_DDP_CHILD_PREFIX = "module."

#: TransformerEngine's per-module opaque state. It appears in ``state_dict()``
#: but not necessarily in the sharded state dict, and holds no weights. This
#: mirrors the producing side's ``extra_state_suffix`` default in
#: :func:`megatron.core.transformer.utils.make_sharded_tensors_for_checkpoint`;
#: keep the two in step if core ever renames it.
_EXTRA_STATE_SUFFIX = "_extra_state"


def _padded_vision_ffn_param_ids(encoder_ddp) -> set:
    """``id()`` of every parameter zero_pad_vision_mlp_channels() (encoder.py) may pad.

    Scoped by exactly the module rule that function uses: every
    :class:`~megatron.core.transformer.mlp.MLP` in the encoder, and inside it
    ``linear_fc1``'s weight and bias plus ``linear_fc2``'s weight. Modules that
    merely reuse the ``linear_fc1``/``linear_fc2`` names without ever being
    padded -- the vision-to-language patch merger, for one -- are not MLPs, and
    must keep the strict global shape validation they have always had. A layer
    already at the real width (which that function skips) stays in the set: its
    shapes match the checkpoint's, so tolerating a mismatch changes nothing.

    Parameters are identified by object identity rather than by state-dict key
    because MCore's sharded state dict carries the parameter itself as
    ``ShardedTensor.data`` (``make_sharded_tensor_for_checkpoint`` hands the
    tensor straight to ``ShardedTensor.from_rank_offsets``). Identity survives
    the DDP and Float16Module wrappers, whose key prefixes do not agree:
    ``Float16Module.sharded_state_dict`` forwards its prefix unchanged while
    DDP contributes a ``module.`` level.
    """
    from megatron.core.transformer.mlp import MLP

    param_ids = set()
    for module in encoder_ddp.modules():
        if not isinstance(module, MLP):
            continue
        for param in (module.linear_fc1.weight, module.linear_fc1.bias, module.linear_fc2.weight):
            if param is not None:
                param_ids.add(id(param))
    return param_ids


def _mark_vision_ffn_padding_shape_mismatch(state_dict: Mapping, encoder_ddp) -> None:
    """Let the padded vision MLP tensors load from a smaller (official,
    unpadded) checkpoint into a --mdp-zero-pad-vision-ffn-padded model.

    Sets ``allow_shape_mismatch = True`` on the ShardedTensor of every
    parameter zero_pad_vision_mlp_channels() may have zero-padded (see
    ``_padded_vision_ffn_param_ids``), so MCore's torch_dist strategy
    zero-initializes the target buffer, then copies only the overlapping (real,
    unpadded) prefix from the checkpoint (see ``strategies/torch.py``'s
    ``_mcore_to_dcp_compatible_tensor``: "if allow_shape_mismatch is True, the
    data is initialized with zeros prior to loading"). Combined with
    zero_pad_vision_mlp_channels() already establishing zero-padding as a
    training-time invariant, this makes the two directions consistent: a
    checkpoint saved from a padded model round-trips exactly, and a checkpoint
    saved from the real (unpadded) official architecture loads cleanly into a
    padded model with the new channels zero-initialized -- exactly the
    invariant zero_pad_vision_mlp_channels() already establishes at
    construction time for training-from-scratch. Symmetric with
    LanguageModule.sharded_state_dict()'s vocab-padding handling, which uses
    the same mechanism.

    The third direction -- writing a padded model back out at the official
    (unpadded) width -- is not implemented: the padded global shape is what
    reaches the checkpoint, so an official-architecture consumer trips
    ``MCoreLoadPlanner._validate_global_shapes``. Truncating the padding
    channels is the only thing such an exporter would have to do, since they
    are guaranteed zero.
    """
    padded_param_ids = _padded_vision_ffn_param_ids(encoder_ddp)
    marked = 0
    for value in state_dict.values():
        if isinstance(value, ShardedTensor) and id(value.data) in padded_param_ids:
            value.allow_shape_mismatch = True
            marked += 1
    if marked == 0:
        raise MdpCheckpointError(
            "MDP: zero_pad_vision_ffn=True but the encoder's sharded state dict "
            "carries no vision MLP linear_fc1/linear_fc2 parameter violates: at "
            "least one padded tensor to mark. The padded parameters must reach "
            "the checkpoint as plain ShardedTensors for an official (unpadded) "
            "checkpoint to be able to zero-fill their padding channels."
        )


def encoder_sharded_state_dict(encoder_ddp, *, vision_ffn_may_be_padded: bool = False) -> Mapping:
    """The encoder's sharded model-weight state with WORLD replica metadata.

    The encoder is fully replicated: its replica domain is WORLD, not the
    decoder's DP-CP group — reusing the decoder metadata here would make
    every PP stage claim a distinct (wrong) replica coordinate.

    ``vision_ffn_may_be_padded`` should be the run's
    ``MdpConfig.zero_pad_vision_ffn`` value: when true, the vision MLP tensors
    that zero_pad_vision_mlp_channels() padded are marked
    shape-mismatch-tolerant (see ``_mark_vision_ffn_padding_shape_mismatch``)
    so an official (unpadded) checkpoint loads into this padded model. Left
    false by default so an unrelated real shape mismatch (a genuine config
    error) still fails loudly instead of being silently zero-filled.
    """
    state_dict = encoder_ddp.sharded_state_dict(
        prefix=ENCODER_STATE_PREFIX, metadata={"dp_cp_group": torch.distributed.group.WORLD}
    )
    if vision_ffn_may_be_padded:
        _mark_vision_ffn_padding_shape_mismatch(state_dict, encoder_ddp)
    return state_dict


def add_encoder_state(
    state_dict: dict, encoder_ddp, *, vision_ffn_may_be_padded: bool = False
) -> dict:
    """Add the encoder weights to a torch_dist checkpoint state dict.

    See :func:`encoder_sharded_state_dict` for ``vision_ffn_may_be_padded``.
    """
    if ENCODER_STATE_KEY in state_dict:
        raise MdpCheckpointError(
            f"MDP: state dict already contains {ENCODER_STATE_KEY!r}; the encoder "
            "state must be contributed exactly once."
        )
    state_dict[ENCODER_STATE_KEY] = encoder_sharded_state_dict(
        encoder_ddp, vision_ffn_may_be_padded=vision_ffn_may_be_padded
    )
    return state_dict


def assert_zero_pad_vision_ffn_resume(args, checkpoint_args) -> None:
    """Reject a resume that would overwrite the vision FFN's zero-padding.

    zero_pad_vision_mlp_channels() (encoder.py) zeroes the alignment-padding
    channels once, at construction time, and the sharded load then writes the
    checkpoint's weights straight back into those same parameters. The
    invariant therefore survives a load only if the checkpoint respects it.
    Two cases do:

    * the checkpoint has the real (official) vision FFN width -- no
      ``--encoder-ffn-hidden-size`` in its args -- so ``allow_shape_mismatch``
      zero-fills the padding channels and copies only the real prefix (see
      ``_mark_vision_ffn_padding_shape_mismatch``);
    * the checkpoint was written by a ``--mdp-zero-pad-vision-ffn`` run *at the
      same padded width*, so its padding channels are zero and stay zero. A
      different padded width is rejected: the tensors are marked
      ``allow_shape_mismatch`` on both sides, so a wider checkpoint would be
      silently truncated into (or a narrower one zero-extended into) a model
      whose real/padding boundary sits elsewhere.

    A checkpoint written at a widened FFN *without* the flag does not: its
    padding channels were randomly initialized and trained as ordinary weights,
    the load restores them verbatim, they take gradient from then on, and the
    model is no longer convertible back to the official architecture -- the
    flag's only reason to exist.

    ``checkpoint_args`` is the args namespace stored in the checkpoint. Both
    ``--mdp-zero-pad-vision-ffn`` and ``--encoder-ffn-hidden-size`` are already
    persisted there, so telling the cases apart needs no extra checkpoint state.
    A checkpoint whose args predate these flags, or that was written outside
    Megatron and carries no args at all, reports neither and is treated as the
    official width: that is the only width such a checkpoint can have been
    written at, since the widened FFN only ever existed behind these flags.
    """
    if not getattr(args, "mdp_zero_pad_vision_ffn", False):
        return
    checkpoint_ffn = getattr(checkpoint_args, "encoder_ffn_hidden_size", None)
    if getattr(checkpoint_args, "mdp_zero_pad_vision_ffn", False):
        live_ffn = getattr(args, "encoder_ffn_hidden_size", None)
        if checkpoint_ffn != live_ffn:
            raise MdpCheckpointError(
                "MDP: --mdp-zero-pad-vision-ffn resuming from a zero-padded checkpoint "
                f"written at --encoder-ffn-hidden-size {checkpoint_ffn} into a model "
                f"built at --encoder-ffn-hidden-size {live_ffn} violates: both padded "
                "widths must be equal. The vision MLP tensors are shape-mismatch "
                "tolerant on both sides, so the load would silently truncate or "
                "zero-extend them across a different real/padding boundary instead of "
                "failing. Build the model at the checkpoint's width."
            )
        return
    if checkpoint_ffn is None:
        return
    raise MdpCheckpointError(
        "MDP: --mdp-zero-pad-vision-ffn resuming from a checkpoint written with "
        f"--encoder-ffn-hidden-size {checkpoint_args.encoder_ffn_hidden_size} but "
        "without --mdp-zero-pad-vision-ffn violates: the checkpoint's vision FFN "
        "padding channels must be zero. That run trained them as ordinary weights, "
        "so loading them restores non-zero padding channels that keep taking "
        "gradient, and the model can no longer be converted back to the official "
        "(unpadded) architecture. Resume without --mdp-zero-pad-vision-ffn, or "
        "load a checkpoint that has the official vision FFN width."
    )


def load_encoder_state(state_dict: Mapping, encoder_ddp, *, strict: bool = True) -> None:
    """Restore the encoder weights from a loaded torch_dist checkpoint.

    ``generate_state_dict`` builds both the save state and the load skeleton,
    so ``dist_checkpointing.load`` has already read the encoder tensors back by
    the time this runs. Nothing else copies them into the encoder module — the
    encoder lives outside the decoder model-chunk list that
    ``load_checkpoint`` iterates — so this is the missing half of the round
    trip.
    """
    if ENCODER_STATE_KEY not in state_dict:
        raise MdpCheckpointError(
            f"MDP: the checkpoint has no {ENCODER_STATE_KEY!r} entry; it was not "
            "written by an MDP run and carries no vision-encoder weights."
        )
    prefix = ENCODER_STATE_PREFIX + _DDP_CHILD_PREFIX
    inner = {}
    for key, value in state_dict[ENCODER_STATE_KEY].items():
        if not key.startswith(prefix):
            raise MdpCheckpointError(
                f"MDP: encoder state key {key!r} does not start with {prefix!r}; the "
                "encoder save and load skeletons have drifted apart."
            )
        inner[key[len(prefix) :]] = value

    # ``strict`` arrives from ``load_checkpoint``'s own parameter, and
    # ``load_state_dict(strict=False)`` merely reports missing keys -- and
    # ``_BaseDataParallel.load_state_dict`` drops even that report, returning
    # ``None``. The encoder is replicated and always written whole, so there is
    # no "empty stage" case to tolerate here the way there is for a decoder
    # chunk: an absent key means the state did not round-trip, and skipping it
    # would resume from the random initialization. Check before delegating.
    #
    # TransformerEngine's ``_extra_state`` entries are exempt: they are present
    # in ``state_dict()`` but the sharded state dict legitimately omits the ones
    # whose modules contribute no persistent extra state, and they carry no
    # weights, so their absence cannot leave a randomly initialized encoder.
    expected = {key for key in encoder_ddp.state_dict() if not key.endswith(_EXTRA_STATE_SUFFIX)}
    missing = sorted(expected - set(inner))
    if missing:
        raise MdpCheckpointError(
            f"MDP: the checkpoint is missing {len(missing)} encoder tensor(s), "
            f"first {missing[:5]}; the encoder would stay randomly initialized. "
            "A non-strict --dist-ckpt-strictness drops the keys the checkpoint "
            "cannot supply, which is how they get here."
        )

    # The check above already enforces the property that matters, so a strict
    # failure below can only come from an ``_extra_state`` or unexpected-key
    # mismatch -- never from an absent weight. Retry non-strictly for those, the
    # way ``load_model_state_dict`` does for every decoder chunk in
    # ``megatron.training.checkpointing``: TransformerEngine changes which
    # ``_extra_state`` entries it publishes between versions, and the encoder
    # must not be stricter about that than the decoder it trains beside.
    try:
        encoder_ddp.load_state_dict(inner, strict=strict)
    except RuntimeError:
        if not strict:
            raise
        encoder_ddp.load_state_dict(inner, strict=False)


def assert_supported_checkpoint_config(args) -> None:
    """Reject checkpoint configurations MDP cannot honor, at startup.

    Optimizer, LR-scheduler, and RNG state round-trip normally; what is left is
    the set of execution modes that are structurally incompatible with the
    two-sharding-domain checkpoint (see the module docstring).
    """
    problems = []
    save_or_load = (
        getattr(args, "save", None) is not None or getattr(args, "load", None) is not None
    )
    if save_or_load:
        # Design doc section 12: only the synchronous, persistent, global
        # torch_dist mode is supported. Asynchronous, non-persistent, and
        # constant-structure caching modes are rejected at startup (scoped to
        # save/load so checkpoint-free runs are unaffected by defaults).
        if getattr(args, "async_save", False):
            problems.append("no --async-save (asynchronous save is unsupported)")
        if getattr(args, "non_persistent_ckpt_type", None) is not None:
            problems.append(
                "no --non-persistent-ckpt-type (non-persistent checkpoints are " "unsupported)"
            )
        if getattr(args, "ckpt_assume_constant_structure", False):
            problems.append(
                "no --ckpt-assume-constant-structure (MDP's plan-derived "
                "structures change per iteration; a cached structure goes stale)"
            )
    if getattr(args, "save", None) is not None:
        # Megatron defaults ckpt_fully_parallel_save=True; the fully-parallel
        # path shards across one DP-CP group for every child, which is wrong
        # for the encoder's WORLD replica domain. Scoped to save/load so runs
        # that never touch a checkpoint are not rejected by the default.
        if getattr(args, "ckpt_fully_parallel_save", False):
            problems.append("--no-ckpt-fully-parallel-save")
    if getattr(args, "load", None) is not None:
        if getattr(args, "ckpt_fully_parallel_load", False):
            problems.append("--no-ckpt-fully-parallel-load (or omit --ckpt-fully-parallel-load)")
    if problems:
        raise MdpCheckpointError(
            "MDP: the checkpoint facade supports the synchronous, persistent, global "
            "torch_dist mode only; run with " + " ".join(problems) + "."
        )
