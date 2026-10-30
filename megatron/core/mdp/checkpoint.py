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
"""

from typing import Mapping

import torch

from megatron.core.mdp.errors import MdpCheckpointError
from megatron.core.transformer.utils import sharded_state_dict_default

#: The state-dict key the encoder state travels under. It sits next to the
#: native ``model``/``model<N>`` keys so the sharded save/load skeleton is
#: symmetric between save and load.
ENCODER_STATE_KEY = "mdp_vision_model"

#: The logical prefix the encoder weights are published under, matching the
#: keys a native (non-MDP) multimodal checkpoint carries.
ENCODER_STATE_PREFIX = "vision_model."

#: ``DistributedDataParallel`` contributes one ``module.`` level below its
#: owner. Checkpoints written before the encoder was published from the
#: wrapped module carry it (``vision_model.module.<param>``); they still load.
_DDP_CHILD_PREFIX = "module."

#: TransformerEngine's per-module opaque state. It appears in ``state_dict()``
#: but not necessarily in the sharded state dict, and holds no weights. This
#: mirrors the producing side's ``extra_state_suffix`` default in
#: :func:`megatron.core.transformer.utils.make_sharded_tensors_for_checkpoint`;
#: keep the two in step if core ever renames it.
_EXTRA_STATE_SUFFIX = "_extra_state"


def encoder_sharded_state_dict(encoder_ddp, *, legacy_ddp_keys: bool = False) -> Mapping:
    """The encoder's sharded model-weight state with WORLD replica metadata.

    The encoder is fully replicated: its replica domain is WORLD, not the
    decoder's DP-CP group — reusing the decoder metadata here would make
    every PP stage claim a distinct (wrong) replica coordinate.

    The tensors are published from the wrapped module, so their keys are
    ``vision_model.<param>`` — the keys of a native (non-MDP) multimodal
    checkpoint, which makes native and MDP checkpoints interchangeable.
    ``legacy_ddp_keys`` builds the load skeleton of a checkpoint written with
    the DDP level in its keys (see :func:`uses_legacy_encoder_keys`).
    """
    metadata = {"dp_cp_group": torch.distributed.group.WORLD}
    if legacy_ddp_keys:
        return encoder_ddp.sharded_state_dict(prefix=ENCODER_STATE_PREFIX, metadata=metadata)
    # What the DDP's own (MegatronModule) recursion does for its ``module`` child, minus the
    # ``module.`` key level: same tensors, same TP group, same replica metadata.
    return sharded_state_dict_default(
        encoder_ddp.module,
        ENCODER_STATE_PREFIX,
        metadata=metadata,
        tp_group=getattr(encoder_ddp, "tp_group", None),
    )


def uses_legacy_encoder_keys(checkpoint_tensor_keys) -> bool:
    """Whether a checkpoint stores the encoder under ``vision_model.module.*``."""
    legacy = ENCODER_STATE_PREFIX + _DDP_CHILD_PREFIX
    return any(key.startswith(legacy) for key in checkpoint_tensor_keys)


def add_encoder_state(state_dict: dict, encoder_ddp, *, legacy_ddp_keys: bool = False) -> dict:
    """Add the encoder weights to a torch_dist checkpoint state dict."""
    if ENCODER_STATE_KEY in state_dict:
        raise MdpCheckpointError(
            f"MDP: state dict already contains {ENCODER_STATE_KEY!r}; the encoder "
            "state must be contributed exactly once."
        )
    state_dict[ENCODER_STATE_KEY] = encoder_sharded_state_dict(
        encoder_ddp, legacy_ddp_keys=legacy_ddp_keys
    )
    return state_dict


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
    # Current skeletons key the tensors ``vision_model.<param>``; skeletons built
    # for a legacy checkpoint keep the DDP level, ``vision_model.module.<param>``.
    legacy = ENCODER_STATE_PREFIX + _DDP_CHILD_PREFIX
    inner = {}
    for key, value in state_dict[ENCODER_STATE_KEY].items():
        if key.startswith(legacy):
            inner[key[len(legacy) :]] = value
        elif key.startswith(ENCODER_STATE_PREFIX):
            inner[key[len(ENCODER_STATE_PREFIX) :]] = value
        else:
            raise MdpCheckpointError(
                f"MDP: encoder state key {key!r} does not start with "
                f"{ENCODER_STATE_PREFIX!r}; the encoder save and load skeletons have "
                "drifted apart."
            )

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
    load_requested = (
        getattr(args, "load", None) is not None
        or getattr(args, "pretrained_checkpoint", None) is not None
    )
    save_or_load = getattr(args, "save", None) is not None or load_requested
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
    if load_requested:
        if getattr(args, "ckpt_fully_parallel_load", False):
            problems.append("--no-ckpt-fully-parallel-load (or omit --ckpt-fully-parallel-load)")
    if problems:
        raise MdpCheckpointError(
            "MDP: the checkpoint facade supports the synchronous, persistent, global "
            "torch_dist mode only; run with " + " ".join(problems) + "."
        )
