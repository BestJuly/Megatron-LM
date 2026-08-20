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


def encoder_sharded_state_dict(encoder_ddp) -> Mapping:
    """The encoder's sharded model-weight state with WORLD replica metadata.

    The encoder is fully replicated: its replica domain is WORLD, not the
    decoder's DP-CP group — reusing the decoder metadata here would make
    every PP stage claim a distinct (wrong) replica coordinate.
    """
    return encoder_ddp.sharded_state_dict(
        prefix=ENCODER_STATE_PREFIX,
        metadata={"dp_cp_group": torch.distributed.group.WORLD},
    )


def add_encoder_state(state_dict: dict, encoder_ddp) -> dict:
    """Add the encoder weights to a torch_dist checkpoint state dict."""
    if ENCODER_STATE_KEY in state_dict:
        raise MdpCheckpointError(
            f"MDP: state dict already contains {ENCODER_STATE_KEY!r}; the encoder "
            "state must be contributed exactly once."
        )
    state_dict[ENCODER_STATE_KEY] = encoder_sharded_state_dict(encoder_ddp)
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
    prefix = ENCODER_STATE_PREFIX + _DDP_CHILD_PREFIX
    inner = {}
    for key, value in state_dict[ENCODER_STATE_KEY].items():
        if not key.startswith(prefix):
            raise MdpCheckpointError(
                f"MDP: encoder state key {key!r} does not start with {prefix!r}; the "
                "encoder save and load skeletons have drifted apart."
            )
        inner[key[len(prefix) :]] = value
    encoder_ddp.load_state_dict(inner, strict=strict)


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
                "no --non-persistent-ckpt-type (non-persistent checkpoints are "
                "unsupported)"
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
