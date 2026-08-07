# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP checkpoint facade: synchronous global torch_dist weight-only save/load.

Logical keys: ``language_model.*`` stays with the decoder chunks (PP/VPP
shards, decoder DP-CP replica metadata, produced by the native checkpoint
path); ``vision_model.*`` comes from the encoder DDP with **encoder WORLD**
replica metadata — one logical copy replicated on every rank. Plans, leaves,
forward handles, autograd graphs, and communication handles are never
persisted; optimizer, LR-scheduler, and RNG state are excluded (weight-only
restart, not exact resume).
"""

from typing import Mapping

import torch

from megatron.core.mdp.errors import MdpCheckpointError

#: The state-dict key the encoder state travels under. It sits next to the
#: native ``model``/``model<N>`` keys so the sharded save/load skeleton is
#: symmetric between save and load.
ENCODER_STATE_KEY = "mdp_vision_model"


def encoder_sharded_state_dict(encoder_ddp) -> Mapping:
    """The encoder's sharded model-weight state with WORLD replica metadata.

    The encoder is fully replicated: its replica domain is WORLD, not the
    decoder's DP-CP group — reusing the decoder metadata here would make
    every PP stage claim a distinct (wrong) replica coordinate.
    """
    return encoder_ddp.sharded_state_dict(
        prefix="vision_model.",
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


def assert_weight_only_checkpoint(args) -> None:
    """Reject non-weight-only checkpoint configurations at startup.

    MDP persists model weights only: optimizer, LR-scheduler, and RNG state
    start fresh after load. The native flags express exactly that.
    """
    problems = []
    if getattr(args, "save", None) is not None:
        if not getattr(args, "no_save_optim", False):
            problems.append("--no-save-optim")
        if not getattr(args, "no_save_rng", False):
            problems.append("--no-save-rng")
    if getattr(args, "load", None) is not None:
        if not getattr(args, "no_load_optim", False):
            problems.append("--no-load-optim")
        if not getattr(args, "no_load_rng", False):
            problems.append("--no-load-rng")
    if problems:
        raise MdpCheckpointError(
            "MDP: the checkpoint facade is a weight-only restart contract; run with "
            + " ".join(problems)
            + ". Optimizer, LR-scheduler, and RNG state are not persisted and start "
            "fresh after load."
        )
