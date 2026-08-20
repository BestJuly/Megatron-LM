# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP composite optimizer: ``[decoder_dense, decoder_expert?, encoder]``.

``ChainedOptimizer`` already computes the combined ``sqrt(sum(member_norm^2))``
and clips every member with the shared total norm, but its ``prepare_grads``
only ORs the members' overflow flags **locally**. The members' grad-stats
groups differ (decoder dense over DP-CP, decoder expert over expert-DP, the
encoder over WORLD), so at PP>1 their local verdicts can genuinely disagree,
and each member updates its own scaler from its member-local flag. Without a
WORLD union before any scaler update, ranks that saw no overflow would grow
their scale while detecting ranks halved theirs — permanently diverging loss
scales and silently corrupted unscaling. :class:`MdpChainedOptimizer` makes
the union explicit and atomic.

Checkpointing needs its own overrides for the same reason. The inherited
``sharded_state_dict`` isolates members with a ``chained_{idx}`` prefix, and
the encoder's index is the decoder member count — which is not guaranteed
equal on every rank (a PP stage without expert parameters contributes one
member instead of two), so the same logical tensor would be written under a
different key per rank. Isolation must come from a **fixed** key instead,
which is also what keeps the encoder's WORLD-sharded ZeRO-1 state from
colliding with the PP-stage-0 decoder optimizer: both compute
``data_parallel_group_idx == 0`` (the encoder's ``mp`` group is ``None`` and
``get_pg_rank(None) == 0``), so their
``optimizer.distributed.dp_group_idx_0.*`` keys are identical without a
prefix. The inherited ``load_state_dict`` additionally pairs members by
position after sorting the keys, which no longer reproduces the member order.
"""

import logging
from contextlib import contextmanager
from typing import List, Optional

import torch

from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding
from megatron.core.mdp.errors import MdpCheckpointError, MdpConfigurationError
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer

logger = logging.getLogger(__name__)

#: Fixed checkpoint key and sharding prefix for the encoder member. Published
#: keys are a compatibility contract — do not rename.
ENCODER_MEMBER_KEY = "mdp_encoder_optimizer"


class MdpChainedOptimizer(ChainedOptimizer):
    """``ChainedOptimizer`` with an explicit WORLD overflow union.

    Args:
        chained_optimizers: the flat member list.
        encoder_member_index: position of the encoder member, or ``None`` when
            the composite has no encoder domain (the inherited, decoder-only
            checkpoint behavior is then used verbatim).
    """

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Unify the overflow flag over WORLD **before** any scaler update.

        Member scaler updates are suppressed while the members unscale and
        check; the flag is MAX-reduced over WORLD; every scaler is then updated
        with the one global verdict. Returns True when the step must be
        skipped (ChainedOptimizer's convention).
        """
        scalers = self._member_grad_scalers()
        with _suppressed_scaler_updates(scalers):
            found_inf = ChainedOptimizer.prepare_grads(self)

        # One flag tensor per iteration, for the all-reduce only; the scalers
        # take the unified Python bool (DynamicGradScaler.update branches on
        # truthiness, so a GPU tensor argument would cost one implicit host
        # sync per scaler — the former per-scaler tensor did exactly that).
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            inf_flag = torch.tensor(
                [1.0 if found_inf else 0.0], dtype=torch.float32, device=_flag_device()
            )
            torch.distributed.all_reduce(inf_flag, op=torch.distributed.ReduceOp.MAX)
            unified = bool(inf_flag.item() > 0.0)
            if unified != found_inf:
                logger.debug(
                    "MDP: overflow flag unified over WORLD: local=%s global=%s", found_inf, unified
                )
            found_inf = unified

        for scaler in scalers:
            scaler.update(found_inf)
        return found_inf

    def _member_grad_scalers(self) -> List:
        """Every member's grad scaler, de-duplicated by identity."""
        scalers, seen = [], set()
        for member in self.chained_optimizers:
            scaler = getattr(member, "grad_scaler", None)
            if scaler is not None and id(scaler) not in seen:
                seen.add(id(scaler))
                scalers.append(scaler)
        return scalers

    #: Check member loss scales for divergence every N calls: torch.equal on
    #: GPU scalars is an implicit host sync, and after prepare_grads every
    #: scaler is driven by the same global verdict, so the invariant holds by
    #: construction between samples.
    LOSS_SCALE_CHECK_INTERVAL = 50

    def __init__(
        self,
        chained_optimizers: List[MegatronOptimizer],
        encoder_member_index: Optional[int] = None,
    ):
        super().__init__(chained_optimizers)
        self._loss_scale_calls = 0
        if encoder_member_index is not None and not 0 <= encoder_member_index < len(
            chained_optimizers
        ):
            raise MdpCheckpointError(
                f"MDP: encoder member index {encoder_member_index} is out of range for "
                f"{len(chained_optimizers)} members."
            )
        self._encoder_member_index = encoder_member_index

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _member_keys(self) -> List[str]:
        """The per-member checkpoint key, in member order.

        Decoder members keep ``chained_{idx}`` so their keys are unchanged from
        the native ``ChainedOptimizer``; the encoder member gets the fixed key.
        """
        return [
            ENCODER_MEMBER_KEY if index == self._encoder_member_index else f"chained_{index}"
            for index in range(len(self.chained_optimizers))
        ]

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        """Per-member sharded state with a fixed, rank-independent encoder key.

        One departure from the inherited implementation: the encoder member is
        keyed and prefixed by :data:`ENCODER_MEMBER_KEY` instead of
        ``chained_{idx}``, and it is prefixed regardless of the sharding type —
        the prefix *is* the isolation between the two sharding domains, not a
        format detail. Decoder members keep the inherited prefixing rule so a
        decoder-only checkpoint stays byte-compatible.
        """
        if self._encoder_member_index is None:
            return super().sharded_state_dict(model_sharded_state_dict, is_loading, **kwargs)

        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

        metadata = kwargs.get("metadata") or {}
        decoder_needs_prefix = (
            "distrib_optim_sharding_type" in metadata
            and metadata["distrib_optim_sharding_type"]
            not in DistributedOptimizer.checkpoint_fully_reshardable_formats
        ) or not metadata.get("chained_optim_avoid_prefix", False)

        self._synchronize_steps()
        sharded_state_dict = {}
        for index, (member, key) in enumerate(zip(self.chained_optimizers, self._member_keys())):
            member_state = member.sharded_state_dict(model_sharded_state_dict, is_loading, **kwargs)
            if index == self._encoder_member_index or decoder_needs_prefix:
                add_prefix_for_sharding(member_state, f"{key}.")
            sharded_state_dict[key] = member_state
        return sharded_state_dict

    def load_state_dict(self, state_dict):
        """Pair members with their state by key, never by position.

        ``ChainedOptimizer.load_state_dict`` zips the member list against
        ``sorted(state_dict.items())``; with the encoder keyed separately that
        ordering is no longer guaranteed to be the member ordering, and a
        silent mis-pairing would load decoder moments into encoder parameters.
        """
        if self._encoder_member_index is None:
            super().load_state_dict(state_dict)
            return
        member_keys = self._member_keys()
        missing = [key for key in member_keys if key not in state_dict]
        if missing:
            raise MdpCheckpointError(
                f"MDP: the checkpoint optimizer state is missing member keys {missing}; "
                f"it holds {sorted(map(str, state_dict))}."
            )
        for member, key in zip(self.chained_optimizers, member_keys):
            member.load_state_dict(state_dict[key])
        self._synchronize_steps()

    def get_loss_scale(self) -> torch.Tensor:
        """The shared loss scale, sample-asserting the members have not diverged.

        The training loop scales the loss by member 0's scale while each
        member unscales with its own, so equal scales are a correctness
        precondition — and after :meth:`prepare_grads` every scaler is driven
        by the same global verdict.
        """
        if self._loss_scale_calls % self.LOSS_SCALE_CHECK_INTERVAL == 0:
            scales = [s.scale for s in self._member_grad_scalers()]
            if len(scales) > 1:
                first = scales[0]
                assert all(torch.equal(first, s) for s in scales[1:]), (
                    "MDP composite optimizer members hold different loss scales "
                    f"({[float(s) for s in scales]}); gradients would be unscaled by "
                    "different factors per domain"
                )
        self._loss_scale_calls += 1
        return super().get_loss_scale()


def _flag_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@contextmanager
def _suppressed_scaler_updates(scalers):
    """Neutralize ``GradScaler.update`` for the duration of the block.

    Members update their scaler inside ``prepare_grads`` with member-local
    flags; MDP substitutes the WORLD-unified flag afterwards, and the only
    seam is to stop the member update and re-run it with the global verdict.
    """
    saved = [(scaler, scaler.update) for scaler in scalers]
    for scaler, _original in saved:
        scaler.update = lambda _found_inf: None
    try:
        yield
    finally:
        for scaler, original in saved:
            scaler.update = original


def build_mdp_composite_optimizer(
    decoder_optimizer: MegatronOptimizer, encoder_optimizer: MegatronOptimizer
) -> MdpChainedOptimizer:
    """Chain the domains as the flat ``[dec_dense, dec_expert?, encoder]``.

    The decoder side may already be a ChainedOptimizer (dense + expert); it is
    flattened rather than nested so the member order is the checkpoint-visible
    sequence the design specifies. ``ChainedOptimizer.__init__`` asserts equal
    member configs, so both sides must be built from the same
    ``OptimizerConfig``.
    """
    members: List[MegatronOptimizer] = list(_flatten(decoder_optimizer))
    encoder_members = list(_flatten(encoder_optimizer))
    if len(encoder_members) != 1:
        raise MdpConfigurationError(
            f"MDP: the encoder side flattened to {len(encoder_members)} optimizer members, "
            "but the checkpoint contract gives the single encoder domain one fixed key "
            f"({ENCODER_MEMBER_KEY!r}). One replicated encoder means one member."
        )
    encoder_member_index = len(members)
    members.extend(encoder_members)
    optimizer = MdpChainedOptimizer(members, encoder_member_index=encoder_member_index)
    logger.info(
        "MDP: composite optimizer with %d members: %s; checkpoint keys %s",
        len(members),
        [type(member).__name__ for member in members],
        optimizer._member_keys(),
    )
    return optimizer


def _flatten(optimizer: MegatronOptimizer):
    if isinstance(optimizer, ChainedOptimizer):
        for member in optimizer.chained_optimizers:
            yield from _flatten(member)
    else:
        yield optimizer
