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
"""

import logging
from contextlib import contextmanager
from typing import List

import torch

from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer

logger = logging.getLogger(__name__)


class MdpChainedOptimizer(ChainedOptimizer):
    """``ChainedOptimizer`` with an explicit WORLD overflow union."""

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

        found_inf = self._unify_over_world(found_inf)

        for scaler in scalers:
            scaler.update(
                torch.tensor(
                    [1.0 if found_inf else 0.0], dtype=torch.float, device=_flag_device()
                )
            )
        return found_inf

    @staticmethod
    def _unify_over_world(found_inf: bool) -> bool:
        if not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
            return found_inf
        flag = torch.tensor(
            [1.0 if found_inf else 0.0], dtype=torch.float32, device=_flag_device()
        )
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        unified = bool(flag.item() > 0.0)
        if unified != found_inf:
            logger.debug(
                "MDP: overflow flag unified over WORLD: local=%s global=%s",
                found_inf,
                unified,
            )
        return unified

    def _member_grad_scalers(self) -> List:
        """Every member's grad scaler, de-duplicated by identity."""
        scalers, seen = [], set()
        for member in self.chained_optimizers:
            scaler = getattr(member, "grad_scaler", None)
            if scaler is not None and id(scaler) not in seen:
                seen.add(id(scaler))
                scalers.append(scaler)
        return scalers

    def get_loss_scale(self) -> torch.Tensor:
        """The shared loss scale, asserting the members have not diverged.

        The training loop scales the loss by member 0's scale while each
        member unscales with its own, so equal scales are a correctness
        precondition — and after :meth:`prepare_grads` every scaler is driven
        by the same global verdict.
        """
        scales = [s.scale for s in self._member_grad_scalers()]
        if len(scales) > 1:
            first = scales[0]
            assert all(torch.equal(first, s) for s in scales[1:]), (
                "MDP composite optimizer members hold different loss scales "
                f"({[float(s) for s in scales]}); gradients would be unscaled by "
                "different factors per domain"
            )
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
    members.extend(_flatten(encoder_optimizer))
    optimizer = MdpChainedOptimizer(members)
    logger.info(
        "MDP: composite optimizer with %d members: %s",
        len(members),
        [type(member).__name__ for member in members],
    )
    return optimizer


def _flatten(optimizer: MegatronOptimizer):
    if isinstance(optimizer, ChainedOptimizer):
        for member in optimizer.chained_optimizers:
            yield from _flatten(member)
    else:
        yield optimizer
