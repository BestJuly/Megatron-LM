# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDP encoder domain: replicated encoder DDP over WORLD, ZeRO-1, and gradient
finalization.

The encoder is fully replicated on every rank and reduced once over WORLD with
prescale 1 (``calculate_per_token_loss=True`` makes the DDP gradient scaling
factor 1.0). The distributed optimizer shards its state over the same WORLD
domain. The encoder never enters the decoder schedule model list.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.mdp.config import (
    MdpConfig,
    apply_encoder_ffn_config,
    apply_encoder_fp8_config,
    apply_encoder_recompute_config,
    validate_effective_vision_config,
)
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.groups import MdpProcessGroups
from megatron.core.mdp.protocols import MdpModelAdapter
from megatron.core.mdp.rank_mapping import MdpRankMap
from megatron.core.process_groups_config import ProcessGroupCollection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncoderDomain:
    """The assembled encoder side: DDP module, optimizer, effective config."""

    encoder_ddp: Any
    encoder_optimizer: Any
    effective_config: Any


def build_encoder_ddp_config(
    decoder_ddp_config: DistributedDataParallelConfig,
) -> DistributedDataParallelConfig:
    """Copy the decoder DDP policy while keeping encoder communication synchronous.

    Decoder gradient-reduce and parameter-gather overlap is owned by the native
    decoder schedule. The encoder has a separate backward/finalization lifecycle in
    P5/P6, so it must not inherit those schedule-driven hooks.

    The two MXFP8 parameter-gather fields are projected for the same reason.
    ``reuse_grad_buf_for_mxfp8_param_ag`` aliases the parameter all-gather
    receive buffer onto the gradient buffer, which is only safe under the
    decoder's staging order: the training loop stages main params into the
    shared buffer before the forward, and the decoder's parameter-gather
    forward pre-hook then all-gathers and zeroes it before backward writes any
    gradient into it. Forcing ``overlap_param_gather=False`` leaves the encoder
    without that pre-hook, so the staged parameters would still be in the shared
    buffer when P5 accumulates encoder gradients into it. ``fp8_param_gather``
    is projected with it because ``DistributedDataParallelConfig.__post_init__``
    couples the two (``reuse_grad_buf_for_mxfp8_param_ag`` asserts it), and that
    assert is the only reader of the field outside Megatron-FSDP, which MDP
    rejects. It does not change what the encoder holds: DDP dispatches on the
    parameter's actual storage, and the encoder's parameters are never
    quantized, because ``apply_encoder_fp8_config`` copies ``fp8``/``fp8_recipe``
    only, never ``fp8_param``.
    """
    return replace(
        decoder_ddp_config,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        align_param_gather=False,
        fp8_param_gather=False,
        reuse_grad_buf_for_mxfp8_param_ag=False,
    )


def build_encoder_pg_collection(
    rank_map: MdpRankMap, *, encoder_cp: int, process_groups: MdpProcessGroups
) -> ProcessGroupCollection:
    """Process groups for the encoder domain.

    With ``encoder_cp=1``: ``dp = dp_cp = intra_dp_cp = intra_dist_opt = WORLD``
    (replicated parameters reduced once over all ranks, ZeRO-1 sharded over the
    same domain), ``tp/pp/ep`` are rank-local singletons, and
    ``mp/expt_dp/tp_ep_pp`` are ``None`` (``get_pg_rank(None) == 0``,
    ``get_pg_size(None) == 1`` — exactly the intended meaning).

    The singleton is created the way Megatron itself does it: each rank calls
    ``new_group`` once with its own rank list. The ``encoder_cp>1`` evolution
    (cp = each logical worker's ranks, dp = workers sharing a cp coordinate)
    changes only this function, but its DDP/ZeRO semantics still require
    revalidation and are rejected here.
    """
    if encoder_cp != 1:
        raise MdpConfigurationError(
            f"MDP: encoder_cp={encoder_cp} violates: encoder_cp == 1. The encoder-CP "
            "group construction requires revalidating MCore CP gradient semantics."
        )
    world = process_groups.world_group
    mine = torch.distributed.new_group(ranks=[torch.distributed.get_rank()])

    pgs = ProcessGroupCollection()
    pgs.dp = world
    pgs.dp_cp = world
    pgs.intra_dp_cp = world
    pgs.intra_dist_opt = world
    pgs.tp = mine
    pgs.pp = mine
    pgs.ep = mine
    pgs.mp = None
    # The encoder has no experts. Set expt_dp explicitly so DDP's fallback
    # does not create another singleton group with a warning.
    pgs.expt_dp = None
    pgs.tp_ep_pp = None
    pgs.inter_dist_opt = None
    return pgs


def zero_pad_vision_mlp_channels(encoder: torch.nn.Module, *, real_ffn_hidden_size: int) -> None:
    """Zero-init the FFN alignment-padding channels on every vision MLP layer.

    ``effective_config.ffn_hidden_size`` (the model actually built) may be larger
    than ``real_ffn_hidden_size`` (the official/checkpoint architecture) when
    ``--encoder-ffn-hidden-size N`` requests a hardware
    alignment target (e.g. MXFP8's 32-token block size: 4304 -> 4320). Every
    :class:`~megatron.core.transformer.mlp.MLP` in the encoder gets its extra
    rows/columns -- ``linear_fc1``'s output rows ``[real_ffn_hidden_size:]`` and
    ``linear_fc2``'s input columns ``[:, real_ffn_hidden_size:]``, plus
    ``linear_fc1``'s bias if present -- zeroed here, once, at construction time.

    No gradient masking or parameter freezing is needed after this: the vision
    MLP is ``linear_fc1 -> activation -> linear_fc2`` with no intermediate
    normalization, so an activation that maps zero to zero keeps the padding
    channels at exactly zero and, by the chain rule, gives them exactly-zero
    gradient on every subsequent backward pass -- a self-stabilizing invariant.
    The two premises that argument rests on -- the activation passes through
    the origin, and the FFN axis is not gated (``linear_fc1`` emits one
    ``ffn_hidden_size``-wide tensor rather than a gate/up pair whose halves the
    padding slice would straddle) -- are checked per layer below, so a later
    vision architecture cannot silently invalidate them.

    TP=1 is assumed for the encoder (true for every current MDP topology, see
    ``build_encoder_pg_collection``), so ``linear_fc1``/``linear_fc2`` weights
    are the full, unsharded ``[ffn_hidden_size, hidden_size]`` /
    ``[hidden_size, ffn_hidden_size]`` tensors on every rank -- no TP shard
    offset accounting is required.
    """
    from megatron.core.transformer.mlp import MLP

    padded = 0
    with torch.no_grad():
        for module in encoder.modules():
            if not isinstance(module, MLP):
                continue
            if module.config.gated_linear_unit:
                raise MdpConfigurationError(
                    "MDP: vision MLP gated_linear_unit=True violates: an ungated "
                    "FFN axis. linear_fc1 would emit a concatenated gate/up pair, "
                    "so the padding slice would straddle both halves instead of "
                    "the alignment channels."
                )
            probe = torch.zeros(1, dtype=torch.float32)
            if not torch.equal(module.config.activation_func(probe), probe):
                raise MdpConfigurationError(
                    f"MDP: vision MLP activation {module.config.activation_func} "
                    "violates: activation(0) == 0. The padding channels stay inert "
                    "only if the activation passes through the origin."
                )
            fc1_out = module.linear_fc1.weight.shape[0]
            fc2_in = module.linear_fc2.weight.shape[1]
            if fc1_out != fc2_in:
                raise MdpConfigurationError(
                    f"MDP: vision MLP linear_fc1 output width {fc1_out} != "
                    f"linear_fc2 input width {fc2_in} violates: the two must "
                    "agree on ffn_hidden_size for zero_pad_vision_ffn to locate "
                    "the padding channels consistently."
                )
            if fc1_out < real_ffn_hidden_size:
                raise MdpConfigurationError(
                    f"MDP: vision MLP ffn_hidden_size {fc1_out} violates: "
                    f">= real_ffn_hidden_size ({real_ffn_hidden_size}). "
                    "zero_pad_vision_ffn only pads up, never down; check the "
                    "--encoder-ffn-hidden-size value."
                )
            if fc1_out == real_ffn_hidden_size:
                continue  # no padding requested for this layer; nothing to zero
            module.linear_fc1.weight.data[real_ffn_hidden_size:, :].zero_()
            if module.linear_fc1.bias is not None:
                module.linear_fc1.bias.data[real_ffn_hidden_size:].zero_()
            module.linear_fc2.weight.data[:, real_ffn_hidden_size:].zero_()
            padded += 1
    if padded == 0:
        raise MdpConfigurationError(
            "MDP: zero_pad_vision_ffn=True found no vision MLP layer whose "
            f"ffn_hidden_size exceeds real_ffn_hidden_size ({real_ffn_hidden_size}) "
            "violates: at least one layer to pad. Check that "
            "--encoder-ffn-hidden-size is actually larger "
            "than the base vision config's ffn_hidden_size."
        )
    logger.info(
        "MDP: zero-padded %d vision MLP layer(s) from real ffn_hidden_size=%d up "
        "to the checkpoint-compatible alignment target.",
        padded,
        real_ffn_hidden_size,
    )


def build_encoder_domain(
    *,
    adapter: MdpModelAdapter,
    model_config,
    mdp_config: MdpConfig,
    ddp_config,
    optimizer_config,
    encoder_pgs: ProcessGroupCollection,
    wrap_mixed_precision: bool = True,
    compat_options=None,
) -> EncoderDomain:
    """Assemble the encoder domain (API design 14.2).

    Order: typed encoder recompute, FP8 and FFN-width config; encoder via the adapter's shared
    factory; the same mixed-precision wrapper depth as the decoder;
    DDP over the encoder process groups; DistributedOptimizer from the DDP
    buffers.
    """
    if getattr(ddp_config, "num_distributed_optimizer_instances", 1) != 1:
        raise MdpConfigurationError(
            "MDP: num_distributed_optimizer_instances != 1 violates: the encoder "
            "shards its optimizer state over WORLD."
        )

    encoder_ddp_config = build_encoder_ddp_config(ddp_config)
    for field_name in (
        "overlap_grad_reduce",
        "overlap_param_gather",
        "align_param_gather",
    ):
        if getattr(encoder_ddp_config, field_name, False):
            raise MdpConfigurationError(
                f"MDP: projected encoder DDP config has {field_name}=True; "
                "decoder schedule-driven communication options must not enter "
                "the encoder P5/P6 domain."
            )

    effective_config = apply_encoder_recompute_config(model_config, mdp_config)
    effective_config = apply_encoder_fp8_config(effective_config, mdp_config, compat_options)
    effective_config = apply_encoder_ffn_config(effective_config, mdp_config)
    validate_effective_vision_config(mdp_config, effective_config, compat_options)
    logger.info(
        "MDP: effective encoder recompute granularity: %s, fp8: %s (recipe %s)",
        mdp_config.encoder_recompute_granularity,
        getattr(effective_config, "fp8", None),
        getattr(effective_config, "fp8_recipe", None) if mdp_config.encoder_fp8 else None,
    )
    encoder = adapter.build_encoder(effective_config, pg_collection=encoder_pgs)
    if mdp_config.zero_pad_vision_ffn:
        zero_pad_vision_mlp_channels(
            encoder, real_ffn_hidden_size=model_config.ffn_hidden_size
        )
    if wrap_mixed_precision and (
        getattr(effective_config, "fp16", False) or getattr(effective_config, "bf16", False)
    ):
        from megatron.core.transformer.module import Float16Module

        encoder = Float16Module(effective_config, encoder.cuda())
    else:
        encoder = encoder.cuda()

    encoder_ddp = DistributedDataParallel(
        config=effective_config,
        ddp_config=encoder_ddp_config,
        module=encoder,
        pg_collection=encoder_pgs,
    )
    assert_encoder_prescale_is_one(encoder_ddp)

    from megatron.core.optimizer import get_megatron_optimizer

    encoder_optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[encoder_ddp],
        pg_collection=encoder_pgs,
        # Megatron cannot derive matching Gloo groups for a caller-built
        # collection.
        use_gloo_process_groups=False,
    )
    return EncoderDomain(
        encoder_ddp=encoder_ddp,
        encoder_optimizer=encoder_optimizer,
        effective_config=effective_config,
    )


def assert_encoder_prescale_is_one(encoder_ddp) -> None:
    """Encoder ranks divide one batch's work; they are not data replicas, so
    WORLD reduction must not pre-divide gradients by W."""
    for buffer in list(encoder_ddp.buffers) + list(encoder_ddp.expert_parallel_buffers):
        if buffer.gradient_scaling_factor != 1.0:
            raise MdpConfigurationError(
                f"MDP: encoder gradient buffer prescale "
                f"{buffer.gradient_scaling_factor} violates: prescale == 1. "
                "calculate_per_token_loss=True must be set before DDP construction."
            )


def assert_parameter_disjointness(
    encoder_ddp, decoder_chunks: Sequence, all_trainable_parameters=None
) -> None:
    """Encoder and decoder parameters must be disjoint (and, when the full set
    is provided, together cover every trainable parameter).

    The load-bearing half is the leak check: a shared parameter would be
    reduced by the decoder finalizer in P4, before P5 produces its encoder
    gradient — silently wrong, never an error.
    """
    encoder_ids = {id(p) for p in encoder_ddp.module.parameters()}
    if not encoder_ids:
        raise MdpConfigurationError("MDP: the encoder has no parameters.")
    decoder_ids = set()
    for index, chunk in enumerate(decoder_chunks):
        leaked = [name for name, p in chunk.named_parameters() if id(p) in encoder_ids]
        if leaked:
            raise MdpConfigurationError(
                f"MDP: decoder chunk {index} contains encoder parameters "
                f"{leaked[:5]}; the native schedule would reduce their gradients "
                "before P5 produces them."
            )
        decoder_ids.update(id(p) for p in chunk.parameters())
    if all_trainable_parameters is not None:
        missing = [
            id(p) for p in all_trainable_parameters
            if id(p) not in encoder_ids and id(p) not in decoder_ids
        ]
        if missing:
            raise MdpConfigurationError(
                f"MDP: {len(missing)} trainable parameters belong to neither domain; "
                "encoder and decoder must cover every trainable parameter."
            )


def finalize_encoder_grads(encoder_ddp, *, globally_reduced_num_tokens: torch.Tensor) -> None:
    """WORLD sum-reduce, then scale by ``1/clamp(T_global, min=1)``.

    ``globally_reduced_num_tokens`` must be the same in-place reduced tensor
    the native decoder finalizer produced (captured via
    ``wrap_finalize_model_grads``); recounting tokens on WORLD would count PP
    replicas more than once. When the count is zero, ``clamp(min=1)`` matches
    the native path's no-scaling behavior (masks already zeroed the numerator).
    """
    encoder_ddp.finish_grad_sync()
    denominator = torch.clamp(globally_reduced_num_tokens.float(), min=1.0)
    # Device-side reciprocal: `.item()` here forced a full host sync between
    # the WORLD reduce-scatter and the scale kernels. The double-precision
    # round trip reproduces `float(1.0 / denominator.item())` bit-exactly
    # (fp32 -> f64 is exact, one f64 divide, one rounding back to fp32), and
    # `grad_data *= tensor` broadcasts the 0-dim fp32 scalar exactly like the
    # Python float the kernel would otherwise receive.
    scale = (1.0 / denominator.double()).float().reshape(())
    encoder_ddp.scale_gradients(scale)
