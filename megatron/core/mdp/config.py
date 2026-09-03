# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDP configuration and compatibility validation.

Pure-compute module: no ``torch.distributed`` calls, no device tensors, no argparse.
The training entry point converts Megatron args into :class:`MdpCompatibilityOptions`;
core reads only that structure so the full rejection list is unit-testable.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from megatron.core.mdp.errors import MdpConfigurationError

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

# The canonical RankGenerator order MDP's rank mapping is derived from.
SUPPORTED_RANK_ORDER = "tp-cp-ep-dp-pp"

# The only checkpoint format supported by the MDP checkpoint facade.
SUPPORTED_CHECKPOINT_MODE = "torch_dist"

ENCODER_RECOMPUTE_GRANULARITIES = (None, "selective", "full", "whole")

# Decoder fp8_recipe values the vision encoder may inherit under --encoder-fp8.
# "delayed" is excluded on purpose, even though it is a valid
# TransformerConfig.fp8_recipe value in general, for two independent reasons:
#
# 1. TEDelayedScaling keeps a persistent, stateful amax-history ring buffer per
#    FP8-enabled module that is pushed to on every fp8_autocast forward call
#    and reduced over that autocast's fp8_group on each depth-0 exit. The MDP
#    encoder forward does not run once per iteration: runtime.py runs it once
#    per chunk (split_encoder_layout() from --mdp-encoder-max-payload-rows), and
#    how many chunks a rank gets depends on the vision segments it was planned
#    (ranks with none run zero encoder forwards). With
#    encoder_recompute_granularity == "whole" every chunk runs twice more:
#    under no_grad in P2 and replayed with autograd in P5 (activation.py
#    EncoderWholeRecomputeHandle). So an encoder's amax history would be pushed
#    and reduced a rank-dependent number of times per iteration, and the P5
#    replay would see a history the P2 forward already advanced. No MDP
#    checkpoint or eval-boundary path resets it either.
# 2. TE keeps one process-global amax buffer and all-reduces every entry in it
#    on each exit from a depth-0 fp8_autocast, without filtering to the
#    autocast being exited. The encoder opens one autocast per layer, and ranks
#    holding no vision segment open none -- so the DECODER's own delayed-scaling
#    buffers, keyed by the decoder's TP x CP x DP group, would see a
#    rank-dependent number of collectives as soon as the encoder runs FP8.
#
# "tensorwise" (Float8CurrentScaling), "blockwise" (Float8BlockScaling) and
# "mxfp8" (MXFP8BlockScaling) derive scale directly from the current tensor,
# keep no cross-forward state and register nothing in TE's global amax buffer,
# so neither mechanism applies. "custom" is excluded as unvalidated: MDP cannot
# tell from args whether the quantizer factory asks for delayed scaling.
ENCODER_COMPATIBLE_FP8_RECIPES: frozenset = frozenset({"tensorwise", "blockwise", "mxfp8"})


@dataclass(frozen=True)
class MdpConfig:
    """User-facing MDP options. See the design doc for field semantics."""

    enable: bool = False
    encoder_cp: int = 1
    encoder_max_payload_rows: Optional[int] = None
    encoder_recompute_granularity: Optional[str] = None
    encoder_recompute_method: Optional[str] = None
    encoder_recompute_num_layers: Optional[int] = None
    encoder_recompute_modules: Optional[tuple[str, ...]] = None
    # Run the vision encoder under the decoder's --fp8-format / --fp8-recipe
    # (apply_encoder_fp8_config copies them from MdpCompatibilityOptions). The
    # encoder has no recipe of its own and never enables FP8 attention. False
    # keeps the encoder in bf16 whatever the decoder's --fp8 says.
    encoder_fp8: bool = False
    # Vision FFN width to build at (e.g. MXFP8's 32-channel block alignment:
    # 4304 -> 4320). Alone it is a raw architecture change ("Approach A",
    # checkpoint-incompatible); with zero_pad_vision_ffn the extra channels are
    # zero-initialized and provably inert ("Approach B", checkpoint-compatible,
    # see zero_pad_vision_mlp_channels in encoder.py).
    encoder_ffn_hidden_size: Optional[int] = None
    zero_pad_vision_ffn: bool = False
    locality_slack_permille: int = 10
    row_alignment: int = 1
    plan_check_interval: int = 1
    debug_plan_payload_check: bool = False
    pixel_locality: bool = False
    overlap_window_capture: bool = False


@dataclass(frozen=True)
class MdpCompatibilityOptions:
    """Snapshot of the Megatron options MDP validates against its support matrix."""

    world_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: int
    expert_parallel_size: int
    rank_order: str
    virtual_pipeline_parallel_size: Optional[int]
    calculate_per_token_loss: bool
    use_distributed_optimizer: bool
    distributed_optimizer_instances: int
    fp16: bool
    bf16: bool
    fsdp_enabled: bool
    cuda_graph_enabled: bool
    activation_offload_enabled: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    overlap_param_gather_with_optimizer_step: bool
    delay_grad_reduce: bool
    checkpoint_mode: str
    save_requested: bool
    load_requested: bool
    overlap_moe_expert_parallel_comm: bool = False
    # The DECODER's --fp8-format (None = decoder in bf16) and fp8_recipe
    # (args.fp8_recipe). TransformerConfig defaults the recipe to "delayed"
    # whether or not --fp8-format was passed, so it only says anything about the
    # run when decoder_fp8 is not None. Under --encoder-fp8 this IS the encoder's
    # recipe too, and the only reader is that path. Decoder FP8 on its own is
    # not an MDP incompatibility and is not otherwise gated.
    decoder_fp8: Optional[str] = None
    decoder_fp8_recipe: Optional[str] = None
    # args.reuse_grad_buf_for_mxfp8_param_ag. Rejected outright under MDP; see
    # validate_mdp_config for the composite-optimizer mechanism.
    reuse_grad_buf_for_mxfp8_param_ag: bool = False


def _is_positive_int(value: Any) -> bool:
    """True only for a real positive integer (bool is an int subclass in Python)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def encoder_fp8_align_size(fp8_recipe: Optional[str]) -> int:
    """Row/channel alignment the encoder's FP8 GEMMs require under ``fp8_recipe``.

    Delegates to MCore's own table rather than restating the block sizes. The
    import is function-local so this module keeps the torch-free import surface
    its own tests rely on.
    """
    from megatron.core.enums import Fp8Recipe
    from megatron.core.fp8_utils import get_fp8_align_size

    return get_fp8_align_size(Fp8Recipe(fp8_recipe))


def _reject(option: str, value: Any, condition: str, why: str, suggestion: str = "") -> None:
    message = f"MDP: {option}={value!r} violates: {condition}. {why}"
    if suggestion:
        message += f" Suggested value: {suggestion}."
    raise MdpConfigurationError(message)


def validate_mdp_config(config: MdpConfig, options: MdpCompatibilityOptions) -> None:
    """Reject every configuration outside the current MDP support matrix.

    Call after Megatron argument post-processing and before creating MDP process
    groups or model weights. Raises :class:`MdpConfigurationError` with the option,
    its current value, the violated condition, and a suggested value when one exists.
    """
    if not config.enable:
        return

    # --- MdpConfig field validation ---
    if config.encoder_cp != 1:
        _reject(
            "encoder_cp",
            config.encoder_cp,
            "encoder_cp == 1",
            "Encoder context parallelism is a registered extension hook, not an "
            "implemented capability.",
            "1",
        )
    if config.encoder_max_payload_rows is not None and config.encoder_max_payload_rows <= 0:
        _reject(
            "encoder_max_payload_rows",
            config.encoder_max_payload_rows,
            "None or a positive integer",
            "The chunk cap is measured in patch rows.",
            "None",
        )
    granularity = config.encoder_recompute_granularity
    if granularity not in ENCODER_RECOMPUTE_GRANULARITIES:
        _reject(
            "encoder_recompute_granularity",
            granularity,
            f"one of {ENCODER_RECOMPUTE_GRANULARITIES}",
            "Encoder recompute supports native MCore selective/full Transformer "
            "checkpointing and Design-Doc whole-encoder replay.",
            "None",
        )

    native_options = {
        "encoder_recompute_method": config.encoder_recompute_method,
        "encoder_recompute_num_layers": config.encoder_recompute_num_layers,
        "encoder_recompute_modules": config.encoder_recompute_modules,
    }
    if granularity in (None, "whole"):
        for option, value in native_options.items():
            if value is not None:
                _reject(
                    option,
                    value,
                    f"None when encoder_recompute_granularity == {granularity!r}",
                    "Native Transformer recompute details do not apply when encoder "
                    "recompute is disabled or spans the whole encoder.",
                    "None",
                )
    elif granularity == "selective":
        for option in ("encoder_recompute_method", "encoder_recompute_num_layers"):
            value = native_options[option]
            if value is not None:
                _reject(
                    option,
                    value,
                    "None when encoder_recompute_granularity == 'selective'",
                    "Selective recompute is configured only by encoder_recompute_modules.",
                    "None",
                )
    elif config.encoder_recompute_modules is not None:
        _reject(
            "encoder_recompute_modules",
            config.encoder_recompute_modules,
            "None when encoder_recompute_granularity == 'full'",
            "Module selection applies only to selective recompute.",
            "None",
        )
    if config.encoder_ffn_hidden_size is not None and not _is_positive_int(
        config.encoder_ffn_hidden_size
    ):
        _reject(
            "encoder_ffn_hidden_size",
            config.encoder_ffn_hidden_size,
            "None or a positive integer",
            "encoder_ffn_hidden_size is the width the encoder FFN is built at (and, "
            "with zero_pad_vision_ffn, the width the checkpoint architecture is "
            "zero-padded up to).",
            "None",
        )
    if config.zero_pad_vision_ffn and config.encoder_ffn_hidden_size is None:
        _reject(
            "zero_pad_vision_ffn",
            config.zero_pad_vision_ffn,
            "encoder_ffn_hidden_size is set",
            "zero_pad_vision_ffn pads the vision FFN's real (checkpoint) hidden size "
            "up to encoder_ffn_hidden_size; with no target there is nothing to pad "
            "to.",
            "--encoder-ffn-hidden-size <alignment target>",
        )
    if config.encoder_fp8:
        if options.decoder_fp8 is None:
            _reject(
                "encoder_fp8",
                config.encoder_fp8,
                "decoder FP8 enabled (--fp8-format)",
                "--encoder-fp8 inherits the decoder's --fp8-format/--fp8-recipe; "
                "enable decoder FP8 first. Encoder-only FP8 is not supported: it "
                "measured as pure launch overhead on the encoder, and it would need "
                "its own recipe plumbing.",
                "False",
            )
        if options.decoder_fp8_recipe not in ENCODER_COMPATIBLE_FP8_RECIPES:
            _reject(
                "fp8_recipe",
                options.decoder_fp8_recipe,
                f"fp8_recipe in {sorted(ENCODER_COMPATIBLE_FP8_RECIPES)} while the "
                "encoder runs FP8",
                "The encoder inherits this recipe. 'delayed' scaling keeps a stateful "
                "amax history that the encoder's per-chunk, rank-dependent forward "
                "count -- doubled by P2/P5 whole-encoder replay -- would push unevenly, "
                "and TE all-reduces its global amax buffer on every depth-0 autocast "
                "exit, so the decoder's own delayed buffers would see rank-dependent "
                "collective counts (see ENCODER_COMPATIBLE_FP8_RECIPES). 'custom' is "
                "unvalidated.",
                "--fp8-recipe tensorwise",
            )
        align = encoder_fp8_align_size(options.decoder_fp8_recipe)
        if config.encoder_ffn_hidden_size is not None and config.encoder_ffn_hidden_size % align:
            _reject(
                "encoder_ffn_hidden_size",
                config.encoder_ffn_hidden_size,
                f"encoder_ffn_hidden_size % {align} == 0 for fp8_recipe "
                f"'{options.decoder_fp8_recipe}'",
                "TE quantizes the vision FFN weights on the first encoder forward "
                "and aborts when a GEMM dimension is not a multiple of the recipe's "
                "block size.",
                str((config.encoder_ffn_hidden_size + align - 1) // align * align),
            )
    if not (0 <= config.locality_slack_permille < 1000):
        _reject(
            "locality_slack_permille",
            config.locality_slack_permille,
            "0 <= locality_slack_permille < 1000",
            "The LPT near-equal-load window is expressed in per-mille.",
            "10",
        )
    if config.row_alignment < 1:
        _reject(
            "row_alignment",
            config.row_alignment,
            "row_alignment >= 1",
            "Row capacity alignment must be a positive integer (1 in production; "
            "tests may use 16).",
            "1",
        )
    if config.plan_check_interval < 1:
        _reject(
            "plan_check_interval",
            config.plan_check_interval,
            "plan_check_interval >= 1",
            "The plan consistency check must never be fully disabled: an undetected "
            "plan mismatch degrades from a diagnosable error into a collective hang.",
            "1",
        )
    if config.overlap_window_capture and options.tensor_parallel_size != 1:
        _reject(
            "overlap_window_capture",
            config.overlap_window_capture,
            "tensor_parallel_size == 1",
            "The capture path performs a TP broadcast per microbatch; running it "
            "on the prefetch thread concurrently with the schedule's NCCL calls "
            "is only validated without tensor parallelism.",
            "False",
        )

    # --- parallel dimensions and rank mapping preconditions ---
    if options.rank_order != SUPPORTED_RANK_ORDER:
        _reject(
            "rank_order",
            options.rank_order,
            f"rank_order == '{SUPPORTED_RANK_ORDER}'",
            "MDP rank mapping is derived from the default RankGenerator order and "
            "has not been validated against other orders.",
            SUPPORTED_RANK_ORDER,
        )
    if options.tensor_parallel_size != 1:
        _reject(
            "tensor_parallel_size",
            options.tensor_parallel_size,
            "TP == 1",
            "The current MDP support matrix requires TP=1.",
            "1",
        )
    if options.context_parallel_size != 1:
        _reject(
            "context_parallel_size",
            options.context_parallel_size,
            "decoder CP == 1",
            "Decoder context parallelism is a registered extension hook, not an "
            "implemented capability.",
            "1",
        )
    model_parallel = (
        options.tensor_parallel_size
        * options.pipeline_parallel_size
        * options.context_parallel_size
    )
    if options.world_size <= 0 or options.world_size % model_parallel != 0:
        _reject(
            "world_size",
            options.world_size,
            "world_size % (TP * PP * CP) == 0",
            f"TP * PP * CP = {model_parallel} must evenly divide the world size to "
            "form outer data-parallel planning groups.",
        )
    if options.overlap_moe_expert_parallel_comm:
        if options.expert_parallel_size <= 1:
            _reject(
                "overlap_moe_expert_parallel_comm",
                options.overlap_moe_expert_parallel_comm,
                "EP > 1",
                "Decoder EP communication overlap requires expert parallelism.",
                "expert_parallel_size > 1",
            )
        if (
            options.pipeline_parallel_size > 1
            and options.virtual_pipeline_parallel_size is None
        ):
            _reject(
                "overlap_moe_expert_parallel_comm",
                options.overlap_moe_expert_parallel_comm,
                "VPP enabled when PP > 1",
                "The native combined 1F1B EP-overlap schedule is interleaved "
                "when pipeline parallelism is enabled.",
                "virtual_pipeline_parallel_size > 1",
            )

    # --- training semantics ---
    if not options.calculate_per_token_loss:
        _reject(
            "calculate_per_token_loss",
            options.calculate_per_token_loss,
            "calculate_per_token_loss == True",
            "Encoder gradient normalization reuses the decoder finalizer's global "
            "token count; with per-token loss off the decoder normalizes by "
            "1/num_microbatches and the derivation collapses.",
            "True",
        )
    if not options.use_distributed_optimizer:
        _reject(
            "use_distributed_optimizer",
            options.use_distributed_optimizer,
            "use_distributed_optimizer == True",
            "The encoder domain uses ZeRO-1 (DistributedOptimizer) over WORLD.",
            "True",
        )
    if options.distributed_optimizer_instances != 1:
        _reject(
            "distributed_optimizer_instances",
            options.distributed_optimizer_instances,
            "distributed_optimizer_instances == 1",
            "Multiple distributed-optimizer instances are not validated with the "
            "MDP composite optimizer.",
            "1",
        )
    if not (options.fp16 or options.bf16):
        _reject(
            "fp16/bf16",
            (options.fp16, options.bf16),
            "fp16 or bf16 mixed precision enabled",
            "MDP is validated on the bf16 main path (fp16 for overflow-union tests).",
            "bf16",
        )

    # --- unsupported feature rejections ---
    if options.fsdp_enabled:
        _reject(
            "fsdp_enabled",
            options.fsdp_enabled,
            "FSDP/HSDP disabled",
            "MDP requires the standard DistributedDataParallel gradient-buffer path.",
            "False",
        )
    if options.cuda_graph_enabled:
        _reject(
            "cuda_graph_enabled",
            options.cuda_graph_enabled,
            "full-iteration CUDA graphs disabled",
            "MDP buffers are not captured graph-safe in this version.",
            "False",
        )
    if options.activation_offload_enabled:
        _reject(
            "activation_offload_enabled",
            options.activation_offload_enabled,
            "CPU activation offload disabled",
            "Offload is not validated against the retained encoder forward graph.",
            "False",
        )
    if options.overlap_param_gather and not options.overlap_grad_reduce:
        _reject(
            "overlap_param_gather",
            options.overlap_param_gather,
            "overlap_param_gather requires overlap_grad_reduce",
            "MDP preserves the native decoder DDP overlap contract; the encoder "
            "uses a separate synchronous DDP configuration.",
            "enable overlap_grad_reduce or disable overlap_param_gather",
        )
    if options.overlap_param_gather_with_optimizer_step:
        _reject(
            "overlap_param_gather_with_optimizer_step",
            options.overlap_param_gather_with_optimizer_step,
            "overlap_param_gather_with_optimizer_step == False",
            "The MDP composite optimizer appends the encoder optimizer after the "
            "decoder optimizers. Dispatching a decoder parameter gather while later "
            "members are still stepping crosses the decoder/encoder domain boundary.",
            "False",
        )
    if options.reuse_grad_buf_for_mxfp8_param_ag:
        _reject(
            "reuse_grad_buf_for_mxfp8_param_ag",
            options.reuse_grad_buf_for_mxfp8_param_ag,
            "reuse_grad_buf_for_mxfp8_param_ag == False",
            "ChainedOptimizer._should_defer_mxfp8_param_sync() answers True as soon "
            "as any chained member has overlap_param_gather=False; MDP's encoder "
            "member always does (build_encoder_ddp_config), so the DECODER would be "
            "moved onto the deferred MXFP8 param-sync path whatever its own setting.",
            "False",
        )
    if options.delay_grad_reduce:
        _reject(
            "delay_grad_reduce",
            options.delay_grad_reduce,
            "delay_grad_reduce == False",
            "Encoder gradient reduction runs synchronously in P5.",
            "False",
        )

    # --- checkpoint restrictions (only when a save or load is requested) ---
    if (options.save_requested or options.load_requested) and (
        options.checkpoint_mode != SUPPORTED_CHECKPOINT_MODE
    ):
        _reject(
            "checkpoint_mode",
            options.checkpoint_mode,
            f"checkpoint_mode == '{SUPPORTED_CHECKPOINT_MODE}'",
            "Only the synchronous global torch_dist checkpoint is "
            "supported; fully-parallel, local, asynchronous, non-persistent, and "
            "constant-structure caching modes are rejected.",
            SUPPORTED_CHECKPOINT_MODE,
        )


def apply_encoder_recompute_config(
    base_config: "TransformerConfig", config: MdpConfig
) -> "TransformerConfig":
    """Apply native encoder recompute settings through TransformerConfig validation.

    Whole recompute is implemented by the MDP phase machine rather than nested
    MCore checkpointing, so it leaves the vision TransformerConfig unchanged.
    """
    granularity = config.encoder_recompute_granularity
    if granularity in (None, "whole"):
        return base_config

    modules = config.encoder_recompute_modules
    return dataclasses.replace(
        base_config,
        recompute_granularity=granularity,
        recompute_method=config.encoder_recompute_method,
        recompute_num_layers=config.encoder_recompute_num_layers,
        recompute_modules=list(modules) if modules is not None else None,
    )


def apply_encoder_fp8_config(
    base_config: "TransformerConfig",
    config: MdpConfig,
    options: Optional[MdpCompatibilityOptions],
) -> "TransformerConfig":
    """Run the encoder under the decoder's FP8 format and recipe when asked.

    The FP8 context itself is opened per layer by ``TransformerBlock.forward``
    off these fields, so this is the whole of the wiring: both the P2 forward
    and, under whole-encoder replay, the P5 replay call the same encoder and
    therefore see the same recipe. FP8 attention is not offered for the encoder,
    so that field is left at the base config's value (False).
    """
    if not config.encoder_fp8:
        return base_config
    if options is None or options.decoder_fp8 is None:
        raise MdpConfigurationError(
            "MDP: encoder_fp8=True violates: decoder FP8 enabled. The encoder "
            "inherits the decoder's recipe, and build_encoder_domain was given no "
            "compatibility snapshot (or one without decoder FP8) to inherit from."
        )
    return dataclasses.replace(
        base_config, fp8=options.decoder_fp8, fp8_recipe=options.decoder_fp8_recipe
    )


def apply_encoder_ffn_config(
    base_config: "TransformerConfig", config: MdpConfig
) -> "TransformerConfig":
    """Build the vision FFN at ``encoder_ffn_hidden_size`` when one is set.

    The base config keeps the checkpoint architecture's width, which is what
    ``zero_pad_vision_mlp_channels`` and the checkpoint facade read as the real
    width; only the built encoder is widened.
    """
    if config.encoder_ffn_hidden_size is None:
        return base_config
    return dataclasses.replace(base_config, ffn_hidden_size=config.encoder_ffn_hidden_size)


def validate_effective_vision_config(
    config: MdpConfig,
    effective_config: "TransformerConfig",
    options: Optional[MdpCompatibilityOptions] = None,
) -> None:
    """Reject unsupported combinations visible only after adapter resolution.

    ``options`` is the compatibility snapshot the encoder inherits its FP8
    recipe from; it may be omitted only when ``config.encoder_fp8`` is False.
    """
    recompute_granularity = getattr(effective_config, "recompute_granularity", None)
    if (
        config.encoder_recompute_granularity == "whole"
        and recompute_granularity is not None
    ):
        _reject(
            "effective vision recompute_granularity",
            recompute_granularity,
            "None when encoder_recompute_granularity == 'whole'",
            "Whole-encoder replay cannot wrap native Transformer recompute; "
            "otherwise the vision Transformer is replayed twice in P5.",
            "None",
        )
    # Cross-check the built config against what validate_mdp_config saw. The
    # gates above run on MdpConfig/MdpCompatibilityOptions before the vision
    # config exists; an adapter or override path that sets fp8 on the vision
    # config on its own would otherwise train an encoder in a state the support
    # matrix never validated (or silently miss one it was told to build).
    expected_fp8 = None
    expected_recipe = None
    if config.encoder_fp8:
        if options is None:
            raise MdpConfigurationError(
                "MDP: validate_effective_vision_config needs the compatibility "
                "snapshot when encoder_fp8=True; the encoder recipe is inherited from it."
            )
        if options.decoder_fp8 is None:
            # Self-sufficient: without this, an off decoder would compare
            # None == None against a bf16 vision config and pass.
            _reject(
                "encoder_fp8",
                config.encoder_fp8,
                "decoder FP8 enabled (--fp8-format)",
                "The encoder inherits the decoder's FP8 format and recipe; with the "
                "decoder in bf16 there is nothing to inherit.",
                "False",
            )
        expected_fp8 = options.decoder_fp8
        expected_recipe = options.decoder_fp8_recipe
    effective_fp8 = getattr(effective_config, "fp8", None)
    if effective_fp8 != expected_fp8:
        _reject(
            "effective vision fp8",
            effective_fp8,
            f"== {expected_fp8!r} (the decoder's --fp8-format under --encoder-fp8, "
            "else None)",
            "The vision TransformerConfig's FP8 state must come from --encoder-fp8 "
            "alone; the MDP support-matrix gates ran against MdpConfig and the "
            "compatibility snapshot, not against this config.",
            repr(expected_fp8),
        )
    if effective_fp8 is not None:
        effective_recipe = getattr(effective_config, "fp8_recipe", None)
        if effective_recipe != expected_recipe:
            _reject(
                "effective vision fp8_recipe",
                effective_recipe,
                f"== {expected_recipe!r} (the decoder's --fp8-recipe)",
                "ENCODER_COMPATIBLE_FP8_RECIPES was checked against the decoder's "
                "recipe; a different recipe on the built config bypasses it.",
                repr(expected_recipe),
            )
        # The width the FFN is actually built at, whether it came from
        # encoder_ffn_hidden_size (already gated in validate_mdp_config) or from
        # the base vision config -- the shipped Qwen3.5-VL 4304 is not a
        # multiple of MXFP8's 32, and without this check that configuration
        # passes every validator and dies inside TE on the first forward.
        align = encoder_fp8_align_size(effective_recipe)
        effective_ffn = getattr(effective_config, "ffn_hidden_size", None)
        if effective_ffn is not None and effective_ffn % align:
            aligned = (effective_ffn + align - 1) // align * align
            _reject(
                "effective vision ffn_hidden_size",
                effective_ffn,
                f"ffn_hidden_size % {align} == 0 for encoder fp8_recipe {effective_recipe!r}",
                "TE quantizes the vision FFN weights on the first encoder forward and "
                "aborts when a GEMM dimension is not a multiple of the recipe's block "
                "size.",
                f"--encoder-ffn-hidden-size {aligned} (with --mdp-zero-pad-vision-ffn to "
                "keep official checkpoints loadable)",
            )
