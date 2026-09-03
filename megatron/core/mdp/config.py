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

# CUDA graph implementations MDP accepts. "local" and "transformer_engine" are the
# per-layer ("partial") implementations: they graph individual decoder submodules and
# leave the schedule, the encoder, and every MDP collective in eager mode. MDP's P4
# runs the unmodified decoder schedule, so the decoder layers a per-layer graph owns
# are never touched by the bridge.
SUPPORTED_CUDA_GRAPH_IMPLS: frozenset = frozenset({"none", "local", "transformer_engine"})

# The one implementation that is structurally incompatible with the phase machine: a
# single graph over the whole forward-backward path would swallow P4 together with the
# Python-level control flow P1-P3/P5 depend on.
REJECTED_CUDA_GRAPH_IMPL = "full_iteration"

# Encoder recompute granularities. "selective"/"full" delegate to native MCore
# Transformer checkpointing on the vision config; "whole" is MDP's own
# Design-Doc whole-encoder replay in P5. These typed arguments replace the
# earlier free-form vision config override channel, which no longer exists.
ENCODER_RECOMPUTE_GRANULARITIES = (None, "selective", "full", "whole")


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
    locality_slack_permille: int = 10
    row_alignment: int = 1
    plan_check_interval: int = 1
    debug_plan_payload_check: bool = False
    pixel_locality: bool = False
    overlap_window_capture: bool = False
    greedy_packing: bool = False
    greedy_packing_approximate_resume: bool = False
    buffer_pool: bool = True


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
    fp8_enabled: bool
    cuda_graph_impl: str
    activation_offload_enabled: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    overlap_param_gather_with_optimizer_step: bool
    delay_grad_reduce: bool
    checkpoint_mode: str
    save_requested: bool
    load_requested: bool
    overlap_moe_expert_parallel_comm: bool = False
    sequence_parallel: bool = False
    sequence_packing_scheduler: Optional[str] = None
    thd_static_packing: bool = False
    max_seqlen_per_dp_cp_rank: Optional[int] = None
    thd_max_packed_sequences: Optional[int] = None
    # max(micro_batch_size, eval_micro_batch_size): the largest number of samples
    # the collator can be handed in one microbatch without greedy packing.
    max_samples_per_microbatch: int = 1


def thd_row_alignment(options: "MdpCompatibilityOptions") -> int:
    """Row alignment the MDP collator pads each packed sample to.

    Mirrors ``pack_or_pad_batch``'s ``divisible_by`` (zigzag CP wants an even
    per-rank split; SP additionally splits across TP). The greedy token budget
    must be a multiple of this, or a full bin cannot be partitioned legally.
    """
    from megatron.core.packed_seq_params import thd_collate_row_alignment

    return thd_collate_row_alignment(
        context_parallel_size=options.context_parallel_size,
        tensor_model_parallel_size=options.tensor_parallel_size,
        sequence_parallel=options.sequence_parallel,
    )


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
    _validate_packing(config, options)

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
    if options.fp8_enabled:
        _reject(
            "fp8_enabled",
            options.fp8_enabled,
            "FP8 disabled",
            "FP8/MXFP8 gradient-buffer reuse is not validated with MDP; row-aligned "
            "allocation is only a future-facing hook, not an FP8 recipe.",
            "False",
        )
    _validate_cuda_graph_options(config, options)
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


def _validate_cuda_graph_options(
    config: MdpConfig, options: MdpCompatibilityOptions
) -> None:
    """Accept per-layer CUDA graphs; keep full-iteration graphs rejected.

    Per-layer graphs own individual decoder submodules. P4 replays the captured
    microbatches through the *unmodified* decoder schedule, and the bridge only ever
    touches the decoder's embedding leaves (P3) and their gradients (P5) - never the
    transformer layers - so a per-layer graph and the phase machine do not interact.
    A full-iteration graph would instead capture P4 itself.
    """
    impl = options.cuda_graph_impl or "none"
    if impl == REJECTED_CUDA_GRAPH_IMPL:
        _reject(
            "cuda_graph_impl",
            impl,
            f"cuda_graph_impl != '{REJECTED_CUDA_GRAPH_IMPL}'",
            "A full-iteration graph captures the decoder schedule itself, so the "
            "Python-level phase machine around it (pixel/embedding/gradient exchange, "
            "per-iteration plan, dynamic vision item counts) cannot run. Per-layer "
            "graphs are supported instead.",
            "transformer_engine",
        )
    if impl not in SUPPORTED_CUDA_GRAPH_IMPLS:
        _reject(
            "cuda_graph_impl",
            impl,
            f"cuda_graph_impl in {sorted(SUPPORTED_CUDA_GRAPH_IMPLS)}",
            "MDP validates against the known CUDA graph implementations only.",
            "none",
        )
    if impl == "none":
        return

    if config.overlap_window_capture:
        # EXPERIMENT ESCAPE HATCH. Default path is unchanged: without the env
        # var this still raises exactly as before. It exists because the
        # rejection below rests on a *predicted* race that was never measured,
        # and grading that prediction requires running the combination.
        # A clean run does NOT prove safety -- the failure mode is described as
        # timing-dependent, so absence of a crash in N iterations is weak
        # evidence. Do not remove the guard on the strength of one green run.
        import os

        if os.environ.get("MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS") == "1":
            print(
                "MDP WARNING: overlap_window_capture + per-layer CUDA graphs is "
                "normally REJECTED (config.py). Running anyway because "
                "MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS=1. This is an experiment; "
                "the guarded failure is a capture-vs-prefetch race and is "
                "timing-dependent.",
                flush=True,
            )
        else:
            _reject(
                "overlap_window_capture",
                config.overlap_window_capture,
                "overlap_window_capture == False when per-layer CUDA graphs are enabled",
                "Window prefetch runs H2D copies and allocations from a background thread "
                "on a side CUDA stream while graph capture is in flight, which "
                "cudaStreamCaptureModeGlobal treats as an unsafe concurrent action. The "
                "prefetch started in iteration N is still running when the capture step "
                "runs, so the conflict is timing-dependent rather than reproducible.",
                "False",
            )

    if not options.thd_static_packing:
        _reject(
            "thd_static_packing",
            options.thd_static_packing,
            "thd_static_packing == True when per-layer CUDA graphs are enabled",
            "MDP requires a THD-packed decoder (window.py asserts qkv_format == 'thd'), "
            "and a CUDA graph replays into fixed-size static input buffers: "
            "[max_seqlen_per_dp_cp_rank, 1, H] hidden_states plus cu_seqlens of "
            "thd_max_packed_sequences + 1 entries. Only --thd-static-packing makes MDP's "
            "collator emit that shape; without it every microbatch has a different "
            "packed token count and replay fails on the first mismatch. "
            "--sequence-packing-scheduler, which produces the same contract for the "
            "decoder-only path, is not usable under MDP (see validate_mdp_config).",
            "--thd-static-packing --pad-packed-seq-alignment max "
            "--max-seqlen-per-dp-cp-rank <N> --thd-max-packed-sequences <K>",
        )


def greedy_max_real_sequences(options: "MdpCompatibilityOptions") -> Optional[int]:
    """Real sequences a greedy bin may hold, or ``None`` for no cap.

    ``thd_max_packed_sequences`` is the *final* static THD capacity. Under
    ``--thd-static-packing`` the padding tail is represented as an ordinary
    dummy sequence appended to ``cu_seqlens``, so one slot must be reserved for
    it -- exactly what ``_get_scheduler_max_real_num_seqs`` does for
    ``dp_balanced``. Without the reservation a bin filled to the cap overflows
    the ``thd_max_packed_sequences + 1`` entry budget and dies inside
    ``_pad_cu_seqlens``.
    """
    cap = options.thd_max_packed_sequences
    if cap is None:
        return None
    return int(cap) - 1 if options.thd_static_packing else int(cap)


def _validate_packing(config: MdpConfig, options: MdpCompatibilityOptions) -> None:
    """Reject packing configurations MDP cannot honor.

    ``--sequence-packing-scheduler`` is rejected outright, not merely untested:
    ``training.py`` wraps the data iterator whenever it is set, and
    ``DpBalancedScheduler.run`` then asserts on GPT-only sample keys, deletes
    every key outside those six (dropping ``pixel_values`` / ``image_grid_thw``),
    and reroutes samples across DP with an all-to-all that has no notion of
    variable-size pixel payloads. Without this rejection the run dies deep inside
    an assert about a missing ``tokens`` key.

    Also enforces the two properties the static/greedy packing paths depend on:
    the ``cu_seqlens`` capacity leaves a slot for the static padding tail, and
    greedy packing is not silently combined with checkpointing (its sample
    buffer is not checkpointed).
    """
    if options.sequence_packing_scheduler is not None:
        _reject(
            "sequence_packing_scheduler",
            options.sequence_packing_scheduler,
            "sequence_packing_scheduler is None",
            "MCore's packing schedulers assert on GPT-only sample keys, drop the "
            "pixel payload, and reroute samples across DP without pixel awareness. "
            "MDP owns its packing (--mdp-greedy-packing).",
            "None",
        )
    if options.thd_static_packing and not config.greedy_packing:
        # Without greedy packing a microbatch is exactly micro_batch_size samples
        # (eval_micro_batch_size on the eval loaders), and the static padding tail
        # is appended to cu_seqlens as one more ordinary sequence, so the pack
        # needs that many + 2 entries against a capacity of
        # thd_max_packed_sequences + 1. greedy_packing makes the same
        # reservation, through greedy_max_real_sequences().
        cap = options.thd_max_packed_sequences
        samples = options.max_samples_per_microbatch
        if cap is not None and cap < samples + 1:
            _reject(
                "thd_max_packed_sequences",
                cap,
                f"thd_max_packed_sequences >= max(micro_batch_size, "
                f"eval_micro_batch_size) + 1 ({samples} + 1)",
                "Under --thd-static-packing the padding tail is appended to "
                "cu_seqlens as an ordinary dummy sequence, so one slot of the "
                "thd_max_packed_sequences + 1 capacity is reserved for it; a full "
                "microbatch would otherwise overflow it inside _pad_cu_seqlens.",
                str(samples + 1),
            )
    if not config.greedy_packing:
        return
    if (options.save_requested or options.load_requested) and (
        not config.greedy_packing_approximate_resume
    ):
        # The greedy stream buffers samples across iterations: the underlying
        # iterator advances by a whole batch_sampler batch while only part of it
        # has been drained into bins. That buffer is not checkpointed, and the
        # sampler is positioned from a single global consumed_train_samples that
        # cannot express per-DP-rank drain counts, so a resume may skip or repeat
        # samples. Greedy packing is a benchmarking path; make that explicit
        # rather than silently corrupting a resume.
        _reject(
            "greedy_packing",
            config.greedy_packing,
            "--save / --load is not combined with --mdp-greedy-packing",
            "The greedy sample buffer is not checkpointed and the sampler cannot be "
            "repositioned per DP rank, so a resume may skip or repeat samples. Pass "
            "--mdp-greedy-packing-approximate-resume to accept that, or drop "
            "--mdp-greedy-packing for runs that checkpoint.",
            "False",
        )
    if options.max_seqlen_per_dp_cp_rank is None:
        _reject(
            "max_seqlen_per_dp_cp_rank",
            options.max_seqlen_per_dp_cp_rank,
            "max_seqlen_per_dp_cp_rank is set when --mdp-greedy-packing is on",
            "The greedy token budget is max_seqlen_per_dp_cp_rank x "
            "context_parallel_size; there is no default for it.",
        )
    alignment = thd_row_alignment(options)
    budget = options.max_seqlen_per_dp_cp_rank * options.context_parallel_size
    if budget % alignment != 0:
        _reject(
            "max_seqlen_per_dp_cp_rank",
            options.max_seqlen_per_dp_cp_rank,
            f"the greedy token budget ({budget}) is divisible by the collator row "
            f"alignment ({alignment})",
            "A bin filled to the budget must still split legally across CP/SP ranks; "
            "discovering this inside TransformerEngine gives a far worse error.",
        )
    minimum = 2 if options.thd_static_packing else 1
    if (
        options.thd_max_packed_sequences is not None
        and options.thd_max_packed_sequences < minimum
    ):
        _reject(
            "thd_max_packed_sequences",
            options.thd_max_packed_sequences,
            f"thd_max_packed_sequences >= {minimum}",
            "It caps the real sequences per greedy bin; under --thd-static-packing "
            "one slot is reserved for the padding tail's dummy sequence.",
            "8",
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


def validate_effective_vision_config(
    config: MdpConfig, effective_config: "TransformerConfig"
) -> None:
    """Reject unsupported combinations visible only after adapter resolution."""
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
