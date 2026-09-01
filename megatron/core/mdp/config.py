# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDP configuration, compatibility validation, and the vision config override channel.

Pure-compute module: no ``torch.distributed`` calls, no device tensors, no argparse.
The training entry point converts Megatron args into :class:`MdpCompatibilityOptions`;
core reads only that structure so the full rejection list is unit-testable.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

from megatron.core.mdp.errors import MdpConfigurationError

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

# The canonical RankGenerator order MDP's rank mapping is derived from.
SUPPORTED_RANK_ORDER = "tp-cp-ep-dp-pp"

# The only checkpoint format supported by the MDP checkpoint facade.
SUPPORTED_CHECKPOINT_MODE = "torch_dist"

# Keys that may be overridden on the vision TransformerConfig. Field semantics and
# cross-field validation are delegated entirely to MCore's own __post_init__.
VISION_CONFIG_OVERRIDE_ALLOWLIST: frozenset = frozenset(
    {"recompute_granularity", "recompute_method", "recompute_num_layers", "recompute_modules"}
)


@dataclass(frozen=True)
class MdpConfig:
    """User-facing MDP options. See the design doc for field semantics."""

    enable: bool = False
    encoder_cp: int = 1
    encoder_max_payload_rows: Optional[int] = None
    vision_config_overrides: tuple = ()
    locality_slack_permille: int = 10
    row_alignment: int = 1
    plan_check_interval: int = 1
    debug_plan_payload_check: bool = False
    pixel_locality: bool = False
    overlap_window_capture: bool = False
    greedy_packing: bool = False
    greedy_packing_approximate_resume: bool = False


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
    sequence_parallel: bool = False
    cp_partition_mode: str = "zigzag"
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
    if options.context_parallel_size > 1:
        return (
            options.tensor_parallel_size * options.context_parallel_size * 2
            if options.sequence_parallel
            else options.context_parallel_size * 2
        )
    return options.tensor_parallel_size if options.sequence_parallel else 1


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
    _validate_override_entries(config.vision_config_overrides)
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
    if options.context_parallel_size < 1:
        _reject(
            "context_parallel_size",
            options.context_parallel_size,
            "context_parallel_size >= 1",
            "Decoder context parallelism is supported; the size must be positive.",
            "1",
        )
    if options.context_parallel_size > 1 and options.cp_partition_mode != "zigzag":
        _reject(
            "cp_partition_mode",
            options.cp_partition_mode,
            "cp_partition_mode == 'zigzag'",
            "MDP routes each vision item's rows with an integer inverse of the "
            "zigzag partition (megatron.core.mdp.cp_partition). Under "
            "'contiguous' the decoder would slice its sequence differently from "
            "the plan, delivering every embedding to the wrong rank without any "
            "shape error.",
            "zigzag",
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
            "FP8/MXFP8 gradient-buffer reuse is not validated with MDP; the vision "
            "config override channel is reserved for a future FP8 recipe.",
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
    if (
        options.thd_static_packing
        and options.context_parallel_size > 1
        and options.max_seqlen_per_dp_cp_rank is not None
        and options.max_seqlen_per_dp_cp_rank % 2 != 0
    ):
        # build_static_thd_metadata splits the padding tail with a bare
        # `dummy_seq_len % (2 * cp) == 0` assert mid-iteration. The static target
        # is max_seqlen_per_dp_cp_rank * cp, so an odd per-rank length makes that
        # assert unsatisfiable. Reject at startup, in both packing modes, rather
        # than a hundred iterations in.
        _reject(
            "max_seqlen_per_dp_cp_rank",
            options.max_seqlen_per_dp_cp_rank,
            "max_seqlen_per_dp_cp_rank is even under --thd-static-packing with CP > 1",
            "The static THD target is max_seqlen_per_dp_cp_rank x "
            "context_parallel_size and its padding tail must still divide by "
            "2 * context_parallel_size.",
            str(options.max_seqlen_per_dp_cp_rank + 1),
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


def _validate_override_entries(overrides: Sequence) -> None:
    """Shared structural validation for vision config override entry sequences."""
    seen = set()
    previous_key = None
    for entry in overrides:
        if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str)):
            raise MdpConfigurationError(
                f"MDP: vision config override entry {entry!r} violates: entries are "
                "(key, value) tuples with a string key."
            )
        key = entry[0]
        if key not in VISION_CONFIG_OVERRIDE_ALLOWLIST:
            raise MdpConfigurationError(
                f"MDP: vision config override key {key!r} violates: key in allowlist "
                f"{sorted(VISION_CONFIG_OVERRIDE_ALLOWLIST)}. Overrides outside the "
                "current support matrix are rejected."
            )
        if key in seen:
            raise MdpConfigurationError(
                f"MDP: vision config override key {key!r} violates: keys are unique."
            )
        if previous_key is not None and key < previous_key:
            raise MdpConfigurationError(
                f"MDP: vision config override key {key!r} violates: entries are "
                "key-sorted. A canonical, immutable, sorted sequence is required so "
                "cross-rank consistency assertions and startup logs can consume it "
                "directly."
            )
        seen.add(key)
        previous_key = key


def apply_vision_config_overrides(
    base_config: "TransformerConfig", overrides: Sequence
) -> "TransformerConfig":
    """Build the vision TransformerConfig from the decoder base plus the override entries.

    Field-level and cross-field validation are delegated to MCore's own
    ``__post_init__`` via ``dataclasses.replace``; MDP does not duplicate those rules.
    """
    _validate_override_entries(overrides)
    if not overrides:
        return base_config
    return dataclasses.replace(base_config, **dict(overrides))
