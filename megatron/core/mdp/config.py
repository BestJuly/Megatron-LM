# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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

# Keys that may be overridden on the vision TransformerConfig. Field semantics and
# cross-field validation are delegated entirely to MCore's own __post_init__.
VISION_CONFIG_OVERRIDE_ALLOWLIST: frozenset = frozenset(
    {
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
        "recompute_modules",
    }
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
    delay_grad_reduce: bool
    checkpoint_mode: str
    save_requested: bool
    load_requested: bool
    sequence_parallel: bool = False
    sequence_packing_scheduler: Optional[str] = None
    thd_static_packing: bool = False
    max_seqlen_per_dp_cp_rank: Optional[int] = None
    thd_max_packed_sequences: Optional[int] = None


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
    _validate_cuda_graph_options(config, options)
    if options.activation_offload_enabled:
        _reject(
            "activation_offload_enabled",
            options.activation_offload_enabled,
            "CPU activation offload disabled",
            "Offload is not validated against the retained encoder forward graph.",
            "False",
        )
    if options.overlap_grad_reduce:
        _reject(
            "overlap_grad_reduce",
            options.overlap_grad_reduce,
            "overlap_grad_reduce == False",
            "Encoder communication must not overlap the decoder schedule or the "
            "optimizer step.",
            "False",
        )
    if options.overlap_param_gather:
        _reject(
            "overlap_param_gather",
            options.overlap_param_gather,
            "overlap_param_gather == False",
            "Encoder communication must not overlap the decoder schedule or the "
            "optimizer step.",
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
            "Only the synchronous global torch_dist weight-only checkpoint is "
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
    if not config.greedy_packing:
        return
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
