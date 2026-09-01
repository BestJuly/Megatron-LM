# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-compute tests for MdpConfig validation and the vision config override
channel. No distributed state, no CUDA."""

import dataclasses

import pytest

from megatron.core.mdp.config import (
    VISION_CONFIG_OVERRIDE_ALLOWLIST,
    MdpCompatibilityOptions,
    MdpConfig,
    apply_vision_config_overrides,
    greedy_max_real_sequences,
    validate_mdp_config,
)
from megatron.core.mdp.errors import MdpConfigurationError


def _options(**overrides):
    base = dict(
        world_size=8,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        context_parallel_size=1,
        expert_parallel_size=1,
        rank_order="tp-cp-ep-dp-pp",
        virtual_pipeline_parallel_size=None,
        calculate_per_token_loss=True,
        use_distributed_optimizer=True,
        distributed_optimizer_instances=1,
        fp16=False,
        bf16=True,
        fsdp_enabled=False,
        fp8_enabled=False,
        cuda_graph_enabled=False,
        activation_offload_enabled=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        delay_grad_reduce=False,
        overlap_moe_expert_parallel_comm=False,
        checkpoint_mode="torch_dist",
        save_requested=False,
        load_requested=False,
        max_samples_per_microbatch=4,
    )
    base.update(overrides)
    return MdpCompatibilityOptions(**base)


def test_valid_configuration_passes():
    validate_mdp_config(MdpConfig(enable=True), _options())


def test_decoder_ep_overlap_configuration_passes_with_vpp():
    validate_mdp_config(
        MdpConfig(enable=True),
        _options(
            expert_parallel_size=2,
            pipeline_parallel_size=4,
            virtual_pipeline_parallel_size=2,
            overlap_moe_expert_parallel_comm=True,
        ),
    )


@pytest.mark.parametrize(
    "option_kwargs",
    [
        dict(expert_parallel_size=1, virtual_pipeline_parallel_size=2),
        dict(expert_parallel_size=2, virtual_pipeline_parallel_size=None),
    ],
)
def test_decoder_ep_overlap_rejects_missing_native_parallelism(option_kwargs):
    with pytest.raises(MdpConfigurationError, match="overlap_moe_expert_parallel_comm"):
        validate_mdp_config(
            MdpConfig(enable=True),
            _options(overlap_moe_expert_parallel_comm=True, **option_kwargs),
        )


def test_disabled_mdp_skips_all_checks():
    validate_mdp_config(MdpConfig(enable=False), _options(fsdp_enabled=True, bf16=False))


@pytest.mark.parametrize(
    "config_kwargs, match",
    [
        (dict(encoder_cp=2), "encoder_cp"),
        (dict(encoder_max_payload_rows=0), "encoder_max_payload_rows"),
        (dict(locality_slack_permille=1000), "locality_slack_permille"),
        (dict(locality_slack_permille=-1), "locality_slack_permille"),
        (dict(row_alignment=0), "row_alignment"),
        (dict(plan_check_interval=0), "plan_check_interval"),
        (dict(vision_config_overrides=(("nonexistent_key", 1),)), "allowlist"),
        (
            dict(
                vision_config_overrides=(
                    ("recompute_granularity", "full"),
                    ("recompute_granularity", "full"),
                )
            ),
            "unique",
        ),
        (
            dict(
                vision_config_overrides=(
                    ("recompute_num_layers", 1),
                    ("recompute_granularity", "full"),
                )
            ),
            "sorted",
        ),
    ],
)
def test_invalid_mdp_config_fields_rejected(config_kwargs, match):
    with pytest.raises(MdpConfigurationError, match=match):
        validate_mdp_config(MdpConfig(enable=True, **config_kwargs), _options())


@pytest.mark.parametrize(
    "option_kwargs, match",
    [
        (dict(rank_order="tp-ep-dp-pp-cp"), "rank_order"),
        (dict(tensor_parallel_size=2), "tensor_parallel_size"),
        (
            dict(context_parallel_size=2, cp_partition_mode="contiguous"),
            "cp_partition_mode",
        ),
        (dict(world_size=6, pipeline_parallel_size=4), "world_size"),
        (dict(calculate_per_token_loss=False), "calculate_per_token_loss"),
        (dict(use_distributed_optimizer=False), "use_distributed_optimizer"),
        (dict(distributed_optimizer_instances=2), "distributed_optimizer_instances"),
        (dict(bf16=False), "fp16/bf16"),
        (dict(fsdp_enabled=True), "fsdp"),
        (dict(fp8_enabled=True), "fp8"),
        (dict(cuda_graph_enabled=True), "cuda_graph"),
        (dict(activation_offload_enabled=True), "activation_offload"),
        (dict(overlap_param_gather=True), "overlap_param_gather"),
        (
            dict(
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                overlap_param_gather_with_optimizer_step=True,
            ),
            "overlap_param_gather_with_optimizer_step",
        ),
        (dict(delay_grad_reduce=True), "delay_grad_reduce"),
        (
            dict(checkpoint_mode="fully_parallel", save_requested=True),
            "checkpoint_mode",
        ),
        (
            dict(checkpoint_mode="local", load_requested=True),
            "checkpoint_mode",
        ),
    ],
)
def test_rejection_list(option_kwargs, match):
    with pytest.raises(MdpConfigurationError, match=match):
        validate_mdp_config(MdpConfig(enable=True), _options(**option_kwargs))


def test_unsupported_checkpoint_mode_allowed_without_save_or_load():
    validate_mdp_config(MdpConfig(enable=True), _options(checkpoint_mode="local"))


def test_fp16_configuration_accepted_for_overflow_tests():
    validate_mdp_config(MdpConfig(enable=True), _options(bf16=False, fp16=True))


@pytest.mark.parametrize(
    "option_kwargs",
    [dict(overlap_grad_reduce=True), dict(overlap_grad_reduce=True, overlap_param_gather=True)],
)
def test_native_decoder_ddp_overlap_is_supported(option_kwargs):
    validate_mdp_config(MdpConfig(enable=True), _options(**option_kwargs))


def test_error_messages_carry_option_value_and_suggestion():
    try:
        validate_mdp_config(
            MdpConfig(enable=True), _options(calculate_per_token_loss=False)
        )
    except MdpConfigurationError as error:
        message = str(error)
        assert "calculate_per_token_loss=False" in message
        assert "Suggested value: True" in message
    else:
        pytest.fail("expected MdpConfigurationError")


# ---------------------- vision config overrides ----------------------


@dataclasses.dataclass
class _FakeTransformerConfig:
    recompute_granularity: object = None
    recompute_method: object = None
    recompute_num_layers: object = None
    recompute_modules: object = None
    hidden_size: int = 64

    def __post_init__(self):
        if self.recompute_granularity not in (None, "selective", "full"):
            raise ValueError(f"bad recompute_granularity {self.recompute_granularity}")


def test_apply_overrides_uses_dataclasses_replace():
    base = _FakeTransformerConfig()
    result = apply_vision_config_overrides(
        base,
        (("recompute_granularity", "full"), ("recompute_num_layers", 1)),
    )
    assert result is not base
    assert result.recompute_granularity == "full"
    assert result.recompute_num_layers == 1
    assert base.recompute_granularity is None


def test_apply_overrides_empty_returns_base():
    base = _FakeTransformerConfig()
    assert apply_vision_config_overrides(base, ()) is base


def test_apply_overrides_delegates_field_validation_to_post_init():
    with pytest.raises(ValueError, match="bad recompute_granularity"):
        apply_vision_config_overrides(
            _FakeTransformerConfig(), (("recompute_granularity", "everything"),)
        )


def test_apply_overrides_rejects_keys_outside_allowlist():
    assert "hidden_size" not in VISION_CONFIG_OVERRIDE_ALLOWLIST
    with pytest.raises(MdpConfigurationError, match="allowlist"):
        apply_vision_config_overrides(_FakeTransformerConfig(), (("hidden_size", 128),))


# ---------------------- args snapshot (integration) ----------------------


def _fake_args(**overrides):
    from types import SimpleNamespace

    base = dict(
        world_size=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        use_tp_pp_dp_mapping=False,
        virtual_pipeline_model_parallel_size=None,
        calculate_per_token_loss=True,
        use_distributed_optimizer=True,
        num_distributed_optimizer_instances=1,
        fp16=False,
        bf16=True,
        use_torch_fsdp2=False,
        use_custom_fsdp=False,
        use_megatron_fsdp=False,
        fp8=None,
        cuda_graph_impl="none",
        cpu_offloading=False,
        fine_grained_activation_offloading=False,
        offload_optimizer_states=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        delay_grad_reduce=False,
        overlap_moe_expert_parallel_comm=False,
        ckpt_format="torch_dist",
        save=None,
        load=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_snapshot_reports_the_real_rank_order():
    # --use-tp-pp-dp-mapping switches initialize_model_parallel to
    # 'tp-cp-ep-pp-dp'; the snapshot must report it so the rank-order guard
    # fires instead of building planning groups that do not match the real
    # decoder replicas.
    from megatron.core.mdp.integration import compatibility_options_from_args

    default_options = compatibility_options_from_args(_fake_args())
    assert default_options.rank_order == "tp-cp-ep-dp-pp"
    validate_mdp_config(MdpConfig(enable=True), default_options)

    remapped_options = compatibility_options_from_args(
        _fake_args(use_tp_pp_dp_mapping=True)
    )
    assert remapped_options.rank_order == "tp-cp-ep-pp-dp"
    with pytest.raises(MdpConfigurationError, match="rank_order"):
        validate_mdp_config(MdpConfig(enable=True), remapped_options)


def test_snapshot_reports_decoder_ep_overlap():
    from megatron.core.mdp.integration import compatibility_options_from_args

    options = compatibility_options_from_args(
        _fake_args(overlap_moe_expert_parallel_comm=True)
    )
    assert options.overlap_moe_expert_parallel_comm is True


def test_snapshot_reports_optimizer_step_param_gather_overlap():
    from megatron.core.mdp.integration import compatibility_options_from_args

    options = compatibility_options_from_args(
        _fake_args(overlap_param_gather_with_optimizer_step=True)
    )
    assert options.overlap_param_gather_with_optimizer_step
    with pytest.raises(MdpConfigurationError, match="overlap_param_gather_with_optimizer_step"):
        validate_mdp_config(MdpConfig(enable=True), options)


def test_snapshot_takes_the_larger_of_the_train_and_eval_microbatch_sizes():
    # Both loaders hand the collator a whole microbatch, and the eval one may be
    # the larger of the two, so the static cu_seqlens capacity must cover it.
    from megatron.core.mdp.integration import compatibility_options_from_args

    options = compatibility_options_from_args(
        _fake_args(micro_batch_size=4, eval_micro_batch_size=16)
    )
    assert options.max_samples_per_microbatch == 16


# ---------------------------------------------------------------------------
# Decoder context parallelism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_decoder_context_parallelism_is_accepted(cp_size):
    validate_mdp_config(
        MdpConfig(enable=True), _options(world_size=8 * cp_size, context_parallel_size=cp_size)
    )


def test_contiguous_cp_partition_is_rejected_only_when_cp_is_on():
    # At CP=1 the partition mode is inert, so it must not gate the run; at CP>1
    # it decides which rank owns each row and MDP inverts the zigzag map only.
    validate_mdp_config(
        MdpConfig(enable=True), _options(cp_partition_mode="contiguous")
    )
    with pytest.raises(MdpConfigurationError, match="cp_partition_mode"):
        validate_mdp_config(
            MdpConfig(enable=True),
            _options(
                world_size=16, context_parallel_size=2, cp_partition_mode="contiguous"
            ),
        )


def test_static_packing_rejects_an_odd_per_rank_length_under_cp():
    # build_static_thd_metadata asserts dummy_seq_len % (2*cp) == 0 mid-run;
    # this must fail at startup instead, and in BOTH packing modes -- the greedy
    # branch used to be the only place any CP alignment was checked.
    for greedy in (False, True):
        with pytest.raises(MdpConfigurationError, match="max_seqlen_per_dp_cp_rank"):
            validate_mdp_config(
                MdpConfig(enable=True, greedy_packing=greedy),
                _options(
                    world_size=16,
                    context_parallel_size=2,
                    thd_static_packing=True,
                    max_seqlen_per_dp_cp_rank=4097,
                    thd_max_packed_sequences=8,
                ),
            )


# ---------------------------------------------------------------------------
# Packing: greedy token budget and the MCore scheduler rejection
# ---------------------------------------------------------------------------


def test_mcore_packing_scheduler_is_rejected():
    # Not merely untested: training.py wraps the data iterator whenever this is
    # set, and DpBalancedScheduler.run then asserts on GPT-only sample keys and
    # drops pixel_values / image_grid_thw.
    with pytest.raises(MdpConfigurationError, match="sequence_packing_scheduler"):
        validate_mdp_config(
            MdpConfig(enable=True), _options(sequence_packing_scheduler="dp_balanced")
        )


def test_greedy_packing_requires_a_token_budget():
    with pytest.raises(MdpConfigurationError, match="max_seqlen_per_dp_cp_rank"):
        validate_mdp_config(MdpConfig(enable=True, greedy_packing=True), _options())


def test_greedy_packing_accepts_a_valid_budget():
    validate_mdp_config(
        MdpConfig(enable=True, greedy_packing=True),
        _options(max_seqlen_per_dp_cp_rank=8192, thd_max_packed_sequences=8),
    )


def test_greedy_budget_must_match_the_collator_row_alignment():
    # SP splits the packed rows across TP, so the budget must divide by TP.
    with pytest.raises(MdpConfigurationError, match="row alignment"):
        validate_mdp_config(
            MdpConfig(enable=True, greedy_packing=True),
            _options(
                tensor_parallel_size=4,
                sequence_parallel=True,
                max_seqlen_per_dp_cp_rank=8190,
            ),
        )


def test_greedy_packing_rejects_a_zero_sequence_cap():
    with pytest.raises(MdpConfigurationError, match="thd_max_packed_sequences"):
        validate_mdp_config(
            MdpConfig(enable=True, greedy_packing=True),
            _options(max_seqlen_per_dp_cp_rank=8192, thd_max_packed_sequences=0),
        )


def test_greedy_packing_is_independent_of_static_packing():
    # Task 2 needs greedy + eager integer alignment (no static pad) as its
    # honest baseline, so all four corners of the 2x2 must validate.
    for greedy in (False, True):
        for static in (False, True):
            validate_mdp_config(
                MdpConfig(enable=True, greedy_packing=greedy),
                _options(
                    thd_static_packing=static,
                    max_seqlen_per_dp_cp_rank=8192,
                    thd_max_packed_sequences=8,
                ),
            )


def test_static_packing_reserves_a_sequence_slot_for_the_padding_tail():
    # thd_max_packed_sequences is the FINAL cu_seqlens capacity. Under static
    # packing the tail becomes an ordinary dummy sequence, so a bin filled to
    # the full cap would need cap + 2 entries and die inside _pad_cu_seqlens.
    eager = _options(max_seqlen_per_dp_cp_rank=8192, thd_max_packed_sequences=8)
    static = _options(
        max_seqlen_per_dp_cp_rank=8192, thd_max_packed_sequences=8, thd_static_packing=True
    )
    assert greedy_max_real_sequences(eager) == 8
    assert greedy_max_real_sequences(static) == 7
    assert greedy_max_real_sequences(_options()) is None


def test_static_packing_needs_room_for_a_real_sequence_and_the_dummy():
    with pytest.raises(MdpConfigurationError, match="thd_max_packed_sequences >= 2"):
        validate_mdp_config(
            MdpConfig(enable=True, greedy_packing=True),
            _options(
                max_seqlen_per_dp_cp_rank=8192,
                thd_max_packed_sequences=1,
                thd_static_packing=True,
            ),
        )


def test_static_only_packing_reserves_the_dummy_slot_for_a_full_microbatch():
    # Without greedy packing a microbatch is exactly micro_batch_size samples
    # (eval_micro_batch_size on the eval loaders), and the padding tail adds one
    # more sequence, so a cap equal to that count overflows the cu_seqlens
    # capacity inside _pad_cu_seqlens.
    options = _options(
        max_samples_per_microbatch=8,
        thd_static_packing=True,
        max_seqlen_per_dp_cp_rank=8192,
        thd_max_packed_sequences=8,
    )
    with pytest.raises(MdpConfigurationError, match="eval_micro_batch_size\\) \\+ 1"):
        validate_mdp_config(MdpConfig(enable=True), options)
    validate_mdp_config(
        MdpConfig(enable=True), dataclasses.replace(options, thd_max_packed_sequences=9)
    )


def test_static_only_packing_slot_check_does_not_apply_without_static_packing():
    validate_mdp_config(
        MdpConfig(enable=True),
        _options(
            max_samples_per_microbatch=8,
            max_seqlen_per_dp_cp_rank=8192,
            thd_max_packed_sequences=8,
        ),
    )


@pytest.mark.parametrize("checkpoint_kwargs", [dict(save_requested=True), dict(load_requested=True)])
def test_greedy_packing_is_rejected_with_checkpointing(checkpoint_kwargs):
    # The greedy sample buffer carries across iterations and is not
    # checkpointed, and the sampler cannot be repositioned per DP rank.
    options = _options(
        max_seqlen_per_dp_cp_rank=8192, thd_max_packed_sequences=8, **checkpoint_kwargs
    )
    with pytest.raises(MdpConfigurationError, match="greedy_packing"):
        validate_mdp_config(MdpConfig(enable=True, greedy_packing=True), options)
    validate_mdp_config(
        MdpConfig(enable=True, greedy_packing=True, greedy_packing_approximate_resume=True),
        options,
    )


def test_checkpointing_without_greedy_packing_is_unaffected():
    validate_mdp_config(
        MdpConfig(enable=True), _options(save_requested=True, load_requested=True)
    )
