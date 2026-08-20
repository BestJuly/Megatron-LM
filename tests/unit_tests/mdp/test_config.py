# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-compute tests for MdpConfig validation and the vision config override
channel. No distributed state, no CUDA."""

import dataclasses

import pytest

from megatron.core.mdp.config import (
    VISION_CONFIG_OVERRIDE_ALLOWLIST,
    MdpCompatibilityOptions,
    MdpConfig,
    apply_vision_config_overrides,
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
        delay_grad_reduce=False,
        checkpoint_mode="torch_dist",
        save_requested=False,
        load_requested=False,
    )
    base.update(overrides)
    return MdpCompatibilityOptions(**base)


def test_valid_configuration_passes():
    validate_mdp_config(MdpConfig(enable=True), _options())


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
        (dict(context_parallel_size=2), "context_parallel_size"),
        (dict(world_size=6, pipeline_parallel_size=4), "world_size"),
        (dict(calculate_per_token_loss=False), "calculate_per_token_loss"),
        (dict(use_distributed_optimizer=False), "use_distributed_optimizer"),
        (dict(distributed_optimizer_instances=2), "distributed_optimizer_instances"),
        (dict(bf16=False), "fp16/bf16"),
        (dict(fsdp_enabled=True), "fsdp"),
        (dict(fp8_enabled=True), "fp8"),
        (dict(cuda_graph_enabled=True), "cuda_graph"),
        (dict(activation_offload_enabled=True), "activation_offload"),
        (dict(overlap_grad_reduce=True), "overlap_grad_reduce"),
        (dict(overlap_param_gather=True), "overlap_param_gather"),
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
        delay_grad_reduce=False,
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
