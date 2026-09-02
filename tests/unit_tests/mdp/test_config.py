# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-compute tests for MDP and encoder recompute configuration.

No distributed state or CUDA.
"""

import dataclasses

import pytest

from megatron.core.mdp.config import (
    MdpCompatibilityOptions,
    MdpConfig,
    apply_encoder_recompute_config,
    validate_effective_vision_config,
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
        (
            dict(encoder_recompute_granularity="partial"),
            "encoder_recompute_granularity",
        ),
        (dict(encoder_recompute_method="uniform"), "encoder_recompute_method"),
        (dict(encoder_recompute_num_layers=1), "encoder_recompute_num_layers"),
        (dict(encoder_recompute_modules=("mlp",)), "encoder_recompute_modules"),
        (
            dict(
                encoder_recompute_granularity="whole",
                encoder_recompute_method="uniform",
            ),
            "encoder_recompute_method",
        ),
        (
            dict(
                encoder_recompute_granularity="selective",
                encoder_recompute_method="uniform",
            ),
            "encoder_recompute_method",
        ),
        (
            dict(
                encoder_recompute_granularity="selective",
                encoder_recompute_num_layers=1,
            ),
            "encoder_recompute_num_layers",
        ),
        (
            dict(
                encoder_recompute_granularity="full",
                encoder_recompute_modules=("mlp",),
            ),
            "encoder_recompute_modules",
        ),
        (dict(locality_slack_permille=1000), "locality_slack_permille"),
        (dict(locality_slack_permille=-1), "locality_slack_permille"),
        (dict(row_alignment=0), "row_alignment"),
        (dict(plan_check_interval=0), "plan_check_interval"),
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


def test_whole_encoder_recompute_without_native_options_is_valid():
    validate_mdp_config(
        MdpConfig(enable=True, encoder_recompute_granularity="whole"), _options()
    )


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


# ---------------------- encoder recompute config ----------------------


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


def test_apply_full_encoder_recompute_uses_dataclasses_replace():
    base = _FakeTransformerConfig()
    result = apply_encoder_recompute_config(
        base,
        MdpConfig(
            enable=True,
            encoder_recompute_granularity="full",
            encoder_recompute_method="uniform",
            encoder_recompute_num_layers=1,
        ),
    )
    assert result is not base
    assert result.recompute_granularity == "full"
    assert result.recompute_method == "uniform"
    assert result.recompute_num_layers == 1
    assert base.recompute_granularity is None


def test_apply_selective_encoder_recompute_copies_modules_to_a_list():
    result = apply_encoder_recompute_config(
        _FakeTransformerConfig(),
        MdpConfig(
            enable=True,
            encoder_recompute_granularity="selective",
            encoder_recompute_modules=("core_attn", "mlp"),
        ),
    )
    assert result.recompute_granularity == "selective"
    assert result.recompute_modules == ["core_attn", "mlp"]


@pytest.mark.parametrize("granularity", [None, "whole"])
def test_disabled_and_whole_recompute_leave_transformer_config_unchanged(granularity):
    base = _FakeTransformerConfig()
    config = MdpConfig(enable=True, encoder_recompute_granularity=granularity)
    assert apply_encoder_recompute_config(base, config) is base


@pytest.mark.parametrize("recompute_granularity", ["full", "selective"])
def test_whole_encoder_recompute_rejects_effective_vision_recompute(
    recompute_granularity,
):
    with pytest.raises(
        MdpConfigurationError, match="effective vision recompute_granularity"
    ):
        validate_effective_vision_config(
            MdpConfig(enable=True, encoder_recompute_granularity="whole"),
            _FakeTransformerConfig(recompute_granularity=recompute_granularity),
        )


def test_apply_encoder_recompute_delegates_field_validation_to_post_init():
    with pytest.raises(ValueError, match="bad recompute_granularity"):
        apply_encoder_recompute_config(
            _FakeTransformerConfig(),
            MdpConfig(enable=True, encoder_recompute_granularity="everything"),
        )


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


@pytest.mark.parametrize(
    "arg_overrides, expected",
    [
        (
            dict(
                encoder_recompute_granularity="selective",
                encoder_recompute_modules=["core_attn", "mlp"],
            ),
            ("selective", None, None, ("core_attn", "mlp")),
        ),
        (
            dict(
                encoder_recompute_granularity="full",
                encoder_recompute_method="uniform",
                encoder_recompute_num_layers=1,
            ),
            ("full", "uniform", 1, None),
        ),
        (
            dict(encoder_recompute_granularity="whole"),
            ("whole", None, None, None),
        ),
    ],
)
def test_encoder_recompute_options_are_snapshotted_from_args(arg_overrides, expected):
    from megatron.core.mdp.integration import mdp_config_from_args

    config = mdp_config_from_args(_fake_args(mdp_enable=True, **arg_overrides))
    actual = (
        config.encoder_recompute_granularity,
        config.encoder_recompute_method,
        config.encoder_recompute_num_layers,
        config.encoder_recompute_modules,
    )
    assert actual == expected
