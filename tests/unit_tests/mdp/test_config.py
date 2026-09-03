# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-compute tests for MDP and encoder recompute configuration.

No distributed state or CUDA.
"""

import dataclasses

import pytest

from megatron.core.mdp.config import (
    MdpCompatibilityOptions,
    MdpConfig,
    apply_encoder_fp8_config,
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


@pytest.mark.parametrize(
    "decoder_fp8, decoder_recipe, encoder_fp8, match",
    [
        # The encoder inherits the decoder's recipe, so the accepting rows also
        # pin that every ENCODER_COMPATIBLE_FP8_RECIPES member is a real
        # Fp8Recipe value. decoder_recipe is never None: args.fp8_recipe defaults
        # to "delayed", so a None column would exercise these gates unreachably.
        ("hybrid", "tensorwise", True, None),
        ("hybrid", "blockwise", True, None),
        ("e4m3", "mxfp8", True, None),
        # Nothing to inherit: encoder-only FP8 is not supported.
        (None, "delayed", True, "enable decoder FP8 first"),
        (None, "tensorwise", True, "enable decoder FP8 first"),
        # A delayed-scaling (or custom) decoder cannot lend its recipe to the
        # encoder: the encoder's per-chunk, rank-dependent forward count would
        # push the amax history unevenly, and TE's global amax reduction on
        # every depth-0 autocast exit would give the decoder's own buffers
        # rank-dependent collective counts. --fp8-format hybrid alone leaves the
        # recipe at "delayed", so this is the most common FP8 command line.
        ("hybrid", "delayed", True, "fp8_recipe="),
        ("hybrid", "custom", True, "fp8_recipe="),
        # With the encoder off the decoder recipe is unrestricted (PR #55).
        ("hybrid", "delayed", False, None),
        (None, "delayed", False, None),
    ],
)
def test_encoder_fp8_gates(decoder_fp8, decoder_recipe, encoder_fp8, match):
    """The rejections config.py guards behind ``if config.encoder_fp8``."""
    config = MdpConfig(enable=True, encoder_fp8=encoder_fp8)
    options = _options(decoder_fp8=decoder_fp8, decoder_fp8_recipe=decoder_recipe)
    if match is None:
        validate_mdp_config(config, options)
    else:
        with pytest.raises(MdpConfigurationError, match=match):
            validate_mdp_config(config, options)


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
    fp8: object = None
    fp8_recipe: object = "delayed"
    ffn_hidden_size: int = 4320

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
        # Megatron's *production* default: TransformerConfig sets fp8_recipe to
        # "delayed" whether or not --fp8-format was passed. A getattr fallback
        # of None would disarm the decoder-amax gate in every args-level test.
        fp8_recipe="delayed",
        encoder_fp8=False,
        cuda_graph_impl="none",
        cpu_offloading=False,
        fine_grained_activation_offloading=False,
        offload_optimizer_states=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        delay_grad_reduce=False,
        overlap_moe_expert_parallel_comm=False,
        reuse_grad_buf_for_mxfp8_param_ag=False,
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


@pytest.mark.parametrize(
    "flag", ["overlap_param_gather_with_optimizer_step", "reuse_grad_buf_for_mxfp8_param_ag"]
)
def test_snapshot_carries_the_flags_the_rejections_read(flag):
    """compatibility_options_from_args() is the only place these args become
    MdpCompatibilityOptions state, and every field below defaults to False, so
    a key read under the wrong name leaves validate_mdp_config's rejection
    permanently inert -- and the suite green."""
    from megatron.core.mdp.integration import compatibility_options_from_args

    options = compatibility_options_from_args(_fake_args(**{flag: True}))
    assert getattr(options, flag), (
        f"compatibility_options_from_args() must snapshot args.{flag}, or "
        "validate_mdp_config's rejection of it can never fire on a real run"
    )
    with pytest.raises(MdpConfigurationError, match=flag):
        validate_mdp_config(MdpConfig(enable=True), options)


def test_decoder_fp8_is_accepted_by_the_support_matrix():
    # --fp8 configures the decoder; the encoder only inherits it under
    # --encoder-fp8. The snapshot carries the decoder format and recipe for
    # that path, and validate_mdp_config must not reject decoder-only FP8.
    from megatron.core.mdp.integration import compatibility_options_from_args

    options = compatibility_options_from_args(_fake_args(fp8="hybrid"))
    assert options.decoder_fp8 == "hybrid"
    # args.fp8_recipe carries the recipe the encoder inherits, and
    # test_encoder_fp8_gates builds decoder_fp8_recipe by hand, so this is the
    # one place args.fp8_recipe -> decoder_fp8_recipe is pinned.
    assert options.decoder_fp8_recipe == "delayed"
    validate_mdp_config(MdpConfig(enable=True), options)


@pytest.mark.parametrize("decoder_recipe, match", [("delayed", "fp8_recipe="), ("mxfp8", None)])
def test_decoder_amax_gate_fires_end_to_end_from_args(decoder_recipe, match):
    """--fp8-format hybrid leaves --fp8-recipe at its "delayed" default, so the
    single most common FP8 command line is exactly the one the recipe gate has
    to reject once the encoder inherits it. A typo in the args.fp8_recipe read
    makes decoder_fp8_recipe None -- not in ENCODER_COMPATIBLE_FP8_RECIPES --
    which would reject even the valid row, and a typo in the --encoder-fp8 read
    leaves MdpConfig.encoder_fp8 False and every encoder gate inert; the two
    rows together pin all three reads."""
    from megatron.core.mdp.integration import mdp_config_from_args, validate_from_args

    args = _fake_args(mdp_enable=True, fp8="hybrid", fp8_recipe=decoder_recipe, encoder_fp8=True)
    assert mdp_config_from_args(args).encoder_fp8
    if match is None:
        validate_from_args(args)
    else:
        with pytest.raises(MdpConfigurationError, match=match):
            validate_from_args(args)


@pytest.mark.parametrize(
    "encoder_fp8, decoder_fp8, decoder_recipe, effective_fp8, effective_recipe, effective_ffn, match",
    [
        # fp8 on the built config that MdpConfig never asked for: the gates ran
        # against MdpConfig, so this encoder was never validated.
        (False, "hybrid", "mxfp8", "hybrid", "mxfp8", 4320, "effective vision fp8="),
        # Asked for and lost on the way: the run would silently be bf16.
        (True, "hybrid", "mxfp8", None, "delayed", 4320, "effective vision fp8="),
        # Same format, a recipe other than the decoder's: the recipe gate was
        # checked on the decoder's recipe, so this one bypassed it.
        (True, "hybrid", "mxfp8", "hybrid", "tensorwise", 4320, "effective vision fp8_recipe"),
        (True, "hybrid", "mxfp8", "hybrid", "mxfp8", 4320, None),
        (False, None, "delayed", None, "delayed", 4320, None),
        # The base vision config's own width, with no --encoder-ffn-hidden-size
        # in play: Qwen3.5-VL's 4304 is not a multiple of MXFP8's 32 and would
        # otherwise pass every validator and die inside TE on the first forward.
        (True, "hybrid", "mxfp8", "hybrid", "mxfp8", 4304, "effective vision ffn_hidden_size"),
        (True, "hybrid", "tensorwise", "hybrid", "tensorwise", 4304, None),
        (False, None, "delayed", None, "delayed", 4304, None),
    ],
)
def test_effective_vision_config_fp8_must_match_mdp_config(
    encoder_fp8, decoder_fp8, decoder_recipe, effective_fp8, effective_recipe, effective_ffn, match
):
    """validate_effective_vision_config is where encoder FP8 becomes observable;
    it must agree with the snapshot the encoder inherits from, and the built FFN
    width must be one the recipe can quantize."""
    config = MdpConfig(enable=True, encoder_fp8=encoder_fp8)
    options = _options(decoder_fp8=decoder_fp8, decoder_fp8_recipe=decoder_recipe)
    effective = _FakeTransformerConfig(
        fp8=effective_fp8, fp8_recipe=effective_recipe, ffn_hidden_size=effective_ffn
    )
    if match is None:
        validate_effective_vision_config(config, effective, options)
    else:
        with pytest.raises(MdpConfigurationError, match=match):
            validate_effective_vision_config(config, effective, options)


@pytest.mark.parametrize(
    "options, match",
    [
        # No snapshot at all: the encoder cannot know what to inherit.
        (None, "needs the compatibility snapshot"),
        # A snapshot with the decoder in bf16: nothing to inherit. Without its
        # own reject this would compare None == None against a bf16 vision
        # config and pass.
        ("decoder_off", "decoder FP8 enabled"),
    ],
)
def test_effective_vision_config_is_self_sufficient_under_encoder_fp8(options, match):
    if options == "decoder_off":
        options = _options(decoder_fp8=None, decoder_fp8_recipe="delayed")
    with pytest.raises(MdpConfigurationError, match=match):
        validate_effective_vision_config(
            MdpConfig(enable=True, encoder_fp8=True), _FakeTransformerConfig(), options
        )


def test_apply_encoder_fp8_config_copies_the_decoder_recipe():
    base = _FakeTransformerConfig()
    decoder = _options(decoder_fp8="hybrid", decoder_fp8_recipe="mxfp8")
    assert apply_encoder_fp8_config(base, MdpConfig(enable=True), decoder) is base
    assert apply_encoder_fp8_config(base, MdpConfig(enable=True), None) is base

    result = apply_encoder_fp8_config(base, MdpConfig(enable=True, encoder_fp8=True), decoder)
    assert result is not base
    assert (result.fp8, result.fp8_recipe) == ("hybrid", "mxfp8")
    assert base.fp8 is None
    # (encoder_fp8 with the decoder in bf16 is rejected upstream, by
    # validate_mdp_config and validate_effective_vision_config, and tested
    # there; the apply-level refusal is defense in depth only.)
    # The two adapters compose: recompute first, FP8 second, neither undoes the other.
    both = apply_encoder_fp8_config(
        apply_encoder_recompute_config(
            base,
            MdpConfig(
                enable=True,
                encoder_recompute_granularity="full",
                encoder_recompute_method="uniform",
                encoder_recompute_num_layers=1,
            ),
        ),
        MdpConfig(enable=True, encoder_fp8=True),
        _options(decoder_fp8="e4m3", decoder_fp8_recipe="tensorwise"),
    )
    assert (both.recompute_granularity, both.fp8, both.fp8_recipe) == ("full", "e4m3", "tensorwise")


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
