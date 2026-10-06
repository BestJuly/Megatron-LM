# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for the multimodal_dev HybridModel migration."""

import math
from types import SimpleNamespace

import pytest

from examples.multimodal_dev.models.qwen35_vl import factory
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    MROPE_SECTION,
    ROTARY_PERCENT,
    get_qwen35_vl_language_config,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_linear_attention_pattern,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_hybrid_total_layer_count,
    parse_hybrid_pattern,
)

# The four-block cadence documented in the README and emitted by
# scripts/run_qwen35_vl.sh: three GatedDeltaNet blocks then one full-attention
# block, each block contributing an attention layer and an MLP layer.
MOE_PATTERN_4_BLOCKS = "GEGEGE*E"
DENSE_PATTERN_4_BLOCKS = "G-G-G-*-"
# The former GPT path built the same cadence from --linear-attention-freq 4.
QWEN35_LINEAR_ATTENTION_FREQ = 4


def _args(**overrides):
    values = {
        "hybrid_layer_pattern": "GEGEGE*E/*E",
        "untie_embeddings_and_output_weights": True,
        "padded_vocab_size": 1024,
        "max_position_embeddings": 4096,
        "image_token_id": 42,
        "position_embedding_type": "mrope",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _language_config(**overrides):
    values = {"mrope_section": list(MROPE_SECTION)}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_factory_requires_hybrid_layer_pattern():
    with pytest.raises(ValueError, match="requires --hybrid-layer-pattern"):
        factory.build_model(
            _args(hybrid_layer_pattern=None),
            language_config=_language_config(),
            vision_config=SimpleNamespace(),
        )


def test_factory_accepts_architecture_mrope_section(monkeypatch):
    """The architecture value passes the factory check."""
    monkeypatch.setattr(
        "examples.multimodal_dev.models.qwen35_vl.model.Qwen35VLModel",
        lambda **kwargs: "model",
    )

    assert (
        factory.build_model(
            _args(),
            language_config=_language_config(),
            vision_config=SimpleNamespace(),
        )
        == "model"
    )


def test_factory_rejects_mrope_section_with_wrong_partition():
    """A section with the right total width but the wrong T/H/W split is rejected.

    [12, 10, 10] sums to the same 32 as MROPE_SECTION, so no generic
    TransformerConfig check catches it, but it misplaces the channel boundaries.
    """
    wrong_split = [12, 10, 10]
    assert sum(wrong_split) == sum(MROPE_SECTION)

    with pytest.raises(ValueError, match="requires --mrope-section 11 11 10"):
        factory.build_model(
            _args(),
            language_config=_language_config(mrope_section=wrong_split),
            vision_config=SimpleNamespace(),
        )


def test_factory_builds_decoder_through_the_hook(monkeypatch):
    """The factory never constructs the decoder itself: it calls the registry
    hook, pinning only the architectural RoPE constants."""
    from examples.multimodal_dev.models.qwen35_vl.configuration import ROTARY_BASE

    captured = {}
    hook_calls = []

    def fake_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(decoder=kwargs["language_model_builder"]())

    def fake_hook(**overrides):
        hook_calls.append(overrides)
        return "decoder"

    monkeypatch.setattr(
        "examples.multimodal_dev.models.qwen35_vl.model.Qwen35VLModel", fake_model
    )
    result = factory.build_model(
        _args(),
        language_config=_language_config(),
        vision_config=SimpleNamespace(),
        language_model_builder=fake_hook,
    )

    assert result.decoder == "decoder"
    assert hook_calls == [{"rotary_percent": ROTARY_PERCENT, "rotary_base": ROTARY_BASE}]
    assert captured["image_token_id"] == 42
    # Decoder-shaped kwargs belong to the builder now, not the model class.
    for gone in ("hybrid_stack_spec", "hybrid_layer_pattern", "vocab_size"):
        assert gone not in captured


def test_factory_rejects_non_mrope_position_embedding():
    """A recipe declaring plain rope must fail loudly, not build an MRoPE decoder."""
    with pytest.raises(ValueError, match="requires --position-embedding-type mrope"):
        factory.build_model(
            _args(position_embedding_type="rope"),
            language_config=_language_config(),
            vision_config=SimpleNamespace(),
            language_model_builder=lambda **_: pytest.fail("decoder must not be built"),
        )


def test_default_language_model_builder_goes_through_hybrid_builder(monkeypatch):
    """``build_language_model`` = hybrid_config_from_args + HybridModelBuilder,
    with the caller's TransformerConfig, overrides applied, and placement kept."""
    import megatron.training.argument_utils as argument_utils
    import megatron.training.models.hybrid as hybrid
    from examples.multimodal_dev.models.base import build_language_model

    seen = {}

    class FakeConfig(SimpleNamespace):
        pass

    def fake_config_from_args(args, config=None):
        seen["config"] = config
        return FakeConfig(rotary_percent=1.0, rotary_base=10000)

    class FakeBuilder:
        def __init__(self, model_config):
            seen["model_config"] = model_config

        def build_model(self, pg_collection, pre_process, post_process, vp_stage):
            seen["placement"] = (pg_collection, pre_process, post_process, vp_stage)
            return "decoder"

    monkeypatch.setattr(argument_utils, "hybrid_config_from_args", fake_config_from_args)
    monkeypatch.setattr(hybrid, "HybridModelBuilder", FakeBuilder)
    language_config = object()
    pg = object()

    out = build_language_model(
        SimpleNamespace(), language_config, pre_process=True, post_process=False,
        vp_stage=1, pg_collection=pg, rotary_percent=0.25,
    )

    assert out == "decoder"
    assert seen["config"] is language_config
    assert seen["model_config"].rotary_percent == 0.25
    assert seen["model_config"].rotary_base == 10000
    assert seen["placement"] == (pg, True, False, 1)
    with pytest.raises(AttributeError, match="no field 'not_a_field'"):
        build_language_model(
            SimpleNamespace(), language_config, pre_process=True, post_process=True,
            pg_collection=pg, not_a_field=1,
        )


def test_mrope_section_matches_rotary_dimension():
    """--mrope-section must sum to half the rotary dimension of the decoder."""
    kv_channels = 256  # every language variant in configuration.py
    rotary_dim = int(kv_channels * ROTARY_PERCENT)

    assert sum(MROPE_SECTION) == rotary_dim // 2


@pytest.mark.parametrize("pattern", [MOE_PATTERN_4_BLOCKS, DENSE_PATTERN_4_BLOCKS])
def test_pattern_reproduces_gpt_block_structure(pattern):
    """Each hybrid block is one attention layer followed by one MLP layer."""
    layers = list(parse_hybrid_pattern(pattern).main_pattern)

    attention_layers = layers[0::2]
    mlp_layers = layers[1::2]
    assert all(s in (Symbols.GDN, Symbols.ATTENTION) for s in attention_layers)
    assert all(s in (Symbols.MLP, Symbols.MOE) for s in mlp_layers)
    assert len(attention_layers) == len(mlp_layers)
    # The GPT path applied the same MLP kind to every block.
    assert len(set(mlp_layers)) == 1


@pytest.mark.parametrize("pattern", [MOE_PATTERN_4_BLOCKS, DENSE_PATTERN_4_BLOCKS])
def test_pattern_matches_gpt_linear_attention_layout(pattern):
    """GDN / full-attention placement matches --linear-attention-freq 4."""
    attention_layers = list(parse_hybrid_pattern(pattern).main_pattern)[0::2]

    gpt_config = SimpleNamespace(
        num_layers=len(attention_layers),
        linear_attention_freq=QWEN35_LINEAR_ATTENTION_FREQ,
        experimental_attention_variant="gated_delta_net",
    )
    # get_linear_attention_pattern returns 1 for linear attention, 0 for SDPA.
    expected = [
        Symbols.GDN if is_linear else Symbols.ATTENTION
        for is_linear in get_linear_attention_pattern(gpt_config)
    ]
    assert attention_layers == expected


def test_mtp_pattern_replicates_final_block():
    """GPT MTP copied the final decoder layer; the MTP pattern must match it."""
    parsed = parse_hybrid_pattern(f"{MOE_PATTERN_4_BLOCKS}/{MOE_PATTERN_4_BLOCKS[-2:]}")

    assert parsed.mtp_num_depths == 1
    assert parsed.mtp_pattern == parsed.main_pattern[-2:]


def test_language_config_num_layers_is_hybrid_layer_count():
    """The variant config must be expressed in hybrid layers, not GPT blocks."""
    # The proxy variant is four Qwen blocks, i.e. eight hybrid layers.
    config = get_qwen35_vl_language_config("proxy")

    assert config.num_layers == get_hybrid_total_layer_count(MOE_PATTERN_4_BLOCKS)
    # Without is_hybrid_model the doubled count would also scale the output-layer
    # init std, so the config must declare which unit num_layers is in.
    assert config.is_hybrid_model
    assert config.output_layer_init_method.keywords["std"] == pytest.approx(
        config.init_method_std / math.sqrt(config.num_layers)
    )


def test_model_provider_binds_registry_language_model_builder(monkeypatch):
    """The entry point hands the factory the registry's decoder hook, bound to
    this stage's placement; the factory only adds config overrides."""
    import examples.multimodal_dev.pretrain_multimodal as training_entrypoint
    from examples.multimodal_dev.models import MODEL_REGISTRY

    args = SimpleNamespace(model_arch="qwen35_vl", model_variant=None, vision_num_layers=None)
    language_config = SimpleNamespace(apply_rope_fusion=False, bf16=True, fp16=False)
    hook_calls = []

    def hook(hook_args, hook_language_config, **kwargs):
        hook_calls.append((hook_args, hook_language_config, kwargs))
        return "decoder"

    def build_model(*, language_model_builder, **kwargs):
        return language_model_builder(rotary_base=7)

    registry = {
        "vision_config_fn": lambda **kwargs: SimpleNamespace(),
        "model_factory_fn": build_model,
        "language_model_builder_fn": hook,
    }
    monkeypatch.setattr(training_entrypoint, "get_args", lambda: args)
    monkeypatch.setattr(
        training_entrypoint, "core_transformer_config_from_args", lambda args: language_config
    )
    monkeypatch.setitem(MODEL_REGISTRY, "qwen35_vl", registry)

    assert training_entrypoint.model_provider(pre_process=False, post_process=True, vp_stage=1) == (
        "decoder"
    )
    assert hook_calls == [
        (
            args,
            language_config,
            dict(
                pre_process=False,
                post_process=True,
                vp_stage=1,
                pg_collection=None,
                rotary_base=7,
            ),
        )
    ]
