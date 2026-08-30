# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from examples.multimodal_dev.models.qwen35_vl import factory, specs


def test_qwen_language_spec_composes_engram(monkeypatch):
    raw_spec = object()
    engram_config = object()
    wrapped_spec = object()
    args = SimpleNamespace()
    language_config = object()

    monkeypatch.setattr(
        factory,
        "get_qwen35_vl_language_spec",
        lambda **kwargs: raw_spec,
    )
    monkeypatch.setattr(
        factory,
        "EngramConfig",
        SimpleNamespace(from_args=lambda actual_args, actual_config: engram_config),
    )

    def apply_engram(actual_spec, actual_engram_config):
        assert actual_spec is raw_spec
        assert actual_engram_config is engram_config
        return wrapped_spec

    monkeypatch.setattr(factory, "apply_engram_to_layer_spec", apply_engram)

    result = factory._build_language_spec(args, language_config, vp_stage=None)

    assert result is wrapped_spec


def test_qwen_language_spec_is_unchanged_without_engram(monkeypatch):
    raw_spec = object()
    args = SimpleNamespace()
    language_config = object()

    monkeypatch.setattr(
        factory,
        "get_qwen35_vl_language_spec",
        lambda **kwargs: raw_spec,
    )
    monkeypatch.setattr(
        factory,
        "EngramConfig",
        SimpleNamespace(from_args=lambda actual_args, actual_config: None),
    )

    assert factory._build_language_spec(args, language_config, vp_stage=None) is raw_spec


def test_vision_rope_wrapper_forwards_current_mcore_arguments(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_apply(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(specs, "_apply_rope_fp32", fake_apply)

    result = specs._apply_rope_fp32_no_cp(
        "tokens",
        "frequencies",
        "config",
        cu_seqlens="cu_seqlens",
        mscale=0.5,
        cp_group="language_cp_group",
        mla_rotary_interleaved=True,
        inverse=True,
        mla_output_remove_interleaving=True,
        max_seqlen=128,
    )

    assert result is sentinel
    assert captured["args"] == ("tokens", "frequencies", "config")
    assert captured["kwargs"] == {
        "cu_seqlens": "cu_seqlens",
        "mscale": 0.5,
        "cp_group": specs._NO_CP_GROUP,
        "mla_rotary_interleaved": True,
        "inverse": True,
        "mla_output_remove_interleaving": True,
        "max_seqlen": 128,
    }
