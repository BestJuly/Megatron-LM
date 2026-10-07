# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.8-VL model package: registry entry, vision config, and the factory's argument checks.

Numerical coverage (HF parity, MDP vs native, the proxy matrix) lives in the pilot's
validation harness; these tests pin the wiring that would otherwise fail silently.
"""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.models import MODEL_REGISTRY, registry_for
from examples.multimodal_dev.models.qwen35_vl.configuration import QWEN35_VL_IMAGE_TOKEN_ID
from examples.multimodal_dev.models.qwen35_vl.mdp_adapter import (
    build_collate_spec,
    build_mdp_adapter,
)
from examples.multimodal_dev.models.qwen38_vl import (
    MROPE_SECTION,
    QWEN38_VL_EOS_TOKEN_ID,
    QWEN38_VL_IMAGE_TOKEN_ID,
    check_qwen38_vl_args,
    get_qwen38_vl_vision_config,
    resolve_image_token_id,
)


def _args(**overrides):
    base = dict(
        hybrid_layer_pattern="GEGEGEQE/QE",
        position_embedding_type="mrope",
        enable_mhc_connections=True,
        mhc_connection_variant="gated_residual",
        use_packed_sequence=True,
        qsa_use_sparse_attention=True,
        engram_enabled=True,
        engram_variant="qwen",
        engram_eos_token_id=QWEN38_VL_EOS_TOKEN_ID,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


_CONFIG = SimpleNamespace(mrope_section=list(MROPE_SECTION))


def test_registry_entry_reuses_the_qwen35_vision_side():
    entry = registry_for(SimpleNamespace(model_arch="qwen38_vl"))
    assert entry is MODEL_REGISTRY["qwen38_vl"]
    # Identical ViT / glue in HF qwen4_exp: the collate spec and MDP adapter are shared.
    assert entry["vision_collate_spec_fn"] is build_collate_spec
    assert entry["mdp_adapter_fn"] is build_mdp_adapter
    # Decoder through the default HybridModelBuilder hook (no custom builder).
    assert "language_model_builder_fn" not in entry
    assert {"mock", "mdp_mock"} <= set(entry["dataset_providers"])


def test_token_ids_match_qwen35():
    assert QWEN38_VL_IMAGE_TOKEN_ID == QWEN35_VL_IMAGE_TOKEN_ID
    assert resolve_image_token_id(SimpleNamespace(image_token_id=None)) == 248056
    assert resolve_image_token_id(SimpleNamespace(image_token_id=7)) == 7


@pytest.mark.parametrize(("variant", "layers"), [("full", 27), ("proxy", 4), (None, 27)])
def test_vision_config_is_the_large_qwen_vit(variant, layers):
    cfg = get_qwen38_vl_vision_config(variant=variant)
    assert (cfg.num_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.ffn_hidden_size) == (
        layers,
        1152,
        16,
        4304,
    )
    assert get_qwen38_vl_vision_config(num_layers_override=2, variant="full").num_layers == 2
    with pytest.raises(ValueError, match="Unknown Qwen3.8-VL variant"):
        get_qwen38_vl_vision_config(variant="9b")


def test_factory_accepts_the_reference_configuration():
    check_qwen38_vl_args(_args(), _CONFIG)
    check_qwen38_vl_args(_args(engram_enabled=False, engram_variant=None), _CONFIG)
    # The dense-mask QSA bridge is fine for BSHD.
    check_qwen38_vl_args(_args(use_packed_sequence=False, qsa_use_sparse_attention=False), _CONFIG)


@pytest.mark.parametrize(
    ("overrides", "config", "match"),
    [
        (dict(hybrid_layer_pattern=None), _CONFIG, "hybrid-layer-pattern"),
        (dict(position_embedding_type="rope"), _CONFIG, "position-embedding-type mrope"),
        ({}, SimpleNamespace(mrope_section=[3, 3, 2]), "mrope-section 11 11 10"),
        (dict(enable_mhc_connections=False), _CONFIG, "gated_residual"),
        (dict(mhc_connection_variant="mhc"), _CONFIG, "gated_residual"),
        (dict(qsa_use_sparse_attention=False), _CONFIG, "qsa-use-sparse-attention"),
        (dict(engram_variant="deepseek"), _CONFIG, "engram-variant qwen"),
        (dict(engram_eos_token_id=0), _CONFIG, "engram-eos-token-id"),
    ],
)
def test_factory_rejects_a_different_decoder(overrides, config, match):
    with pytest.raises(ValueError, match=match):
        check_qwen38_vl_args(_args(**overrides), config)
