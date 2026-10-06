# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared code must not privilege one architecture (CPU only)."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.models import MODEL_REGISTRY, registry_for
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    resolve_image_token_id,
)


@pytest.mark.parametrize("model_arch", [None, "not_a_model"])
def test_registry_has_no_implicit_default_arch(model_arch):
    with pytest.raises(ValueError, match="Unknown model arch"):
        registry_for(SimpleNamespace(model_arch=model_arch))


def test_registry_lookup():
    assert registry_for(SimpleNamespace(model_arch="qwen35_vl")) is MODEL_REGISTRY["qwen35_vl"]


def test_cli_has_no_architecture_defaults():
    import argparse

    parser = add_multimodal_args(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args([])  # --model-arch is required
    args = parser.parse_args(["--model-arch", "qwen35_vl"])
    assert args.image_token_id is None


@pytest.mark.parametrize(
    ("cli_value", "expected"), [(None, QWEN35_VL_IMAGE_TOKEN_ID), (17, 17)]
)
def test_image_token_id_falls_back_to_the_architecture(cli_value, expected):
    assert resolve_image_token_id(SimpleNamespace(image_token_id=cli_value)) == expected
    assert resolve_image_token_id(SimpleNamespace()) == QWEN35_VL_IMAGE_TOKEN_ID
