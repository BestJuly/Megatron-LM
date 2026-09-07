# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CLI contract tests for multimodal encoder recompute."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.arguments import (
    encoder_recompute_overrides_from_args,
    validate_encoder_recompute_args,
)


_DEFAULTS = {
    "encoder_recompute_granularity": None,
    "encoder_recompute_method": None,
    "encoder_recompute_num_layers": None,
    "encoder_recompute_modules": None,
}


def _args(*, mdp_enable, **overrides):
    values = dict(_DEFAULTS)
    values.update(overrides)
    return SimpleNamespace(mdp_enable=mdp_enable, **values)


@pytest.mark.parametrize(
    "granularity, options",
    [
        (
            "selective",
            {"encoder_recompute_modules": ["core_attn", "mlp"]},
        ),
        (
            "full",
            {
                "encoder_recompute_method": "uniform",
                "encoder_recompute_num_layers": 1,
            },
        ),
    ],
)
def test_native_transformer_recompute_is_accepted_without_mdp(granularity, options):
    validate_encoder_recompute_args(
        _args(
            mdp_enable=False,
            encoder_recompute_granularity=granularity,
            **options,
        )
    )


def test_whole_encoder_recompute_requires_mdp():
    with pytest.raises(RuntimeError, match="whole requires --mdp-enable"):
        validate_encoder_recompute_args(
            _args(
                mdp_enable=False,
                encoder_recompute_granularity="whole",
            )
        )


def test_whole_encoder_recompute_is_accepted_with_mdp():
    validate_encoder_recompute_args(
        _args(
            mdp_enable=True,
            encoder_recompute_granularity="whole",
        )
    )


@pytest.mark.parametrize(
    "granularity, option, value",
    [
        (None, "encoder_recompute_method", "uniform"),
        (None, "encoder_recompute_num_layers", 1),
        (None, "encoder_recompute_modules", ["mlp"]),
        ("whole", "encoder_recompute_method", "uniform"),
        ("selective", "encoder_recompute_method", "uniform"),
        ("selective", "encoder_recompute_num_layers", 1),
        ("full", "encoder_recompute_modules", ["mlp"]),
    ],
)
def test_encoder_recompute_rejects_options_for_other_modes(granularity, option, value):
    with pytest.raises(RuntimeError, match="cannot be used with"):
        validate_encoder_recompute_args(
            _args(
                mdp_enable=True,
                encoder_recompute_granularity=granularity,
                **{option: value},
            )
        )


def test_native_encoder_recompute_overrides_are_typed():
    overrides = encoder_recompute_overrides_from_args(
        _args(
            mdp_enable=False,
            encoder_recompute_granularity="selective",
            encoder_recompute_modules=("core_attn", "mlp"),
        )
    )

    assert overrides == {
        "recompute_granularity": "selective",
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": ["core_attn", "mlp"],
    }


@pytest.mark.parametrize("granularity", [None, "whole"])
def test_non_native_recompute_modes_do_not_override_transformer_config(granularity):
    assert (
        encoder_recompute_overrides_from_args(
            _args(
                mdp_enable=granularity == "whole",
                encoder_recompute_granularity=granularity,
            )
        )
        == {}
    )
