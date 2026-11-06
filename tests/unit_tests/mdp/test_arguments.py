# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CLI contract tests for multimodal encoder recompute."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.arguments import validate_encoder_recompute_args


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
    "name, value",
    [
        ("encoder_recompute_granularity", "whole"),
        ("encoder_recompute_method", "uniform"),
        ("encoder_recompute_num_layers", 1),
        ("encoder_recompute_modules", ["mlp"]),
    ],
)
def test_encoder_recompute_flags_require_mdp(name, value):
    with pytest.raises(RuntimeError, match="currently require --mdp-enable"):
        validate_encoder_recompute_args(_args(mdp_enable=False, **{name: value}))


def test_encoder_recompute_flags_are_accepted_with_mdp():
    validate_encoder_recompute_args(
        _args(
            mdp_enable=True,
            encoder_recompute_granularity="full",
            encoder_recompute_method="uniform",
            encoder_recompute_num_layers=1,
        )
    )


def test_unset_encoder_recompute_flags_are_accepted_without_mdp():
    validate_encoder_recompute_args(_args(mdp_enable=False))
