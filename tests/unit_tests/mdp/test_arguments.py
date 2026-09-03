# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CLI contract tests for the multimodal encoder recompute and FP8 arguments."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.arguments import (
    encoder_fp8_overrides_from_args,
    encoder_recompute_overrides_from_args,
    validate_encoder_fp8_args,
    validate_encoder_recompute_args,
)

_DEFAULTS = {
    "encoder_recompute_granularity": None,
    "encoder_recompute_method": None,
    "encoder_recompute_num_layers": None,
    "encoder_recompute_modules": None,
    "encoder_fp8": False,
    "fp8": None,
    "fp8_recipe": "delayed",
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


@pytest.mark.parametrize(
    "mdp_enable, options, match",
    [
        # Nothing to inherit: encoder-only FP8 is not supported.
        (True, {"encoder_fp8": True}, "requires --fp8-format"),
        # Payload-row alignment lives in the MDP adapter; the native path would
        # abort inside TE on the first unaligned vision batch.
        (False, {"encoder_fp8": True, "fp8": "hybrid"}, "requires --mdp-enable"),
        (True, {"encoder_fp8": True, "fp8": "hybrid", "fp8_recipe": "mxfp8"}, None),
        (False, {}, None),
    ],
)
def test_encoder_fp8_args_matrix(mdp_enable, options, match):
    args = _args(mdp_enable=mdp_enable, **options)
    if match is None:
        validate_encoder_fp8_args(args)
        overrides = encoder_fp8_overrides_from_args(args)
        # The decoder's format and recipe, nothing else -- no FP8 attention.
        assert overrides == ({"fp8": "hybrid", "fp8_recipe": "mxfp8"} if options else {})
    else:
        with pytest.raises(RuntimeError, match=match):
            validate_encoder_fp8_args(args)
