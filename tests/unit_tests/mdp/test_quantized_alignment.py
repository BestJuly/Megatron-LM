# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-compute tests for ``quantized_row_alignment``. No distributed state,
no CUDA.

The function lives in ``examples/multimodal_dev/forward_step.py`` next to the
collation code it feeds, but it only reads an ``args`` namespace, so it is
tested here instead of in ``examples/multimodal_dev/tests/``: everything there
collates real tensors on ``device="cuda"`` behind a module-scoped
``Utils.initialize_model_parallel`` fixture, and the directory sits outside
``testpaths`` in ``pyproject.toml``, so none of it runs on CPU CI.
"""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.forward_step import _row_alignment, quantized_row_alignment

# Recipes with no derivable row multiple, each with the substring its rejection
# must name: --fp4-format is mutually exclusive with --fp8-format and has no
# ``get_fp8_align_size`` counterpart, and "custom" reports 16 there though a
# custom block-scaled quantizer may need 32 as MXFP8 does.
UNDERIVABLE_RECIPES = [
    pytest.param(dict(fp4="e2m1"), "fp4", id="fp4"),
    pytest.param(dict(fp8="hybrid", fp8_recipe="custom"), "custom", id="custom_fp8_recipe"),
]


def _alignment_args(*, packed=True, **overrides):
    """``args`` in the shape ``quantized_row_alignment`` reads it."""
    fields = dict(fp4=None, fp8=None, fp8_recipe=None, use_packed_sequence=packed)
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestQuantizedRowAlignment:
    """``quantized_row_alignment`` is the only source of ``pad_to_multiple``."""

    def test_no_quantization_means_no_alignment(self):
        assert quantized_row_alignment(_alignment_args()) is None

    @pytest.mark.parametrize("fp8_recipe, expected", [("delayed", 16), ("mxfp8", 32)])
    def test_recipe_selects_the_row_multiple(self, fp8_recipe, expected):
        """``get_fp8_align_size`` branches on MXFP8 alone -- it quantizes in
        32-element blocks -- and every other recipe falls through to the 16 the
        backward wgrad GEMM needs. ``delayed`` is --fp8-recipe's default and
        stands for that else branch."""
        args = _alignment_args(fp8="hybrid", fp8_recipe=fp8_recipe)
        assert quantized_row_alignment(args) == expected

    @pytest.mark.parametrize("overrides, match", UNDERIVABLE_RECIPES)
    def test_underivable_recipes_are_rejected_on_the_packed_path(self, overrides, match):
        """Packed sequences are concatenated at the samples' own lengths, so
        this return value is their only aligner and an underivable recipe has
        to raise."""
        with pytest.raises(NotImplementedError, match=match):
            quantized_row_alignment(_alignment_args(**overrides))

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(fp8="hybrid", fp8_recipe="delayed"),
            dict(fp8="hybrid", fp8_recipe="mxfp8"),
            dict(fp8="hybrid", fp8_recipe="custom"),
            dict(fp4="e2m1"),
        ],
        ids=["fp8", "mxfp8", "custom", "fp4"],
    )
    def test_bshd_gets_no_alignment_for_any_recipe(self, overrides):
        """Without --use-packed-sequence this contributes nothing, so BSHD
        collation is byte-identical to base -- including the recipes that raise
        on the packed path. MDP requires packed sequences and never reaches
        BSHD; the native path is out of scope."""
        assert quantized_row_alignment(_alignment_args(packed=False, **overrides)) is None

    def test_computed_once_per_argument_set(self):
        """get_batch asks per microbatch; equal launch args must not recompute."""
        _row_alignment.cache_clear()
        args = _alignment_args(fp8="hybrid", fp8_recipe="mxfp8")
        assert quantized_row_alignment(args) == 32
        assert quantized_row_alignment(args) == 32
        assert _row_alignment.cache_info().misses == 1
        assert _row_alignment.cache_info().hits == 1
