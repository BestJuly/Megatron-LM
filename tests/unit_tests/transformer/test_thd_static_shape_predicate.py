# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The shared "are THD shapes static" predicate behind the THD CUDA-graph path."""

from types import SimpleNamespace

import pytest

from megatron.core.packed_seq_params import thd_shapes_are_static


def _config(**overrides):
    base = dict(
        sequence_packing_scheduler=None,
        dynamic_context_parallel=False,
        thd_static_packing=False,
        pad_packed_seq_alignment=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(sequence_packing_scheduler="dp_balanced"),
        dict(sequence_packing_scheduler="default_dynamic_cp"),
        dict(dynamic_context_parallel=True),
        dict(thd_static_packing=True),
    ],
)
def test_every_fixed_shape_producer_is_recognized(overrides):
    assert thd_shapes_are_static(_config(**overrides))


def test_no_producer_means_dynamic_shapes():
    assert not thd_shapes_are_static(_config())


def test_alignment_alone_does_not_imply_static_shapes():
    """Existing GPT --sft runs set an alignment without a scheduler.

    Deriving the predicate from ``pad_packed_seq_alignment`` would silently
    change their behavior; an explicit opt-in cannot.
    """
    assert not thd_shapes_are_static(_config(pad_packed_seq_alignment="max"))
    assert not thd_shapes_are_static(_config(pad_packed_seq_alignment=4096))


def test_absent_attributes_are_tolerated():
    """Callers pass both TransformerConfig and the argparse Namespace."""
    assert not thd_shapes_are_static(SimpleNamespace())
    assert thd_shapes_are_static(SimpleNamespace(thd_static_packing=True))
