# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for multimodal FLOPs accounting."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev import forward_step


class _Wrapper:
    """Minimal DDP-like wrapper for exercising wrapped-model lookup."""

    def __init__(self, module):
        self.module = module


@pytest.mark.parametrize(
    ("vp_stage", "expected_calls"),
    [(None, ["decoder", "vision"]), (0, ["decoder", "vision"]), (1, [])],
)
def test_workload_stats_report_once_per_physical_rank(monkeypatch, vp_stage, expected_calls):
    """Only the canonical VPP chunk reports a physical rank's micro-batch."""
    calls = []
    monkeypatch.setattr(forward_step, "accumulate_flops_stats", lambda _: calls.append("decoder"))
    monkeypatch.setattr(
        forward_step,
        "accumulate_vision_flops_stats_from_items",
        lambda _: calls.append("vision"),
    )

    model = _Wrapper(SimpleNamespace(vp_stage=vp_stage))
    forward_step._accumulate_workload_stats(
        model,
        packed_seq_params=object(),
        vision_items=[object()],
    )

    assert calls == expected_calls


def test_workload_stats_native_path_uses_grid_metadata(monkeypatch):
    """The native path keeps reporting grid-based vision statistics."""
    calls = []
    grid = object()
    monkeypatch.setattr(forward_step, "accumulate_flops_stats", lambda _: calls.append("decoder"))
    monkeypatch.setattr(
        forward_step,
        "accumulate_vision_flops_stats_from_grids",
        lambda value: calls.append(("vision_grid", value)),
    )

    forward_step._accumulate_workload_stats(
        SimpleNamespace(vp_stage=0),
        packed_seq_params=object(),
        image_grid_thw=grid,
    )

    assert calls == ["decoder", ("vision_grid", grid)]
