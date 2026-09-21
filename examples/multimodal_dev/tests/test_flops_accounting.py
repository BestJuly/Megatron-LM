# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for multimodal FLOPs accounting."""

from types import SimpleNamespace

import pytest
import torch

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
    monkeypatch.setattr(
        forward_step, "accumulate_flops_stats", lambda *_, **__: calls.append("decoder")
    )
    monkeypatch.setattr(
        forward_step, "accumulate_vision_flops_stats_from_items", lambda _: calls.append("vision")
    )

    model = _Wrapper(SimpleNamespace(vp_stage=vp_stage))
    forward_step._accumulate_workload_stats(
        model, packed_seq_params=object(), vision_items=[object()]
    )

    assert calls == expected_calls


def test_workload_stats_native_path_uses_grid_metadata(monkeypatch):
    """The native path keeps reporting grid-based vision statistics."""
    calls = []
    grid = object()
    monkeypatch.setattr(
        forward_step, "accumulate_flops_stats", lambda *_, **__: calls.append("decoder")
    )
    monkeypatch.setattr(
        forward_step,
        "accumulate_vision_flops_stats_from_grids",
        lambda value: calls.append(("vision_grid", value)),
    )

    forward_step._accumulate_workload_stats(
        SimpleNamespace(vp_stage=0), packed_seq_params=object(), image_grid_thw=grid
    )

    assert calls == ["decoder", ("vision_grid", grid)]


@pytest.mark.parametrize("source", ["mdp", "native_none", "native_empty"])
def test_text_only_microbatch_reports_zero_vision_work(monkeypatch, source):
    calls = []
    monkeypatch.setattr(forward_step, "_update_vision_stats", lambda *values: calls.append(values))
    if source == "mdp":
        forward_step.accumulate_vision_flops_stats_from_items([])
    else:
        grid = None if source == "native_none" else torch.empty(0, 3, dtype=torch.long)
        forward_step.accumulate_vision_flops_stats_from_grids(grid)
    assert calls == [(0.0, 0.0)]


@pytest.mark.parametrize("source", ["mdp", "native"])
def test_vision_stats_reduce_with_text_only_dp_replicas(source):
    import megatron.training.training as training
    from megatron.core import parallel_state
    from tests.unit_tests.test_utilities import Utils

    if Utils.world_size < 4 or Utils.world_size % 2:
        pytest.skip("requires an even world of at least four ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    try:
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        # First iteration has vision on half the replicas; the second is all text.
        for all_text in (False, True):
            training.reset_vision_stats_in_iteration()
            has_vision = not all_text and dp_rank % 2 == 0
            if source == "mdp":
                items = [SimpleNamespace(grid_thw=(1, 8, 8))] if has_vision else []
                forward_step.accumulate_vision_flops_stats_from_items(items)
            else:
                grid = torch.tensor([[1, 8, 8]], device="cuda") if has_vision else None
                forward_step.accumulate_vision_flops_stats_from_grids(grid)

            # Fail on every rank before the faulty conditional collective could hang.
            active = torch.tensor(int(training._vision_stats_active), device="cuda")
            torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MIN)
            assert active.item() == 1
            replicas_with_vision = 0 if all_text else (dp_size + 1) // 2
            assert training.consume_vision_stats_in_iteration() == (
                64 * replicas_with_vision,
                4096 * replicas_with_vision,
            )
    finally:
        training.reset_vision_stats_in_iteration()
        Utils.destroy_model_parallel()
