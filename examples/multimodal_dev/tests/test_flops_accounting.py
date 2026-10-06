# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for multimodal FLOPs accounting."""

from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev import forward_step
from examples.multimodal_dev.models.vision_base import grid_workload_fn


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
        forward_step, "accumulate_vision_flops_stats", lambda _: calls.append("vision")
    )

    model = _Wrapper(SimpleNamespace(vp_stage=vp_stage))
    forward_step._accumulate_workload_stats(
        model, packed_seq_params=object(), shape_keys=torch.tensor([[1, 4, 4]])
    )

    assert calls == expected_calls


def test_workload_stats_forward_shape_keys_and_real_cu_seqlens(monkeypatch):
    """Both arms hand their shape keys and the pre-tail cu_seqlens through verbatim."""
    calls = []
    keys = torch.tensor([[1, 4, 4]])
    real = torch.tensor([0, 7], dtype=torch.int32)
    monkeypatch.setattr(
        forward_step,
        "accumulate_flops_stats",
        lambda _psp, real_cu_seqlens=None: calls.append(("decoder", real_cu_seqlens)),
    )
    monkeypatch.setattr(
        forward_step, "accumulate_vision_flops_stats", lambda value: calls.append(("vision", value))
    )

    forward_step._accumulate_workload_stats(
        SimpleNamespace(vp_stage=0), packed_seq_params=object(), shape_keys=keys,
        real_cu_seqlens=real,
    )

    assert calls == [("decoder", real), ("vision", keys)]


def _install_grid_spec(monkeypatch, merge=2):
    """Route ``build_vision_collate_spec`` to a grid spec without a registry/args."""
    from examples.multimodal_dev import mdp_adapter

    spec = SimpleNamespace(workload_of=grid_workload_fn(merge))
    monkeypatch.setattr(mdp_adapter, "build_vision_collate_spec", lambda _args: spec)
    monkeypatch.setattr(forward_step, "get_args", lambda: SimpleNamespace())


@pytest.mark.parametrize("source", ["mdp", "native_none", "native_empty"])
def test_text_only_microbatch_reports_zero_vision_work(monkeypatch, source):
    """A text-only DP rank must still enter the global stats reduction."""
    calls = []
    monkeypatch.setattr(forward_step, "_update_vision_stats", lambda *values: calls.append(values))
    if source == "mdp":
        keys = forward_step.shape_keys_from_vision_items([])
    elif source == "native_none":
        keys = None
    else:
        keys = torch.empty(0, 3, dtype=torch.long)
    forward_step.accumulate_vision_flops_stats(keys)
    assert calls == [(0.0, 0.0, 0.0)]


@pytest.mark.parametrize("source", ["mdp", "native"])
def test_vision_stats_reduce_with_text_only_dp_replicas(monkeypatch, source):
    import megatron.training.training as training
    from megatron.core import parallel_state
    from tests.unit_tests.test_utilities import Utils

    if Utils.world_size < 4 or Utils.world_size % 2:
        pytest.skip("requires an even world of at least four ranks")
    _install_grid_spec(monkeypatch)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    try:
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        # First iteration has vision on half the replicas; the second is all text.
        for all_text in (False, True):
            training.reset_vision_stats_in_iteration()
            has_vision = not all_text and dp_rank % 2 == 0
            if source == "mdp":
                items = [SimpleNamespace(shape_key=(1, 8, 8))] if has_vision else []
                keys = forward_step.shape_keys_from_vision_items(items)
            else:
                keys = torch.tensor([[1, 8, 8]], device="cuda") if has_vision else None
            forward_step.accumulate_vision_flops_stats(keys)

            # Fail on every rank before the faulty conditional collective could hang.
            active = torch.tensor(int(training._vision_stats_active), device="cuda")
            torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MIN)
            assert active.item() == 1
            replicas_with_vision = 0 if all_text else (dp_size + 1) // 2
            assert training.consume_vision_stats_in_iteration() == (
                64 * replicas_with_vision,
                4096 * replicas_with_vision,
                16 * replicas_with_vision,
            )
    finally:
        training.reset_vision_stats_in_iteration()
        Utils.destroy_model_parallel()


def test_shape_keys_from_vision_items_round_trip():
    """MDP records carry shape keys as plain tuples on every rank."""
    items = [
        SimpleNamespace(shape_key=(1, 4, 4)),
        SimpleNamespace(shape_key=(2, 4, 8)),
    ]
    keys = forward_step.shape_keys_from_vision_items(items)
    assert keys.tolist() == [[1, 4, 4], [2, 4, 8]]
    assert forward_step.shape_keys_from_vision_items([]) is None


def test_grid_workload_matches_the_closed_form():
    """The grid family reports patch rows, frame-wise L^2, and merged rows.

    Frames, not whole items, are the attention chunks: a ``(2, 4, 8)`` item is
    two chunks of 32, so the ``L^2`` term is ``2 * 32**2``, not ``64**2``.
    """
    keys = torch.tensor([[1, 4, 4], [2, 4, 8]])
    rows, attn_sq, merged = grid_workload_fn(2)(keys)
    assert rows.item() == 1 * 16 + 2 * 32
    assert attn_sq.item() == 1 * 16**2 + 2 * 32**2
    assert merged.item() == 1 * 2 * 2 + 2 * 2 * 4


def test_grid_workload_is_device_resident():
    """No host sync on the hot path: the reductions stay where the input is."""
    keys = torch.tensor([[1, 4, 4]])
    rows, attn_sq, merged = grid_workload_fn(2)(keys)
    for value in (rows, attn_sq, merged):
        assert value.device == keys.device
        assert value.dim() == 0
