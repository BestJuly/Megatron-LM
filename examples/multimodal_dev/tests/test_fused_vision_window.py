# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev import forward_step, fused_vision_window
from examples.multimodal_dev.data import energon_mdp
from examples.multimodal_dev.data.energon_mdp import (
    MDPWindowMaterializingIterator,
    loader_prepartition_window_size,
)
from examples.multimodal_dev.mdp_image_materialize import encode_image_descriptors
from examples.multimodal_dev.mdp_pipeline_sidecar import cp_fused_vision_requested
from examples.multimodal_dev.models import base
from examples.multimodal_dev.sidecar_prefetch import (
    image_vision_pack_plan,
    sidecar_prefetch_window_count,
)


def test_full_step_window_uses_global_num_microbatches():
    assert sidecar_prefetch_window_count(True, current_microbatch=0, num_microbatches=16) == 16
    assert sidecar_prefetch_window_count(True, current_microbatch=1, num_microbatches=16) == 0


@pytest.mark.parametrize("inner_scope", ["cp", "pp_cp"])
def test_loader_full_step_window_is_sixteen_microbatches(inner_scope):
    args = SimpleNamespace(
        mdp_fused_vision_window=True,
        mdp_vision_encoder_max_sequence_length=262_144,
        global_batch_size=512,
        micro_batch_size=1,
        data_parallel_size=32,
    )

    assert (
        loader_prepartition_window_size(args, loader_prepartition=True, inner_scope=inner_scope)
        == 16
    )


@pytest.mark.parametrize("inner_scope", ["cp", "pp_cp"])
@pytest.mark.parametrize(("fused_window", "expected"), [(False, 1), (True, 4)])
def test_loader_window_contract_matches_cp_scopes(inner_scope, fused_window, expected):
    args = SimpleNamespace(
        mdp_fused_vision_window=fused_window,
        mdp_vision_encoder_max_sequence_length=262_144,
        global_batch_size=8,
        micro_batch_size=1,
        data_parallel_size=2,
    )

    assert (
        loader_prepartition_window_size(args, loader_prepartition=True, inner_scope=inner_scope)
        == expected
    )


def test_fused_window_accepts_post_broadcast_assignment_dict():
    from examples.multimodal_dev.fused_vision_window import _assignment_rows

    assert _assignment_rows(
        {"_mdp_prepartitioned_assignment": {1: [(2, 3)], 0: [(0, 1), (0, 2)]}}
    ) == [[0, 0, 1], [0, 0, 2], [1, 2, 3]]


def _lazy_batch(batch_id, descriptor_ids, raw_patch_counts):
    descriptors = [{"id": int(descriptor_id)} for descriptor_id in descriptor_ids]
    return {
        "batch_id": int(batch_id),
        "tokens": torch.tensor([batch_id], dtype=torch.long),
        "image_grid_thw": torch.tensor(
            [(1, 1, int(count)) for count in raw_patch_counts], dtype=torch.long
        ),
        "image_cu_seqlens": torch.tensor([0, len(raw_patch_counts)], dtype=torch.int32),
        "_mdp_image_descriptors_json": encode_image_descriptors(descriptors),
    }


def _planning_iterator(batches, *, rank, across_items):
    return MDPWindowMaterializingIterator(
        iter(batches),
        lookahead_microbatches=2,
        prefetch_windows=1,
        rank=rank,
        world=2,
        pixel_dim=3,
        patch_size=1,
        spatial_merge_size=1,
        lpt_hidden_size=8,
        balance_across_microbatches=across_items,
        materialize_workers=1,
    )


def test_full_window_lpt_differs_from_per_microbatch_planning(monkeypatch):
    monkeypatch.setattr(
        energon_mdp,
        "materialize_descriptor",
        lambda _descriptor, grid, *, pixel_dim, patch_size: torch.zeros(
            int(grid[0]) * int(grid[1]) * int(grid[2]), pixel_dim
        ),
    )
    batches = [_lazy_batch(0, [0, 1], [100, 1]), _lazy_batch(1, [2, 3], [99, 98])]
    fused = _planning_iterator(deepcopy(batches), rank=0, across_items=True)
    per_microbatch = _planning_iterator(deepcopy(batches), rank=0, across_items=False)

    fused_assignments = [next(fused)["_mdp_prepartitioned_assignment"].tolist() for _ in range(2)]
    per_microbatch_assignments = [
        next(per_microbatch)["_mdp_prepartitioned_assignment"].tolist() for _ in range(2)
    ]

    assert fused_assignments == [[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]]]
    assert per_microbatch_assignments == [[[0, 0, 0], [1, 0, 1]], [[0, 0, 0], [1, 0, 1]]]


def test_json_lazy_descriptors_materialize_only_selected_owner_in_order(monkeypatch):
    materialized = []

    def materialize(descriptor, grid, *, pixel_dim, patch_size):
        del patch_size
        materialized.append(int(descriptor["id"]))
        raw_patches = int(grid[0]) * int(grid[1]) * int(grid[2])
        return torch.full((raw_patches, pixel_dim), float(descriptor["id"]))

    monkeypatch.setattr(energon_mdp, "materialize_descriptor", materialize)
    iterator = _planning_iterator(
        [_lazy_batch(10, [10, 1], [100, 1]), _lazy_batch(20, [8], [80])], rank=1, across_items=True
    )

    first = next(iterator)
    second = next(iterator)

    assert [first["batch_id"], second["batch_id"]] == [10, 20]
    assert sorted(materialized) == [1, 8]
    assert 10 not in materialized
    assert first["pixel_values"].shape == (1, 3)
    assert second["pixel_values"].shape == (80, 3)
    assert "_mdp_image_descriptors_json" not in first
    assert "_mdp_image_descriptors_json" not in second
    assert iterator._closed is True


def test_planning_iterator_closes_after_materialization_failure(monkeypatch):
    def fail_materialization(*_args, **_kwargs):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(energon_mdp, "materialize_descriptor", fail_materialization)
    iterator = _planning_iterator([_lazy_batch(0, [0], [1])], rank=0, across_items=True)

    with pytest.raises(RuntimeError, match="decode failed"):
        next(iterator)

    assert iterator._closed is True


def test_planning_iterator_closes_when_initial_planning_fails(monkeypatch):
    shutdown_calls = []

    class TrackingExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 1

        def shutdown(self, *, wait, cancel_futures):
            shutdown_calls.append((wait, cancel_futures))

    def fail_planning(*_args, **_kwargs):
        raise RuntimeError("planning failed")

    monkeypatch.setattr(energon_mdp, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr(energon_mdp, "assign_images_lpt", fail_planning)

    with pytest.raises(RuntimeError, match="planning failed"):
        _planning_iterator(
            [_lazy_batch(0, [0], [1])],
            rank=0,
            across_items=True,
        )

    assert shutdown_calls == [(False, True)]


def test_raw_patch_pack_plan_honors_262144_cap():
    lengths = [[200_000, 70_000], [192_000, 64_000, 1]]

    plan = image_vision_pack_plan(lengths, 262_144)

    assert plan == [[0, 4], [2, 1], [3]]
    flat = [length for microbatch in lengths for length in microbatch]
    assert all(sum(flat[index] for index in pack) <= 262_144 for pack in plan)


def test_raw_patch_pack_plan_keeps_oversized_image_unsplit():
    assert image_vision_pack_plan([[300_000, 100_000, 100_000]], 262_144) == [[0], [1, 2]]


def test_pipeline_hook_enqueues_full_window_only_once(monkeypatch):
    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model._pipeline_sidecar_enabled = True
    model.vp_stage = None
    args = SimpleNamespace(
        mdp_fused_vision_window=True, mdp_vision_encoder_max_sequence_length=262_144
    )
    calls = []

    monkeypatch.setattr(base, "get_args", lambda: args)
    monkeypatch.setattr(base.parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1)

    def build_window(**kwargs):
        calls.append(kwargs)
        return [
            {
                "batch": {"microbatch": index},
                "vision_embeddings": torch.empty(0, 1),
                "forward_only": False,
            }
            for index in range(kwargs["count"])
        ]

    monkeypatch.setattr(forward_step, "build_mdp_pp_cp_sidecar_cache_window", build_window)
    monkeypatch.setattr(
        forward_step,
        "build_mdp_pp_cp_sidecar_cache",
        lambda **_kwargs: pytest.fail("full fused window used the single-batch path"),
    )

    for microbatch in range(16):
        model.pipeline_sidecar_pre_forward(
            data_iterator=object(),
            current_microbatch=microbatch,
            num_microbatches=16,
            forward_only=False,
        )

    assert len(calls) == 1
    assert calls[0]["count"] == 16
    assert len(model._mdp_pp_cp_sidecar_cache) == 16


def test_pipeline_hook_uses_fused_builder_for_single_microbatch(monkeypatch):
    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model._pipeline_sidecar_enabled = True
    model.vp_stage = None
    args = SimpleNamespace(
        mdp_fused_vision_window=True, mdp_vision_encoder_max_sequence_length=131_072
    )
    calls = []

    monkeypatch.setattr(base, "get_args", lambda: args)

    def build_window(**kwargs):
        calls.append(kwargs)
        return [
            {
                "batch": {"microbatch": 0},
                "vision_embeddings": torch.empty(0, 1),
                "forward_only": False,
            }
        ]

    monkeypatch.setattr(forward_step, "build_mdp_pp_cp_sidecar_cache_window", build_window)
    monkeypatch.setattr(
        forward_step,
        "build_mdp_pp_cp_sidecar_cache",
        lambda **_kwargs: pytest.fail("fused mode used the non-fused builder"),
    )

    model.pipeline_sidecar_pre_forward(
        data_iterator=object(), current_microbatch=0, num_microbatches=1, forward_only=False
    )

    assert len(calls) == 1
    assert calls[0]["count"] == 1
    assert calls[0]["max_sequence_length"] == 131_072
    assert len(model._mdp_pp_cp_sidecar_cache) == 1


@pytest.mark.parametrize("inner_scope", ["cp", "pp_cp"])
def test_cp_only_fused_window_is_requested(inner_scope):
    args = SimpleNamespace(
        mdp_encoder_mode=True,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        mdp_inner_dp_scope=inner_scope,
        text_only=False,
        use_packed_sequence=True,
        mdp_fused_vision_window=True,
        mdp_vision_encoder_max_sequence_length=262_144,
    )
    assert cp_fused_vision_requested(args)


@pytest.mark.parametrize("inner_scope", ["cp", "pp_cp"])
def test_cp_only_window_off_does_not_enable_fused_sidecar(inner_scope):
    args = SimpleNamespace(
        mdp_encoder_mode=True,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        mdp_inner_dp_scope=inner_scope,
        text_only=False,
        use_packed_sequence=True,
        mdp_fused_vision_window=False,
        mdp_vision_encoder_max_sequence_length=262_144,
    )

    assert not cp_fused_vision_requested(args)


def test_pp2_pp_cp_does_not_select_cp_only_fused_sidecar():
    args = SimpleNamespace(
        mdp_encoder_mode=True,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        mdp_inner_dp_scope="pp_cp",
        text_only=False,
        use_packed_sequence=True,
        mdp_fused_vision_window=True,
        mdp_vision_encoder_max_sequence_length=262_144,
    )

    assert not cp_fused_vision_requested(args)


class _BackwardHarness(torch.nn.Module):
    pre_process = True
    _pipeline_sidecar_enabled = True

    mdp_pp_cp_sidecar_activate_cache = base.MultimodalModel.mdp_pp_cp_sidecar_activate_cache
    pipeline_sidecar_post_backward = base.MultimodalModel.pipeline_sidecar_post_backward

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def mdp_pp_cp_sidecar_compute_vision(self, *, pixel_values, image_grid_thw, mdp_cp_local_plan):
        del image_grid_thw, mdp_cp_local_plan
        return self.vision_model(pixel_values)


@pytest.mark.parametrize("backward_mode", ["retain", "recompute"])
def test_fused_backward_cache_matches_direct_gradient(monkeypatch, backward_mode):
    monkeypatch.setattr(
        "examples.multimodal_dev.mdp_batch.apply_mdp_prepartition",
        lambda *, pixel_values, image_grid_thw, **_kwargs: (pixel_values, image_grid_thw),
    )
    torch.manual_seed(2026)
    vision_model = torch.nn.Linear(3, 2, bias=False)
    reference = deepcopy(vision_model)
    harness = _BackwardHarness(vision_model)
    inputs = torch.randn(4, 3)
    first_weight = torch.randn(2, 2)
    second_weight = torch.randn(2, 2)

    if backward_mode == "retain":
        output = vision_model(inputs)
    else:
        with torch.no_grad():
            output = vision_model(inputs)
    first_leaf = output[:2].detach().requires_grad_(True)
    second_leaf = output[2:].detach().requires_grad_(True)
    state = fused_vision_window._PackBackwardState(
        pixel_values=inputs,
        image_grid_thw=torch.tensor([[1, 1, 2], [1, 1, 2]]),
        assignment={0: [(0, 0), (1, 1)]},
        image_indices=[0, 1],
        row_counts=[2, 2],
        remaining_microbatches=2,
        backward_mode=backward_mode,
        retained_output=output if backward_mode == "retain" else None,
    )

    harness.mdp_pp_cp_sidecar_activate_cache(
        {
            "vision_embeddings": first_leaf,
            "fused_backward_entries": [fused_vision_window._BackwardEntry(state, 0, first_leaf)],
            "forward_only": False,
        }
    )
    harness.mdp_pp_cp_sidecar_activate_cache(
        {
            "vision_embeddings": second_leaf,
            "fused_backward_entries": [fused_vision_window._BackwardEntry(state, 1, second_leaf)],
            "forward_only": False,
        }
    )

    (first_leaf * first_weight).sum().backward()
    harness.pipeline_sidecar_post_backward()
    assert vision_model.weight.grad is None
    (second_leaf * second_weight).sum().backward()
    harness.pipeline_sidecar_post_backward()

    expected = reference(inputs)
    ((expected[:2] * first_weight).sum() + (expected[2:] * second_weight).sum()).backward()

    torch.testing.assert_close(vision_model.weight.grad, reference.weight.grad)
    assert state.done is True
    assert state.remaining_microbatches == 0
