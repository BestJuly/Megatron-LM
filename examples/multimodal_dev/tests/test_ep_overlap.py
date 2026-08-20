# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contract tests for multimodal decoder EP communication overlap."""

from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev import forward_step
from examples.multimodal_dev.models.base import MultimodalModel
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule


class _FakeLanguageModel(torch.nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_calls = 0
        self.forward_inputs = None
        self.plan_inputs = None
        self.plan = object()

    def embedding(self, input_ids, position_ids):
        self.embedding_calls += 1
        seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        return torch.zeros(seq_length, batch_size, self.hidden_size)

    def forward(self, **kwargs):
        self.forward_inputs = kwargs
        return kwargs["decoder_input"]

    def build_schedule_plan(self, **kwargs):
        self.plan_inputs = kwargs
        return self.plan


class _FakeVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixel_values, image_grid_thw):
        self.calls += 1
        return pixel_values + 1


def _make_model(*, pre_process=True):
    model = MultimodalModel.__new__(MultimodalModel)
    MegatronModule.__init__(model, config=SimpleNamespace(sequence_parallel=False))
    model.image_token_id = 7
    model.pre_process = pre_process
    model.post_process = False
    model.vp_stage = 0
    model.language_model = _FakeLanguageModel()
    model.vision_model = _FakeVisionModel() if pre_process else None
    return model


@pytest.fixture(autouse=True)
def _single_rank_parallel_state(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)


def test_schedule_plan_uses_mdp_leaf_without_reencoding_pixels():
    model = _make_model()
    input_ids = torch.tensor([[7, 3, 7]])
    vision_leaf = torch.tensor(
        [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]], requires_grad=True
    )
    padding_mask = torch.tensor([[False, True, False]])

    plan = model.build_schedule_plan(
        input_ids=input_ids,
        vision_embeddings=vision_leaf,
        pixel_values=torch.full_like(vision_leaf, -1),
        padding_mask=padding_mask,
    )

    assert plan is model.language_model.plan
    assert model.vision_model.calls == 0
    plan_inputs = model.language_model.plan_inputs
    assert torch.equal(plan_inputs["decoder_input"][[0, 2], 0], vision_leaf)
    assert plan_inputs["padding_mask"] is padding_mask

    plan_inputs["decoder_input"].sum().backward()
    assert torch.equal(vision_leaf.grad, torch.ones_like(vision_leaf))


def test_schedule_plan_keeps_native_vision_preprocessing_outside_decoder_plan():
    model = _make_model()
    input_ids = torch.tensor([[7, 3, 7]])
    pixel_values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    plan = model.build_schedule_plan(input_ids=input_ids, pixel_values=pixel_values)

    assert plan is model.language_model.plan
    assert model.vision_model.calls == 1
    assert torch.equal(
        model.language_model.plan_inputs["decoder_input"][[0, 2], 0], pixel_values + 1
    )


def test_schedule_plan_input_preparation_matches_eager_forward():
    model_inputs = dict(
        input_ids=torch.tensor([[7, 3, 7]]),
        position_ids=None,
        attention_mask=None,
        labels=torch.tensor([[3, 7, 4]]),
        loss_mask=torch.ones(1, 3),
        padding_mask=torch.tensor([[False, True, False]]),
        pixel_values=torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        image_grid_thw=torch.tensor([[1, 1, 2]]),
        packed_seq_params=None,
    )

    eager_model = _make_model()
    eager_model(**model_inputs)

    scheduled_model = _make_model()
    plan = scheduled_model.build_schedule_plan(**model_inputs)

    assert plan is scheduled_model.language_model.plan
    assert eager_model.vision_model.calls == scheduled_model.vision_model.calls == 1
    for name in (
        "input_ids",
        "position_ids",
        "attention_mask",
        "decoder_input",
        "labels",
        "loss_mask",
        "padding_mask",
        "packed_seq_params",
    ):
        eager_value = eager_model.language_model.forward_inputs[name]
        scheduled_value = scheduled_model.language_model.plan_inputs[name]
        if eager_value is None:
            assert scheduled_value is None
        else:
            torch.testing.assert_close(scheduled_value, eager_value)


def test_schedule_plan_non_preprocess_chunk_skips_embedding_and_vision():
    model = _make_model(pre_process=False)

    plan = model.build_schedule_plan(
        input_ids=torch.tensor([[7, 3, 7]]), decoder_input=torch.ones(3, 1, 4)
    )

    assert plan is model.language_model.plan
    assert model.language_model.embedding_calls == 0
    assert model.language_model.plan_inputs["decoder_input"] is None


class _ForwardModel:
    def __init__(self):
        self.forward_calls = 0
        self.plan_calls = 0
        self.inputs = None

    def __call__(self, **kwargs):
        self.forward_calls += 1
        self.inputs = kwargs
        return "eager-output"

    def build_schedule_plan(self, **kwargs):
        self.plan_calls += 1
        self.inputs = kwargs
        return "schedule-plan"


def test_mdp_forward_step_switches_only_when_overlap_scheduler_requests_plan(monkeypatch):
    vision_leaf = torch.ones(2, 4, requires_grad=True)
    runtime = SimpleNamespace(storage=SimpleNamespace(get_leaf=lambda microbatch_id: vision_leaf))
    record = SimpleNamespace(
        microbatch_id=5,
        text_only=False,
        vision_items=(),
        decoder_packed_seq_params=object(),
        model_payload={"input_ids": torch.tensor([[7, 3, 7]]), "loss_mask": torch.ones(1, 3)},
    )
    monkeypatch.setattr(forward_step, "_accumulate_workload_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(forward_step, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(forward_step, "is_pipeline_last_stage", lambda: False)
    monkeypatch.setattr(
        forward_step, "get_args", lambda: SimpleNamespace(overlap_moe_expert_parallel_comm=True)
    )

    eager_model = _ForwardModel()
    eager_output, _ = forward_step.mdp_forward_step(runtime, iter([record]), eager_model)
    assert eager_output == "eager-output"
    assert eager_model.forward_calls == 1
    assert eager_model.plan_calls == 0

    plan_model = _ForwardModel()
    plan, _ = forward_step.mdp_forward_step(
        runtime, iter([record]), plan_model, return_schedule_plan=True
    )
    assert plan == "schedule-plan"
    assert plan_model.forward_calls == 0
    assert plan_model.plan_calls == 1
    assert plan_model.inputs["vision_embeddings"] is vision_leaf
    assert plan_model.inputs["pixel_values"] is None
    assert plan_model.inputs["packed_seq_params"] is record.decoder_packed_seq_params
