# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib
from types import SimpleNamespace

import torch


def test_pp_vision_sum_precedes_distributed_optimizer_sync(monkeypatch):
    finalize_module = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    events = []
    config = SimpleNamespace(timers=None, moe_router_enable_expert_bias=False)
    model = SimpleNamespace(
        finish_grad_sync=lambda force_all_reduce=False: events.append("dp_sync")
    )
    groups = SimpleNamespace(tp=object(), pp=object(), embd=None, pos_embd=None, dp_cp=object())

    monkeypatch.setattr(finalize_module, "get_model_config", lambda _model: config)
    monkeypatch.setattr(finalize_module, "_has_mdp_pp_cp_inner", lambda _model: True)
    monkeypatch.setattr(
        finalize_module,
        "_allreduce_mdp_pp_cp_vision_grads",
        lambda _model, _group: events.append("pp_sum"),
    )
    for name in (
        "_allreduce_conditional_embedding_grads",
        "_allreduce_non_tensor_model_parallel_grads",
        "_allreduce_word_embedding_grads",
        "_allreduce_position_embedding_grads",
        "reset_model_temporary_tensors",
    ):
        monkeypatch.setattr(finalize_module, name, lambda *_args, **_kwargs: None)

    finalize_module.finalize_model_grads([model], pg_collection=groups)

    assert events == ["pp_sum", "dp_sync"]


def test_pp_vision_sum_materializes_missing_grad_as_zero(monkeypatch):
    finalize_module = importlib.import_module("megatron.core.distributed.finalize_model_grads")

    class VisionContainer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._mdp_pp_cp_inner = True
            self.vision_model = torch.nn.Linear(2, 2)

    model = VisionContainer()
    model.vision_model.weight.grad = torch.ones_like(model.vision_model.weight)
    model.vision_model.bias.grad = None

    monkeypatch.setattr(finalize_module, "get_pg_size", lambda _group: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group=None: tensor.mul_(2))

    finalize_module._allreduce_mdp_pp_cp_vision_grads([model], pp_group=object())

    torch.testing.assert_close(
        model.vision_model.weight.grad, torch.full_like(model.vision_model.weight, 2.0)
    )
    torch.testing.assert_close(
        model.vision_model.bias.grad, torch.zeros_like(model.vision_model.bias)
    )
