# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib
from types import SimpleNamespace

import torch

from examples.multimodal_dev import pretrain_multimodal
from examples.multimodal_dev import forward_step as forward_step_module
from examples.multimodal_dev.mdp_pipeline_sidecar import (
    mark_downstream_pp_vision_params_shared,
    pp_cp_replicated_vision_requested,
)
from examples.multimodal_dev.models import base
from examples.multimodal_dev.models.qwen35_vl import factory as qwen35_factory
from megatron.core.packed_seq_params import PackedSeqParams


def test_model_provider_forwards_pipeline_stage_contract(monkeypatch):
    observed = {}
    expected_model = SimpleNamespace(vision_model=None)
    args = SimpleNamespace(
        model_arch="qwen35_vl",
        model_variant="0.8b",
        vision_num_layers=None,
        recompute_vision=False,
        mdp_encoder_mode=False,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        mdp_inner_dp_scope="cp",
        text_only=False,
        use_packed_sequence=True,
        micro_batch_size=1,
    )
    language_config = SimpleNamespace(bf16=True, fp16=False)
    vision_config = SimpleNamespace()

    def build_model(**kwargs):
        observed.update(kwargs)
        return expected_model

    from examples.multimodal_dev.models import MODEL_REGISTRY

    registry = MODEL_REGISTRY["qwen35_vl"]
    monkeypatch.setitem(registry, "model_factory_fn", build_model)
    monkeypatch.setitem(
        registry,
        "vision_config_fn",
        lambda **_kwargs: vision_config,
    )
    monkeypatch.setitem(registry, "post_language_config_fn", None)
    monkeypatch.setitem(registry, "vision_flops_fn", None)
    monkeypatch.setattr(pretrain_multimodal, "get_args", lambda: args)
    monkeypatch.setattr(
        pretrain_multimodal,
        "core_transformer_config_from_args",
        lambda _args: language_config,
    )

    result = pretrain_multimodal.model_provider(
        pre_process=False,
        post_process=True,
    )

    assert result is expected_model
    assert observed["pre_process"] is False
    assert observed["post_process"] is True


def test_factory_builds_downstream_vision_only_for_pp_cp_replica(monkeypatch):
    observed = []

    class FakeModel:
        def __init__(self, **kwargs):
            observed.append(kwargs)

    monkeypatch.setattr(
        "examples.multimodal_dev.models.qwen35_vl.model.Qwen35VLModel",
        FakeModel,
    )
    monkeypatch.setattr(
        "examples.multimodal_dev.models.qwen35_vl.specs.get_qwen35_vl_language_spec",
        lambda **_kwargs: "spec",
    )
    args = SimpleNamespace(
        mtp_num_layers=None,
        transformer_impl="transformer_engine",
        untie_embeddings_and_output_weights=True,
        padded_vocab_size=248320,
        max_position_embeddings=32768,
        image_token_id=248056,
        mdp_encoder_mode=False,
        pipeline_model_parallel_size=2,
        mdp_inner_dp_scope="cp",
        text_only=False,
    )

    qwen35_factory.build_model(
        args,
        language_config=object(),
        vision_config=object(),
        pre_process=False,
        post_process=True,
    )
    args.mdp_encoder_mode = True
    args.mdp_inner_dp_scope = "pp_cp"
    qwen35_factory.build_model(
        args,
        language_config=object(),
        vision_config=object(),
        pre_process=False,
        post_process=True,
    )

    assert observed[0]["build_vision_model"] is False
    assert observed[1]["build_vision_model"] is True


def _args(**overrides):
    values = {
        "mdp_encoder_mode": True,
        "pipeline_model_parallel_size": 2,
        "mdp_inner_dp_scope": "pp_cp",
        "text_only": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pp_cp_replication_selection_is_explicit():
    assert pp_cp_replicated_vision_requested(_args())
    assert not pp_cp_replicated_vision_requested(
        _args(mdp_inner_dp_scope="cp")
    )
    assert not pp_cp_replicated_vision_requested(
        _args(pipeline_model_parallel_size=1)
    )
    assert not pp_cp_replicated_vision_requested(
        _args(mdp_encoder_mode=False)
    )


def test_checkpoint_replica_id_replaces_only_pipeline_coordinate():
    assert base._replace_pp_replica_id(0, 2) == 2
    assert base._replace_pp_replica_id((0, 3, 4), 2) == (2, 3, 4)


def test_sharded_state_marks_only_replicated_vision(monkeypatch):
    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.vision_model = torch.nn.Linear(2, 2, bias=False)
    model.language_model = torch.nn.Linear(2, 2, bias=False)
    model._mdp_pp_cp_inner = True

    def sharded_state_dict_default(module, prefix, *_args):
        del module
        return {prefix + "weight": SimpleNamespace(replica_id=(0, 7, 9))}

    monkeypatch.setattr(base, "sharded_state_dict_default", sharded_state_dict_default)
    monkeypatch.setattr(
        base.parallel_state,
        "get_pipeline_model_parallel_rank",
        lambda: 1,
    )

    state = model.sharded_state_dict()

    assert state["vision_model.weight"].replica_id == (1, 7, 9)
    assert state["language_model.weight"].replica_id == (0, 7, 9)


def test_only_downstream_vision_replica_is_shared_for_norms():
    pp0 = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    pp1 = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))

    mark_downstream_pp_vision_params_shared(pp0, pp_rank=0)
    mark_downstream_pp_vision_params_shared(pp1, pp_rank=1)

    assert all(
        not getattr(parameter, "shared", False)
        for parameter in pp0.vision_model.parameters()
    )
    assert all(
        getattr(parameter, "shared", False)
        for parameter in pp1.vision_model.parameters()
    )


def test_non_consuming_stage_runs_zero_gradient_sidecar_backward():
    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model._pipeline_sidecar_enabled = True
    leaf = torch.tensor([2.0], requires_grad=True)
    dependency = leaf * 3.0

    model.mdp_pp_cp_sidecar_activate_cache(
        {"vision_embeddings": dependency}
    )
    model.pipeline_sidecar_post_backward()

    torch.testing.assert_close(leaf.grad, torch.zeros_like(leaf))


def test_non_consuming_frozen_vision_requests_zero_dependency_gather(monkeypatch):
    class FrozenVision(torch.nn.Linear):
        def forward(self, pixel_values, image_grid_thw):
            del image_grid_thw
            return super().forward(pixel_values)

    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model._mdp_pp_cp_inner = True
    model.vision_model = FrozenVision(3, 2, bias=False)
    model.vision_model.requires_grad_(False)
    model.language_model = torch.nn.Linear(2, 2, bias=False)
    model._mdp_rank_assignment = {0: [(0, 0)]}
    model._mdp_rank_assignment_row_counts = [1]
    observed = {}

    def gather(**kwargs):
        observed.update(kwargs)
        assert kwargs["local_embeddings"].requires_grad is True
        return kwargs["local_embeddings"].reshape(-1)[:1].sum() * 0.0

    monkeypatch.setattr(base, "get_mdp_images_to_language_group", lambda _model: object())
    monkeypatch.setattr(base, "gather_to_inner_dp_zero", gather)

    result = model._run_mdp_vision_bridge(
        pixel_values=torch.ones(1, 3),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
    )

    assert result.ndim == 0
    assert observed["return_zero_dependency_only"] is True


def test_forward_only_sidecar_does_not_leave_backward_dependency():
    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model._pipeline_sidecar_enabled = True

    model.mdp_pp_cp_sidecar_activate_cache(
        {
            "vision_embeddings": torch.tensor([0.0]),
            "forward_only": True,
        }
    )

    assert not hasattr(model, "_mdp_pp_cp_sidecar_backward_cache")


def test_pp0_consumes_cached_embedding_once_via_cp_local_plan(monkeypatch):
    class FakeLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_input = None
            self.forward_decoder_input = None

        def embedding(self, input_ids, position_ids=None):
            del position_ids
            self.embedding_input = input_ids
            return torch.zeros(input_ids.shape[1], 1, 2)

        def forward(self, **kwargs):
            self.forward_decoder_input = kwargs["decoder_input"]
            return kwargs["decoder_input"]

    class FailVision(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("cached PP x CP forward must not rerun vision")

    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.pre_process = True
    model.vision_model = FailVision()
    model.language_model = FakeLanguageModel()
    model._mdp_enabled = True
    model._mdp_pp_cp_inner = True
    active = torch.arange(8, dtype=torch.float32).view(4, 2)
    model._mdp_pp_cp_active_vision_embeddings = active
    plan = {"image_positions": [0, 3], "input_shape": (1, 4)}
    observed = {}
    decoder_input = torch.full((2, 1, 2), 7.0)

    monkeypatch.setattr(
        base.parallel_state, "get_context_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        base.MultimodalModel,
        "_run_mdp_vision_bridge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sidecar cache must replace the forward-time bridge")
        ),
    )
    monkeypatch.setattr(
        base.MultimodalModel,
        "_partition_cp_local_input_ids",
        lambda _self, input_ids, _packed: input_ids[:, :2],
    )

    def merge(
        _self,
        input_ids,
        text_embeddings,
        vision_embeddings,
        packed_seq_params=None,
        mdp_cp_local_plan=None,
    ):
        del input_ids, text_embeddings, packed_seq_params
        observed["vision_embeddings"] = vision_embeddings
        observed["plan"] = mdp_cp_local_plan
        return decoder_input

    monkeypatch.setattr(base.MultimodalModel, "_cp_local_merge_decoder_input", merge)

    def split(_self, **kwargs):
        observed["already_partitioned"] = kwargs[
            "decoder_input_already_cp_partitioned"
        ]
        return (
            kwargs["decoder_input"],
            kwargs["input_ids"],
            kwargs["labels"],
            kwargs["loss_mask"],
            kwargs["attention_mask"],
            kwargs["position_ids"],
        )

    monkeypatch.setattr(base.MultimodalModel, "_cp_split_for_forward", split)
    monkeypatch.setattr(
        base.MultimodalModel,
        "_thd_mrope_no_cp_override",
        lambda _self, _packed: contextlib.nullcontext(),
    )

    result = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        position_ids=torch.arange(4).view(1, 4),
        mdp_cp_local_plan=plan,
    )

    assert observed["vision_embeddings"] is active
    assert observed["plan"] is plan
    assert observed["already_partitioned"] is True
    assert model._mdp_pp_cp_active_vision_embeddings is None
    assert result is decoder_input


def test_sidecar_compute_uses_local_grid_but_forward_keeps_global_grid(monkeypatch):
    global_grid = torch.tensor([[1, 4, 4], [1, 8, 4]])
    local_grid = torch.tensor([[1, 8, 4]])
    local_pixels = torch.ones(32, 3, dtype=torch.bfloat16)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "pixel_values": torch.zeros(0, 3, dtype=torch.bfloat16),
        "image_grid_thw": global_grid,
        "_mdp_prepartitioned_assignment": {0: [(0, 0)], 1: [(0, 1)]},
        "_mdp_prepartitioned_row_counts": [4, 8],
        "_mdp_prepartitioned_image_grid_thw": local_grid,
        "_mdp_cp_local_plan": {"image_positions": [0, 1], "input_shape": (1, 4)},
    }
    observed = {}

    class Model:
        def mdp_pp_cp_sidecar_compute_vision(self, **kwargs):
            observed.update(kwargs)
            return torch.ones(12, 2)

    monkeypatch.setattr(forward_step_module, "get_batch", lambda _iterator: batch)
    monkeypatch.setattr(
        forward_step_module,
        "apply_mdp_prepartition",
        lambda **_kwargs: (local_pixels, local_grid),
    )

    cache = forward_step_module.build_mdp_pp_cp_sidecar_cache(
        data_iterator=object(), model=Model()
    )

    assert observed["pixel_values"] is local_pixels
    assert observed["image_grid_thw"] is local_grid
    assert cache["batch"]["image_grid_thw"] is global_grid
    assert cache["batch"]["pixel_values"] is None


def test_forward_step_does_not_prepartition_sidecar_batch_twice(monkeypatch):
    plan = {"image_positions": [1], "input_shape": (1, 4)}
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "loss_mask": torch.ones(1, 4),
        "position_ids": torch.arange(4).view(1, 4),
        "pixel_values": None,
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "_mdp_cp_local_plan": plan,
        "_mdp_pp_cp_sidecar_applied": True,
    }
    observed = {}

    class Model:
        def __call__(self, **kwargs):
            observed.update(kwargs)
            return torch.ones(1, 4)

    monkeypatch.setattr(
        forward_step_module,
        "get_args",
        lambda: SimpleNamespace(mdp_encoder_mode=True),
    )
    monkeypatch.setattr(
        forward_step_module,
        "_pop_mdp_pp_cp_sidecar_cache",
        lambda _model: {"batch": batch},
    )
    monkeypatch.setattr(
        forward_step_module,
        "apply_mdp_prepartition",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("sidecar batch must not be prepartitioned twice")
        ),
    )
    monkeypatch.setattr(
        forward_step_module, "get_context_parallel_world_size", lambda: 1
    )

    output, _loss = forward_step_module.forward_step(None, Model())

    assert output.shape == (1, 4)
    assert observed["mdp_cp_local_plan"] is plan


def _generic_sidecar_batch():
    logical_cu_seqlens = torch.tensor([0, 8, 12], dtype=torch.int32)
    physical_cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical_cu_seqlens,
        cu_seqlens_kv=logical_cu_seqlens,
        cu_seqlens_q_padded=physical_cu_seqlens,
        cu_seqlens_kv_padded=physical_cu_seqlens,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        total_tokens=16,
    )
    return {
        "input_ids": torch.arange(16).view(1, 16),
        "labels": torch.arange(16).view(1, 16),
        "loss_mask": torch.ones(1, 16),
        "position_ids": torch.arange(16).view(1, 16),
        "attention_mask": torch.ones(1, 16, dtype=torch.bool),
        "pixel_values": torch.ones(4, 3),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "packed_seq_params": packed_seq_params,
    }


def test_mdp_off_sidecar_consumes_only_pp0_and_broadcasts_packed_metadata(
    monkeypatch,
):
    model = SimpleNamespace(_pp_cp_batch_sidecar=True)
    source_batch = _generic_sidecar_batch()
    role = {"first": True, "last": False}
    wire = {}
    get_batch_calls = []

    monkeypatch.setattr(
        forward_step_module.mpu,
        "is_pipeline_first_stage",
        lambda **_kwargs: role["first"],
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "is_pipeline_last_stage",
        lambda **_kwargs: role["last"],
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "get_pipeline_model_parallel_group",
        lambda: object(),
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "get_pipeline_model_parallel_first_rank",
        lambda: 0,
    )

    def get_batch(_iterator):
        get_batch_calls.append("pp0")
        return source_batch

    def broadcast(batch, **_kwargs):
        if batch is not None:
            wire["batch"] = batch
        return wire["batch"]

    monkeypatch.setattr(forward_step_module, "get_batch", get_batch)
    monkeypatch.setattr(
        forward_step_module, "broadcast_data_batch_from_rank", broadcast
    )

    source_cache = forward_step_module.build_mdp_pp_cp_sidecar_cache(
        data_iterator=object(), model=model
    )
    assert get_batch_calls == ["pp0"]
    assert source_cache["batch"]["pixel_values"] is source_batch["pixel_values"]
    assert "pixel_values" not in wire["batch"]
    assert "packed_seq_params" not in wire["batch"]

    role.update(first=False, last=True)
    receiver_cache = forward_step_module.build_mdp_pp_cp_sidecar_cache(
        data_iterator=None, model=model
    )

    assert get_batch_calls == ["pp0"]
    receiver_batch = receiver_cache["batch"]
    assert "pixel_values" not in receiver_batch
    assert torch.equal(
        receiver_batch["packed_seq_params"].cu_seqlens_q,
        torch.tensor([0, 8, 12], dtype=torch.int32),
    )
    assert torch.equal(
        receiver_batch["packed_seq_params"].cu_seqlens_q_padded,
        torch.tensor([0, 8, 16], dtype=torch.int32),
    )
    assert receiver_batch["packed_seq_params"].total_tokens == 16


def test_generic_sidecar_keeps_pp2_endpoints_and_trims_pp4_middle(monkeypatch):
    batch = _generic_sidecar_batch()
    role = {"first": True, "last": False}
    monkeypatch.setattr(
        forward_step_module.mpu,
        "is_pipeline_first_stage",
        lambda **_kwargs: role["first"],
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "is_pipeline_last_stage",
        lambda **_kwargs: role["last"],
    )

    assert forward_step_module._stage_forward_view_from_full_batch(batch) is batch
    role.update(first=False, last=True)
    assert forward_step_module._stage_forward_view_from_full_batch(batch) is batch

    role.update(first=False, last=False)
    middle = forward_step_module._stage_forward_view_from_full_batch(batch)
    assert set(middle) == {
        "position_ids",
        "attention_mask",
        "packed_seq_params",
    }
    assert "input_ids" not in middle
    assert "labels" not in middle
    assert "loss_mask" not in middle
    assert "pixel_values" not in middle


def test_generic_sidecar_keeps_custom_layout_mtp_middle_batch(monkeypatch):
    batch = _generic_sidecar_batch()
    batch["tokens"] = batch["input_ids"].clone()
    config = SimpleNamespace(pipeline_model_parallel_layout=object())
    observed = {}

    monkeypatch.setattr(
        forward_step_module.mpu,
        "is_pipeline_first_stage",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "is_pipeline_last_stage",
        lambda **_kwargs: False,
    )

    def is_mtp_rank(received_config, *, ignore_virtual, vp_stage):
        observed.update(
            config=received_config,
            ignore_virtual=ignore_virtual,
            vp_stage=vp_stage,
        )
        return True

    monkeypatch.setattr(forward_step_module, "mtp_on_this_rank", is_mtp_rank)

    middle = forward_step_module._stage_forward_view_from_full_batch(
        batch, vp_stage=3, config=config
    )

    assert middle is batch
    assert observed == {
        "config": config,
        "ignore_virtual": False,
        "vp_stage": 3,
    }
    for key in ("input_ids", "tokens", "labels", "loss_mask"):
        assert middle[key] is batch[key]


def test_forward_step_accepts_generic_pp4_middle_stage_view(monkeypatch):
    batch = {
        "position_ids": torch.arange(16).view(1, 16),
        "packed_seq_params": _generic_sidecar_batch()["packed_seq_params"],
        "_mdp_pp_cp_sidecar_applied": True,
    }
    observed = {}

    class MiddleStageModel:
        def __call__(self, **kwargs):
            observed.update(kwargs)
            return torch.ones(16, 1, 2)

    monkeypatch.setattr(
        forward_step_module,
        "get_args",
        lambda: SimpleNamespace(mdp_encoder_mode=False),
    )
    monkeypatch.setattr(
        forward_step_module,
        "_pop_mdp_pp_cp_sidecar_cache",
        lambda _model: {"batch": batch, "vision_embeddings": None},
    )
    monkeypatch.setattr(
        forward_step_module, "get_context_parallel_world_size", lambda: 1
    )

    output, _loss = forward_step_module.forward_step(None, MiddleStageModel())

    assert output.shape == (16, 1, 2)
    assert observed["input_ids"] is None
    assert observed["labels"] is None
    assert observed["loss_mask"] is None
    assert observed["packed_seq_params"] is batch["packed_seq_params"]


def test_generic_sidecar_window_builds_n_caches_without_vision(monkeypatch):
    batches = [_generic_sidecar_batch() for _ in range(3)]
    calls = []
    monkeypatch.setattr(
        base,
        "get_args",
        lambda: SimpleNamespace(
            mdp_vision_prefetch_microbatches=1,
            mdp_vision_encoder_max_sequence_length=0,
        ),
    )
    monkeypatch.setattr(
        base.parallel_state,
        "get_pipeline_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "is_pipeline_first_stage",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "is_pipeline_last_stage",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "get_pipeline_model_parallel_group",
        lambda: object(),
    )
    monkeypatch.setattr(
        forward_step_module.mpu,
        "get_pipeline_model_parallel_first_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        forward_step_module,
        "get_batch",
        lambda _iterator: (calls.append("batch"), batches.pop(0))[1],
    )
    monkeypatch.setattr(
        forward_step_module,
        "broadcast_data_batch_from_rank",
        lambda batch, **_kwargs: batch,
    )
    monkeypatch.setattr(
        base.MultimodalModel,
        "mdp_pp_cp_sidecar_compute_vision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generic MDP-off windows must not run vision precompute")
        ),
    )
    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.vp_stage = None
    model._pipeline_sidecar_enabled = True
    model._pp_cp_batch_sidecar = True

    for microbatch in range(3):
        model.pipeline_sidecar_pre_forward(
            data_iterator=object(),
            current_microbatch=microbatch,
            num_microbatches=3,
        )

    caches = list(model._mdp_pp_cp_sidecar_cache)
    assert len(caches) == 3
    assert all(cache["vision_embeddings"] is None for cache in caches)
    assert calls == ["batch", "batch", "batch"]


def test_generic_sidecar_post_backward_is_noop_without_vision_dependency():
    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model._pipeline_sidecar_enabled = True
    model._pp_cp_batch_sidecar = True

    model.pipeline_sidecar_post_backward()
