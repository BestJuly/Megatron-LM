from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev import forward_step, mdp_model_setup


def _args(**overrides):
    values = dict(
        mdp_encoder_mode=True,
        mdp_inner_dp_scope="cp",
        context_parallel_size=1,
        text_only=False,
        use_packed_sequence=True,
        freeze_ViT=False,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_packed_trainable_vision_disables_overlap_before_ddp_setup():
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args()

    result = mdp_model_setup.configure_mdp_model(model, args)

    assert result is model
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False
    assert args.mdp_encoder_mode is False


def test_qwen3_like_text_only_cp_mock_stays_on_normal_path():
    model = SimpleNamespace()
    args = _args(
        context_parallel_size=2, dataset_provider="mock", micro_batch_size=8, model_arch="qwen3"
    )

    result = mdp_model_setup.configure_mdp_model(model, args)

    assert result is model
    assert args.text_only is True
    assert args.mdp_encoder_mode is False
    assert model._mdp_enabled is False
    assert model._mdp_inner_dp_group is None
    assert args.overlap_grad_reduce is True
    assert args.overlap_param_gather is True
    assert args.overlap_param_gather_with_optimizer_step is True


def test_qwen3_like_text_only_cp_mock_uses_normal_batch_dispatch(monkeypatch):
    model = SimpleNamespace()
    args = _args(
        context_parallel_size=2, dataset_provider="mock", micro_batch_size=8, text_only=True
    )
    mdp_model_setup.configure_mdp_model(model, args)

    normal_batch = {"cu_seqlens": torch.tensor([0, 1], dtype=torch.int32)}
    real_tensor = torch.tensor

    def cpu_tensor(*values, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*values, **kwargs)

    monkeypatch.setattr(forward_step, "get_args", lambda: args)
    monkeypatch.setattr(forward_step, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(forward_step, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(forward_step, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(forward_step, "get_tensor_model_parallel_group", object)
    monkeypatch.setattr(forward_step.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(forward_step.torch.distributed, "broadcast", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(forward_step, "broadcast_data_batch", lambda data, device: data)
    monkeypatch.setattr(
        forward_step, "_prepare_prepacked_batch", lambda batch: {"normal_batch": batch}
    )
    monkeypatch.setattr(
        forward_step,
        "_get_mdp_prepartitioned_batch",
        lambda _iterator: pytest.fail("text-only qwen3 must not use MDP batch dispatch"),
    )

    result = forward_step.get_batch(iter([normal_batch]))

    assert result == {"normal_batch": normal_batch}


def test_mdp_off_energon_marks_packed_and_disables_overlap_before_ddp_setup():
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(mdp_encoder_mode=False, dataset_provider="energon", use_packed_sequence=False)

    mdp_model_setup.configure_mdp_model(model, args)

    assert args.use_packed_sequence is True
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False
    assert model._mdp_enabled is False


@pytest.mark.parametrize("dataset_provider", ["blend", "energon", "mock", "mock_mdp"])
def test_descriptor_backed_mdp_providers_are_allowed_and_marked_packed(
    monkeypatch, dataset_provider
):
    group = object()
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(
        context_parallel_size=2,
        dataset_provider=dataset_provider,
        micro_batch_size=1,
        use_packed_sequence=False,
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _group: [0, 1])
    monkeypatch.setattr(mdp_model_setup.ps, "get_context_parallel_group", lambda: group)

    result = mdp_model_setup.configure_mdp_model(model, args)

    assert result is model
    assert args.use_packed_sequence is True
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False
    assert model._mdp_enabled is True


def test_direct_blend_packing_is_marked_before_ddp_when_mdp_is_off():
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(
        mdp_encoder_mode=False,
        dataset_provider="blend",
        pack_samples_per_item=2,
        use_packed_sequence=False,
    )

    mdp_model_setup.configure_mdp_model(model, args)

    assert args.use_packed_sequence is True
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False


@pytest.mark.parametrize("mdp_requested", [False, True])
def test_mdp_off_packed_pipeline_enables_generic_batch_sidecar_without_vision(mdp_requested):
    model = SimpleNamespace(vision_model=None)
    args = _args(
        mdp_encoder_mode=mdp_requested,
        context_parallel_size=1,
        pipeline_model_parallel_size=4,
        dataset_provider="mock",
        micro_batch_size=1,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )

    result = mdp_model_setup.configure_mdp_model(model, args)

    assert result is model
    assert args.mdp_encoder_mode is False
    assert model._mdp_enabled is False
    assert model._pp_cp_batch_sidecar is True
    assert model._pipeline_sidecar_enabled is True


def test_qwen3_text_only_pipeline_never_enables_generic_sidecar():
    model = SimpleNamespace(vision_model=None)
    args = _args(
        model_arch="qwen3",
        context_parallel_size=1,
        pipeline_model_parallel_size=2,
        dataset_provider="mock",
        micro_batch_size=1,
    )

    result = mdp_model_setup.configure_mdp_model(model, args)

    assert result is model
    assert args.text_only is True
    assert args.mdp_encoder_mode is False
    assert model._mdp_enabled is False
    assert model._pp_cp_batch_sidecar is False
    assert model._pipeline_sidecar_enabled is False
    assert args.overlap_grad_reduce is True
    assert args.overlap_param_gather is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_packed_sequence": False}, "requires --use-packed-sequence"),
        ({"micro_batch_size": 2}, "requires micro_batch_size=1"),
    ],
)
def test_mdp_off_pp_rejects_unsupported_thd_contract(override, message):
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(
        mdp_encoder_mode=False,
        context_parallel_size=1,
        pipeline_model_parallel_size=2,
        dataset_provider="mock",
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        **override,
    )

    with pytest.raises(RuntimeError, match=message):
        mdp_model_setup.configure_mdp_model(model, args)


def test_cp_scope_wires_megatron_context_group(monkeypatch):
    group = object()
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(context_parallel_size=2, overlap_grad_reduce=False, overlap_param_gather=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _group: [2, 3])
    monkeypatch.setattr(mdp_model_setup.ps, "get_context_parallel_group", lambda: group)

    mdp_model_setup.configure_mdp_model(model, args)

    assert model._mdp_enabled is True
    assert model._mdp_inner_dp_group is group


def test_pp_cp_scope_delegates_to_replicated_sidecar(monkeypatch):
    from examples.multimodal_dev import mdp_pipeline_sidecar

    group = object()
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(
        context_parallel_size=2,
        pipeline_model_parallel_size=2,
        mdp_inner_dp_scope="pp_cp",
        dataset_provider="energon",
        micro_batch_size=1,
    )

    def configure_sidecar(inner_model, _args):
        inner_model._mdp_inner_dp_group = group
        inner_model._mdp_enabled = True
        inner_model._mdp_pp_cp_inner = True
        inner_model._pipeline_sidecar_enabled = True
        return True

    monkeypatch.setattr(
        mdp_pipeline_sidecar, "configure_pp_cp_replicated_vision", configure_sidecar
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 4)

    result = mdp_model_setup.configure_mdp_model(model, args)

    assert result is model
    assert model._mdp_pp_cp_inner is True
    assert model._pipeline_sidecar_enabled is True
    assert args.overlap_grad_reduce is False
    assert args.overlap_param_gather is False
    assert args.overlap_param_gather_with_optimizer_step is False


def test_pipeline_parallelism_is_reserved_for_following_sidecar_pr():
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(
        context_parallel_size=2,
        pipeline_model_parallel_size=2,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    with pytest.raises(RuntimeError, match="pipeline_model_parallel_size=1"):
        mdp_model_setup.configure_mdp_model(model, args)


def test_pp_cp_scope_requires_pipeline_parallelism():
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(
        context_parallel_size=2,
        pipeline_model_parallel_size=1,
        mdp_inner_dp_scope="pp_cp",
        dataset_provider="energon",
        micro_batch_size=1,
    )
    with pytest.raises(RuntimeError, match="requires CP>1 fused vision prefetch"):
        mdp_model_setup.configure_mdp_model(model, args)


@pytest.mark.parametrize("inner_scope", ["cp", "pp_cp"])
def test_pp1_fused_window_uses_cp_group_and_sidecar(monkeypatch, inner_scope):
    group = object()
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(
        context_parallel_size=2,
        pipeline_model_parallel_size=1,
        mdp_inner_dp_scope=inner_scope,
        dataset_provider="energon",
        micro_batch_size=1,
        mdp_fused_vision_window=True,
        mdp_vision_encoder_max_sequence_length=262_144,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _group: [0, 1])
    monkeypatch.setattr(mdp_model_setup.ps, "get_context_parallel_group", lambda: group)

    result = mdp_model_setup.configure_mdp_model(model, args)

    assert result is model
    assert model._mdp_enabled is True
    assert model._mdp_inner_dp_group is group
    assert model._mdp_pp_cp_inner is False
    assert model._mdp_cp_fused_sidecar is True
    assert model._pipeline_sidecar_enabled is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"micro_batch_size": 2}, "micro_batch_size=1"),
        ({"dataset_provider": "cord_v2"}, "descriptor-backed"),
        ({"dynamic_context_parallel": True}, "static context parallelism"),
        ({"use_megatron_fsdp": True}, "does not support FSDP"),
    ],
)
def test_cp_runtime_rejects_unsupported_configs(override, message):
    model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2))
    args = _args(
        context_parallel_size=2, overlap_grad_reduce=False, overlap_param_gather=False, **override
    )
    with pytest.raises(RuntimeError, match=message):
        mdp_model_setup.configure_mdp_model(model, args)
