from types import SimpleNamespace

import pytest
import torch


def _provider_args(**overrides):
    values = {
        "mdp_encoder_mode": True,
        "mdp_inner_dp_scope": "cp",
        "total_seq_length": 128,
        "seq_length": 128,
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "patch_size": 1,
        "temporal_patch_size": 1,
        "spatial_merge_size": 2,
        "image_min_pixels": 0,
        "image_max_pixels": 0,
        "vision_hidden_size": 8,
        "context_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "world_size": 2,
        "rank": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cyclic_loader_rejects_an_empty_epoch():
    from examples.multimodal_dev.data.qwen35_energon import provider

    iterator = provider._CyclicDataIterator([])

    with pytest.raises(RuntimeError, match="produced no batches"):
        next(iterator)


def test_provider_assigns_cp_rank_as_loader_prepartition_owner(monkeypatch):
    from examples.multimodal_dev.data.qwen35_energon import provider

    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(provider.parallel_state, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(provider.parallel_state, "get_context_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        provider.parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(provider.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)

    encoder = provider._task_encoder(_provider_args(), tokenizer=object())

    assert encoder.mdp_loader_prepartition is True
    assert encoder.mdp_loader_prepartition_rank == 1
    assert encoder.mdp_loader_prepartition_world == 2
    assert encoder.mdp_loader_prepartition_encoder_stage is True
    assert encoder.mdp_loader_prepartition_materialize is True


def test_provider_keeps_pr2_full_materialization_when_mdp_is_off(monkeypatch):
    from examples.multimodal_dev.data.qwen35_energon import provider

    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: False)
    encoder = provider._task_encoder(_provider_args(mdp_encoder_mode=False), tokenizer=object())

    assert encoder.mdp_loader_prepartition is False
    assert encoder.mdp_loader_prepartition_rank == 0
    assert encoder.mdp_loader_prepartition_world == 1


def test_provider_rejects_pipeline_parallelism_for_cp_local_scope(monkeypatch):
    from examples.multimodal_dev.data.qwen35_energon import provider

    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: False)
    with pytest.raises(ValueError, match="requires PP=1"):
        provider._task_encoder(_provider_args(pipeline_model_parallel_size=2), tokenizer=object())


def test_provider_uses_pp_cp_inner_rank_for_loader_owner(monkeypatch):
    from examples.multimodal_dev.data.qwen35_energon import provider

    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: False)
    encoder = provider._task_encoder(
        _provider_args(
            mdp_inner_dp_scope="pp_cp",
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
            world_size=4,
            rank=3,
        ),
        tokenizer=object(),
    )

    assert encoder.mdp_loader_prepartition is True
    assert encoder.mdp_loader_prepartition_rank == 3
    assert encoder.mdp_loader_prepartition_world == 4
    assert encoder.mdp_loader_prepartition_encoder_stage is True


@pytest.mark.parametrize("inner_scope", ["cp", "pp_cp"])
def test_fused_pp1_cp_window_uses_lazy_train_materialization_only(monkeypatch, inner_scope):
    from examples.multimodal_dev.data.qwen35_energon import provider

    args = _provider_args(
        mdp_inner_dp_scope=inner_scope,
        rank=1,
        mdp_fused_vision_window=True,
        mdp_vision_encoder_max_sequence_length=262_144,
        mdp_loader_prepartition_prefetch_windows=2,
        global_batch_size=512,
        data_parallel_size=32,
        micro_batch_size=1,
        energon_path="unused",
        dataloader_type="external",
        energon_packing_buffer_size=4,
        energon_max_samples_per_sequence=4,
        energon_prefetch_factor=2,
        energon_shuffle_buffer_size=4,
        dataset_split="train",
        num_workers=2,
        eval_iters=2,
    )
    encoders = []
    wrappers = []

    class FakeEncoder:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            encoders.append(self)

    def wrap_train(source, **kwargs):
        wrappers.append((source, kwargs))
        return ("planned", source)

    monkeypatch.setattr(provider, "get_args", lambda: args)
    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(provider, "_parallel_layout", lambda _args: (2, 1, 1, 0))
    monkeypatch.setattr(provider, "_worker_config", lambda _args: object())
    monkeypatch.setattr(provider, "_tokenizer", lambda _args: object())
    monkeypatch.setattr(provider, "Qwen35EnergonTaskEncoder", FakeEncoder)
    monkeypatch.setattr(provider, "get_train_dataset", lambda *_args, **_kwargs: "train-ds")
    monkeypatch.setattr(provider, "get_savable_loader", lambda *_args, **_kwargs: "train-loader")
    monkeypatch.setattr(provider, "_CyclicDataIterator", lambda source: ("cyclic", source))
    monkeypatch.setattr(provider, "MDPWindowMaterializingIterator", wrap_train)
    monkeypatch.setattr(
        provider, "get_val_datasets", lambda *_args, **_kwargs: [("validation-ds", 1.0)]
    )
    monkeypatch.setattr(provider, "get_loader", lambda *_args, **_kwargs: ["validation"])

    train, validation, test = provider.train_valid_test_datasets_provider(None)

    assert encoders[0].mdp_loader_prepartition_materialize is False
    assert encoders[1].mdp_loader_prepartition_materialize is True
    assert encoders[0].mdp_loader_prepartition_rank == 1
    assert encoders[0].mdp_loader_prepartition_world == 2
    assert len(wrappers) == 1
    assert wrappers[0][1]["lookahead_microbatches"] == 16
    assert wrappers[0][1]["prefetch_windows"] == 2
    assert wrappers[0][1]["balance_across_microbatches"] is True
    assert train[0] == "planned"
    assert list(validation) == ["validation"]
    assert test is None


@pytest.mark.parametrize("inner_scope", ["cp", "pp_cp"])
def test_pp1_cp_scopes_share_fused_planning_contract(inner_scope):
    from examples.multimodal_dev.data.qwen35_energon import provider

    layout = provider._resolve_mdp_layout(
        _provider_args(
            mdp_inner_dp_scope=inner_scope,
            rank=1,
            mdp_fused_vision_window=True,
            mdp_vision_encoder_max_sequence_length=262_144,
            global_batch_size=512,
            data_parallel_size=32,
            micro_batch_size=1,
        ),
        cp_size=2,
        cp_rank=1,
        pp_size=1,
        pp_rank=0,
    )

    assert layout.prepartition_rank == 1
    assert layout.prepartition_world == 2
    assert layout.prepartition_encoder_stage is True
    assert layout.planning_microbatches == 16
    assert layout.use_planning_prefetch is True


def test_pp1_cp_scope_window_off_keeps_fused_planning_off():
    from examples.multimodal_dev.data.qwen35_energon import provider

    layout = provider._resolve_mdp_layout(
        _provider_args(
            mdp_inner_dp_scope="cp",
            mdp_fused_vision_window=False,
            mdp_vision_encoder_max_sequence_length=262_144,
        ),
        cp_size=2,
        cp_rank=1,
        pp_size=1,
        pp_rank=0,
    )

    assert layout.planning_microbatches == 1
    assert layout.use_planning_prefetch is False


def test_pp2_cp2_pp_cp_fused_layout_is_unchanged():
    from examples.multimodal_dev.data.qwen35_energon import provider

    layout = provider._resolve_mdp_layout(
        _provider_args(
            mdp_inner_dp_scope="pp_cp",
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
            world_size=4,
            rank=3,
            mdp_fused_vision_window=True,
            mdp_vision_encoder_max_sequence_length=262_144,
            global_batch_size=512,
            data_parallel_size=32,
            micro_batch_size=1,
        ),
        cp_size=2,
        cp_rank=1,
        pp_size=2,
        pp_rank=1,
    )

    assert layout.prepartition_rank == 3
    assert layout.prepartition_world == 4
    assert layout.prepartition_encoder_stage is True
    assert layout.planning_microbatches == 16
    assert layout.use_planning_prefetch is True


def test_pp1_pp_cp_requires_a_fused_window():
    from examples.multimodal_dev.data.qwen35_energon import provider

    with pytest.raises(ValueError, match="requires CP>1 fused vision prefetch"):
        provider._resolve_mdp_layout(
            _provider_args(
                mdp_inner_dp_scope="pp_cp",
                mdp_fused_vision_window=False,
                mdp_vision_encoder_max_sequence_length=262_144,
            ),
            cp_size=2,
            cp_rank=0,
            pp_size=1,
            pp_rank=0,
        )


def test_json_lazy_descriptors_materialize_only_on_the_cp_owner(monkeypatch):
    from examples.multimodal_dev.data.qwen35_energon import task_encoder as module

    calls = []

    def fake_materialize(descriptor, grid, *, pixel_dim, patch_size):
        calls.append((descriptor["kind"], descriptor["id"], tuple(grid)))
        rows = int(grid[0]) * int(grid[1]) * int(grid[2])
        return torch.full((rows, pixel_dim), float(descriptor["id"]))

    monkeypatch.setattr(module, "materialize_descriptor", fake_materialize)
    encoder = module.Qwen35EnergonTaskEncoder(
        tokenizer=object(),
        seq_length=128,
        patch_size=1,
        temporal_patch_size=1,
        spatial_merge_size=2,
        mdp_loader_prepartition=True,
        mdp_loader_prepartition_rank=1,
        mdp_loader_prepartition_world=2,
        mdp_loader_prepartition_encoder_stage=True,
        mdp_loader_prepartition_materialize=True,
        mdp_lpt_hidden_size=8,
    )
    docs = [
        {
            "num_images": 2,
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 8, 4]], dtype=torch.long),
            "_mdp_image_descriptors": [
                {"kind": "zip_image", "id": 10, "materializer": "unused"},
                {"kind": "parquet_column_image", "id": 20, "materializer": "unused"},
            ],
            "_mdp_image_owner_ranks": [0, 1],
        }
    ]
    output = {"_mdp_image_descriptors_json": "descriptor metadata"}

    result = encoder._attach_prepartition(output, docs)

    assert calls == [("parquet_column_image", 20, (1, 8, 4))]
    assert result["_mdp_image_descriptors_json"] == "descriptor metadata"
    assert result["_mdp_prepartitioned_assignment"].tolist() == [[0, 0, 0], [1, 0, 1]]
    assert result["_mdp_prepartitioned_row_counts"].tolist() == [4, 8]
    assert result["_mdp_prepartitioned_image_grid_thw"].tolist() == [[1, 8, 4]]
    assert result["pixel_values"].shape == (32, 3)
    assert result["pixel_values"].dtype == torch.bfloat16


def test_runtime_strips_json_descriptors_after_owner_materialization():
    from examples.multimodal_dev import mdp_batch

    original = {
        "_mdp_image_descriptors_json": "descriptor metadata",
        "_mdp_prepartitioned_assignment": torch.tensor([[0, 0, 0]]),
        "pixel_values": torch.ones(4, 3),
    }

    stripped = mdp_batch._strip_mdp_image_descriptors(original)

    assert "_mdp_image_descriptors_json" not in stripped
    assert "_mdp_image_descriptors_json" in original
    assert "_mdp_prepartitioned_assignment" in stripped
    torch.testing.assert_close(stripped["pixel_values"], original["pixel_values"])
