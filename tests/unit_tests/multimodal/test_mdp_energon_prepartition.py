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


def test_provider_assigns_cp_rank_as_loader_prepartition_owner(monkeypatch):
    from examples.multimodal_dev.data.qwen35_energon import provider

    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(provider.parallel_state, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(provider.parallel_state, "get_context_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        provider.parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1
    )

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
