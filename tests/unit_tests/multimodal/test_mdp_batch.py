from types import SimpleNamespace

import torch

from examples.multimodal_dev import forward_step, mdp_batch


def test_cp_scope_enables_loader_prepartition(monkeypatch):
    monkeypatch.setattr(
        mdp_batch,
        "get_args",
        lambda: SimpleNamespace(mdp_encoder_mode=True, mdp_inner_dp_scope="cp"),
    )
    assert mdp_batch._loader_prepartition_enabled() is True


def test_pp_cp_scope_is_not_enabled_before_sidecar_pr(monkeypatch):
    monkeypatch.setattr(
        mdp_batch,
        "get_args",
        lambda: SimpleNamespace(mdp_encoder_mode=True, mdp_inner_dp_scope="pp_cp"),
    )
    assert mdp_batch._loader_prepartition_enabled() is False


def test_fetch_normalizes_assignment_after_tp_metadata_broadcast(monkeypatch):
    replacements = {
        "get_args": lambda: SimpleNamespace(mdp_encoder_mode=True, mdp_inner_dp_scope="cp"),
        "get_tensor_model_parallel_src_rank": lambda: 0,
        "get_tensor_model_parallel_group": lambda: object(),
        "get_tensor_model_parallel_rank": lambda: 1,
        "get_tensor_model_parallel_world_size": lambda: 2,
        "model_parallel_is_initialized": lambda: True,
        "_broadcast_data_batch_and_side_metadata_from_rank": (
            lambda *_args, **_kwargs: (
                {"input_ids": torch.tensor([[1]])},
                (True, None, None, {"0": [["1", "2"]]}, [3], [7, 11]),
            )
        ),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(mdp_batch, name, replacement)

    batch = mdp_batch.fetch_and_broadcast(iter(()))

    assert batch["_mdp_prepartitioned_assignment"] == {0: [(1, 2)]}
    assert batch["_mdp_prepartitioned_row_counts"] == [3]
    assert batch["_mdp_prepartitioned_local_raw_counts"] == [7, 11]


def test_apply_prepartition_uses_owner_local_loader_payload(monkeypatch):
    class DummyModel:
        _mdp_enabled = True

    monkeypatch.setattr(mdp_batch, "unwrap_model", lambda model: model)
    model = DummyModel()
    local_pixels = torch.arange(2 * 3, dtype=torch.float32).view(2, 3)
    local_grid = torch.tensor([[1, 1, 2]], dtype=torch.long)

    pixel_values, image_grid_thw = mdp_batch.apply_mdp_prepartition(
        model=model,
        pixel_values=local_pixels,
        image_grid_thw=torch.tensor([[1, 1, 2], [1, 2, 2]], dtype=torch.long),
        image_grid_thw_rows=[(1, 1, 2), (1, 2, 2)],
        prepartitioned_assignment={0: [(0, 0)], 1: [(0, 1)]},
        prepartitioned_row_counts=[2, 4],
        prepartitioned_image_grid_thw=local_grid,
    )

    torch.testing.assert_close(pixel_values, local_pixels)
    torch.testing.assert_close(image_grid_thw, local_grid)
    assert model._mdp_rank_assignment == {0: [(0, 0)], 1: [(0, 1)]}
    assert model._mdp_rank_assignment_row_counts == [2, 4]


def test_mdp_batch_reuses_pr2_logical_and_physical_thd_boundaries(monkeypatch):
    raw = {
        "input_ids": torch.arange(8, dtype=torch.long),
        "tokens": torch.arange(8, dtype=torch.long),
        "labels": torch.arange(8, dtype=torch.long),
        "loss_mask": torch.ones(8),
        "position_ids": torch.arange(24, dtype=torch.long).view(3, 8),
        "pixel_values": torch.zeros(0, 3),
        "image_grid_thw": torch.zeros(0, 3, dtype=torch.long),
        "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "max_seqlen": torch.tensor(4, dtype=torch.int32),
        "_mdp_cp_local_plan": {
            "input_shape": (1, 8),
            "image_positions": [],
            "cu_seqlens": [0, 3, 5],
            "cu_seqlens_padded": [0, 4, 8],
        },
    }
    monkeypatch.setattr(forward_step, "fetch_and_broadcast", lambda _iterator: dict(raw))

    batch = forward_step._get_mdp_prepartitioned_batch(iter(()))
    packed = batch["packed_seq_params"]

    assert batch["input_ids"].shape == (1, 8)
    assert batch["position_ids"].shape == (3, 1, 8)
    assert packed.cu_seqlens_q.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_kv.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert packed.total_tokens == 8
    assert batch["_mdp_cp_local_plan"]["cu_seqlens"] == [0, 3, 5]
