# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import importlib.util as _ilu
import os
import unittest

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_MB_PATH = os.path.join(_REPO_ROOT, "examples", "multimodal_dev", "modality_bridge.py")
_spec = _ilu.spec_from_file_location("modality_bridge_under_test", _MB_PATH)
_mb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mb)


IMAGE = 99


def test_all_empty_cp_batch_keeps_vision_zero_dependency(monkeypatch):
    group = object()
    monkeypatch.setattr(_mb.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(_mb.dist, "get_rank", lambda group: 0)

    vision_parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    local_zero_dep = vision_parameter.sum() * 0.0
    gathered = _mb.gather_to_inner_dp_zero(
        local_embeddings=torch.empty(0, 3),
        rank_assignment={0: []},
        encoder_dp_group=group,
        global_per_image_row_counts=[],
        local_zero_dep=local_zero_dep,
    )

    assert tuple(gathered.shape) == (0, 3)
    assert gathered.requires_grad
    gathered.sum().backward()
    torch.testing.assert_close(vision_parameter.grad, torch.zeros_like(vision_parameter))


def _local_cp_split_tensor(tensor, seq_dim, cp_size, cp_rank):
    S = tensor.shape[seq_dim]
    if S % (2 * cp_size) != 0:
        raise AssertionError("sequence length must divide 2*cp_size")
    tensor = tensor.view(
        *tensor.shape[:seq_dim], 2 * cp_size, S // (2 * cp_size), *tensor.shape[seq_dim + 1 :]
    )
    index = torch.tensor(
        [cp_rank, 2 * cp_size - cp_rank - 1], dtype=torch.int64, device=tensor.device
    )
    tensor = tensor.index_select(seq_dim, index)
    return tensor.view(*tensor.shape[:seq_dim], -1, *tensor.shape[seq_dim + 2 :])


class TestSelectVisionRowsForCpRank(unittest.TestCase):

    def test_selected_rows_preserve_cp_local_order(self):
        vision_embeddings = torch.arange(30, dtype=torch.float32).view(10, 3)
        row_ids = torch.tensor([5, 1, 7, 1], dtype=torch.int64)
        selected = _mb.select_vision_rows_for_cp_rank(vision_embeddings, row_ids)
        self.assertTrue(torch.equal(selected, vision_embeddings[row_ids]))

    def test_empty_selection_returns_empty_hidden_tensor(self):
        vision_embeddings = torch.arange(12, dtype=torch.float16).view(4, 3)
        selected = _mb.select_vision_rows_for_cp_rank(
            vision_embeddings, torch.empty((0,), dtype=torch.int64)
        )
        self.assertEqual(selected.shape, (0, 3))
        self.assertEqual(selected.dtype, vision_embeddings.dtype)
        self.assertEqual(selected.device, vision_embeddings.device)

    def test_out_of_range_row_id_raises(self):
        vision_embeddings = torch.arange(12, dtype=torch.float32).view(4, 3)
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            _mb.select_vision_rows_for_cp_rank(
                vision_embeddings, torch.tensor([0, 4], dtype=torch.int64)
            )


def _scatter_like_model(input_ids, text_embeddings, vision_embeddings):
    combined = text_embeddings.transpose(0, 1).contiguous()
    mask = (input_ids == IMAGE).unsqueeze(-1).expand_as(combined)
    combined = combined.masked_scatter(mask, vision_embeddings)
    return combined.transpose(0, 1).contiguous()


class TestCpLocalMergeEquivalence(unittest.TestCase):

    def test_bshd_cp_local_merge_equals_full_merge_then_cp_split(self):
        input_ids = torch.tensor([[1, IMAGE, 2, IMAGE, 3, IMAGE, 4, IMAGE]], dtype=torch.long)
        text_embeddings = torch.arange(8 * 1 * 3, dtype=torch.float32).view(8, 1, 3)
        vision_embeddings = torch.arange(4 * 3, dtype=torch.float32).view(4, 3) + 1000.0

        full_merged = _scatter_like_model(input_ids, text_embeddings, vision_embeddings)
        image_positions = [
            int(x) for x in input_ids.reshape(-1).eq(IMAGE).nonzero(as_tuple=False).view(-1)
        ]

        for cp_rank in (0, 1):
            expected = _local_cp_split_tensor(full_merged, seq_dim=0, cp_size=2, cp_rank=cp_rank)
            text_cp = _local_cp_split_tensor(text_embeddings, seq_dim=0, cp_size=2, cp_rank=cp_rank)
            image_positions_cp, cp_rows, global_rows = (
                _mb.cp_local_image_positions_and_row_ids_from_cpu_metadata(
                    image_positions=image_positions,
                    input_shape=tuple(input_ids.shape),
                    cp_size=2,
                    cp_rank=cp_rank,
                )
            )
            self.assertEqual(global_rows, 4)
            vision_cp = _mb.select_vision_rows_for_cp_rank(vision_embeddings, cp_rows)
            actual = _mb.scatter_vision_rows_at_positions(text_cp, vision_cp, image_positions_cp)
            self.assertTrue(torch.equal(actual, expected))

    def test_cpu_metadata_row_ids_are_monotonic_in_masked_scatter_order(self):
        input_ids = torch.tensor([[IMAGE, 1, IMAGE, 2], [IMAGE, IMAGE, 3, 4]], dtype=torch.long)
        image_positions = [
            int(x) for x in input_ids.reshape(-1).eq(IMAGE).nonzero(as_tuple=False).view(-1)
        ]
        positions, row_ids, total = _mb.cp_local_image_positions_and_row_ids_from_cpu_metadata(
            image_positions=image_positions,
            input_shape=tuple(input_ids.shape),
            cp_size=1,
            cp_rank=0,
        )
        self.assertTrue(torch.equal(positions, torch.tensor([0, 2, 4, 5], dtype=torch.int64)))
        self.assertTrue(torch.equal(row_ids, torch.tensor([0, 1, 2, 3])))
        self.assertEqual(total, 4)


if __name__ == "__main__":
    unittest.main()
