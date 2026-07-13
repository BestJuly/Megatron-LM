# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
blend_dataset = pytest.importorskip("examples.multimodal_dev.data.blend_dataset")


class _FakeBackend:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx % len(self.samples)]


class _FakeTokenizer:
    eod = 151643

    def tokenize(self, text):
        ids = []
        for word in (text or "").split():
            ids.append(100 + (sum(word.encode("utf-8")) % 1000))
        return ids


def _make_dataset(
    seq_length=256,
    pack_samples_per_item=1,
    pack_scan_multiplier=1,
    images=None,
    metadata_only_batch=False,
    mdp_loader_prepartition=False,
    mdp_loader_prepartition_rank=0,
    mdp_loader_prepartition_world=1,
    dataloader_sequence_packing=False,
    dataloader_dp_rank=0,
):
    if images is None:
        images = [Image.new("RGB", (32, 32), color=(128, 64, 32))]
    samples = [
        blend_dataset._RawSample(images=images, text="short real sample " * 2),
        blend_dataset._RawSample(images=images, text="another real sample " * 2),
        blend_dataset._RawSample(images=images, text="third real sample " * 2),
        blend_dataset._RawSample(images=images, text="fourth real sample " * 2),
    ]
    dataset = blend_dataset.Qwen35VLDataset.__new__(blend_dataset.Qwen35VLDataset)
    dataset.seq_length = seq_length
    dataset.vocab_size = blend_dataset.QWEN35_VL_VOCAB_SIZE
    dataset.image_token_id = blend_dataset.QWEN35_VL_IMAGE_TOKEN_ID
    dataset.video_token_id = blend_dataset.QWEN35_VL_VIDEO_TOKEN_ID
    dataset.vision_start_token_id = blend_dataset.QWEN35_VL_VISION_START_TOKEN_ID
    dataset.patch_size = 16
    dataset.temporal_patch_size = 2
    dataset.spatial_merge_size = 2
    dataset.image_size_max = 32
    dataset.image_max_pixels = 0
    dataset.image_min_pixels = 0
    dataset.cp_size = 2
    dataset.emit_cu_seqlens = True
    dataset.align = 64
    dataset.pack_samples_per_item = pack_samples_per_item
    dataset.pack_scan_multiplier = pack_scan_multiplier
    dataset._pack_start_stride = pack_samples_per_item * pack_scan_multiplier
    dataset._pack_scan_span = max(
        dataset._pack_start_stride, (seq_length + dataset.align - 1) // dataset.align
    )
    dataset.dataloader_sequence_packing = bool(dataloader_sequence_packing)
    dataset.dataloader_dp_rank = int(dataloader_dp_rank)
    dataset._pixel_dim = 3 * dataset.temporal_patch_size * dataset.patch_size * dataset.patch_size
    dataset.metadata_only_batch = bool(metadata_only_batch)
    dataset.mdp_loader_prepartition = bool(mdp_loader_prepartition)
    dataset.mdp_loader_prepartition_rank = int(mdp_loader_prepartition_rank)
    dataset.mdp_loader_prepartition_world = int(mdp_loader_prepartition_world)
    dataset.mdp_loader_prepartition_encoder_stage = True
    dataset.mdp_loader_prepartition_hidden = 1280
    dataset.tokenizer = _FakeTokenizer()
    dataset._backends = [("fake", _FakeBackend(samples))]
    dataset._index = [(0, i) for i in range(len(samples))]
    dataset._virtual_len = 128
    return dataset


def _packing_dataset(seq_length=16, cp_size=2):
    dataset = blend_dataset.Qwen35VLDataset.__new__(blend_dataset.Qwen35VLDataset)
    dataset.seq_length = int(seq_length)
    dataset.cp_size = int(cp_size)
    dataset.image_token_id = 10_001
    dataset.video_token_id = 10_002
    dataset.vision_start_token_id = 10_003
    dataset.spatial_merge_size = 2
    dataset._pixel_dim = 1536
    dataset.emit_cu_seqlens = True
    dataset.metadata_only_batch = False
    return dataset


def _doc(values):
    input_ids = torch.tensor(values, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "real_len": int(input_ids.numel()),
        "content_len": int(input_ids.numel()),
        "pixel_values": torch.zeros(0, 1536),
        "image_grid_thw": torch.zeros(0, 3, dtype=torch.long),
        "num_images": 0,
        "num_patches": 0,
        "_mdp_image_descriptors": [],
    }


def test_direct_blend_builds_cp_aligned_thd_container(monkeypatch):
    monkeypatch.setattr(blend_dataset, "_MIMO_PACK_PAD_MULTIPLE", 4)

    def fake_rope_index(*, input_ids, **_kwargs):
        positions = torch.arange(input_ids.shape[-1], dtype=torch.long)
        return positions.view(1, 1, -1).repeat(3, 1, 1), 0

    monkeypatch.setattr(blend_dataset, "get_rope_index", fake_rope_index)
    dataset = _packing_dataset()

    batch = dataset._finalize_packed_container([_doc([1, 2, 3, 4, 5]), _doc([11, 12, 13])])

    assert batch["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, 0, 0, 11, 12, 13, 0, 0, 0, 0, 0]
    assert batch["tokens"].data_ptr() == batch["input_ids"].data_ptr()
    assert batch["cu_seqlens"].tolist() == [0, 8, 12]
    assert batch["cu_seqlens_padded"].tolist() == [0, 8, 16]
    assert int(batch["max_seqlen"].item()) == 8
    assert batch["image_cu_seqlens"].tolist() == [0, 0, 0]
    assert batch["pixel_cu_seqlens"].tolist() == [0, 0, 0]
    assert batch["position_ids"].shape == (3, 16)
    assert batch["position_ids"][0].tolist() == list(range(8)) + list(range(8))
    assert batch["loss_mask"].tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert torch.equal(batch["labels"] == -100, batch["loss_mask"] == 0)


def test_direct_blend_materializes_only_the_lpt_owner(monkeypatch):
    dataset = _packing_dataset()
    dataset.patch_size = 16
    dataset.mdp_loader_prepartition = True
    dataset.mdp_loader_prepartition_rank = 1
    dataset.mdp_loader_prepartition_world = 2
    dataset.mdp_loader_prepartition_encoder_stage = True
    dataset.mdp_loader_prepartition_hidden = 1152

    materialized = []

    def fake_materialize(descriptor, grid, *, pixel_dim, patch_size):
        materialized.append((descriptor["name"], tuple(grid), pixel_dim, patch_size))
        rows = int(grid[0]) * int(grid[1]) * int(grid[2])
        return torch.full((rows, pixel_dim), float(descriptor["value"]))

    monkeypatch.setattr(blend_dataset, "materialize_descriptor", fake_materialize)
    grids = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long)
    out = {"image_grid_thw": grids, "pixel_values": torch.zeros(0, 1536)}

    result = dataset._attach_loader_prepartition(
        out, [{"name": "large", "value": 1}, {"name": "small", "value": 2}]
    )

    assert materialized == [("small", (1, 2, 2), 1536, 16)]
    assert result["pixel_values"].shape == (4, 1536)
    assert torch.all(result["pixel_values"] == 2)
    assert result["_mdp_prepartitioned_image_grid_thw"].tolist() == [[1, 2, 2]]
    assert result["_mdp_prepartitioned_assignment"].tolist() == [[0, 0, 0], [1, 0, 1]]
    assert result["_mdp_prepartitioned_row_counts"].tolist() == [4, 1]
    assert result["_mdp_prepartitioned_local_raw_counts"].tolist() == [4]


def test_direct_mock_provider_emits_prepartitioned_thd_item(monkeypatch):
    args = SimpleNamespace(
        dataset_backend="mock",
        dataset_root=None,
        dataset_subsets=None,
        dataset_split="train",
        pack_samples_per_item=1,
        pack_scan_multiplier=1,
        dataloader_sequence_packing=False,
        micro_batch_size=1,
        mdp_encoder_mode=True,
        mdp_inner_dp_scope="cp",
        context_parallel_size=2,
        pipeline_model_parallel_size=1,
        total_seq_length=4096,
        padded_vocab_size=blend_dataset.QWEN35_VL_VOCAB_SIZE,
        image_token_id=blend_dataset.QWEN35_VL_IMAGE_TOKEN_ID,
        use_packed_sequence=True,
        dynamic_context_parallel=False,
    )
    monkeypatch.setattr(blend_dataset, "get_args", lambda: args)
    monkeypatch.setattr(blend_dataset, "get_tokenizer", _FakeTokenizer)
    monkeypatch.setattr(blend_dataset.parallel_state, "is_initialized", lambda: False)

    train, _, _ = blend_dataset.train_valid_test_datasets_provider([1, 0, 0])
    sample = train[0]

    assert sample["input_ids"].shape == (4096,)
    assert sample["cu_seqlens_padded"].tolist()[-1] == 4096
    assert "_mdp_prepartitioned_assignment" in sample
    assert "_mdp_prepartitioned_row_counts" in sample
    assert "_mdp_prepartitioned_local_raw_counts" in sample
    assert sample["pixel_values"].shape[1] == 1536


def test_direct_blend_rejects_prepacked_micro_batch_greater_than_one(monkeypatch):
    args = SimpleNamespace(
        dataset_backend="mock",
        micro_batch_size=2,
        use_packed_sequence=True,
        pack_samples_per_item=1,
        dataloader_sequence_packing=False,
    )
    monkeypatch.setattr(blend_dataset, "get_args", lambda: args)

    with pytest.raises(ValueError, match="micro-batch-size 1"):
        blend_dataset.train_valid_test_datasets_provider([1, 0, 0])


class TestBlendDatasetMegatronPackingMetadata(unittest.TestCase):
    def test_static_packed_metadata_keeps_container_padding_out_of_cu_seqlens(self):
        sample = _make_dataset().__getitem__(0)

        cu = sample["cu_seqlens"].tolist()
        cu_padded = sample["cu_seqlens_padded"].tolist()

        self.assertEqual(len(cu), 2)
        self.assertEqual(cu[0], 0)
        self.assertEqual(cu_padded, [0, 256])
        self.assertLess(cu[-1], 256)
        self.assertEqual(sample["max_seqlen"].item(), 256)
        self.assertEqual(sample["input_ids"].shape, (256,))
        self.assertEqual(sample["position_ids"].shape, (3, 256))
        self.assertEqual(sample["image_cu_seqlens"].tolist(), [0, 1])
        self.assertGreater(sample["pixel_cu_seqlens"].tolist()[-1], 0)

    def test_dataset_side_pack_emits_multi_doc_fixed_length_item(self):
        dataset = _make_dataset(seq_length=512, pack_samples_per_item=4)
        sample = dataset.__getitem__(0)

        cu = sample["cu_seqlens"].tolist()
        cu_padded = sample["cu_seqlens_padded"].tolist()

        self.assertGreater(len(cu), 5)
        self.assertGreater(len(cu) - 1, dataset.pack_samples_per_item)
        self.assertEqual(len(cu_padded), len(cu))
        self.assertEqual(cu_padded[:-1], cu[:-1])
        self.assertEqual(cu_padded[-1], 512)
        self.assertEqual(sample["input_ids"].shape, (512,))
        self.assertEqual(cu[-1], 512)
        self.assertEqual(sample["position_ids"].shape, (3, 512))
        self.assertEqual(
            sample["max_seqlen"].item(),
            max(cu_padded[i + 1] - cu_padded[i] for i in range(len(cu_padded) - 1)),
        )
        self.assertEqual(sample["image_cu_seqlens"].shape[0], len(cu))
        self.assertEqual(sample["pixel_cu_seqlens"].shape[0], len(cu))
        if cu[-1] < 512:
            self.assertEqual(sample["loss_mask"][cu[-1] :].sum().item(), 0.0)

    def test_dataset_side_pack_resets_position_ids_per_doc(self):
        dataset = _make_dataset(seq_length=512, pack_samples_per_item=4)
        sample = dataset.__getitem__(0)
        cu = sample["cu_seqlens"].tolist()

        for start in cu[:-1]:
            self.assertEqual(sample["position_ids"][0, start].item(), 0)
            self.assertEqual(sample["position_ids"][1, start].item(), 0)
            self.assertEqual(sample["position_ids"][2, start].item(), 0)

    def test_dataloader_sequence_packing_uses_fixed_length_packed_item(self):
        dataset = _make_dataset(seq_length=512, dataloader_sequence_packing=True)
        sample = dataset.__getitem__(0)
        cu = sample["cu_seqlens"].tolist()
        cu_padded = sample["cu_seqlens_padded"].tolist()

        self.assertGreater(len(cu), 2)
        self.assertEqual(cu_padded[:-1], cu[:-1])
        self.assertEqual(cu_padded[-1], 512)
        self.assertEqual(sample["input_ids"].shape, (512,))
        self.assertEqual(cu[-1], 512)
        self.assertEqual(sample["position_ids"].shape, (3, 512))

    def test_loader_prepartition_materializes_rank_local_pixels(self):
        images = [
            Image.new("RGB", (32, 32), color=(128, 64, 32)),
            Image.new("RGB", (64, 64), color=(32, 64, 128)),
        ]
        dataset_rank0 = _make_dataset(
            seq_length=512,
            images=images,
            metadata_only_batch=True,
            mdp_loader_prepartition=True,
            mdp_loader_prepartition_rank=0,
            mdp_loader_prepartition_world=2,
        )
        dataset_rank1 = _make_dataset(
            seq_length=512,
            images=images,
            metadata_only_batch=True,
            mdp_loader_prepartition=True,
            mdp_loader_prepartition_rank=1,
            mdp_loader_prepartition_world=2,
        )

        sample_rank0 = dataset_rank0.__getitem__(0)
        sample_rank1 = dataset_rank1.__getitem__(0)

        self.assertIn("_mdp_prepartitioned_assignment", sample_rank0)
        self.assertIn("_mdp_prepartitioned_row_counts", sample_rank0)
        self.assertEqual(sample_rank0["image_grid_thw"].shape[0], 2)
        self.assertEqual(sample_rank1["image_grid_thw"].shape[0], 2)
        self.assertEqual(sample_rank0["_mdp_prepartitioned_image_grid_thw"].shape[1], 3)
        self.assertEqual(sample_rank1["_mdp_prepartitioned_image_grid_thw"].shape[1], 3)
        total_local_patches = (
            sample_rank0["pixel_values"].shape[0] + sample_rank1["pixel_values"].shape[0]
        )
        self.assertEqual(total_local_patches, int(sample_rank0["image_grid_thw"].prod(dim=1).sum()))

    def test_qwen_pixel_budget_resizes_to_visual_token_range(self):
        dataset = _make_dataset()
        dataset.image_size_max = 0
        dataset.image_min_pixels = 256 * 32 * 32
        dataset.image_max_pixels = 1280 * 32 * 32

        self.assertEqual(dataset._resize_hw(32, 32), (512, 512))
        height, width = dataset._resize_hw(4096, 4096)
        self.assertLessEqual(height * width, dataset.image_max_pixels)
        self.assertEqual(height % 32, 0)
        self.assertEqual(width % 32, 0)
