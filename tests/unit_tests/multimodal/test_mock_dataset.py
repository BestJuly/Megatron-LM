# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Focused fixed-mock tests for the regular CP1 / MDP-off provider path."""

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import default_collate

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.data.mock import (
    MockQwen35VLDataset,
    mock_collate_fn,
    train_valid_test_datasets_provider,
)
from examples.multimodal_dev.forward_step import _prepare_prepacked_batch
from examples.multimodal_dev.mdp_image_materialize import (
    decode_image_descriptors,
    materialize_descriptor,
)
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)


def _build(**overrides):
    kwargs = {
        "num_samples": 2,
        "seq_length": 4096,
        "image_seq_length": None,
        "image_size": 224,
        "num_images": 1,
        "layout": "single",
        "pack_num_docs": 1,
        "text_only": False,
        "metadata_only_batch": True,
    }
    kwargs.update(overrides)
    return MockQwen35VLDataset(**kwargs)


def test_fixed_mock_arguments_are_registered():
    parser = add_multimodal_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--image-size-w",
            "448",
            "--image-sizes-h",
            "224_448",
            "--image-sizes-w",
            "448_224",
            "--num-images",
            "2",
            "--mock-layout",
            "interleaved",
            "--mock-pack-num-docs",
            "1",
            "--text-only",
        ]
    )

    assert args.image_size_w == 448
    assert args.image_sizes_h == "224_448"
    assert args.image_sizes_w == "448_224"
    assert args.num_images == 2
    assert args.mock_layout == "interleaved"
    assert args.mock_pack_num_docs == 1
    assert args.text_only is True


@pytest.mark.parametrize(
    ("image_size", "tokens_per_image"),
    [(224, 98), (448, 392), (672, 882), (896, 1568), (1344, 3528)],
)
def test_tokens_per_image_are_derived_from_the_patch_grid(image_size, tokens_per_image):
    dataset = _build(image_size=image_size)

    assert dataset.tokens_per_image == tokens_per_image
    assert dataset.image_seq_length == tokens_per_image


def test_explicit_image_sequence_length_must_match_the_grid():
    with pytest.raises(ValueError, match="image_seq_length"):
        _build(image_seq_length=256)


@pytest.mark.parametrize("layout", ["single", "interleaved"])
def test_layout_preserves_image_token_counts(layout):
    sample = _build(num_images=4, layout=layout)[0]

    assert sample["input_ids"].shape == torch.Size([4096])
    assert int((sample["input_ids"] == QWEN35_VL_IMAGE_TOKEN_ID).sum()) == 4 * 98
    assert int((sample["input_ids"] == QWEN35_VL_VISION_START_TOKEN_ID).sum()) == 4


def test_interleaved_layout_separates_image_blocks_with_text():
    input_ids = _build(num_images=4, layout="interleaved")[0]["input_ids"]
    starts = (input_ids == QWEN35_VL_VISION_START_TOKEN_ID).nonzero(as_tuple=True)[0]

    assert len(starts) == 4
    assert torch.all(starts[1:] - starts[:-1] > 99)


def test_rectangular_and_per_image_sizes_define_each_grid():
    dataset = _build(num_images=3, image_sizes_h=[224, 448, 224], image_sizes_w=[448, 224, 224])
    sample = dataset[0]

    assert dataset.tokens_per_image_list == [196, 196, 98]
    assert sample["image_grid_thw"].tolist() == [[2, 14, 28], [2, 28, 14], [2, 14, 14]]
    assert int((sample["input_ids"] == QWEN35_VL_IMAGE_TOKEN_ID).sum()) == 490


def test_metadata_only_batch_defers_and_materializes_each_image():
    sample = _build(num_images=2, image_sizes_h=[224, 448], image_sizes_w=[448, 224])[0]

    assert sample["pixel_values"].shape == torch.Size([0, 1536])
    descriptors = decode_image_descriptors(sample["_mdp_image_descriptors_json"])
    assert len(descriptors) == 2
    for descriptor, grid in zip(descriptors, sample["image_grid_thw"].tolist()):
        patches = materialize_descriptor(descriptor, grid, pixel_dim=1536, patch_size=16)
        assert patches.shape == (grid[0] * grid[1] * grid[2], 1536)


def test_materialized_batch_has_all_rectangular_patch_rows():
    sample = _build(
        num_images=3,
        image_sizes_h=[224, 448, 224],
        image_sizes_w=[448, 224, 224],
        metadata_only_batch=False,
    )[0]

    assert sample["pixel_values"].shape == torch.Size([1960, 1536])
    assert sample["image_grid_thw"].shape == torch.Size([3, 3])


def test_fixed_multi_doc_pack_emits_prepacked_boundaries():
    sample = _build(pack_num_docs=4, metadata_only_batch=False)[0]

    assert sample["input_ids"].shape == torch.Size([4096])
    assert sample["position_ids"].shape == torch.Size([3, 4096])
    assert sample["pixel_values"].shape == torch.Size([4 * 392, 1536])
    assert sample["image_grid_thw"].shape == torch.Size([4, 3])
    assert sample["cu_seqlens"].tolist() == [0, 1024, 2048, 3072, 4096]
    assert int(sample["max_seqlen"].item()) == 1024

    prepared = _prepare_prepacked_batch(default_collate([sample]))
    params = prepared["packed_seq_params"]
    assert params.cu_seqlens_q.tolist() == sample["cu_seqlens"].tolist()
    assert params.max_seqlen_q == 1024
    assert params.total_tokens == 4096


def test_fixed_pack_preserves_variable_image_grids_per_document():
    sample = _build(
        pack_num_docs=2,
        num_images=2,
        image_sizes_h=[224, 448],
        image_sizes_w=[448, 224],
        metadata_only_batch=False,
    )[0]

    assert sample["image_grid_thw"].tolist() == [[2, 14, 28], [2, 28, 14], [2, 14, 28], [2, 28, 14]]
    assert sample["pixel_values"].shape == torch.Size([3136, 1536])
    assert sample["image_cu_seqlens"].tolist() == [0, 2, 4]
    assert sample["pixel_cu_seqlens"].tolist() == [0, 1568, 3136]


@pytest.mark.parametrize(
    "overrides",
    [
        {"pack_num_docs": 3},
        {"pack_num_docs": 4, "layout": "interleaved"},
        {"pack_num_docs": 8, "image_size": 896},
        {"num_images": 2, "image_sizes_h": [224]},
        {"image_size": 230},
    ],
)
def test_invalid_fixed_mock_configuration(overrides):
    with pytest.raises(ValueError):
        _build(**overrides)


def test_text_only_ignores_image_layout_and_pack_shape():
    sample = _build(num_images=4, layout="single", pack_num_docs=4, text_only=True)[0]

    assert sample["input_ids"].shape == torch.Size([4096])
    assert sample["position_ids"].tolist() == list(range(4096))
    assert sample["pixel_values"].shape == torch.Size([0, 1536])
    assert sample["image_grid_thw"].shape == torch.Size([0, 3])
    assert sample["cu_seqlens"].tolist() == [0, 4096]
    assert "_mdp_image_descriptors_json" not in sample

    collated = mock_collate_fn(
        [_build(seq_length=32, text_only=True)[0], _build(seq_length=32, text_only=True)[0]]
    )
    assert collated["position_ids"].shape == torch.Size([2, 32])


def test_provider_plumbs_fixed_axes_and_keeps_regular_mock_materialized(monkeypatch):
    args = SimpleNamespace(
        total_seq_length=4096,
        padded_vocab_size=248320,
        image_token_id=248056,
        image_size=224,
        image_size_w=None,
        image_sizes_h="224_448",
        image_sizes_w="448_224",
        num_images=2,
        mock_layout="single",
        mock_pack_num_docs=2,
        text_only=False,
    )
    monkeypatch.setattr("megatron.training.get_args", lambda: args)

    train, valid, test = train_valid_test_datasets_provider([1, 1, 1])

    assert len(train) == len(valid) == len(test) == 1
    assert train.tokens_per_image_list == [196, 196]
    assert train.pack_num_docs == 2
    sample = train[0]
    assert sample["pixel_values"].shape == torch.Size([3136, 1536])
    assert "_mdp_image_descriptors_json" not in sample


def test_qwen3vl_launcher_forwards_fixed_mock_overrides(tmp_path):
    repository_root = Path(__file__).parents[3]
    script = repository_root / "examples" / "multimodal_dev" / "scripts" / "dev_qwen3vl_gb200.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--gpus",
            "1",
            "--nnodes",
            "1",
            "--results-dir",
            str(tmp_path),
            "fixed-mock",
            "ep=1",
            "dispatcher_backend=alltoall",
            "image_size_w=448",
            "image_sizes_h=224_448",
            "image_sizes_w=448_224",
            "num_images=2",
            "mock_layout=interleaved",
            "mock_pack_num_docs=1",
            "text_only=1",
            "use_packed_sequence=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    command = next(line for line in result.stdout.splitlines() if line.startswith("CMD: "))
    expected = (
        "--image-size-w 448",
        "--image-sizes-h 224_448",
        "--image-sizes-w 448_224",
        "--num-images 2",
        "--mock-layout interleaved",
        "--mock-pack-num-docs 1",
        "--text-only",
    )
    for option in expected:
        assert option in command
    assert "--pack-sequences" not in command
    assert "--use-packed-sequence" not in command
