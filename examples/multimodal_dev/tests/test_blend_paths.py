# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse

import pytest

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.data.blend import RawDatasetBlend


def test_direct_blend_arguments_are_registered():
    parser = add_multimodal_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--dataset-provider",
            "blend",
            "--dataset-backend",
            "mantis-instruct,pixmo-docs",
            "--dataset-subsets",
            "multi_vqa,charts",
            "--dataset-root",
            "/mnt/datasets",
            "--dataset-split",
            "train",
            "--image-size-max",
            "896",
            "--pack-samples-per-item",
            "4",
            "--pack-scan-multiplier",
            "2",
        ]
    )

    assert args.dataset_provider == "blend"
    assert args.dataset_backend == "mantis-instruct,pixmo-docs"
    assert args.dataset_subsets == "multi_vqa,charts"
    assert args.dataset_root == "/mnt/datasets"
    assert args.dataset_split == "train"
    assert args.image_size_max == 896
    assert args.pack_samples_per_item == 4
    assert args.pack_scan_multiplier == 2


def test_non_mock_blend_requires_dataset_root():
    with pytest.raises(ValueError, match="dataset root is required"):
        RawDatasetBlend._normalize_roots(["mantis-instruct"], None)


def test_multi_backend_root_uses_portable_subdirectories(tmp_path):
    roots = RawDatasetBlend._normalize_roots(
        ["mantis-instruct", "pixmo-docs", "m4-instruct"], str(tmp_path)
    )

    assert roots == {
        "mantis-instruct": str(tmp_path / "mantis" / "Mantis-Instruct"),
        "pixmo-docs": str(tmp_path / "pixmo" / "pixmo-docs"),
        "m4-instruct": str(tmp_path / "m4" / "M4-Instruct-Data"),
    }
