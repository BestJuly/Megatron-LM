# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest


def _dataset_for_grid_cap(max_pixels=0):
    pytest.importorskip("torch")
    from examples.multimodal_dev.data.blend_dataset import Qwen35VLDataset

    dataset = Qwen35VLDataset.__new__(Qwen35VLDataset)
    dataset.image_size_max = 0
    dataset.image_max_pixels = int(max_pixels)
    dataset.image_min_pixels = 0
    dataset.patch_size = 16
    dataset.spatial_merge_size = 2
    return dataset


def test_descriptor_grid_is_preserved_without_resize_cap():
    dataset = _dataset_for_grid_cap(max_pixels=0)
    descriptor = {"grid_thw": [1, 1024, 1024], "width": 4096, "height": 4096}

    assert dataset._grid_from_descriptor_or_image(None, descriptor) == (1, 1024, 1024)


def test_descriptor_grid_uses_width_height_when_resize_cap_enabled():
    dataset = _dataset_for_grid_cap(max_pixels=327680)
    descriptor = {"grid_thw": [1, 1024, 1024], "width": 4096, "height": 4096}

    grid = dataset._grid_from_descriptor_or_image(None, descriptor)
    merge = dataset.spatial_merge_size
    visual_tokens = grid[0] * (grid[1] // merge) * (grid[2] // merge)

    assert grid != (1, 1024, 1024)
    assert visual_tokens < 8192
