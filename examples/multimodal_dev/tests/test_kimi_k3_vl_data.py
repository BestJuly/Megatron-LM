# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kimi-K3 VL collate spec, MDP adapter geometry and dataset invariants (CPU)."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.data.mdp_scenarios import build_scenarios
from examples.multimodal_dev.models import registry_for
from examples.multimodal_dev.models.kimi_k3_vl.configuration import (
    KIMI_K3_IMAGE_TOKEN_ID,
    KIMI_K3_MEDIA_BEGIN_TOKEN_ID,
)
from examples.multimodal_dev.models.kimi_k3_vl.data import (
    PIXEL_DIM,
    KimiFixedMockDataset,
    KimiMdpMockDataset,
    expand_media_pad,
    kimi_scenarios,
)
from examples.multimodal_dev.models.kimi_k3_vl.mdp_adapter import (
    PAYLOAD_WIDTH,
    KimiK3VLMdpAdapter,
    build_collate_spec,
    kimi_workload,
)
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.protocols import CapturedVisionItem

ARGS = SimpleNamespace(image_token_id=None, model_arch="kimi_k3_vl")


def test_registry_entry_is_complete():
    entry = registry_for(ARGS)
    for field in (
        "model_factory_fn",
        "vision_config_fn",
        "vision_flops_fn",
        "vision_collate_spec_fn",
        "mdp_adapter_fn",
    ):
        assert entry[field] is not None
    assert set(entry["dataset_providers"]) == {"mock", "mdp_mock", "cord_v2"}


@pytest.mark.parametrize("grid", [(1, 16, 20), (2, 8, 8), (4, 28, 42)])
def test_collate_spec_rows_pool_frames(grid):
    spec = build_collate_spec(ARGS)
    t, h, w = grid
    assert spec.image_token_id == KIMI_K3_IMAGE_TOKEN_ID
    assert spec.payload_rows_of(grid) == t * h * w
    # MoonViT averages over frames before the 2x2 merge: output rows ignore t.
    assert spec.output_rows_of(grid) == (h // 2) * (w // 2)
    spec.check(grid)


@pytest.mark.parametrize(
    "grid", [(1, 15, 20), (5, 8, 8), (1, 514, 8), (0, 8, 8)], ids=["odd_h", "t5", "side", "t0"]
)
def test_collate_spec_rejects_bad_grids(grid):
    with pytest.raises(ValueError):
        build_collate_spec(ARGS).check(grid)


def test_workload_uses_whole_item_attention():
    keys = torch.tensor([[1, 16, 20], [4, 8, 8]])
    rows, attn, out = kimi_workload(keys)
    assert rows.item() == 320 + 256
    assert attn.item() == 320**2 + 256**2  # one sequence per item, all frames
    assert out.item() == 80 + 16


def test_adapter_validates_item_geometry():
    adapter = KimiK3VLMdpAdapter(ARGS, out_hidden_size=7168)
    assert adapter.payload_width == PAYLOAD_WIDTH == 588
    good = CapturedVisionItem(
        sample_id=0,
        image_ordinal=0,
        shape_key=(2, 8, 12),
        payload_row_start=0,
        payload_rows=192,
        output_rows=24,
        decoder_positions=tuple(range(24)),
    )
    adapter.validate_item(good)
    bad = dataclasses.replace(good, output_rows=48)
    with pytest.raises(MdpConfigurationError):
        adapter.validate_item(bad)


def test_mdp_mock_samples_are_consistent():
    scenarios = kimi_scenarios(build_scenarios())
    assert {t for grids, _ in scenarios for t, _, _ in grids} == {1, 2, 4}
    ds = KimiMdpMockDataset(num_samples=32, vocab_size=163840, scenarios=scenarios)
    spec = build_collate_spec(ARGS)
    for idx in range(len(ds)):
        sample = ds[idx]
        grids = [tuple(g) for g in sample["image_grid_thw"].tolist()]
        ids = sample["input_ids"]
        assert int((ids == KIMI_K3_IMAGE_TOKEN_ID).sum()) == sum(map(spec.output_rows_of, grids))
        assert int((ids == KIMI_K3_MEDIA_BEGIN_TOKEN_ID).sum()) == len(grids)
        assert sample["pixel_values"].shape == (sum(map(spec.payload_rows_of, grids)), PIXEL_DIM)
        assert torch.equal(sample["labels"][:-1], ids[1:])


def test_fixed_mock_rounds_to_merge_multiple():
    ds = KimiFixedMockDataset(
        num_samples=2,
        seq_length=1024,
        image_size=450,
        vocab_size=163840,
        image_token_id=KIMI_K3_IMAGE_TOKEN_ID,
    )
    sample = ds[0]
    assert ds.grid == (1, 32, 32)  # 450 px -> 448 px -> 32 x 32 patches
    assert int((sample["input_ids"] == KIMI_K3_IMAGE_TOKEN_ID).sum()) == 256
    assert sample["input_ids"].shape[0] == 1024


def test_expand_media_pad():
    pad = KIMI_K3_IMAGE_TOKEN_ID
    assert expand_media_pad([1, pad, 2, pad], [3, 1], pad) == [1, pad, pad, pad, 2, pad]
    with pytest.raises(ValueError):
        expand_media_pad([1, pad], [2, 2], pad)
