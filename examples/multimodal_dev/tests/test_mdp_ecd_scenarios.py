# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``--mdp-mock-scenario ecd``: the ECD-like MDP mock pool (CPU only).

Every sample carries exactly one still image of 64-256 decoder tokens, total lengths
follow the measured ECD distribution (<= 2047), the pool is deterministic, and the
default pool is unchanged by the new switch.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset
from examples.multimodal_dev.data.mdp_scenarios import (
    ECD_MAX_TOTAL_TOKENS,
    ECD_POOL_SIZE,
    build_ecd_scenarios,
    build_scenario_pool,
    build_scenarios,
    scenario_totals,
)

IMAGE_TOKEN_ID = 248056
VISION_START_TOKEN_ID = 248053


def test_pool_is_deterministic():
    assert build_ecd_scenarios() == build_ecd_scenarios()
    assert build_scenario_pool("ecd") == build_ecd_scenarios()


def test_every_sample_is_one_still_image_within_bounds():
    pool = build_ecd_scenarios()
    assert len(pool) == ECD_POOL_SIZE
    totals, images = [], []
    for grids, text_chunks in pool:
        assert len(grids) == 1 and len(text_chunks) == 2
        t, h, w = grids[0]
        assert t == 1 and h % 2 == 0 and w % 2 == 0
        total, image_tokens = scenario_totals((grids, text_chunks))
        assert 64 <= image_tokens <= 256
        assert image_tokens + 2 <= total <= ECD_MAX_TOTAL_TOKENS <= 2048
        totals.append(total)
        images.append(image_tokens)
    # Measured ECD: mean total 1038, mean image tokens 240.
    assert 950 <= sum(totals) / len(totals) <= 1130
    assert 230 <= sum(images) / len(images) <= 250


def test_dataset_samples_match_the_scenarios():
    pool = build_ecd_scenarios()
    ds = MdpThdMockDataset(
        num_samples=16,
        vocab_size=248320,
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        scenarios=pool,
    )
    for i in range(16):
        sample = ds[i]
        total, image_tokens = scenario_totals(pool[i])
        assert sample["input_ids"].numel() == total
        assert int((sample["input_ids"] == IMAGE_TOKEN_ID).sum()) == image_tokens
        assert sample["image_grid_thw"].shape == (1, 3)


def test_default_pool_unchanged_and_ecd_rejects_length_config():
    assert build_scenario_pool("default") == build_scenarios()
    with pytest.raises(ValueError, match="cannot be combined"):
        build_scenario_pool("ecd", length_config={"mode": "distribution"})
    with pytest.raises(ValueError, match="unknown"):
        build_scenario_pool("bogus")
