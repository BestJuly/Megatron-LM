# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``--mdp-mock-vision-config-json``: per-sample vision share for long samples.

Pure CPU. The scenario pool is checked through ``scenario_totals``; a few
samples are generated in full to tie those counts to the emitted tensors.

Run with::

    python -m pytest -q examples/multimodal_dev/tests/test_mdp_vision_share.py
"""

import hashlib
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.data.mdp_scenarios import (
    SPATIAL_MERGE_SIZE,
    build_scenarios,
    scenario_totals,
)
from examples.multimodal_dev.models.kimi_k3_vl.data import (
    KimiMdpMockDataset,
    kimi_scenarios,
    kimi_vision_config,
)
from examples.multimodal_dev.models.kimi_k3_vl.mdp_adapter import validate_grid

VISION = {"vision_fraction": [0.10, 0.20], "item_tokens": [256, 4096], "text_only_every": 0}
GREEDY = {
    "mode": "distribution",
    "type": "lognormal",
    "format": "thd",
    "min_seq_len": 1024,
    "max_seq_len": 4096,
    "mean_seq_len": 2048,
    "lognormal_sigma": 1.1,
}
# repr() digests of the pools built without a vision config at the commit that
# introduced it (bb3671087); the option must leave them untouched.
DEFAULT_POOL_SHA256 = "b0b5e06760de571addefe91920f35caa42663cada3afb556f11261090838ac22"
GREEDY_POOL_SHA256 = "a51c8ce7a20c5d3ed2eea5aedccdfff2bb4126676f4f338c2db011a9d3bae9e5"


def _fixed(length):
    return {**GREEDY, "min_seq_len": length, "max_seq_len": length, "mean_seq_len": length}


def _kimi_pool(length, vision=VISION):
    return kimi_scenarios(
        build_scenarios(length_config=_fixed(length), vision_config=kimi_vision_config(vision))
    )


def _sha(pool):
    return hashlib.sha256(repr(pool).encode()).hexdigest()


def test_pools_without_vision_config_are_unchanged():
    assert _sha(build_scenarios()) == DEFAULT_POOL_SHA256
    assert _sha(build_scenarios(length_config=GREEDY)) == GREEDY_POOL_SHA256


@pytest.mark.parametrize("length", [16384, 32768, 65536])
def test_kimi_pool_is_exact_length_with_vision_share(length):
    pool = _kimi_pool(length)
    for grids, _ in pool:
        assert grids, "text_only_every=0 must leave no text-only sample"
        total, image = scenario_totals((grids, _), frames_pooled=True)
        assert total == length
        assert 0.10 <= image / total <= 0.20
        for grid in grids:
            validate_grid(grid)
            t, h, w = grid
            assert t in (1, 2, 4)
            assert h % SPATIAL_MERGE_SIZE == 0 and w % SPATIAL_MERGE_SIZE == 0


def test_kimi_samples_match_the_pool_counts():
    pool = _kimi_pool(16384)
    dataset = KimiMdpMockDataset(vocab_size=163840, scenarios=pool)
    for idx in range(2):
        sample = dataset[idx]
        grids = pool[idx][0]
        total, image = scenario_totals(pool[idx], frames_pooled=True)
        ids = sample["input_ids"]
        assert ids.numel() == total
        assert int((ids == dataset.image_token_id).sum()) == image
        assert sample["pixel_values"].shape == (
            sum(t * h * w for t, h, w in grids),
            dataset.pixel_dim,
        )


def test_pool_is_deterministic():
    assert _sha(_kimi_pool(32768)) == _sha(_kimi_pool(32768))


def test_text_only_cadence():
    pool = build_scenarios(vision_config={"text_only_every": 4, "item_tokens": [8, 64]})
    assert [not grids for grids, _ in pool] == [i % 4 == 3 for i in range(len(pool))]


def test_unpooled_frames_count_per_frame():
    pool = build_scenarios(
        length_config=_fixed(8192), vision_config={"video_fraction": 1.0, "video_frames": [2]}
    )
    for scenario in pool:
        total, image = scenario_totals(scenario)
        assert total == 8192 and 0.10 <= image / total <= 0.20
        # A video item counts every frame; one too small to split falls back to t=1.
        assert {t for t, _, _ in scenario[0]} <= {1, 2}
    assert any(t == 2 for grids, _ in pool for t, _, _ in grids)


@pytest.mark.parametrize(
    "bad",
    [
        {"vision_fraction": [0.3, 0.2]},
        {"item_tokens": [2, 64]},
        {"item_tokens": [256, 8192], "max_merged_side": 64},
        {"max_items": 0},
        {"video_frames": [1]},
        {"not_a_key": 1},
    ],
)
def test_invalid_vision_config_is_rejected(bad):
    with pytest.raises(ValueError):
        build_scenarios(vision_config=bad)


def test_sample_too_short_for_an_item_is_rejected():
    with pytest.raises(ValueError, match="cannot hold"):
        build_scenarios(length_config=_fixed(1024), vision_config=VISION)
