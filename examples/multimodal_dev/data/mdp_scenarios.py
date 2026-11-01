# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Randomized scenario pool for ``MdpThdMockDataset``.

A scenario is a ``(grids, text_chunk_lengths)`` pair, where ``grids`` are
``(t, h, w)`` vision items in patch units and ``text_chunk_lengths`` are the
text runs around them. The pool is what makes the mock dataset representative:

* ~4/5 multimodal samples, 1-3 vision items with random ``(t, h, w)`` grids
  (``h``/``w`` divisible by the spatial merge size, ~25% multi-frame video
  items), total image decoder tokens per sample in ``IMAGE_TOKENS_RANGE``.
* ~1/5 text-only samples.
* Every sample's total token count (text + image slots + one vision-start
  sentinel per item) lands in ``TOTAL_TOKENS_RANGE``.

The point of the randomization is *heterogeneity*: variable sequence lengths
and variable grids are what exercise THD packing, the MDP planner's load
balancing, and the vision encoder's grid-keyed caches. A handful of tiny
fixed scenarios keeps the code paths alive but measures nothing.

Determinism: a fixed ``GENERATOR_SEED`` drives a private ``random.Random``,
so the pool is byte-identical across processes, ranks, and runs -- required
because every MDP rank rebuilds the same window independently.

Run ``python -m examples.multimodal_dev.data.mdp_scenarios`` to dump the pool.
"""

import random

GENERATOR_SEED = 2026
POOL_SIZE = 64
SPATIAL_MERGE_SIZE = 2

TOTAL_TOKENS_RANGE = (1000, 2000)
IMAGE_TOKENS_RANGE = (100, 500)
TEXT_ONLY_EVERY = 5  # every 5th scenario is text-only (~1/5 of the pool)
MIN_TOTAL_TOKENS = 64  # floor for externally supplied length distributions
MAX_GRID_DRAWS = 200  # rejection-sampling cap; see _multimodal_scenario


def draw_total_token_lengths(length_config, count, seed=GENERATOR_SEED):
    """``count`` per-sample total token lengths from a length distribution.

    ``length_config`` is the ``--mdp-mock-dataset-config-json`` payload, the same
    schema ``--varlen-mock-dataset-config-json`` uses, so MDP and the
    decoder-only reference can be pointed at identical distributions.

    Determinism matters more than usual here: every MDP rank rebuilds the
    scenario pool independently, so a divergent draw becomes a collective hang,
    not a wrong number. Two things guarantee it. ``MockSFTLowLevelDataset``
    seeds NumPy's global RNG with its own fixed class seed, so its length vector
    is identical everywhere; and the *selection* out of that vector runs on a
    private ``random.Random(seed)`` rather than on whatever global state the
    process happens to hold. The global NumPy state is saved and restored so the
    draw leaves no side effect on other datasets.
    """
    import numpy as np

    from megatron.training.datasets.sft_dataset import MockSFTLowLevelDataset

    # MockSFTLowLevelDataset refuses "file"/"distribution" mode without
    # vocab_size >= 2, because it can also synthesize token ids. MDP reads only
    # ``.sequence_lengths``, which is drawn from the length distribution alone
    # and never touches the vocabulary, so supply a nominal value when the
    # config does not carry one. This cannot shift the draw: vocab_size is not
    # an input to generate_lognormal_samples().
    kwargs = dict(length_config)
    kwargs.setdefault("vocab_size", 2)

    state = np.random.get_state()
    try:
        low_level = MockSFTLowLevelDataset(**kwargs)
    finally:
        np.random.set_state(state)
    lengths = low_level.sequence_lengths
    rng = random.Random(seed)
    return [
        max(MIN_TOTAL_TOKENS, int(lengths[rng.randrange(len(lengths))])) for _ in range(count)
    ]


def _item_tokens(grid, merge=SPATIAL_MERGE_SIZE):
    t, h, w = grid
    return t * (h // merge) * (w // merge)


def _random_grid(rng, max_tokens):
    """One vision item grid (t, h, w) whose decoder tokens fit in max_tokens.

    h, w are in patch units and must be divisible by the merge size; the item
    occupies t*(h/merge)*(w/merge) decoder slots.
    """
    # ~25% of items are multi-frame video (t in 2-3), rest still images.
    t = rng.choice((2, 3)) if rng.random() < 0.25 else 1
    # A short budget cannot carry a 2x2 merged grid per frame. Clamp after the
    # draw rather than before it, so the rng stream is unchanged and only
    # otherwise-impossible cases behave differently.
    while t > 1 and max_tokens // t < 4:
        t -= 1
    # Choose merged spatial extent (h/2, w/2) so tokens = t * mh * mw fills the
    # share as closely as the factorization allows without exceeding it.
    budget = max_tokens // t
    assert budget >= 4, f"budget {budget} too small for a 2x2 merged grid"
    mh = rng.randint(2, min(16, budget // 2))
    mw = max(2, min(16, budget // mh))
    return (t, mh * SPATIAL_MERGE_SIZE, mw * SPATIAL_MERGE_SIZE)


def _image_token_bounds(total_target):
    """Image-token window for a sample of ``total_target`` tokens.

    Identical to ``IMAGE_TOKENS_RANGE`` for the built-in [1000, 2000] pool; the
    clamps only bind for the short samples an external length distribution can
    produce, where a fixed 100-500 image budget would leave no room for text.
    """
    high = max(4, min(IMAGE_TOKENS_RANGE[1], total_target // 2))
    low = max(4, min(IMAGE_TOKENS_RANGE[0], high))
    return low, high


def _multimodal_scenario(rng, total_target):
    image_low, image_high = _image_token_bounds(total_target)

    # Rejection-sample grids until the image token total lands in range (the
    # grid factorization can undershoot a share, so one draw is not enough).
    # The window is a *quality* constraint -- text fills whatever remains, so
    # any draw yields an exact total. Short samples from an external length
    # distribution leave a narrow window, so after MAX_GRID_DRAWS attempts any
    # structurally valid draw is accepted rather than looping.
    attempts = 0
    while True:
        attempts += 1
        image_target = rng.randint(image_low, image_high)
        # Every item needs at least a 2x2 merged grid, so the item count is
        # bounded by the drawn budget (always 3 for the built-in pool).
        num_items = rng.randint(1, max(1, min(3, image_target // 4)))
        grids = []
        remaining = image_target
        for i in range(num_items):
            items_left = num_items - i
            # Leave at least 4 tokens (minimum 2x2 merged still) per later item.
            share = remaining - 4 * (items_left - 1) if items_left > 1 else remaining
            if items_left > 1:
                lo = max(4, share // 2)
                share = rng.randint(lo, share)
            grids.append(_random_grid(rng, share))
            remaining -= _item_tokens(grids[-1])
        image_tokens = sum(_item_tokens(g) for g in grids)
        # k items need k+1 text chunks, each at least 1 token.
        text_total = total_target - image_tokens - num_items
        if text_total < num_items + 1:
            continue
        if image_low <= image_tokens <= image_high or attempts >= MAX_GRID_DRAWS:
            break

    # Text fills the remainder: total = text + image_tokens + num_items sentinels.
    cuts = sorted(rng.sample(range(1, text_total), num_items))
    bounds = [0] + cuts + [text_total]
    text_chunks = tuple(bounds[i + 1] - bounds[i] for i in range(num_items + 1))

    return (tuple(grids), text_chunks)


def _text_only_scenario(rng, total_target):
    return ((), (total_target,))


def scenario_totals(scenario, merge=SPATIAL_MERGE_SIZE):
    """(total_tokens, image_tokens) of one scenario as the dataset builds it."""
    grids, text_chunks = scenario
    image_tokens = sum(_item_tokens(g, merge) for g in grids)
    total = sum(text_chunks) + image_tokens + len(grids)
    return total, image_tokens


def build_scenarios(pool_size=POOL_SIZE, seed=GENERATOR_SEED, length_config=None):
    """Deterministic pool of ``pool_size`` scenarios.

    The first five entries deliberately cover every structural case the
    dataset contract needs -- multi-image, multi-frame video, variable grids,
    and a text-only sample -- so a caller that only looks at a short prefix
    still sees all of them.

    Args:
        pool_size: Number of scenarios.
        seed: Drives the private ``random.Random``; the pool is byte-identical
            across processes and ranks for a given seed.
        length_config: Optional ``--mdp-mock-dataset-config-json`` payload. When
            given, each sample's *total* token length is drawn from that
            distribution and the vision/text split is fitted to it; when
            ``None``, totals are drawn uniformly from ``TOTAL_TOKENS_RANGE`` as
            before (the default pool is unchanged).
    """
    rng = random.Random(seed)
    totals = (
        None
        if length_config is None
        else draw_total_token_lengths(length_config, pool_size, seed=seed)
    )
    pool = []
    for i in range(pool_size):
        text_only = i % TEXT_ONLY_EVERY == TEXT_ONLY_EVERY - 1
        # Drawn here rather than inside the scenario builders so both branches
        # consume the same rng budget with and without a length distribution.
        total_target = rng.randint(*TOTAL_TOKENS_RANGE) if totals is None else totals[i]
        if text_only:
            scenario = _text_only_scenario(rng, total_target)
        else:
            scenario = _multimodal_scenario(rng, total_target)
        total, image_tokens = scenario_totals(scenario)
        assert total == total_target, f"scenario {i}: total {total} != target {total_target}"
        grids = scenario[0]
        if grids:
            _, image_high = _image_token_bounds(total_target)
            assert 4 <= image_tokens <= image_high, (
                f"scenario {i}: image tokens {image_tokens} outside [4, {image_high}]"
            )
            for t, h, w in grids:
                assert h % SPATIAL_MERGE_SIZE == 0 and w % SPATIAL_MERGE_SIZE == 0
        else:
            assert image_tokens == 0
        pool.append(scenario)
    return tuple(pool)


# ---------------------------------------------------------------------------
# ECD-like pool (``--mdp-mock-scenario ecd``)
# ---------------------------------------------------------------------------
# Measured on the first 8,192 complete samples of nvidia/Nemotron-Image-Training-v3
# ``ecd/ecd.jsonl`` @ 7656391d (chart QA conversations; images from
# ChartFoundation/ECD-10k-Images @ 284489f1) under the Qwen/Qwen3.8-Flash-Next @ de4b8e4d
# processor with 65,536-262,144 image pixels, keeping whole samples whose total length
# (aligned to 8) fits 2048 -- the selection of Megatron-Bridge PR #6354's image-SFT
# subset. 12,729 rows scanned, 4,537 overlength skipped. Every sample has exactly one
# still image; image tokens are p50 240 / p1 225 / max 256 (most charts hit the pixel
# cap), total length mean 1038, p50 918, max 2047.

#: Total-length quantiles at 0, 2, ..., 100 % (51 entries). Lengths are drawn by
#: linear interpolation between neighbouring quantiles (inverse CDF).
ECD_TOTAL_LENGTH_QUANTILES = (
    419, 522, 556, 583, 603, 619, 636, 651, 665, 678, 690, 703, 717, 729, 742, 754, 769,
    785, 799, 814, 829, 846, 863, 882, 898, 918, 939, 960, 984, 1008, 1033, 1060, 1091,
    1123, 1156, 1195, 1237, 1278, 1321, 1365, 1407, 1453, 1509, 1572, 1629, 1686, 1745,
    1801, 1876, 1962, 2047,
)

#: (h, w) image grids in patch units (t = 1) with their sample counts out of 8,192.
ECD_IMAGE_GRIDS = (
    (24, 40, 2055), (28, 34, 996), (26, 38, 924), (26, 36, 885), (22, 44, 878),
    (34, 28, 424), (30, 32, 301), (20, 48, 210), (28, 36, 143), (18, 54, 139),
    (24, 42, 125), (38, 26, 123), (32, 32, 117), (24, 38, 104), (22, 42, 99),
    (32, 30, 93), (36, 26, 91), (16, 58, 67), (18, 52, 56), (40, 24, 52), (22, 46, 48),
    (20, 46, 43), (18, 50, 23), (30, 34, 22), (44, 22, 22), (36, 28, 19), (28, 32, 15),
    (20, 50, 15), (26, 34, 11), (16, 60, 9), (14, 64, 8), (28, 30, 8), (18, 56, 7),
    (16, 62, 7), (34, 30, 6), (28, 28, 5), (26, 30, 5), (16, 56, 5), (38, 24, 5),
    (42, 22, 4), (16, 64, 4), (30, 30, 4), (48, 20, 4), (42, 24, 2), (20, 34, 2),
    (46, 20, 2), (58, 16, 1), (50, 20, 1), (14, 66, 1), (54, 18, 1), (50, 18, 1),
)

#: Text tokens before the image (``<|im_start|>user\n`` -- ECD puts the image first).
ECD_PREFIX_TEXT_TOKENS = 3
ECD_POOL_SIZE = 1024
ECD_MAX_TOTAL_TOKENS = ECD_TOTAL_LENGTH_QUANTILES[-1]


def _ecd_total_length(rng):
    u = rng.random() * (len(ECD_TOTAL_LENGTH_QUANTILES) - 1)
    i = min(int(u), len(ECD_TOTAL_LENGTH_QUANTILES) - 2)
    lo, hi = ECD_TOTAL_LENGTH_QUANTILES[i], ECD_TOTAL_LENGTH_QUANTILES[i + 1]
    return int(round(lo + (u - i) * (hi - lo)))


def build_ecd_scenarios(pool_size=ECD_POOL_SIZE, seed=GENERATOR_SEED):
    """Deterministic pool of ECD-like scenarios: one still image + text per sample.

    Image grid and total length are drawn independently from the measured ECD tables
    above (the image token count is nearly constant, so the joint distribution is
    essentially the product). Layout: ``ECD_PREFIX_TEXT_TOKENS`` text tokens, the
    vision-start sentinel, the image rows, then the remaining text (question + answer).
    No text-only samples, no video, total length <= ``ECD_MAX_TOTAL_TOKENS``.
    """
    rng = random.Random(seed)
    grids = [(1, h, w) for h, w, _ in ECD_IMAGE_GRIDS]
    weights = [n for _, _, n in ECD_IMAGE_GRIDS]
    pool = []
    for i in range(pool_size):
        grid = rng.choices(grids, weights=weights)[0]
        total = _ecd_total_length(rng)
        image_tokens = _item_tokens(grid)
        tail = total - image_tokens - 1 - ECD_PREFIX_TEXT_TOKENS
        assert tail >= 1, f"ECD scenario {i}: total {total} too short for {grid}"
        scenario = ((grid,), (ECD_PREFIX_TEXT_TOKENS, tail))
        assert scenario_totals(scenario) == (total, image_tokens)
        pool.append(scenario)
    return tuple(pool)


def build_scenario_pool(kind="default", length_config=None):
    """Scenario pool selected by ``--mdp-mock-scenario`` (``default`` or ``ecd``)."""
    if kind == "ecd":
        if length_config is not None:
            raise ValueError(
                "--mdp-mock-scenario ecd draws lengths from the measured ECD distribution; "
                "it cannot be combined with --mdp-mock-dataset-config-json."
            )
        return build_ecd_scenarios()
    if kind != "default":
        raise ValueError(f"unknown --mdp-mock-scenario {kind!r}; expected 'default' or 'ecd'")
    return build_scenarios(length_config=length_config)


if __name__ == "__main__":
    pool = build_scenarios()
    totals = []
    image_totals = []
    n_text_only = 0
    n_video_items = 0
    n_items_hist = {}
    for i, scenario in enumerate(pool):
        grids, _ = scenario
        total, image_tokens = scenario_totals(scenario)
        totals.append(total)
        if grids:
            image_totals.append(image_tokens)
            n_video_items += sum(1 for t, _, _ in grids if t > 1)
        else:
            n_text_only += 1
        n_items_hist[len(grids)] = n_items_hist.get(len(grids), 0) + 1
        print(f"[{i:02d}] total={total:4d} image={image_tokens:3d} grids={grids}")
    print()
    print(f"pool={len(pool)} text_only={n_text_only} items_hist={sorted(n_items_hist.items())}")
    print(f"total tokens: min={min(totals)} max={max(totals)} mean={sum(totals)/len(totals):.0f}")
    print(
        f"image tokens (multimodal): min={min(image_totals)} max={max(image_totals)} "
        f"mean={sum(image_totals)/len(image_totals):.0f} video_items={n_video_items}"
    )
