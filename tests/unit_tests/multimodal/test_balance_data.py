# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for per-image LPT balance data.

Pure-python, login-node tests. No torch, no CUDA, no model loading.
Loads ``balance_data.py`` directly via importlib to skip the package
``__init__.py`` which chains torch imports.
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_BD_PATH = os.path.join(_REPO_ROOT, "examples", "multimodal_dev", "balance_data.py")
_spec = _ilu.spec_from_file_location("balance_data_under_test", _BD_PATH)
_bd = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bd)

compute_image_flops = _bd.compute_image_flops
balance_per_image_lpt = _bd.balance_per_image_lpt


# Qwen3-VL anchor settings used throughout the tests.
PATCH = 16
HIDDEN = 1280


def _compute_per_rank_load(assignment, raw_batch):
    loads = {}
    for rank, pairs in assignment.items():
        total = 0
        for sample_idx, img_idx in pairs:
            img_size, patch, hidden = raw_batch[sample_idx][img_idx]
            total += compute_image_flops(img_size, patch, hidden)
        loads[rank] = total
    return loads


class TestComputeImageFlops(unittest.TestCase):
    """Verify the corrected O5 FLOPs proxy formula."""

    def test_compute_image_flops_attention_dominates_large_img(self):
        """Attention term (O(num_patches^2 * hidden)) must dominate FFN
        (O(num_patches * hidden^2)) once num_patches > hidden.

        At img=1024, patch=16, hidden=1280: num_patches = (1024/16)^2 = 4096,
        which is > hidden=1280 → attention term should be the larger of the two.
        """
        img_size = 1024
        num_patches = (img_size // PATCH) ** 2  # 4096
        attention = num_patches * num_patches * HIDDEN
        ffn = num_patches * HIDDEN * HIDDEN
        total = compute_image_flops(img_size, PATCH, HIDDEN)
        self.assertEqual(total, attention + ffn)
        # Sanity: attention must be the larger term when num_patches > hidden.
        self.assertGreater(num_patches, HIDDEN)
        self.assertGreater(attention, ffn)

    def test_compute_image_flops_rejects_invalid_args(self):
        with self.assertRaises(ValueError):
            compute_image_flops(0, PATCH, HIDDEN)
        with self.assertRaises(ValueError):
            compute_image_flops(256, 0, HIDDEN)
        with self.assertRaises(ValueError):
            compute_image_flops(256, PATCH, 0)
        # Non-divisible img_size / patch must error.
        with self.assertRaises(ValueError):
            compute_image_flops(257, PATCH, HIDDEN)


class TestLPT(unittest.TestCase):
    """Verify the greedy min-heap LPT scheduler."""

    def test_lpt_single_image_per_sample(self):
        """4 samples each with 1 image of varying size on 4 ranks.

        Sizes: 256, 384, 512, 1024 → flops asc roughly 16/81/256/4096-quartic.
        LPT sorts desc, so first iteration assigns the 1024-px image. With
        4 ranks and 4 images, each rank ends up with exactly 1 image.
        """
        raw = [
            [(256, PATCH, HIDDEN)],
            [(384, PATCH, HIDDEN)],
            [(512, PATCH, HIDDEN)],
            [(1024, PATCH, HIDDEN)],
        ]
        assignment = balance_per_image_lpt(raw, num_ranks=4)

        # Each rank gets exactly one image.
        for r in range(4):
            self.assertIn(r, assignment)
            self.assertEqual(len(assignment[r]), 1)

        # All 4 samples covered.
        owners = {sample_idx for r, pairs in assignment.items() for sample_idx, _ in pairs}
        self.assertEqual(owners, {0, 1, 2, 3})

    def test_lpt_variable_image_count(self):
        """Realistic Mantis-shaped distribution: most samples 1 img, a few
        with 2-3 imgs. All images same size for determinism.

        With 6 samples and 9 images on 4 ranks, the load split is even.
        """
        raw = [
            [(512, PATCH, HIDDEN)],  # s0: 1 img
            [(512, PATCH, HIDDEN)],  # s1: 1 img
            [(512, PATCH, HIDDEN), (512, PATCH, HIDDEN)],  # s2: 2 imgs
            [(512, PATCH, HIDDEN)],  # s3: 1 img
            [(512, PATCH, HIDDEN), (512, PATCH, HIDDEN), (512, PATCH, HIDDEN)],  # s4: 3 imgs
            [(512, PATCH, HIDDEN)],  # s5: 1 img
        ]
        assignment = balance_per_image_lpt(raw, num_ranks=4)
        loads = _compute_per_rank_load(assignment, raw)

        # Total images = 9, distributed across 4 ranks.
        total_imgs = sum(len(v) for v in assignment.values())
        self.assertEqual(total_imgs, 9)

        # Equal-size workload → max - min <= 1 image worth of flops.
        per_image_flops = compute_image_flops(512, PATCH, HIDDEN)
        self.assertLessEqual(max(loads.values()) - min(loads.values()), per_image_flops)

    def test_lpt_extreme_size_variance(self):
        """Image-size variance 256..1024 — exercise the LPT win.

        Construct a bimodal distribution: 4 large (1024-px) + 4 small (256-px)
        images on 4 ranks. Greedy LPT will pair each large with a small,
        yielding 1 large + 1 small per rank.
        """
        raw = [[(1024, PATCH, HIDDEN)]] * 4 + [[(256, PATCH, HIDDEN)]] * 4
        assignment = balance_per_image_lpt(raw, num_ranks=4)
        loads = _compute_per_rank_load(assignment, raw)

        # Each rank should have exactly 2 images (1 large + 1 small).
        for r in range(4):
            self.assertEqual(len(assignment[r]), 2)

        # Perfect pairing → all loads identical.
        flops_1024 = compute_image_flops(1024, PATCH, HIDDEN)
        flops_256 = compute_image_flops(256, PATCH, HIDDEN)
        expected_per_rank = flops_1024 + flops_256
        for r in range(4):
            self.assertEqual(loads[r], expected_per_rank)

    def test_lpt_load_balance_within_4_over_3_bound(self):
        """Graham 1969: LPT max load <= (4/3 - 1/(3m)) * OPT.

        With heterogeneous workload (8 ranks, 20 images of mixed sizes),
        verify the empirical max_load / mean_load ratio stays below the
        4/3 bound (1.333). Mean is a lower bound on OPT, so checking
        max / mean <= 4/3 is a *necessary* (slightly stronger) condition.
        """
        sizes_pixels = [256, 384, 512, 768, 1024]
        raw = []
        # 20 samples, each with 1 image; sizes cycle deterministically.
        for i in range(20):
            raw.append([(sizes_pixels[i % len(sizes_pixels)], PATCH, HIDDEN)])

        num_ranks = 8
        assignment = balance_per_image_lpt(raw, num_ranks=num_ranks)
        loads = _compute_per_rank_load(assignment, raw)

        max_load = max(loads.values())
        mean_load = sum(loads.values()) / num_ranks
        # 4/3 - 1/(3*8) = 4/3 - 1/24 ≈ 1.2917.
        bound = 4.0 / 3.0 - 1.0 / (3.0 * num_ranks)
        self.assertLessEqual(max_load / mean_load, bound + 1e-9)


class TestEdgeCases(unittest.TestCase):
    """Verify edge cases (empty batch, empty samples, single rank)."""

    def test_empty_raw_batch(self):
        assignment = balance_per_image_lpt([], num_ranks=4)
        self.assertEqual(set(assignment.keys()), {0, 1, 2, 3})
        for r in range(4):
            self.assertEqual(assignment[r], [])

    def test_samples_with_no_images(self):
        """Samples with empty image lists must be skipped silently."""
        raw = [[], [(512, PATCH, HIDDEN)], []]  # s0: no imgs  # s1: 1 img  # s2: no imgs
        assignment = balance_per_image_lpt(raw, num_ranks=2)
        # Exactly one image distributed across 2 ranks.
        total = sum(len(v) for v in assignment.values())
        self.assertEqual(total, 1)

    def test_num_ranks_must_be_positive(self):
        with self.assertRaises(ValueError):
            balance_per_image_lpt([[(256, PATCH, HIDDEN)]], num_ranks=0)


if __name__ == "__main__":
    unittest.main()
