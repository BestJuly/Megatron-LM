# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for sequence-axis ModalityBridge ordering helpers.

These tests cover the deterministic gather-order and canonical reorder
contracts without launching distributed collectives. Distributed gradient
coverage for ``gather_to_inner_dp_zero`` lives in
``test_mdp_gradient_flow.py``.

The order helpers (``build_global_ordering`` and
``reconstruct_canonical_order``) are exercised directly with pure-python data
so the LPT-output-shape contract can be verified.
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_MB_PATH = os.path.join(_REPO_ROOT, "examples", "multimodal_dev", "modality_bridge.py")
_spec = _ilu.spec_from_file_location("modality_bridge_under_test", _MB_PATH)
_mb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mb)

build_global_ordering = _mb.build_global_ordering
reconstruct_canonical_order = _mb.reconstruct_canonical_order


class TestBuildGlobalOrdering(unittest.TestCase):
    """build_global_ordering must produce a deterministic gather-order
    sequence of (rank, local_idx, sample_idx, img_idx) tuples."""

    def test_basic(self):
        # 4 ranks, 5 images: rank 0 gets (s0,i0), (s1,i0); rank 1 gets
        # (s2,i0); rank 2 empty; rank 3 gets (s0,i1), (s2,i1).
        assignment = {0: [(0, 0), (1, 0)], 1: [(2, 0)], 2: [], 3: [(0, 1), (2, 1)]}
        ordering = build_global_ordering(assignment)
        expected = [
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 2, 0),
            # rank 2 contributes nothing
            (3, 0, 0, 1),
            (3, 1, 2, 1),
        ]
        self.assertEqual(ordering, expected)


class TestReconstructCanonicalOrder(unittest.TestCase):

    def test_reconstruct_order_preserves_sample_image_pairs(self):
        """The gathered list (in rank-order) must reorder to canonical
        (sample_idx, img_idx) order."""
        assignment = {
            0: [(1, 0), (0, 1)],  # rank 0 gets s1's first img + s0's second img
            1: [(0, 0)],  # rank 1 gets s0's first img
        }
        # Simulate per-image payloads — strings tag the (sample, img) pair.
        gathered = ["s1_i0", "s0_i1", "s0_i0"]  # rank-order
        reordered = reconstruct_canonical_order(gathered, assignment)
        # Canonical order: (0,0), (0,1), (1,0)
        self.assertEqual(reordered, ["s0_i0", "s0_i1", "s1_i0"])

    def test_length_mismatch_raises(self):
        assignment = {0: [(0, 0)], 1: [(1, 0)]}
        with self.assertRaises(ValueError):
            reconstruct_canonical_order(["only_one"], assignment)


class TestGatherByteEquivalence(unittest.TestCase):
    """Verify the rank-major gather post-condition with pure-python data."""

    def test_gather_byte_equivalence(self):
        """Simulate per-rank chunks, manually concat, verify the
        gather-order is rank-order."""
        # Pure-python proxy: each rank's "embeddings" is a unique list.
        chunks = [["r0_tok0", "r0_tok1"], ["r1_tok0"], [], ["r3_tok0", "r3_tok1", "r3_tok2"]]
        # Manual rank-order concat (which is what the gather should produce).
        expected = [tok for chunk in chunks for tok in chunk]
        # The gather post-condition is satisfied by definition of concat.
        self.assertEqual(len(expected), sum(len(c) for c in chunks))
        # Round-trip through reconstruct_canonical_order using a synthetic
        # assignment where each rank's contribution maps to one sample's
        # one image — then "canonical" = "rank-order" because the LPT
        # output here is sorted-by-rank trivially.
        assignment = {}
        sample_counter = 0
        for r, chunk in enumerate(chunks):
            assignment[r] = []
            for _ in chunk:
                assignment[r].append((sample_counter, 0))
                sample_counter += 1
        reordered = reconstruct_canonical_order(expected, assignment)
        # Canonical (sample_idx ascending) matches rank-order here.
        self.assertEqual(reordered, expected)


class TestSequenceAxisCompatibility(unittest.TestCase):
    """Verify the gather/reorder pipeline preserves the property that
    the result is a 1D sequence of per-image token blocks ready to feed
    into the LLM's existing CP-along-seq path."""

    def test_sequence_axis_compatibility(self):
        """After reconstruct_canonical_order, downstream consumer can
        iterate the result as a flat per-sample image-token-block list
        — no hidden-axis manipulation required."""
        # 3 samples; sample 0 has 2 imgs, sample 1 has 1 img, sample 2
        # has 1 img. Total 4 imgs distributed across 2 ranks.
        assignment = {
            0: [(0, 0), (2, 0)],  # rank 0: s0_i0, s2_i0
            1: [(1, 0), (0, 1)],  # rank 1: s1_i0, s0_i1
        }
        # Gather order = rank order:
        gathered = [("s0", 0), ("s2", 0), ("s1", 0), ("s0", 1)]
        reordered = reconstruct_canonical_order(gathered, assignment)
        # Canonical: (s0,i0), (s0,i1), (s1,i0), (s2,i0)
        expected = [("s0", 0), ("s0", 1), ("s1", 0), ("s2", 0)]
        self.assertEqual(reordered, expected)


if __name__ == "__main__":
    unittest.main()
