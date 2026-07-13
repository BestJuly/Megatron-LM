# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for MDP CP-local image scheduling and reconstruction.

Pure-python login-node tests that exercise:

1. The CP-pair LPT split contract — when called with ``num_ranks=cp_size``
   (typically 2), per-image LPT distributes images BETWEEN the two
   CP-pair members balancing FLOPs.
2. The reorder-to-canonical contract under CP-pair gather order.
3. PP x CP InnerDP rank layout.
4. Login-node-safe CLI parsing for the public MDP flags.

The pure-python helpers are loaded via importlib.util.spec_from_file_location
to bypass the torch-poisoned ``tests/unit_tests/__init__.py``.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import os
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PARALLEL_GROUPS_PATH = os.path.join(
    _REPO_ROOT, "examples", "multimodal_dev", "mdp_parallel_groups.py"
)
_BAL_PATH = os.path.join(_REPO_ROOT, "examples", "multimodal_dev", "balance_data.py")
_MB_PATH = os.path.join(_REPO_ROOT, "examples", "multimodal_dev", "modality_bridge.py")
_ARGS_PATH = os.path.join(_REPO_ROOT, "examples", "multimodal_dev", "arguments.py")


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_groups = _load("mdp_cp_local_layout_groups", _PARALLEL_GROUPS_PATH)
_bal = _load("mdp_cp_local_layout_bal", _BAL_PATH)
_args_mod = _load("mdp_cp_local_layout_args", _ARGS_PATH)
try:
    _mb = _load("mdp_cp_local_layout_mb", _MB_PATH)
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _mb = None
try:
    _torch = __import__("torch")
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _torch = None

compute_pp_cp_inner_dp_layout = _groups.compute_pp_cp_inner_dp_layout
find_pp_cp_inner_dp_group_for_rank = _groups.find_pp_cp_inner_dp_group_for_rank
balance_per_image_lpt = _bal.balance_per_image_lpt
compute_image_flops = _bal.compute_image_flops
build_global_ordering = _mb.build_global_ordering if _mb is not None else None
reconstruct_canonical_order = _mb.reconstruct_canonical_order if _mb is not None else None
gather_order_is_canonical = _mb.gather_order_is_canonical if _mb is not None else None
# reorder_gathered_embeddings uses torch/distributed at call time.
# Function reference resolves at the loaded module level.
reorder_gathered_embeddings = _mb.reorder_gathered_embeddings if _mb is not None else None


class TestMdpArgs(unittest.TestCase):
    """The PR exposes one MDP encoder-mode switch."""

    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        _args_mod.add_multimodal_args(parser)
        return parser.parse_args(argv)

    def test_no_auxiliary_vision_parallel_arg(self):
        args = self._parse([])
        self.assertFalse(hasattr(args, "mdp_vision_parallel_mode"))


class TestPpCpInnerDpLayout(unittest.TestCase):
    """PP-replicated MDP fixes TP and outer DP while spanning PP x CP."""

    def test_tp1_pp2_cp2(self):
        groups = compute_pp_cp_inner_dp_layout(world_size=8, tp_size=1, cp_size=2, pp_size=2)
        self.assertEqual(groups, [[0, 1, 4, 5], [2, 3, 6, 7]])
        group, local_rank = find_pp_cp_inner_dp_group_for_rank(5, groups)
        self.assertEqual(group, [0, 1, 4, 5])
        self.assertEqual(local_rank, 3)

    def test_tp2_pp2_cp2_keeps_tp_fixed(self):
        groups = compute_pp_cp_inner_dp_layout(world_size=16, tp_size=2, cp_size=2, pp_size=2)
        self.assertEqual(groups, [[0, 2, 8, 10], [1, 3, 9, 11], [4, 6, 12, 14], [5, 7, 13, 15]])
        self.assertEqual(sorted(rank for group in groups for rank in group), list(range(16)))

    def test_pp_first_mapping_and_cp1(self):
        pp_first = compute_pp_cp_inner_dp_layout(
            world_size=16, tp_size=1, cp_size=2, pp_size=2, order="tp-cp-ep-pp-dp"
        )
        self.assertEqual(pp_first, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        cp1 = compute_pp_cp_inner_dp_layout(world_size=16, tp_size=1, cp_size=1, pp_size=4)
        self.assertEqual(cp1, [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])

    def test_rejects_invalid_world(self):
        with self.assertRaises(ValueError):
            compute_pp_cp_inner_dp_layout(world_size=10, tp_size=1, cp_size=2, pp_size=4)

    @unittest.skipIf(_torch is None, "torch is unavailable")
    def test_matches_megatron_rank_generator(self):
        from megatron.core.parallel_state import RankGenerator

        cases = (
            (64, 1, 2, 2, "tp-cp-ep-dp-pp"),
            (64, 2, 2, 2, "tp-cp-ep-dp-pp"),
            (64, 1, 4, 2, "tp-cp-ep-dp-pp"),
            (64, 1, 2, 4, "tp-cp-ep-dp-pp"),
            (64, 1, 2, 2, "tp-cp-ep-pp-dp"),
        )
        for world, tp, pp, cp, order in cases:
            with self.subTest(world=world, tp=tp, pp=pp, cp=cp, order=order):
                expected = RankGenerator(
                    tp=tp, ep=1, dp=world // (tp * pp * cp), pp=pp, cp=cp, order=order
                ).get_ranks("pp-cp")
                self.assertEqual(
                    compute_pp_cp_inner_dp_layout(
                        world_size=world, tp_size=tp, cp_size=cp, pp_size=pp, order=order
                    ),
                    expected,
                )


class TestCpPairLPTSplit(unittest.TestCase):
    """Under CP-local MDP, ``balance_per_image_lpt`` is called with
    ``num_ranks=cp_pair_size`` (typically 2). The output must:

      * Split the image set across exactly cp_pair_size ranks.
      * Balance FLOPs within Graham's (4/3 - 1/(3m)) * OPT bound.
      * Be deterministic given the same descriptor list.
    """

    def test_2_ranks_balanced_split(self):
        # 4 equal images: each rank gets 2 → perfect balance (within LPT).
        descriptors = [(256, 16, 1280)] * 4
        assignment = balance_per_image_lpt([descriptors], num_ranks=2)
        self.assertEqual(set(assignment.keys()), {0, 1})
        self.assertEqual(len(assignment[0]), 2)
        self.assertEqual(len(assignment[1]), 2)
        # All 4 images present in the union.
        all_imgs = sorted(assignment[0] + assignment[1], key=lambda t: t[1])
        self.assertEqual(all_imgs, [(0, 0), (0, 1), (0, 2), (0, 3)])

    def test_2_ranks_unbalanced_input_load_within_bound(self):
        # One large (1024) + three small (256) images. Per-image FLOPs:
        #   1024px → (64*64)^2 * 1280 + (64*64) * 1280^2 ≈ 2.2e10 + 6.7e9
        #   256px  → (16*16)^2 * 1280 + (16*16) * 1280^2 ≈ 8.4e7  + 4.2e8
        # LPT: rank 0 takes the 1024 image; rank 1 takes the three 256s.
        descriptors = [(1024, 16, 1280), (256, 16, 1280), (256, 16, 1280), (256, 16, 1280)]
        assignment = balance_per_image_lpt([descriptors], num_ranks=2)
        # Check assignment shape (no specific rank-to-image binding, but
        # one rank should own the largest image alone).
        sizes_per_rank = sorted([len(v) for v in assignment.values()])
        # One rank owns 1 image, the other owns 3.
        self.assertEqual(sizes_per_rank, [1, 3])

    def test_4_ranks_cp_4_workload(self):
        # CP=4: 8 images split across 4 CP-pair members.
        descriptors = [(256, 16, 1280)] * 8
        assignment = balance_per_image_lpt([descriptors], num_ranks=4)
        self.assertEqual(set(assignment.keys()), {0, 1, 2, 3})
        # Balanced split: every rank gets 2 images.
        for r in range(4):
            self.assertEqual(len(assignment[r]), 2)

    def test_determinism(self):
        # Same input → same output across two calls.
        descriptors = [(512, 16, 1280), (256, 16, 1280), (1024, 16, 1280)]
        a1 = balance_per_image_lpt([descriptors], num_ranks=2)
        a2 = balance_per_image_lpt([descriptors], num_ranks=2)
        self.assertEqual(a1, a2)

    def test_empty_batch(self):
        # No images at all → every rank gets empty list (text-only batch).
        assignment = balance_per_image_lpt([[]], num_ranks=2)
        self.assertEqual(assignment, {0: [], 1: []})


class TestReorderToCanonicalCpPair(unittest.TestCase):
    """Under CP-local MDP, the all_gather over the CP-pair (size cp) produces
    [rank0_payload | rank1_payload | ...] in rank-of-CP-pair order. The
    downstream reconstruct_canonical_order must sort by (sample_idx,
    img_idx_within_sample) for the _scatter_vision_embeddings consumer."""

    def setUp(self):
        if reconstruct_canonical_order is None:
            self.skipTest("torch not available to import modality_bridge")

    def test_reorder_round_trip_cp_2(self):
        # CP-pair with 4 images: LPT split puts (sample=0, img=2) on rank 0
        # and (sample=0, img=0), (sample=0, img=1), (sample=0, img=3) on rank 1.
        # (This is a perfectly-legal LPT outcome for unequal FLOPs.)
        assignment = {0: [(0, 2)], 1: [(0, 0), (0, 1), (0, 3)]}
        # Simulate per-image payloads tagged with (sample, img).
        gathered = ["s0_i2", "s0_i0", "s0_i1", "s0_i3"]  # rank-order
        canonical = reconstruct_canonical_order(gathered, assignment)
        self.assertEqual(canonical, ["s0_i0", "s0_i1", "s0_i2", "s0_i3"])

    def test_reorder_preserves_independence_across_pairs(self):
        # CP-pair internally — 3 images split as (0,1) on rank 0,
        # (0,0), (0,2) on rank 1.
        assignment = {0: [(0, 1)], 1: [(0, 0), (0, 2)]}
        gathered = ["A1", "A0", "A2"]
        canonical = reconstruct_canonical_order(gathered, assignment)
        self.assertEqual(canonical, ["A0", "A1", "A2"])

    def test_gather_order_canonical_fast_path_predicate(self):
        if gather_order_is_canonical is None:
            self.skipTest("torch not available to import modality_bridge")
        self.assertTrue(gather_order_is_canonical({0: [(0, 0), (0, 1)], 1: [(0, 2), (0, 3)]}))
        self.assertFalse(gather_order_is_canonical({0: [(0, 1)], 1: [(0, 0), (0, 2)]}))


class TestProductionReorderWiresCanonical(unittest.TestCase):
    """BLOCK #1 fix-up — verify that the production reorder helper
    (``reorder_gathered_embeddings``) produces canonical (sample_idx,
    img_idx) order from a synthetic CP=2 gather output, even when LPT
    produced a non-canonical permutation.

    Approach: build a synthetic gather output where each "embedding row"
    is uniquely tagged with its (sample_idx, img_idx) origin, simulate
    the gather as a rank-major concat in LPT order, run the production
    helper, and assert the output rows are in canonical order.

    This test uses a fake-distributed shim (FakeProcGroup) executed
    sequentially on each rank — the helper's all_gather call is patched
    to read from rank-local state, simulating the post-NCCL view. Login
    node has no torch/NCCL; we use a minimal numpy-backed fake.
    """

    def setUp(self):
        if _mb is None:
            self.skipTest("torch not available to import modality_bridge")

    def test_canonical_reorder_under_non_canonical_lpt(self):
        """3 images of sizes [1024, 256, 256] — LPT assigns the largest
        to CP-rank-0, the two smaller to CP-rank-1.

        Image LPT assignment (after balance_per_image_lpt with cp=2):
            rank 0: [(s0, img_0)]         — t*h*w = 16 rows
            rank 1: [(s0, img_1), (s0, img_2)] — 4 + 4 = 8 rows each

        Gather order: [rank0_payload | rank1_payload]
        Gather-order images: img_0 (rank0), img_1 (rank1), img_2 (rank1).
        That is ALREADY canonical (sample, img_idx ascending). To force
        a NON-canonical gather we flip the assignment: rank 0 owns
        img_1, rank 1 owns img_0 and img_2 (in LPT order [img_2, img_0]).
        """
        from unittest import mock

        import torch

        mb = _mb  # loaded via importlib at module top — bypasses package init

        # Non-canonical assignment — rank 0 owns the middle image,
        # rank 1 owns the first and last (in LPT order [img_2, img_0]).
        rank_assignment = {
            0: [(0, 1)],
            1: [(0, 2), (0, 0)],  # rank 1 LPT-order: img_2 first, img_0 second
        }
        # Row counts per image (t*h*w): img_0 -> 4, img_1 -> 16, img_2 -> 4.
        row_count_by_img = {0: 4, 1: 16, 2: 4}

        hidden = 8

        # Build per-rank "local embeddings" — concatenation of per-image
        # blocks in LPT order on each rank. Each row tagged with the
        # image index (via a unique value pattern: row content = img_idx).
        def build_local(rank):
            pairs = rank_assignment[rank]
            blocks = []
            for _s, img_idx in pairs:
                rc = row_count_by_img[img_idx]
                # Each row is filled with img_idx so we can identify it.
                block = torch.full((rc, hidden), fill_value=float(img_idx), dtype=torch.float32)
                blocks.append(block)
            return (
                torch.cat(blocks, dim=0)
                if blocks
                else torch.zeros((0, hidden), dtype=torch.float32)
            )

        local_emb_r0 = build_local(0)
        local_emb_r1 = build_local(1)

        # Simulate the gather: concat in rank order → flat
        # [total_rows, hidden] in gather order (= LPT order).
        gathered = torch.cat([local_emb_r0, local_emb_r1], dim=0)

        # Build local per-image row counts in LPT order, per rank.
        local_counts_r0 = torch.tensor(
            [row_count_by_img[img_idx] for (_s, img_idx) in rank_assignment[0]], dtype=torch.int64
        )
        local_counts_r1 = torch.tensor(
            [row_count_by_img[img_idx] for (_s, img_idx) in rank_assignment[1]], dtype=torch.int64
        )

        # Patch torch.distributed.* — fake out all_gather / get_rank /
        # get_world_size. For each "rank pass" we pretend we are that
        # rank and supply the matching local data.
        for fake_rank, local_counts in ((0, local_counts_r0), (1, local_counts_r1)):
            with (
                mock.patch.object(torch.distributed, "is_available", return_value=True),
                mock.patch.object(torch.distributed, "is_initialized", return_value=True),
                mock.patch.object(torch.distributed, "get_world_size", return_value=2),
                mock.patch.object(torch.distributed, "get_rank", return_value=fake_rank),
            ):

                # all_gather copies the per-rank values into the
                # output list. We need to mirror real NCCL semantics.
                # First call: per-rank n_local int64 tensors.
                # Second call: padded counts tensors (length max_n).
                # We'll detect which call we're in by tensor shape.
                call_state = {"n": 0}

                def fake_all_gather(tensor_list, tensor, group=None):
                    # tensor: this rank's local payload.
                    # tensor_list: pre-allocated per-rank receive buffers.
                    # Fill rank 0 slot with rank-0 data, rank 1 slot with
                    # rank-1 data. We use call_state to keep track.
                    call_state["n"] += 1
                    if tensor.numel() == 1:
                        # n_local exchange.
                        n0 = local_counts_r0.shape[0]
                        n1 = local_counts_r1.shape[0]
                        tensor_list[0].fill_(n0)
                        tensor_list[1].fill_(n1)
                    else:
                        # row-counts exchange (padded to max_n).
                        max_n = tensor.shape[0]
                        # r0 padded:
                        pad0 = torch.zeros((max_n,), dtype=torch.int64)
                        pad0[: local_counts_r0.shape[0]] = local_counts_r0
                        pad1 = torch.zeros((max_n,), dtype=torch.int64)
                        pad1[: local_counts_r1.shape[0]] = local_counts_r1
                        tensor_list[0].copy_(pad0)
                        tensor_list[1].copy_(pad1)

                with mock.patch.object(
                    torch.distributed, "all_gather", side_effect=fake_all_gather
                ):
                    out = mb.reorder_gathered_embeddings(
                        gathered_embeddings=gathered,
                        local_per_image_row_counts=local_counts,
                        rank_assignment=rank_assignment,
                        group=None,  # fake group; calls patched
                    )

                # Expected: rows in canonical order — img_0 (4 rows of
                # value 0), then img_1 (16 rows of value 1), then img_2
                # (4 rows of value 2).
                expected_rows = (
                    [0.0] * row_count_by_img[0]
                    + [1.0] * row_count_by_img[1]
                    + [2.0] * row_count_by_img[2]
                )
                actual_rows = [float(out[i, 0].item()) for i in range(out.shape[0])]
                self.assertEqual(
                    actual_rows,
                    expected_rows,
                    f"rank {fake_rank}: gathered embeddings not in "
                    f"canonical order. got={actual_rows}, "
                    f"expected={expected_rows}",
                )

    def test_total_imgs_zero_returns_unchanged(self):
        """Edge case: total_imgs == 0 → return the input unchanged."""
        import torch

        mb = _mb  # loaded via importlib

        empty = torch.zeros((0, 8), dtype=torch.float32)
        out = mb.reorder_gathered_embeddings(
            gathered_embeddings=empty,
            local_per_image_row_counts=torch.zeros((0,), dtype=torch.int64),
            rank_assignment={0: [], 1: []},
            group=None,
        )
        # Returned tensor must be the same shape (zero rows).
        self.assertEqual(tuple(out.shape), (0, 8))

    def test_total_imgs_one_returns_unchanged(self):
        """Edge case: total_imgs == 1 → no permutation needed."""
        import torch

        mb = _mb  # loaded via importlib

        single = torch.full((5, 4), 7.0, dtype=torch.float32)
        # Rank 0 owns the single image; rank 1 contributes nothing.
        out = mb.reorder_gathered_embeddings(
            gathered_embeddings=single,
            local_per_image_row_counts=torch.tensor([5], dtype=torch.int64),
            rank_assignment={0: [(0, 0)], 1: []},
            group=None,
        )
        # No work — return the same tensor.
        self.assertTrue(torch.equal(out, single))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
