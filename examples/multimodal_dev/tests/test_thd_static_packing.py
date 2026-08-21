# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed-shape THD batches (``--thd-static-packing``) and the MDP mock length
distribution.

``pack_or_pad_batch`` ends with a TP-group broadcast, so these tests require
``torch.distributed`` to be initialised. Run via::

    torchrun --nproc-per-node 1 -m pytest -q \\
        examples/multimodal_dev/tests/test_thd_static_packing.py
"""

import os
import sys
import types

import numpy as np
import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev import forward_step as fs
from examples.multimodal_dev.data import mdp_scenarios
from megatron.core.packed_seq_params import build_static_thd_metadata
from tests.unit_tests.test_utilities import Utils

MAX_SEQLEN = 64
MAX_PACKED = 8


@pytest.fixture(scope="module", autouse=True)
def _init_model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _fake_args(**overrides):
    base = dict(
        sequence_parallel=False,
        mdp_enable=False,
        thd_static_packing=True,
        max_seqlen_per_dp_cp_rank=MAX_SEQLEN,
        thd_max_packed_sequences=MAX_PACKED,
        thd_tail_padding_policy=None,
        image_token_id=248056,
        vision_spatial_merge_size=2,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


@pytest.fixture
def static_args(monkeypatch):
    """Install a fake ``get_args`` returning a static-packing configuration."""

    def _install(**overrides):
        args = _fake_args(**overrides)
        monkeypatch.setattr(fs, "get_args", lambda: args)
        return args

    return _install


def _make_sample(seq_len, *, base=0, num_patches=4, pixel_dim=8, device="cuda"):
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.long, device=device) + base,
        "labels": torch.arange(seq_len, dtype=torch.long, device=device) + base + 100,
        "loss_mask": torch.ones(seq_len, dtype=torch.float, device=device),
        "pixel_values": torch.full((num_patches, pixel_dim), float(base), device=device),
        "image_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# build_static_thd_metadata -- pure helper
# ---------------------------------------------------------------------------


class TestBuildStaticThdMetadata:
    def test_extend_last_keeps_valid_boundaries(self):
        cu = torch.tensor([0, 5, 12], dtype=torch.int32)
        cu_padded = torch.tensor([0, 6, 14], dtype=torch.int32)
        q, q_padded, real = build_static_thd_metadata(
            cu, cu_padded, target_len=32, max_num_seqs=4, tail_padding_policy="extend_last"
        )
        assert real is None  # no FLOPs override needed
        assert q.tolist() == [0, 5, 12, 12, 12]  # valid vector untouched, then padded
        assert q_padded.tolist() == [0, 6, 32, 32, 32]
        assert q.numel() == q_padded.numel() == 5

    def test_append_dummy_seq_returns_the_real_vector(self):
        cu = torch.tensor([0, 5, 12], dtype=torch.int32)
        cu_padded = torch.tensor([0, 6, 14], dtype=torch.int32)
        q, q_padded, real = build_static_thd_metadata(
            cu, cu_padded, target_len=32, max_num_seqs=4, tail_padding_policy="append_dummy_seq"
        )
        assert real is not None and real.tolist() == [0, 5, 12]
        assert q.tolist() == [0, 5, 12, 30, 30]  # tail became an ordinary sequence
        assert q_padded.tolist() == [0, 6, 14, 32, 32]

    def test_exact_fit_needs_no_tail(self):
        cu = torch.tensor([0, 16, 32], dtype=torch.int32)
        q, q_padded, real = build_static_thd_metadata(
            cu, cu.clone(), target_len=32, max_num_seqs=3, tail_padding_policy="extend_last"
        )
        assert real is None
        assert q.tolist() == [0, 16, 32, 32]

    def test_overflowing_pack_names_the_flag(self):
        cu = torch.tensor([0, 40], dtype=torch.int32)
        with pytest.raises(AssertionError, match="max-seqlen-per-dp-cp-rank"):
            build_static_thd_metadata(
                cu, cu.clone(), target_len=32, max_num_seqs=4, tail_padding_policy="extend_last"
            )

    def test_too_many_sequences_names_the_flag(self):
        cu = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
        with pytest.raises(AssertionError, match="thd_max_packed_sequences"):
            build_static_thd_metadata(
                cu, cu.clone(), target_len=32, max_num_seqs=2, tail_padding_policy="extend_last"
            )

    def test_extend_last_is_rejected_under_cp(self):
        cu = torch.tensor([0, 8], dtype=torch.int32)
        with pytest.raises(AssertionError, match="before CP slicing"):
            build_static_thd_metadata(
                cu,
                cu.clone(),
                target_len=32,
                max_num_seqs=4,
                tail_padding_policy="extend_last",
                cp_size=2,
            )


# ---------------------------------------------------------------------------
# pack_or_pad_batch -- fixed shapes
# ---------------------------------------------------------------------------


# The pinned fast path (--mdp-enable, TP=1) stages into pinned host buffers, so
# it requires the CPU sample tensors a real dataloader produces.
@pytest.mark.parametrize(
    "mdp_enable, sample_device",
    [(False, "cuda"), (True, "cpu")],
    ids=["generic", "pinned"],
)
class TestStaticShapes:
    def test_heterogeneous_batch_has_fixed_shapes(self, static_args, mdp_enable, sample_device):
        static_args(mdp_enable=mdp_enable)
        batch = [
            _make_sample(L, base=i * 1000, device=sample_device)
            for i, L in enumerate([5, 13, 7])
        ]
        packed = fs.pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        for key in ("input_ids", "labels", "loss_mask", "padding_mask"):
            assert packed[key].shape == (1, MAX_SEQLEN), key
        psp = packed["packed_seq_params"]
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        ):
            assert getattr(psp, name).numel() == MAX_PACKED + 1, name
        assert psp.max_seqlen_q == MAX_SEQLEN
        assert psp.max_seqlen_kv == MAX_SEQLEN
        # Derived from the collator's row alignment, not hardcoded: at TP=CP=1 no
        # sample is ever padded, so cu_seqlens and cu_seqlens_padded coincide and
        # there is provably no gap. Claiming True here is not free -- TE disables
        # FlashAttention for THD whenever padding may exist between sequences.
        assert psp.pad_between_seqs is False
        assert torch.equal(psp.cu_seqlens_q, psp.cu_seqlens_q_padded)

    def test_shapes_do_not_depend_on_the_batch(self, static_args, mdp_enable, sample_device):
        static_args(mdp_enable=mdp_enable)
        shapes = set()
        entries = set()
        for lengths in ([3], [5, 13, 7], [11, 11, 11, 11, 2]):
            batch = [
                _make_sample(L, base=i * 1000, device=sample_device)
                for i, L in enumerate(lengths)
            ]
            packed = fs.pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")
            shapes.add(tuple(packed["input_ids"].shape))
            entries.add(int(packed["packed_seq_params"].cu_seqlens_q.numel()))
        assert shapes == {(1, MAX_SEQLEN)}
        assert entries == {MAX_PACKED + 1}

    def test_pad_positions_are_masked_out(self, static_args, mdp_enable, sample_device):
        static_args(mdp_enable=mdp_enable)
        lengths = [5, 13, 7]
        batch = [
            _make_sample(L, base=i * 1000, device=sample_device)
            for i, L in enumerate(lengths)
        ]
        packed = fs.pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        real = sum(lengths)
        assert packed["loss_mask"][0, :real].sum().item() == real
        assert packed["loss_mask"][0, real:].sum().item() == 0
        assert bool(packed["padding_mask"][0, real:].all())
        assert not bool(packed["padding_mask"][0, :real].any())
        assert bool((packed["labels"][0, real:] == -100).all())

    def test_overflow_is_rejected_with_a_named_flag(self, static_args, mdp_enable, sample_device):
        static_args(mdp_enable=mdp_enable)
        batch = [
            _make_sample(MAX_SEQLEN, base=0, device=sample_device),
            _make_sample(MAX_SEQLEN, base=1000, device=sample_device),
        ]
        with pytest.raises(AssertionError, match="max-seqlen-per-dp-cp-rank"):
            fs.pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")


def test_static_packing_off_is_byte_identical(static_args):
    """The opt-in branch must not perturb the default path."""
    batch = [_make_sample(L, base=i * 1000) for i, L in enumerate([5, 13, 7])]
    reference = fs.pack_or_pad_batch(
        [dict(s) for s in batch], use_packed_sequence=True, device="cuda"
    )
    static_args(thd_static_packing=False)
    actual = fs.pack_or_pad_batch(
        [dict(s) for s in batch], use_packed_sequence=True, device="cuda"
    )
    for key in ("input_ids", "labels", "loss_mask", "padding_mask"):
        assert torch.equal(reference[key], actual[key]), key
    assert torch.equal(
        reference["packed_seq_params"].cu_seqlens_q, actual["packed_seq_params"].cu_seqlens_q
    )
    assert reference["packed_seq_params"].max_seqlen_q == actual["packed_seq_params"].max_seqlen_q
    assert "flops_cu_seqlens" not in actual


# ---------------------------------------------------------------------------
# FLOPs accounting
# ---------------------------------------------------------------------------


class TestFlopsAccounting:
    def test_extend_last_reports_only_real_tokens(self, static_args):
        static_args()
        lengths = [5, 13, 7]
        batch = [_make_sample(L, base=i * 1000) for i, L in enumerate(lengths)]
        packed = fs.pack_or_pad_batch(batch, use_packed_sequence=True, device="cuda")

        # extend_last leaves cu_seqlens_q compact, so no override is emitted and
        # the accumulator sees exactly the real tokens.
        assert "flops_cu_seqlens" not in packed
        cu = packed["packed_seq_params"].cu_seqlens_q
        seqlens = (cu[1:] - cu[:-1]).tolist()
        assert sum(seqlens) == sum(lengths)

    def test_dummy_tail_would_pollute_without_the_override(self):
        """The override exists because the dummy tail lands in cu_seqlens_q."""
        lengths = [5, 13, 7]
        cu = torch.tensor([0, 5, 18, 25], dtype=torch.int32)
        q, _, real = build_static_thd_metadata(
            cu,
            cu.clone(),
            target_len=MAX_SEQLEN,
            max_num_seqs=MAX_PACKED,
            tail_padding_policy="append_dummy_seq",
        )
        polluted = (q[1:] - q[:-1]).clamp(min=0).sum().item()
        honest = (real[1:] - real[:-1]).sum().item()
        assert honest == sum(lengths)
        assert polluted == MAX_SEQLEN  # would overstate by the whole pad


# ---------------------------------------------------------------------------
# Mock length distribution
# ---------------------------------------------------------------------------


LOGNORMAL = {
    "mode": "distribution",
    "type": "lognormal",
    "format": "thd",
    "min_seq_len": 256,
    "max_seq_len": 4096,
    "mean_seq_len": 2048,
    "lognormal_sigma": 1.1,
}
DEGENERATE = {
    "mode": "distribution",
    "type": "lognormal",
    "format": "thd",
    "min_seq_len": 1024,
    "max_seq_len": 1024,
    "mean_seq_len": 1024,
    "lognormal_sigma": 1.1,
}


class TestMockLengthDistribution:
    def test_default_pool_is_unchanged(self):
        pool = mdp_scenarios.build_scenarios()
        totals = [mdp_scenarios.scenario_totals(s)[0] for s in pool]
        assert len(pool) == mdp_scenarios.POOL_SIZE
        assert min(totals) >= mdp_scenarios.TOTAL_TOKENS_RANGE[0]
        assert max(totals) <= mdp_scenarios.TOTAL_TOKENS_RANGE[1]

    def test_distribution_controls_the_totals(self):
        pool = mdp_scenarios.build_scenarios(length_config=LOGNORMAL)
        totals = [mdp_scenarios.scenario_totals(s)[0] for s in pool]
        assert min(totals) >= LOGNORMAL["min_seq_len"]
        assert max(totals) <= LOGNORMAL["max_seq_len"]
        # Wider than the built-in [1000, 2000] window, which is the point.
        assert max(totals) - min(totals) > 1000

    def test_degenerate_distribution_is_constant(self):
        pool = mdp_scenarios.build_scenarios(length_config=DEGENERATE)
        totals = {mdp_scenarios.scenario_totals(s)[0] for s in pool}
        assert totals == {1024}

    def test_pool_is_identical_across_ranks(self):
        """Every MDP rank rebuilds the pool independently; a divergence hangs.

        Perturbing NumPy's global RNG between builds stands in for the different
        global state each rank carries.
        """
        first = mdp_scenarios.build_scenarios(length_config=LOGNORMAL)
        np.random.seed(9871)
        np.random.random(1000)
        second = mdp_scenarios.build_scenarios(length_config=LOGNORMAL)
        assert first == second

    def test_the_draw_leaves_no_global_numpy_side_effect(self):
        np.random.seed(4242)
        expected = np.random.random(4).tolist()
        np.random.seed(4242)
        mdp_scenarios.draw_total_token_lengths(LOGNORMAL, 8)
        assert np.random.random(4).tolist() == expected

    def test_dataset_emits_the_requested_lengths(self):
        from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset

        dataset = MdpThdMockDataset(num_samples=16, length_config=DEGENERATE)
        for i in range(16):
            assert dataset[i]["input_ids"].shape[0] == 1024
