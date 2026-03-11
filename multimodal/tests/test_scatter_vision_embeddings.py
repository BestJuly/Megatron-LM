# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for MultimodalModel._scatter_vision_embeddings.

Covers correctness and SP+TP shape consistency.

Run (from Megatron-LM root):
  # Single-process — runs TP=1 tests only:
  python -m pytest multimodal/tests/test_scatter_vision_embeddings.py -v

  # Two-process — also runs SP+TP=2 tests:
  torchrun --nproc_per_node 2 -m pytest multimodal/tests/test_scatter_vision_embeddings.py -v
"""

import types

import pytest
import torch

from tests.unit_tests.test_utilities import Utils

from multimodal.models.base import MultimodalModel

# ---------------------------------------------------------------------------
# Sentinel token ID used as the image placeholder throughout these tests.
# (Matches the default used in run_qwen35_vl.sh / Qwen3.5-VL vocab.)
# ---------------------------------------------------------------------------
IMAGE_TOKEN_ID = 151655


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock(sequence_parallel: bool):
    """Minimal namespace that satisfies _scatter_vision_embeddings's self accesses."""
    obj = types.SimpleNamespace()
    obj.image_token_id = IMAGE_TOKEN_ID
    obj.config = types.SimpleNamespace(sequence_parallel=sequence_parallel)
    return obj


def _scatter(mock, input_ids, text_emb, vis_emb):
    """Call the method as an unbound function on the mock namespace."""
    return MultimodalModel._scatter_vision_embeddings(mock, input_ids, text_emb, vis_emb)


def _reference_scatter(input_ids, text_emb_full, vis_emb):
    """Pure-Python reference: scatter vision tokens into text embeddings.

    No SP logic — operates on the full [S, B, D] tensor.  Used to build
    the expected result that the SP variant must match after gathering.

    Args:
        input_ids:       [B, S] token IDs (CPU or CUDA)
        text_emb_full:   [S, B, D]
        vis_emb:         [N_img, D]  (may be empty when N_img == 0)

    Returns:
        combined: [S, B, D]
    """
    combined = text_emb_full.clone().transpose(0, 1).contiguous()  # [B, S, D]
    image_mask = (input_ids == IMAGE_TOKEN_ID)                      # [B, S]
    mask_expanded = image_mask.unsqueeze(-1).expand_as(combined)
    if mask_expanded.any():
        combined = combined.masked_scatter(mask_expanded, vis_emb)
    return combined.transpose(0, 1).contiguous()                    # [S, B, D]


def _make_inputs(B, S, D, image_positions: dict, seed: int = 42):
    """Create identical input tensors across all ranks.

    Tensors are constructed on CPU (where torch.manual_seed is deterministic
    across processes) and then moved to CUDA, guaranteeing every rank holds
    the same values.

    Args:
        B:               batch size
        S:               sequence length
        D:               hidden dim
        image_positions: {sample_idx: [pos, ...]} mapping image token positions
        seed:            RNG seed

    Returns:
        input_ids   [B, S]    CUDA int64
        text_emb    [S, B, D] CUDA float32
        vis_emb     [N, D]    CUDA float32  (N = total image tokens)
    """
    torch.manual_seed(seed)
    input_ids = torch.randint(0, 1000, (B, S))
    for sample_idx, positions in image_positions.items():
        for pos in positions:
            input_ids[sample_idx, pos] = IMAGE_TOKEN_ID

    n_img = sum(len(v) for v in image_positions.values())
    text_emb = torch.randn(S, B, D)
    vis_emb = torch.randn(n_img, D)

    return input_ids.cuda(), text_emb.cuda(), vis_emb.cuda()


def _world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


# ---------------------------------------------------------------------------
# TP=1, SP=False  —  baseline correctness
# ---------------------------------------------------------------------------

class TestScatterNoSP:
    """Correctness tests with a single GPU / TP=1, no sequence parallelism."""

    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_image_tokens_replaced(self):
        """Image-token positions must contain the corresponding vision embeddings."""
        B, S, D = 2, 16, 64
        # Sample 0: three consecutive image tokens at positions 2-4
        input_ids, text_emb, vis_emb = _make_inputs(B, S, D, {0: [2, 3, 4]})

        output = _scatter(_make_mock(False), input_ids, text_emb, vis_emb)

        assert output.shape == (S, B, D)
        assert torch.allclose(output[2, 0], vis_emb[0])
        assert torch.allclose(output[3, 0], vis_emb[1])
        assert torch.allclose(output[4, 0], vis_emb[2])

    def test_non_image_positions_unchanged(self):
        """Non-image positions must retain the original text embeddings."""
        B, S, D = 2, 16, 64
        input_ids, text_emb, vis_emb = _make_inputs(B, S, D, {0: [5]})

        output = _scatter(_make_mock(False), input_ids, text_emb, vis_emb)

        # Positions 0 and 1 in sample 0 are plain text — should be untouched.
        assert torch.allclose(output[0, 0], text_emb[0, 0])
        assert torch.allclose(output[1, 0], text_emb[1, 0])

    def test_no_image_tokens_is_identity(self):
        """With zero image tokens the output must equal the input exactly."""
        B, S, D = 2, 16, 64
        input_ids, text_emb, _ = _make_inputs(B, S, D, {})
        vis_emb = torch.empty(0, D, device='cuda')

        output = _scatter(_make_mock(False), input_ids, text_emb, vis_emb)

        assert torch.allclose(output, text_emb)

    def test_multi_sample_row_major_order(self):
        """masked_scatter fills in row-major [B, S] order: sample 0 then sample 1."""
        B, S, D = 2, 16, 64
        # 2 image tokens per sample → 4 vision embeddings total
        image_positions = {0: [1, 5], 1: [3, 7]}
        input_ids, text_emb, vis_emb = _make_inputs(B, S, D, image_positions)

        output = _scatter(_make_mock(False), input_ids, text_emb, vis_emb)

        # Row-major fill: sample 0 pos 1, sample 0 pos 5, sample 1 pos 3, sample 1 pos 7
        assert torch.allclose(output[1, 0], vis_emb[0])
        assert torch.allclose(output[5, 0], vis_emb[1])
        assert torch.allclose(output[3, 1], vis_emb[2])
        assert torch.allclose(output[7, 1], vis_emb[3])

    def test_output_shape(self):
        """Output shape is [S, B, D] regardless of image token count."""
        for n_img in [0, 1, 8]:
            B, S, D = 3, 32, 128
            positions = {0: list(range(n_img))} if n_img > 0 else {}
            input_ids, text_emb, vis_emb = _make_inputs(B, S, D, positions)
            output = _scatter(_make_mock(False), input_ids, text_emb, vis_emb)
            assert output.shape == (S, B, D), f"shape mismatch for n_img={n_img}"


# ---------------------------------------------------------------------------
# TP=2, SP=True  —  the bug-fix case
# ---------------------------------------------------------------------------

class TestScatterSP:
    """SP+TP=2 tests that verify the gather→scatter fix.

    Each test skips automatically when fewer than 2 processes are available
    (i.e. when running with torchrun --nproc_per_node 1).
    """

    @pytest.fixture(autouse=True)
    def _setup_tp2(self):
        if _world_size() < 2:
            pytest.skip("SP+TP=2 tests require torchrun --nproc_per_node 2")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    # ------------------------------------------------------------------
    # Internal helpers (use _world_size() == 2 at this point)
    # ------------------------------------------------------------------

    @staticmethod
    def _local_text_emb(text_emb_full, tp_rank, chunk):
        """Simulate what embedding() returns under SP: the rank-local slice."""
        return text_emb_full[tp_rank * chunk:(tp_rank + 1) * chunk]

    @staticmethod
    def _gather_output(output_local):
        """Gather [S/TP, B, D] from all ranks to [S, B, D] for comparison."""
        from megatron.core import tensor_parallel
        return tensor_parallel.gather_from_sequence_parallel_region(
            output_local, tensor_parallel_output_grad=False
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_sp_matches_no_sp_reference(self):
        """SP+TP=2 result, gathered across ranks, must equal the no-SP reference."""
        from megatron.core import parallel_state
        B, S, D = 2, 16, 64
        assert S % 2 == 0
        image_positions = {0: [2, 3, 4], 1: [1, 8]}
        input_ids, text_emb, vis_emb = _make_inputs(B, S, D, image_positions)

        expected = _reference_scatter(input_ids, text_emb, vis_emb)  # [S, B, D]

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        chunk = S // 2
        text_emb_local = self._local_text_emb(text_emb, tp_rank, chunk)  # [S/TP, B, D]

        output_local = _scatter(_make_mock(True), input_ids, text_emb_local, vis_emb)
        output_full = self._gather_output(output_local)  # [S, B, D]

        assert torch.allclose(output_full, expected, atol=1e-5), (
            f"rank {tp_rank}: max diff = {(output_full - expected).abs().max():.2e}"
        )

    def test_sp_output_shape_is_local_chunk(self):
        """With SP=True the output shape is [S/TP, B, D], not [S, B, D]."""
        from megatron.core import parallel_state
        B, S, D = 2, 16, 64
        image_positions = {0: [3], 1: [11]}
        input_ids, text_emb, vis_emb = _make_inputs(B, S, D, image_positions)

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        chunk = S // 2
        text_emb_local = self._local_text_emb(text_emb, tp_rank, chunk)

        output_local = _scatter(_make_mock(True), input_ids, text_emb_local, vis_emb)

        assert output_local.shape == (chunk, B, D), (
            f"expected ({chunk}, {B}, {D}), got {output_local.shape}"
        )

    def test_sp_image_tokens_straddle_chunk_boundary(self):
        """Image tokens split across the SP chunk boundary must be placed correctly.

        Positions 6 and 7 land in rank-0's chunk [0:8]; positions 8 and 9 land
        in rank-1's chunk [8:16].  After gathering, the result must match the
        no-SP reference exactly.
        """
        from megatron.core import parallel_state
        B, S, D = 1, 16, 64
        chunk = S // 2  # = 8
        # Two tokens on each side of the boundary
        image_positions = {0: [6, 7, 8, 9]}
        input_ids, text_emb, vis_emb = _make_inputs(B, S, D, image_positions)

        expected = _reference_scatter(input_ids, text_emb, vis_emb)

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        text_emb_local = self._local_text_emb(text_emb, tp_rank, chunk)

        output_local = _scatter(_make_mock(True), input_ids, text_emb_local, vis_emb)
        output_full = self._gather_output(output_local)

        assert torch.allclose(output_full, expected, atol=1e-5), (
            f"rank {tp_rank}: max diff = {(output_full - expected).abs().max():.2e}"
        )

    def test_sp_no_image_tokens(self):
        """SP+TP=2 with no image tokens: gathered output must equal the input."""
        from megatron.core import parallel_state
        B, S, D = 2, 16, 64
        input_ids, text_emb, _ = _make_inputs(B, S, D, {})
        vis_emb = torch.empty(0, D, device='cuda')

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        chunk = S // 2
        text_emb_local = self._local_text_emb(text_emb, tp_rank, chunk)

        output_local = _scatter(_make_mock(True), input_ids, text_emb_local, vis_emb)
        output_full = self._gather_output(output_local)

        assert torch.allclose(output_full, text_emb, atol=1e-5)

    def test_sp_all_tokens_are_image_tokens(self):
        """Edge case: every position is an image token."""
        from megatron.core import parallel_state
        B, S, D = 1, 8, 32
        assert S % 2 == 0
        # All S positions in sample 0 are image tokens
        image_positions = {0: list(range(S))}
        input_ids, text_emb, vis_emb = _make_inputs(B, S, D, image_positions)

        expected = _reference_scatter(input_ids, text_emb, vis_emb)

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        chunk = S // 2
        text_emb_local = self._local_text_emb(text_emb, tp_rank, chunk)

        output_local = _scatter(_make_mock(True), input_ids, text_emb_local, vis_emb)
        output_full = self._gather_output(output_local)

        assert torch.allclose(output_full, expected, atol=1e-5), (
            f"rank {tp_rank}: max diff = {(output_full - expected).abs().max():.2e}"
        )

    def test_sp_gradient_flows_through_scatter(self):
        """Backward pass: gradients must reach both text_emb_local and vis_emb."""
        from megatron.core import parallel_state
        B, S, D = 2, 16, 64
        image_positions = {0: [3], 1: [10]}
        input_ids, text_emb_full, vis_emb_base = _make_inputs(B, S, D, image_positions)

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        chunk = S // 2

        text_emb_local = (
            self._local_text_emb(text_emb_full, tp_rank, chunk)
            .detach()
            .requires_grad_(True)
        )
        vis_emb = vis_emb_base.detach().requires_grad_(True)

        output_local = _scatter(_make_mock(True), input_ids, text_emb_local, vis_emb)
        output_local.sum().backward()

        assert text_emb_local.grad is not None, "no gradient reached text_emb_local"
        assert vis_emb.grad is not None, "no gradient reached vis_emb"

        # text_emb_local gradient should be 1 at non-image positions and 0 at image positions
        # (masked_scatter zero-fills image positions from vis_emb, so text grad is blocked there)
        local_start = tp_rank * chunk
        local_image_mask = (
            input_ids[:, local_start:local_start + chunk] == IMAGE_TOKEN_ID
        ).T.unsqueeze(-1).expand_as(text_emb_local)  # [S/TP, B, D]

        # Non-image positions: grad should be 1 (identity through transpose+scatter)
        non_image_grad = text_emb_local.grad[~local_image_mask]
        assert torch.allclose(non_image_grad, torch.ones_like(non_image_grad)), (
            "gradient at non-image text positions should be 1"
        )
