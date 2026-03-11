# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for VisionRotaryEmbedding and VisionEncoder._compute_rotary_pos_emb.

Verifies that our Vision RoPE implementation matches the HF reference
(Qwen3_5VisionRotaryEmbedding and Qwen3_5VisionModel.rot_pos_emb).

Run (from Megatron-LM root):
    python -m pytest multimodal/tests/test_vision_rotary_embedding.py -v
"""

import types

import pytest
import torch

from multimodal.models.vision import VisionEncoder, VisionRotaryEmbedding


# ---------------------------------------------------------------------------
# HF reference implementation (inlined to avoid dependency on transformers)
# Matches Qwen3_5VisionRotaryEmbedding + Qwen3_5VisionModel.rot_pos_emb exactly.
# ---------------------------------------------------------------------------

class _HFVisionRotaryEmbedding:
    """Reference: HF Qwen3_5VisionRotaryEmbedding."""

    def __init__(self, dim: int, theta: float = 10000.0):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)  # [seqlen, dim // 2]


def _hf_rot_pos_emb(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    head_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Reference: HF Qwen3_5VisionModel.rot_pos_emb.

    Returns [total_patches, head_dim] raw frequency tensor.
    """
    rotary_emb = _HFVisionRotaryEmbedding(dim=head_dim // 2, theta=theta)
    merge = spatial_merge_size

    pos_ids_list = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos = (
            torch.arange(h).unsqueeze(1).expand(h, w).reshape(h * w)
            .reshape(h // merge, merge, w // merge, merge)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        wpos = (
            torch.arange(w).unsqueeze(0).expand(h, w).reshape(h * w)
            .reshape(h // merge, merge, w // merge, merge)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        pos_ids_list.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids_list, dim=0)       # [total, 2]
    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = rotary_emb(max_hw)                 # [max_hw, head_dim // 4]
    embeddings = freq_table[pos_ids]                # [total, 2, head_dim // 4]
    embeddings = embeddings.flatten(1)              # [total, head_dim // 2]
    return torch.cat((embeddings, embeddings), dim=-1)  # [total, head_dim]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HIDDEN_SIZE = 1152
_NUM_HEADS = 16
_HEAD_DIM = _HIDDEN_SIZE // _NUM_HEADS  # = 72
_MERGE = 2


def _make_stub(spatial_merge_size: int = _MERGE, head_dim: int = _HEAD_DIM):
    """Minimal namespace for calling _compute_rotary_pos_emb as unbound method."""
    stub = types.SimpleNamespace()
    stub.spatial_merge_size = spatial_merge_size
    stub.rot_pos_emb = VisionRotaryEmbedding(dim=head_dim // 2)
    return stub


def _compute(stub, grid_thw):
    return VisionEncoder._compute_rotary_pos_emb(stub, grid_thw)


# ---------------------------------------------------------------------------
# VisionRotaryEmbedding tests
# ---------------------------------------------------------------------------

class TestVisionRotaryEmbedding:
    """1D frequency table: shape and value tests."""

    def test_output_shape(self):
        dim = 36  # head_dim // 2 for the vision encoder
        emb = VisionRotaryEmbedding(dim=dim)
        seqlen = 14
        out = emb(seqlen)
        assert out.shape == (seqlen, dim // 2), f"expected ({seqlen}, {dim // 2}), got {out.shape}"

    def test_matches_hf_values(self):
        """Frequency table values must match HF reference."""
        dim = 36
        seqlen = 20
        our_emb = VisionRotaryEmbedding(dim=dim)
        hf_emb = _HFVisionRotaryEmbedding(dim=dim)
        assert torch.allclose(our_emb(seqlen), hf_emb(seqlen), atol=1e-6)

    def test_position_zero_is_zero(self):
        """Position 0 maps to frequency 0 for all dims."""
        dim = 18
        emb = VisionRotaryEmbedding(dim=dim)
        out = emb(10)
        assert torch.allclose(out[0], torch.zeros(dim // 2))

    def test_monotonically_nondecreasing(self):
        """Frequencies increase (or stay equal) with position for each frequency dim."""
        dim = 18
        emb = VisionRotaryEmbedding(dim=dim)
        out = emb(10)
        diffs = out[1:] - out[:-1]
        assert (diffs >= -1e-6).all(), "frequencies should be non-decreasing with position"


# ---------------------------------------------------------------------------
# _compute_rotary_pos_emb tests
# ---------------------------------------------------------------------------

class TestComputeRotaryPosEmb:
    """2D Vision RoPE: shape and value tests against HF reference."""

    def test_output_shape_single_image(self):
        """Output shape is [t*h*w, head_dim] for a single image."""
        stub = _make_stub()
        grid_thw = torch.tensor([[1, 4, 4]])
        out = _compute(stub, grid_thw)
        assert out.shape == (1 * 4 * 4, _HEAD_DIM), f"got {out.shape}"

    def test_output_shape_multi_image(self):
        """Output shape is [total_patches, head_dim] for multiple images."""
        stub = _make_stub()
        grid_thw = torch.tensor([[1, 4, 4], [1, 6, 6]])
        out = _compute(stub, grid_thw)
        assert out.shape == (1 * 4 * 4 + 1 * 6 * 6, _HEAD_DIM)

    def test_output_shape_temporal(self):
        """Temporal frames multiply the patch count."""
        stub = _make_stub()
        grid_thw = torch.tensor([[3, 4, 4]])
        out = _compute(stub, grid_thw)
        assert out.shape == (3 * 4 * 4, _HEAD_DIM)

    def test_matches_hf_single_image(self):
        """Values must match HF rot_pos_emb for a single image."""
        stub = _make_stub()
        grid_thw = torch.tensor([[1, 4, 4]])
        our = _compute(stub, grid_thw)
        ref = _hf_rot_pos_emb(grid_thw, _MERGE, _HEAD_DIM)
        assert torch.allclose(our, ref, atol=1e-5), f"max diff: {(our - ref).abs().max():.2e}"

    def test_matches_hf_multi_image(self):
        """Values must match HF rot_pos_emb for multiple images."""
        stub = _make_stub()
        grid_thw = torch.tensor([[1, 4, 6], [2, 6, 4]])
        our = _compute(stub, grid_thw)
        ref = _hf_rot_pos_emb(grid_thw, _MERGE, _HEAD_DIM)
        assert torch.allclose(our, ref, atol=1e-5), f"max diff: {(our - ref).abs().max():.2e}"

    def test_matches_hf_rectangular(self):
        """Rectangular grids (h ≠ w) must match HF."""
        stub = _make_stub()
        grid_thw = torch.tensor([[1, 4, 8]])  # 4 rows, 8 cols
        our = _compute(stub, grid_thw)
        ref = _hf_rot_pos_emb(grid_thw, _MERGE, _HEAD_DIM)
        assert torch.allclose(our, ref, atol=1e-5), f"max diff: {(our - ref).abs().max():.2e}"

    def test_block_merge_ordering_differs_from_row_major(self):
        """Block-merge ordering must differ from simple row-major for h > merge_size.

        For a 4x4 grid with merge=2, simple row-major gives positions:
            (0,0),(0,1),(0,2),(0,3),(1,0),...
        Block-merge gives positions:
            (0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3),...
        (first 4 patches are all in the top-left 2x2 block).
        """
        stub = _make_stub()
        grid_thw = torch.tensor([[1, 4, 4]])
        out = _compute(stub, grid_thw)
        ref = _hf_rot_pos_emb(grid_thw, _MERGE, _HEAD_DIM)
        # Our output should match HF (block-merge), not row-major
        assert torch.allclose(out, ref, atol=1e-5)

        # Verify it's actually different from row-major by checking position 2:
        # Row-major: patch 2 → (row=0, col=2)  → different from block-merge (row=1, col=0)
        freq_table = stub.rot_pos_emb(4)  # [4, head_dim // 4]
        # Row 0, col 2 → row-major
        row_major_emb = freq_table[0]  # row=0 freq
        col_major_emb = freq_table[2]  # col=2 freq
        row_major_patch2 = torch.cat([row_major_emb, col_major_emb], dim=-1)
        row_major_patch2 = torch.cat([row_major_patch2, row_major_patch2], dim=-1)
        # Block-merge gives patch 2 → (row=1, col=0)
        assert not torch.allclose(out[2], row_major_patch2, atol=1e-5), (
            "Patch 2 should differ from row-major order"
        )

    def test_temporal_repeat_matches_hf(self):
        """Temporal repeat behavior matches HF reference."""
        stub = _make_stub()
        grid_thw_t2 = torch.tensor([[2, 4, 4]])
        grid_thw_t1 = torch.tensor([[1, 4, 4]])
        out_t2 = _compute(stub, grid_thw_t2)
        out_t1 = _compute(stub, grid_thw_t1)
        patches = 4 * 4
        # Both temporal frames have the same spatial positions
        assert torch.allclose(out_t2[:patches], out_t1, atol=1e-5)
        assert torch.allclose(out_t2[patches:], out_t1, atol=1e-5)

    def test_doubled_output_range(self):
        """Output is row-freq cat col-freq cat row-freq cat col-freq (doubled).

        The final cat((embeddings, embeddings)) means the first half of each
        vector equals the second half.
        """
        stub = _make_stub()
        grid_thw = torch.tensor([[1, 4, 4]])
        out = _compute(stub, grid_thw)
        half = _HEAD_DIM // 2
        assert torch.allclose(out[:, :half], out[:, half:], atol=1e-6), (
            "First half should equal second half (doubled frequencies)"
        )
