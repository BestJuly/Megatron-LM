# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for compute_mrope_position_ids against HF get_rope_index reference.

Verifies that our MRoPE position ID computation matches HF Qwen3VL.get_rope_index
for the relevant cases: single image, multi-image, multi-sample, text-only.

Run (from Megatron-LM root):
    python -m pytest multimodal/tests/test_mrope_position_ids.py -v
"""

import pytest
import torch

from multimodal.models.qwen35_vl import compute_mrope_position_ids

IMAGE_TOKEN = 151655  # Qwen3.5-VL image token id


# ---------------------------------------------------------------------------
# HF reference: simplified get_rope_index for image-only pipeline (no video,
# no attention mask, images ordered sample-0-first then sample-1, etc.)
# Matches HF Qwen3VLForConditionalGeneration.get_rope_index exactly.
# ---------------------------------------------------------------------------

def _hf_get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """HF reference implementation of get_rope_index.

    Returns position_ids [3, B, S] matching HF's output.
    """
    B, S = input_ids.shape
    position_ids = torch.zeros(3, B, S, dtype=torch.long)
    grid_iter = iter(image_grid_thw)  # global iterator, same as HF

    for b in range(B):
        current_pos = 0
        seq = input_ids[b].tolist()
        pos_t, pos_h, pos_w = [], [], []

        i = 0
        while i < S:
            if seq[i] != image_token_id:
                pos_t.append(current_pos)
                pos_h.append(current_pos)
                pos_w.append(current_pos)
                current_pos += 1
                i += 1
            else:
                thw = next(grid_iter)
                t = int(thw[0])
                h = int(thw[1]) // spatial_merge_size
                w = int(thw[2]) // spatial_merge_size
                n = t * h * w
                # temporal: all current_pos
                pos_t.extend([current_pos] * n)
                # height: arange(h).repeat_interleave(w * t)
                pos_h.extend([current_pos + r for r in range(h) for _ in range(w * t)])
                # width: arange(w).repeat(h * t)
                pos_w.extend(list(range(current_pos, current_pos + w)) * (h * t))
                current_pos += max(h, w)
                i += n

        for s_idx in range(S):
            position_ids[0, b, s_idx] = pos_t[s_idx]
            position_ids[1, b, s_idx] = pos_h[s_idx]
            position_ids[2, b, s_idx] = pos_w[s_idx]

    return position_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeMropePositionIds:
    """Tests for compute_mrope_position_ids against HF reference."""

    def _check(self, input_ids, image_grid_thw, merge=2):
        """Helper: compare our output to HF reference."""
        our = compute_mrope_position_ids(
            input_ids, image_grid_thw, IMAGE_TOKEN, merge
        )
        ref = _hf_get_rope_index(input_ids, image_grid_thw, IMAGE_TOKEN, merge)
        assert our.shape == (3,) + input_ids.shape, f"shape {our.shape}"
        assert torch.allclose(our, ref), (
            f"Mismatch\nOurs: {our}\nRef:  {ref}\nDiff: {our - ref}"
        )

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Output is [3, B, S]."""
        B, S = 2, 10
        ids = torch.zeros(B, S, dtype=torch.long)
        out = compute_mrope_position_ids(ids, None, IMAGE_TOKEN)
        assert out.shape == (3, B, S)

    def test_text_only_sequential(self):
        """Pure text: all 3 dims are sequential positions."""
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        out = compute_mrope_position_ids(ids, None, IMAGE_TOKEN)
        expected = torch.arange(5).unsqueeze(0).expand(3, 1, -1)
        assert torch.allclose(out, expected)

    def test_no_image_grid_returns_sequential(self):
        """Empty image_grid_thw is treated as text-only."""
        ids = torch.tensor([[1, IMAGE_TOKEN, IMAGE_TOKEN, 2]])
        out = compute_mrope_position_ids(ids, torch.empty(0, 3, dtype=torch.long), IMAGE_TOKEN)
        # Falls back to sequential (no image info)
        assert out.shape == (3, 1, 4)

    # ------------------------------------------------------------------
    # Single image (B=1)
    # ------------------------------------------------------------------

    def test_b1_single_image_matches_hf(self):
        """B=1 with one 4x4 image (merge=2 -> 2x2 = 4 tokens) matches HF."""
        ids = torch.tensor([[1, 2, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 3]])
        thw = torch.tensor([[1, 4, 4]])
        self._check(ids, thw)

    def test_b1_image_temporal_is_start_position(self):
        """All image tokens share the same temporal position (= text pos before image)."""
        # text_pos=2 when image starts
        ids = torch.tensor([[1, 2, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 3]])
        thw = torch.tensor([[1, 4, 4]])
        out = compute_mrope_position_ids(ids, thw, IMAGE_TOKEN)
        # positions 2-5 are image tokens; temporal should all be 2
        assert out[0, 0, 2:6].unique().numel() == 1, "temporal should be constant over image"
        assert out[0, 0, 2].item() == 2

    def test_b1_image_height_row_major(self):
        """Image height positions follow row-major order (0,0,...,1,1,...) from start_pos."""
        ids = torch.tensor([[IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN]])
        thw = torch.tensor([[1, 4, 4]])  # 2x2 merged
        out = compute_mrope_position_ids(ids, thw, IMAGE_TOKEN)
        # h: [0, 0, 1, 1] (each row repeated w=2 times)
        assert out[1, 0].tolist() == [0, 0, 1, 1]

    def test_b1_image_width_cycles(self):
        """Image width positions cycle 0..w-1 per row."""
        ids = torch.tensor([[IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN]])
        thw = torch.tensor([[1, 4, 4]])  # 2x2 merged
        out = compute_mrope_position_ids(ids, thw, IMAGE_TOKEN)
        # w: [0, 1, 0, 1]
        assert out[2, 0].tolist() == [0, 1, 0, 1]

    def test_b1_text_after_image_position_offset(self):
        """Text after image continues from start_pos + max(h, w)."""
        ids = torch.tensor([[IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 99]])
        thw = torch.tensor([[1, 4, 4]])  # h_merged=2, w_merged=2, max=2
        out = compute_mrope_position_ids(ids, thw, IMAGE_TOKEN)
        # After the 4 image tokens, next text position = 0 + max(2, 2) = 2
        assert out[0, 0, 4].item() == 2
        assert out[1, 0, 4].item() == 2
        assert out[2, 0, 4].item() == 2

    def test_b1_rectangular_image(self):
        """Rectangular images (h != w) match HF."""
        # 6 image tokens: h=4/2=2, w=6/2=3 → 6 tokens
        ids = torch.tensor([[1, IMAGE_TOKEN] * 3 + [IMAGE_TOKEN] * 3 + [99]])
        # Simpler: straight 6 image tokens at start
        ids = torch.tensor([[IMAGE_TOKEN] * 6 + [1]])
        thw = torch.tensor([[1, 4, 6]])  # h_merged=2, w_merged=3
        self._check(ids, thw)

    def test_b1_two_images(self):
        """Two images in the same sequence, with text between them."""
        # [img*4, txt*2, img*4, txt*1]
        img4 = [IMAGE_TOKEN] * 4
        ids = torch.tensor([img4 + [1, 2] + img4 + [3]])
        thw = torch.tensor([[1, 4, 4], [1, 4, 4]])
        self._check(ids, thw)

    # ------------------------------------------------------------------
    # Multi-sample (B > 1) — tests the img_idx global counter fix
    # ------------------------------------------------------------------

    def test_b2_same_grids_matches_hf(self):
        """B=2 with identical grids: both samples get correct positions."""
        ids = torch.tensor([
            [IMAGE_TOKEN] * 4 + [1, 2],
            [IMAGE_TOKEN] * 4 + [3, 4],
        ])
        thw = torch.tensor([[1, 4, 4], [1, 4, 4]])
        self._check(ids, thw)

    def test_b2_different_grids_matches_hf(self):
        """B=2 with different grids: sample 1 uses its own grid (not sample 0's).

        This is the key regression test for the img_idx global counter fix.
        Both samples have 4 image tokens but with different spatial layouts:
          sample 0: grid [1, 4, 4] / merge=2 → h=2, w=2 → 4 tokens; w_max=1
          sample 1: grid [1, 2, 8] / merge=2 → h=1, w=4 → 4 tokens; w_max=3

        Width positions for sample 1 should reach 3, not 1 (which the old
        img_idx-reset bug would produce).
        """
        ids = torch.tensor([
            [IMAGE_TOKEN] * 4 + [1, 2, 3, 4],
            [IMAGE_TOKEN] * 4 + [5, 6, 7, 8],
        ])
        thw = torch.tensor([[1, 4, 4], [1, 2, 8]])  # both yield 4 image tokens
        self._check(ids, thw)

    def test_b2_different_num_images(self):
        """B=2 where sample 0 has 0 images and sample 1 has 1 image."""
        ids = torch.tensor([
            [1, 2, 3, 4, 5, 6],
            [IMAGE_TOKEN] * 4 + [1, 2],
        ])
        thw = torch.tensor([[1, 4, 4]])  # only sample 1's image
        self._check(ids, thw)

    def test_b2_first_sample_no_image_second_has_image(self):
        """img_idx stays at 0 until the first real image is encountered."""
        ids = torch.tensor([
            [10, 11, 12, 13, 14],
            [IMAGE_TOKEN] * 4 + [15],
        ])
        thw = torch.tensor([[1, 4, 4]])
        self._check(ids, thw)

    def test_b3_three_samples_three_grids(self):
        """B=3 each with a different image grid, global iterator advances correctly.

        Each sample has 4 image tokens with different layouts:
          sample 0: [1, 4, 4] / merge=2 → h=2, w=2 (4 tokens)
          sample 1: [1, 2, 8] / merge=2 → h=1, w=4 (4 tokens)
          sample 2: [1, 8, 2] / merge=2 → h=4, w=1 (4 tokens)
        """
        ids = torch.tensor([
            [IMAGE_TOKEN] * 4 + [1, 2],
            [IMAGE_TOKEN] * 4 + [3, 4],
            [IMAGE_TOKEN] * 4 + [5, 6],
        ])
        thw = torch.tensor([[1, 4, 4], [1, 2, 8], [1, 8, 2]])
        self._check(ids, thw)
