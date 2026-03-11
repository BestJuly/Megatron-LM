# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Reusable vision encoder components built from mcore primitives.

Provides PatchEmbed3D, PatchMerger, and VisionEncoder that can be shared
across multimodal models (Qwen3.5-VL, etc.).
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


class PatchEmbed3D(nn.Module):
    """Patch embedding for pre-extracted patches (matches HF Qwen3.5-VL format).

    The HF processor (and our mock data) produces pixel_values as pre-extracted
    flat patches of shape [total_patches, C * temporal_patch_size * patch_h * patch_w].
    This is mathematically identical to Conv3d on the raw image, but the input
    is already flattened so we use nn.Linear (same weight structure, different input
    layout).

    Args:
        in_channels: Number of input channels (3 for RGB).
        hidden_size: Output embedding dimension.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, hidden_size, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: [total_patches, C * temporal_patch_size * patch_h * patch_w]

        Returns:
            Patch embeddings [total_patches, hidden_size].
        """
        return self.proj(pixel_values)


class PatchMerger(nn.Module):
    """Spatial patch merger that reduces the number of visual tokens.

    Merges spatial_merge_size^2 adjacent patches into one token.

    Args:
        hidden_size: Input hidden size from ViT.
        out_hidden_size: Output hidden size (should match language model hidden_size).
        spatial_merge_size: Number of patches to merge per spatial dimension.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        out_hidden_size: int = 3584,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.ln = nn.LayerNorm(merge_dim)
        self.mlp = nn.Sequential(
            nn.Linear(merge_dim, out_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(out_hidden_size, out_hidden_size, bias=True),
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Merge patches spatially.

        Args:
            hidden_states: [total_patches, hidden_size] from ViT blocks.
            grid_thw: [num_images, 3] with (temporal, height, width) per image.

        Returns:
            Merged embeddings [total_merged_patches, out_hidden_size].
        """
        merge = self.spatial_merge_size
        outputs = []
        offset = 0
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            num_patches = t * h * w
            # [t*h*w, D] -> [t, h, w, D]
            patches = hidden_states[offset : offset + num_patches].view(t, h, w, -1)
            # Merge spatial dims: [t, h//m, m, w//m, m, D] -> [t, h//m, w//m, m*m*D]
            h_m, w_m = h // merge, w // merge
            patches = patches.view(t, h_m, merge, w_m, merge, -1)
            patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(
                t * h_m * w_m, -1
            )
            outputs.append(patches)
            offset += num_patches

        merged = torch.cat(outputs, dim=0)
        return self.mlp(self.ln(merged))


class VisionRotaryEmbedding(nn.Module):
    """1D rotary position frequency table for the vision transformer.

    Generates a lookup table of RoPE frequencies for integer positions 0..seqlen-1.
    Use VisionEncoder._compute_rotary_pos_emb to map per-patch 2D (row, col)
    positions to embeddings via table lookup.

    Matches HF Qwen3_5VisionRotaryEmbedding exactly.

    Args:
        dim: Frequency dimension (= head_dim // 2).
        theta: RoPE base frequency.
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """Return a frequency lookup table for positions 0..seqlen-1.

        Args:
            seqlen: Number of positions (typically max(height, width) across all images).

        Returns:
            freqs: [seqlen, dim // 2]
        """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)  # [seqlen, dim // 2]


class VisionEncoder(MegatronModule):
    """Vision encoder built from mcore TransformerBlock.

    Processes image/video inputs through:
    1. PatchEmbed3D (Linear projection of pre-extracted flat patches)
    2. 2D Vision RoPE from (row, col) patch positions
    3. TransformerBlock (N layers of ViT)
    4. PatchMerger (spatial merge to reduce tokens)

    Output dimension matches the language model hidden_size.

    Args:
        config: TransformerConfig for the vision transformer.
        transformer_layer_spec: ModuleSpec for vision transformer layers.
        in_channels: Number of input channels.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        spatial_merge_size: Spatial merge factor in PatchMerger.
        out_hidden_size: Output hidden size (language model hidden_size).
        max_num_positions: Maximum number of position embeddings.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec = None,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 3584,
        max_num_positions: int = 2304,
    ):
        super().__init__(config=config)

        self.hidden_size = config.hidden_size
        self.spatial_merge_size = spatial_merge_size

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            hidden_size=config.hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )

        # Vision rotary embeddings: freq table for positions 0..max_hw-1.
        # dim = head_dim // 2 so that row + col frequencies combine to head_dim.
        head_dim = config.hidden_size // config.num_attention_heads
        self.rot_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        # Transformer blocks
        if transformer_layer_spec is None:
            transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

        self.blocks = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )

        # Patch merger
        self.merger = PatchMerger(
            hidden_size=config.hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    def _compute_rotary_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute 2D Vision RoPE frequencies for all patches.

        Patches are ordered in block-merge order so that spatial_merge_size ×
        spatial_merge_size blocks of patches are grouped together, matching the
        layout expected by PatchMerger and the HF processor output.

        Matches HF Qwen3_5VisionModel.rot_pos_emb exactly.

        Args:
            grid_thw: [num_images, 3] with (temporal, height, width) per image.

        Returns:
            rot_freqs: [total_patches, head_dim] raw frequencies for mcore RoPE.
                Pass as rotary_pos_emb after unsqueezing to [seq, 1, 1, head_dim].
        """
        merge = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rot_pos_emb(max_hw)  # [max_hw, head_dim // 4]

        pos_ids_list = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            device = freq_table.device
            # Block-merge ordered row indices: merge×merge blocks are grouped
            hpos = (
                torch.arange(h, device=device)
                .unsqueeze(1)
                .expand(h, w)
                .reshape(h * w)
                .reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            # Block-merge ordered col indices
            wpos = (
                torch.arange(w, device=device)
                .unsqueeze(0)
                .expand(h, w)
                .reshape(h * w)
                .reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            # Stack (row, col) and repeat for temporal frames
            pos_ids = torch.stack([hpos, wpos], dim=-1).repeat(t, 1)  # [t*h*w, 2]
            pos_ids_list.append(pos_ids)

        pos_ids = torch.cat(pos_ids_list, dim=0)  # [total_patches, 2]
        embeddings = freq_table[pos_ids]           # [total_patches, 2, head_dim // 4]
        embeddings = embeddings.flatten(1)         # [total_patches, head_dim // 2]
        return torch.cat((embeddings, embeddings), dim=-1)  # [total_patches, head_dim]

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: Preprocessed pixel values. Shape depends on preprocessing,
                typically [total_patches, C * temporal_patch * patch_h * patch_w].
            grid_thw: [num_images, 3] with (temporal, height, width) in patch grid units.

        Returns:
            Visual embeddings [total_merged_patches, out_hidden_size].
        """
        # 1. Patch embedding
        hidden_states = self.patch_embed(pixel_values)

        # 2. Compute 2D Vision RoPE frequencies for all patches
        rot_freqs = self._compute_rotary_pos_emb(grid_thw)  # [seq, head_dim]
        rot_freqs = rot_freqs.unsqueeze(1).unsqueeze(1)      # [seq, 1, 1, head_dim]

        # 3. ViT transformer blocks with Vision RoPE
        hidden_states = hidden_states.unsqueeze(1)           # [seq, 1, hidden]
        hidden_states = self.blocks(
            hidden_states=hidden_states,
            attention_mask=None,
            rotary_pos_emb=rot_freqs,
        )
        hidden_states = hidden_states.squeeze(1)             # [seq, hidden]

        # 4. Patch merger
        merged = self.merger(hidden_states, grid_thw)

        return merged
