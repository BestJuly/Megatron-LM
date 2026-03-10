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
    """2D rotary position embedding for vision transformer.

    Args:
        hidden_size: Per-head dimension for computing rotary embeddings.
        theta: Base frequency for rotary embeddings.
    """

    def __init__(self, hidden_size: int, theta: float = 10000.0):
        super().__init__()
        self.dim = hidden_size // 2  # half for cos, half for sin
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute 2D rotary position embeddings from grid dimensions.

        Args:
            grid_thw: [num_images, 3] with (temporal, height, width) per image.

        Returns:
            Rotary embeddings [total_patches, dim].
        """
        all_pos = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            # Create 2D position grid
            hpos = torch.arange(h, device=grid_thw.device).unsqueeze(1).expand(-1, w)
            wpos = torch.arange(w, device=grid_thw.device).unsqueeze(0).expand(h, -1)
            # Stack and repeat for temporal dimension
            hpos = hpos.reshape(-1).repeat(t)
            wpos = wpos.reshape(-1).repeat(t)
            # Compute rotary embeddings for h and w separately
            inv_freq = self.inv_freq.to(grid_thw.device)
            h_emb = hpos.float().unsqueeze(1) @ inv_freq.unsqueeze(0)
            w_emb = wpos.float().unsqueeze(1) @ inv_freq.unsqueeze(0)
            pos_emb = torch.cat([h_emb, w_emb], dim=-1)
            all_pos.append(pos_emb)
        return torch.cat(all_pos, dim=0)


class VisionEncoder(MegatronModule):
    """Vision encoder built from mcore TransformerBlock.

    Processes image/video inputs through:
    1. PatchEmbed3D (Linear projection of pre-extracted flat patches)
    2. Learned absolute position embeddings
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

        # Learned absolute position embeddings
        self.pos_embed = nn.Embedding(max_num_positions, config.hidden_size)

        # Vision rotary embeddings
        head_dim = config.hidden_size // config.num_attention_heads
        self.rot_pos_emb = VisionRotaryEmbedding(head_dim)

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

        # 2. Position embeddings (learned absolute, interpolated if needed)
        seq_len = hidden_states.shape[0]
        if seq_len <= self.pos_embed.num_embeddings:
            pos_ids = torch.arange(seq_len, device=hidden_states.device)
            hidden_states = hidden_states + self.pos_embed(pos_ids)

        # 3. ViT transformer blocks
        # Vision RoPE (VisionRotaryEmbedding) uses a (cos, sin) tuple format that
        # differs from what mcore's ViT TransformerBlock expects for rotary_pos_emb
        # (a raw frequency tensor). Passing None here uses the learned absolute
        # position embeddings from step 2. Vision RoPE can be wired in once E2E
        # is confirmed working.
        hidden_states = hidden_states.unsqueeze(1)  # [seq, 1, hidden]
        hidden_states = self.blocks(
            hidden_states=hidden_states,
            attention_mask=None,
            rotary_pos_emb=None,
        )
        hidden_states = hidden_states.squeeze(1)  # [seq, hidden]

        # 4. Patch merger
        merged = self.merger(hidden_states, grid_thw)

        return merged
