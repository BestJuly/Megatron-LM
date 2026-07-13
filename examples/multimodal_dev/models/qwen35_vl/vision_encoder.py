# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron-native Qwen3.5-VL vision encoder.

Architecture (matches HF ``Qwen3VLVisionModel`` exactly):

  PatchEmbed (Conv3d)
    -> learned position embedding (bilinear interpolation)
    -> 2D Vision RoPE
    -> TransformerBlock x N  (with PackedSeqParams / THD attention)
    -> PatchMerger  (per-token LN -> spatial merge -> MLP)

Key design choices:
  * ``Conv3d`` patch embedding is replicated across TP ranks (no MCore
    equivalent for 3D convolutions).
  * ``PatchMerger`` MLP uses ``ColumnParallelLinear`` / ``RowParallelLinear``
    for TP sharding.
  * Inherits from ``VisionModule``.
  * Expects pixel values in block-merge order (as produced by the HF
    processor) so the merger's simple reshape is correct.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from examples.multimodal_dev.models.qwen35_vl.specs import get_qwen35_vl_vision_spec
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from megatron.training.utils import get_nvtx_range


def _has_partial_storage(tensor: Tensor) -> bool:
    try:
        return tensor.untyped_storage().nbytes() < tensor.numel() * tensor.element_size()
    except RuntimeError:
        return False


def _using_megatron_fsdp() -> bool:
    return bool(getattr(get_args(), "use_megatron_fsdp", False))


class Qwen35VLPatchEmbed(MegatronModule):
    """3D convolution patch embedding matching HF ``Qwen3VLVisionPatchEmbed``.

    Uses ``nn.Conv3d`` with kernel = stride = ``[temporal_patch_size,
    patch_size, patch_size]`` and ``bias=True``.  The module is replicated
    across TP ranks (no MCore equivalent for 3D conv).

    Args:
        config: TransformerConfig (used by MegatronModule base).
        in_channels: Number of input channels (3 for RGB).
        hidden_size: Output embedding dimension.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
    ):
        super().__init__(config=config)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        kernel = [temporal_patch_size, patch_size, patch_size]
        self.proj = torch.nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Forward pass.

        Args:
            pixel_values: ``[total_patches, C * T * pH * pW]``
                pre-extracted flat patches.

        Returns:
            Patch embeddings ``[total_patches, hidden_size]``.
        """
        target_dtype = self.proj.weight.dtype
        if _using_megatron_fsdp() or _has_partial_storage(self.proj.weight):
            pixel_values = pixel_values.to(dtype=target_dtype).view(
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
            return self.proj(pixel_values).flatten(1)

        # The dataloader already materializes non-overlapping flattened
        # patches. The Conv3d kernel therefore degenerates to one linear
        # projection per patch; keeping the Conv3d module preserves checkpoint
        # compatibility while avoiding cuDNN convolution overhead here.
        return F.linear(
            pixel_values.to(dtype=target_dtype),
            self.proj.weight.flatten(1),
            self.proj.bias,
        )


# -------------------------------------------------------------------
# VisionRotaryEmbedding - 1D frequency table
# -------------------------------------------------------------------

class Qwen35VLVisionRotaryEmbedding(MegatronModule):
    """1D rotary position frequency table for the vision transformer.

    Generates RoPE frequencies for integer positions ``0 .. seqlen-1``.
    The encoder maps 2D (row, col) positions to embeddings via table
    lookup.  Matches HF ``Qwen3VLVisionRotaryEmbedding``.

    Args:
        dim: Frequency dimension (``head_dim // 2``).
        theta: RoPE base frequency.
        config: Optional TransformerConfig for MegatronModule base.
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        config: Optional[TransformerConfig] = None,
    ):
        super().__init__(config=config)
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_inv_freq(self, device: torch.device) -> Tensor:
        """Return ``inv_freq`` in float32 on *device*.

        Always recomputes in float32 regardless of the buffer's stored dtype.
        This matches Bridge's lazy-init behaviour where ``inv_freq`` is
        constructed fresh (in float32) on the first forward call, after any
        ``model.bfloat16()`` cast has already occurred.
        """
        return 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0, self.dim, 2,
                    dtype=torch.float32, device=device,
                )
                / self.dim
            )
        )

    def forward(
        self,
        seqlen: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Frequency lookup table for positions ``0 .. seqlen-1``.

        Args:
            seqlen: Number of positions.
            device: Runtime device (required for meta-init safety).

        Returns:
            ``[seqlen, dim // 2]`` frequencies.
        """
        if device is None:
            if self.inv_freq.device.type != "meta":
                device = self.inv_freq.device
            else:
                device = torch.device(
                    "cuda", torch.cuda.current_device()
                )
        inv_freq = self._get_inv_freq(device)
        seq = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
        return torch.outer(seq, inv_freq)


# -------------------------------------------------------------------
# PatchMerger - per-token LN, spatial merge, TP-sharded MLP
# -------------------------------------------------------------------

class Qwen35VLPatchMerger(MegatronModule):
    """Spatial patch merger matching HF ``Qwen3VLVisionPatchMerger``.

    Per-token ``LayerNorm`` on ``hidden_size`` -> reshape to merge
    ``spatial_merge_size ** 2`` adjacent patches -> two-layer MLP
    (``ColumnParallelLinear`` -> GELU -> ``RowParallelLinear``).

    MLP dimensions: ``merge_dim -> merge_dim -> out_hidden_size``
    where ``merge_dim = hidden_size * spatial_merge_size ** 2``.

    Args:
        config: TransformerConfig (provides TP settings, init_method).
        hidden_size: Per-token hidden size from the ViT.
        out_hidden_size: Output dimension (language model hidden_size).
        spatial_merge_size: Merge factor per spatial dimension.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int = 1152,
        out_hidden_size: int = 3584,
        spatial_merge_size: int = 2,
    ):
        super().__init__(config=config)
        self.spatial_merge_size = spatial_merge_size
        self.merge_dim = hidden_size * (spatial_merge_size ** 2)
        merge_dim = self.merge_dim

        self.patch_norm = TENorm(config=config, hidden_size=hidden_size, eps=1e-6)
        self.linear_fc1 = build_module(
            ColumnParallelLinear,
            merge_dim,
            merge_dim,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
        )
        self.linear_fc2 = build_module(
            RowParallelLinear,
            merge_dim,
            out_hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Merge patches spatially.

        Args:
            hidden_states: ``[total_patches, hidden_size]`` in block-merge
                order from the ViT transformer blocks.

        Returns:
            ``[total_merged_patches, out_hidden_size]``.
        """
        with get_nvtx_range()("PatchMerger/patch_norm"):
            hidden_states = self.patch_norm(hidden_states)
        merged = hidden_states.view(-1, self.merge_dim)
        with get_nvtx_range()("PatchMerger/linear_fc1"):
            merged, _ = self.linear_fc1(merged)
        with get_nvtx_range()("PatchMerger/gelu"):
            # NOTE: Official HuggingFace uses default approximate='none' in Qwen3VLVisionPatchMerger.
            merged = torch.nn.functional.gelu(merged, approximate="tanh")
        with get_nvtx_range()("PatchMerger/linear_fc2"):
            merged, _ = self.linear_fc2(merged)
        return merged


# -------------------------------------------------------------------
# Qwen35VLVisionEncoder - top-level encoder module
# -------------------------------------------------------------------

class Qwen35VLVisionEncoder(VisionModule):
    """Megatron-native Qwen3.5-VL vision encoder.

    Processes image / video inputs through:

    1. ``Qwen35VLPatchEmbed``  (Conv3d)
    2. Learned ``nn.Embedding`` position table with bilinear interpolation
    3. 2D Vision RoPE from ``(row, col)`` patch positions
    4. ``TransformerBlock`` x N  with ``PackedSeqParams`` (THD attention)
    5. ``Qwen35VLPatchMerger``

    Output dimension matches the language model ``hidden_size``.

    Args:
        config: Vision ``TransformerConfig``.
        transformer_layer_spec: ``ModuleSpec`` for ViT layers.
        in_channels: Image channels (3 for RGB).
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        spatial_merge_size: Spatial merge factor.
        out_hidden_size: Output dim (language decoder hidden_size).
        max_num_positions: Size of the learned position table.
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
        # Cached so mRoPE position construction does not need to materialize
        # GPU grid metadata on the host for multi-image batches.
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size

        # --- Patch embedding (Conv3d) ---
        self.patch_embed = Qwen35VLPatchEmbed(
            config=config,
            in_channels=in_channels,
            hidden_size=config.hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )

        # --- Learned position embedding with bilinear interpolation ---
        self.pos_embed = torch.nn.Embedding(
            max_num_positions, config.hidden_size,
        )
        self.num_grid_per_side = int(max_num_positions ** 0.5)

        # --- Vision rotary embeddings ---
        head_dim = config.hidden_size // config.num_attention_heads
        self.rot_pos_emb = Qwen35VLVisionRotaryEmbedding(
            head_dim // 2, config=config,
        )
        self._pos_embed_grid_cache: Dict[
            Tuple[int, int, int, str, torch.dtype],
            Tuple[Tensor, Tensor],
        ] = {}
        self._rotary_pos_id_cache: Dict[
            Tuple[int, int, int, str],
            Tensor,
        ] = {}
        self._rotary_freq_grid_cache: Dict[
            Tuple[int, int, int, str],
            Tensor,
        ] = {}
        self._packed_seq_params_grid_cache: Dict[
            Tuple[int, int, int, int, str],
            Tuple[Tensor, Tensor],
        ] = {}

        # --- Transformer blocks ---
        if transformer_layer_spec is None:
            transformer_layer_spec = get_qwen35_vl_vision_spec()

        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            post_layer_norm=False,
        )

        # --- Patch merger ---
        self.merger = Qwen35VLPatchMerger(
            config=config,
            hidden_size=config.hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    def _uniform_grid_spec(
        self,
        grid_thw: Tensor,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return ``(num_images, t, h, w)`` when all grids are identical."""
        num_images = int(grid_thw.shape[0])
        if num_images == 1:
            result = (
                1,
                int(grid_thw[0, 0].item()),
                int(grid_thw[0, 1].item()),
                int(grid_thw[0, 2].item()),
            )
            return result
        all_equal = bool(torch.all(grid_thw == grid_thw[0]).item())
        if all_equal:
            result = (
                num_images,
                int(grid_thw[0, 0].item()),
                int(grid_thw[0, 1].item()),
                int(grid_thw[0, 2].item()),
            )
            return result
        return None

    # ---------------------------------------------------------------
    # Learned position embedding with bilinear interpolation
    # ---------------------------------------------------------------

    def _cached_pos_embed_indices_and_weights(
        self,
        t: int,
        h: int,
        w: int,
    ) -> Tuple[Tensor, Tensor]:
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        key = (int(t), int(h), int(w), str(device), dtype)
        cached = self._pos_embed_grid_cache.get(key)
        if cached is not None:
            return cached

        n = self.num_grid_per_side
        merge = int(self.spatial_merge_size)
        h_idxs = torch.linspace(0, n - 1, int(h))
        w_idxs = torch.linspace(0, n - 1, int(w))

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clip(max=n - 1)
        w_ceil = (w_floor + 1).clip(max=n - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        base_h = h_floor * n
        base_h_ceil = h_ceil * n
        indices = [
            base_h[:, None] + w_floor[None, :],
            base_h[:, None] + w_ceil[None, :],
            base_h_ceil[:, None] + w_floor[None, :],
            base_h_ceil[:, None] + w_ceil[None, :],
        ]
        weights = [
            (1 - dh)[:, None] * (1 - dw)[None, :],
            (1 - dh)[:, None] * dw[None, :],
            dh[:, None] * (1 - dw)[None, :],
            dh[:, None] * dw[None, :],
        ]

        idx_parts = []
        weight_parts = []
        for idx, weight in zip(indices, weights):
            idx = (
                idx.view(1, h, w)
                .expand(t, h, w)
                .view(t, h // merge, merge, w // merge, merge)
                .permute(0, 1, 3, 2, 4)
                .reshape(-1)
            )
            weight = (
                weight.view(1, h, w)
                .expand(t, h, w)
                .view(t, h // merge, merge, w // merge, merge)
                .permute(0, 1, 3, 2, 4)
                .reshape(-1)
            )
            idx_parts.append(idx)
            weight_parts.append(weight)

        idx_tensor = torch.stack(idx_parts, dim=0).to(
            dtype=torch.long,
            device=device,
        )
        weight_tensor = torch.stack(weight_parts, dim=0).to(
            dtype=dtype,
            device=device,
        )
        cached = (idx_tensor, weight_tensor)
        if len(self._pos_embed_grid_cache) >= 64:
            self._pos_embed_grid_cache.clear()
        self._pos_embed_grid_cache[key] = cached
        return cached

    def _single_pos_embed_interpolate(self, t: int, h: int, w: int) -> Tensor:
        idx_tensor, weight_tensor = self._cached_pos_embed_indices_and_weights(
            t, h, w,
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        return pos_embeds.sum(dim=0)

    def _fast_pos_embed_interpolate(
        self,
        grid_thw: Tensor,
        uniform_grid: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tensor:
        """Bilinear interpolation of the learned 2D position table.

        Matches HF ``Qwen3VLVisionModel.fast_pos_embed_interpolate``.
        Adopted from megatron-bridge ``Qwen3VLVisionModel.fast_pos_embed_interpolate``
        which avoids the upfront ``grid_thw.tolist()`` GPU->CPU memcpy that
        deadlocks for some multi-image configs.
        Each per-image (h, w) iteration is via ``zip(grid_hs, grid_ws)`` -
        which yields 0-d GPU tensors that PyTorch implicitly converts to
        Python ints via small per-element syncs (working under multi-image
        whereas a single batch ``.tolist()`` deadlocked).

        Args:
            grid_thw: ``[num_images, 3]`` (T, H, W) in patch-grid units.

        Returns:
            ``[total_patches, hidden_size]`` position embeddings in
            block-merge order.
        """
        if uniform_grid is None:
            uniform_grid = self._uniform_grid_spec(grid_thw)

        if uniform_grid is not None:
            num_images, t, h, w = uniform_grid
            single_image_embeds = self._single_pos_embed_interpolate(t, h, w)
            if int(num_images) == 1:
                return single_image_embeds
            result = single_image_embeds.repeat(int(num_images), 1)
            return result

        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list: List[List[int]] = [[] for _ in range(4)]
        weight_list: List[List[float]] = [[] for _ in range(4)]

        n = self.num_grid_per_side

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            # h, w are 0-d GPU tensors. torch.linspace accepts them as the
            # `steps` argument (PyTorch converts via small per-element sync).
            h_idxs = torch.linspace(0, n - 1, h)
            w_idxs = torch.linspace(0, n - 1, w)

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=n - 1)
            w_ceil = (w_floor + 1).clip(max=n - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * n
            base_h_ceil = h_ceil * n

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                # indices[i] / weights[i] are CPU tensors (torch.linspace
                # default device = cpu); .tolist() here is CPU-only.
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(
            idx_list, dtype=torch.long, device=self.pos_embed.weight.device,
        )
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=self.pos_embed.weight.device,
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # split sizes - each (h*w) is a small product of 0-d tensors -> int via small implicit sync.
        split_sizes = [h * w for h, w in zip(grid_hs, grid_ws)]
        patch_pos_embeds = patch_pos_embeds.split(split_sizes)

        merge = self.spatial_merge_size
        result = []
        for pe, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pe = pe.repeat(t, 1)
            pe = (
                pe.view(t, h // merge, merge, w // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pe)
        result = torch.cat(result)
        return result

    # ---------------------------------------------------------------
    # 2D Vision RoPE
    # ---------------------------------------------------------------

    def _cached_rotary_pos_ids(self, t: int, h: int, w: int, device) -> Tensor:
        key = (int(t), int(h), int(w), str(device))
        cached = self._rotary_pos_id_cache.get(key)
        if cached is not None:
            return cached

        merge = int(self.spatial_merge_size)
        merged_h, merged_w = int(h) // merge, int(w) // merge
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge, device=device)
        intra_col = torch.arange(merge, device=device)

        row_idx = (
            block_rows[:, None, None, None] * merge
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * merge
            + intra_col[None, None, None, :]
        )
        row_idx = row_idx.expand(merged_h, merged_w, merge, merge).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge, merge).reshape(-1)
        pos_ids = torch.stack((row_idx, col_idx), dim=-1)
        if int(t) > 1:
            pos_ids = pos_ids.repeat(int(t), 1)
        if len(self._rotary_pos_id_cache) >= 64:
            self._rotary_pos_id_cache.clear()
        self._rotary_pos_id_cache[key] = pos_ids
        return pos_ids

    def _cached_rotary_freqs_for_grid(
        self,
        t: int,
        h: int,
        w: int,
        device,
    ) -> Tensor:
        key = (int(t), int(h), int(w), str(device))
        cached = self._rotary_freq_grid_cache.get(key)
        if cached is not None:
            return cached

        max_hw = max(int(h), int(w))
        freq_table = self.rot_pos_emb(max_hw, device=device)
        pos_ids = self._cached_rotary_pos_ids(t, h, w, device)
        freqs = freq_table[pos_ids].flatten(1)
        if len(self._rotary_freq_grid_cache) >= 64:
            self._rotary_freq_grid_cache.clear()
        self._rotary_freq_grid_cache[key] = freqs
        return freqs

    def _compute_rotary_pos_emb(
        self,
        grid_thw: Tensor,
        uniform_grid: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tensor:
        """Compute 2D Vision RoPE for all patches in block-merge order.

        Matches HF ``Qwen3VLVisionModel.rot_pos_emb``.
        Adopted from megatron-bridge ``Qwen3VLVisionModel.rot_pos_emb`` -
        avoids the upfront ``grid_thw.tolist()`` GPU->CPU memcpy that
        deadlocks for multi-image configs. Uses GPU-vectorized ``.max()``
        / ``torch.prod`` then a single small ``.item()`` (sync on a
        small derived tensor, NOT on the original grid_thw).

        Args:
            grid_thw: ``[num_images, 3]`` (T, H, W) per image, on GPU.

        Returns:
            ``[total_patches, head_dim // 2]`` raw RoPE frequencies on GPU.
        """
        device = self.rot_pos_emb.inv_freq.device
        if device.type == "meta":
            device = torch.device("cuda", torch.cuda.current_device())

        if uniform_grid is None:
            uniform_grid = self._uniform_grid_spec(grid_thw)
        if uniform_grid is not None:
            num_images, t, h, w = uniform_grid
            single_image_freqs = self._cached_rotary_freqs_for_grid(t, h, w, device)
            if int(num_images) == 1:
                return single_image_freqs
            result = single_image_freqs.repeat(int(num_images), 1)
            return result

        merge = self.spatial_merge_size

        # GPU-vectorized derivations: result is a small (0-d) tensor; the
        # only sync is the final .item() - single-shot sync on a fresh
        # small derived tensor (works under multi-image; .tolist() of the
        # original grid_thw deadlocks on multi-element copy).
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rot_pos_emb(max_hw, device=device)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty(
            (total_tokens, 2), dtype=torch.long, device=device,
        )

        offset = 0
        # Iterate over GPU rows directly. Each `num_frames`/`height`/`width`
        # is a 0-d GPU tensor; `height // merge` is a GPU tensor op; the
        # arange / expand / reshape ops below all stay on GPU. The only
        # sync per row is via `coords.shape[0]` (Python int from .shape -
        # CPU-side metadata, no sync).
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge, width // merge

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge, device=device)
            intra_col = torch.arange(merge, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge
                + intra_col[None, None, None, :]
            )

            row_idx = row_idx.expand(
                merged_h, merged_w, merge, merge,
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge, merge,
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            n_tokens = coords.shape[0]
            pos_ids[offset: offset + n_tokens] = coords
            offset += n_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    # ---------------------------------------------------------------
    # PackedSeqParams for variable-length attention
    # ---------------------------------------------------------------

    @staticmethod
    def _build_packed_seq_params(grid_thw: Tensor) -> PackedSeqParams:
        """Build ``PackedSeqParams`` from grid dimensions - pure GPU path.

        Adopted from megatron-bridge ``Qwen3VLVisionModel.build_packed_seq_params``.
        All ops are GPU tensor ops (``repeat_interleave``, ``cumsum``,
        ``F.pad``, ``.max()``) - NO ``.tolist()`` / ``.item()`` on the
        per-element data; ``max_seqlen_q`` is passed as a 0-d GPU tensor
        (TE / Megatron PackedSeqParams accepts both int and 0-d tensor).

        This avoids the GPU->CPU memcpy deadlock observed under specific
        multi-image (image_size, N) configs.

        Args:
            grid_thw: ``[num_images, 3]`` (T, H, W) on GPU.

        Returns:
            ``PackedSeqParams`` with cu_seqlens / max_seqlen on GPU.
        """
        seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0],
        )
        cu_seqlens = seqlens.cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).int()

        max_seqlen_q = int((grid_thw[:, 1] * grid_thw[:, 2]).max().item())

        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
        )

    def _cached_packed_seq_params_for_uniform_grid(
        self,
        num_images: int,
        t: int,
        h: int,
        w: int,
        device,
    ) -> PackedSeqParams:
        key = (int(num_images), int(t), int(h), int(w), str(device))
        cached = self._packed_seq_params_grid_cache.get(key)
        if cached is None:
            num_seqs = int(num_images) * int(t)
            seqlen = int(h) * int(w)
            cu_seqlens = (
                torch.arange(num_seqs + 1, device=device, dtype=torch.int32)
                * seqlen
            )
            max_seqlen_q = seqlen
            cached = (cu_seqlens, max_seqlen_q)
            if len(self._packed_seq_params_grid_cache) >= 64:
                self._packed_seq_params_grid_cache.clear()
            self._packed_seq_params_grid_cache[key] = cached

        cu_seqlens, max_seqlen_q = cached
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
        )

    def _build_or_get_packed_seq_params(
        self,
        grid_thw: Tensor,
        uniform_grid: Optional[Tuple[int, int, int, int]] = None,
    ) -> PackedSeqParams:
        if uniform_grid is None:
            uniform_grid = self._uniform_grid_spec(grid_thw)
        if uniform_grid is not None:
            num_images, t, h, w = uniform_grid
            result = self._cached_packed_seq_params_for_uniform_grid(
                int(num_images),
                int(t),
                int(h),
                int(w),
                grid_thw.device,
            )
            return result

        result = self._build_packed_seq_params(grid_thw)
        return result

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    def forward(
        self,
        pixel_values: Tensor,
        grid_thw: Tensor,
    ) -> Tensor:
        """Encode images / video frames.

        Args:
            pixel_values: ``[total_patches, C * T * pH * pW]``
                pre-extracted flat patches in block-merge order.
            grid_thw: ``[num_images, 3]`` (T, H, W) in patch-grid units.

        Returns:
            ``[total_merged_patches, out_hidden_size]`` visual embeddings.
        """


        # 1. Patch embedding (Conv3d)
        with get_nvtx_range()("VisionEncoder/patch_embed"):
            hidden_states = self.patch_embed(pixel_values)

        uniform_grid = self._uniform_grid_spec(grid_thw)

        # 2. Learned position embedding (bilinear interpolation)
        with get_nvtx_range()("VisionEncoder/pos_embed_interpolate"):
            pos_embeds = self._fast_pos_embed_interpolate(
                grid_thw, uniform_grid=uniform_grid,
            )
            hidden_states = hidden_states + pos_embeds

        # 3. 2D Vision RoPE
        with get_nvtx_range()("VisionEncoder/vision_rope"):
            rot_freqs = self._compute_rotary_pos_emb(
                grid_thw, uniform_grid=uniform_grid,
            )
            emb = torch.cat((rot_freqs, rot_freqs), dim=-1)
            rot_freqs_expanded = emb.unsqueeze(1).unsqueeze(1)

        # 4. Transformer blocks with PackedSeqParams
        packed_seq_params = self._build_or_get_packed_seq_params(
            grid_thw, uniform_grid=uniform_grid,
        )
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=None,
            rotary_pos_emb=rot_freqs_expanded,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = hidden_states.squeeze(1)

        # 5. Patch merger
        with get_nvtx_range()("VisionEncoder/patch_merger"):
            hidden_states = self.merger(hidden_states)
        return hidden_states
