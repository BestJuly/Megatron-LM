# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-native Qwen3.5-VL vision encoder.

Architecture (matches HF ``Qwen3VLVisionModel`` exactly):

  PatchEmbed (Conv3d)
    → learned position embedding (bilinear interpolation)
    → 2D Vision RoPE
    → TransformerBlock × N  (with PackedSeqParams / THD attention)
    → PatchMerger  (per-token LN → spatial merge → MLP)

Key design choices:
  * ``Conv3d`` patch embedding is replicated across TP ranks (no MCore
    equivalent for 3D convolutions).
  * ``PatchMerger`` MLP uses ``ColumnParallelLinear`` / ``RowParallelLinear``
    for TP sharding.
  * Inherits from ``VisionModule``.
  * Expects pixel values in block-merge order (as produced by the HF
    processor) so the merger's simple reshape is correct.
"""

import os
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

# Experiment kill-switch: QWEN35_VL_GRID_CACHE=0 restores the original
# per-grid loop implementations (pre grid-cache behavior) for A/B runs.
_GRID_CACHE_ENABLED = os.environ.get("QWEN35_VL_GRID_CACHE", "1") != "0"


@contextmanager
def _nvtx(name: str):
    """Timeline tag for one vision-encoder sub-phase (native and MDP paths)."""
    torch.cuda.nvtx.range_push(f"vision_encoder.{name}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

from examples.multimodal_dev.models.qwen35_vl.vision_pos_cache import GridCache
from examples.multimodal_dev.observability import backward_range_begin, backward_range_end
from megatron.core.models.common.vision_module.vision_module import (
    VisionModule,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

# -------------------------------------------------------------------
# PatchEmbed — Conv3d (replicated, no TP sharding)
# -------------------------------------------------------------------

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
        pixel_values = pixel_values.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(pixel_values.to(dtype=target_dtype)).view(
            -1, self.hidden_size
        )


# -------------------------------------------------------------------
# VisionRotaryEmbedding — 1D frequency table
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
# PatchMerger — per-token LN, spatial merge, TP-sharded MLP
# -------------------------------------------------------------------

class Qwen35VLPatchMerger(MegatronModule):
    """Spatial patch merger matching HF ``Qwen3VLVisionPatchMerger``.

    Per-token ``LayerNorm`` on ``hidden_size`` → reshape to merge
    ``spatial_merge_size ** 2`` adjacent patches → two-layer MLP
    (``ColumnParallelLinear`` → GELU → ``RowParallelLinear``).

    MLP dimensions: ``merge_dim → merge_dim → out_hidden_size``
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
        hidden_states = self.patch_norm(hidden_states)
        merged = hidden_states.view(-1, self.merge_dim)
        merged, _ = self.linear_fc1(merged)
        # Match official HuggingFace Qwen3VLVisionPatchMerger (default approximate='none').
        merged = torch.nn.functional.gelu(merged, approximate="none")
        merged, _ = self.linear_fc2(merged)
        return merged


# -------------------------------------------------------------------
# Qwen35VLVisionEncoder — top-level encoder module
# -------------------------------------------------------------------

class _GatherChunkAlongSeq(torch.autograd.Function):
    """All-gather the encoder-CP shards and restore chunk order.

    Forward: every rank holds ``[L/e, H]`` (its zigzag chunks, compacted); the
    gather produces rank-major ``[L, H]`` which ``perm`` un-zigzags back into
    chunk order.

    Backward is a **reduce-scatter**, and that is the load-bearing half. After
    the gather every rank runs the merger on the full sequence, but only the
    designated rank's output feeds the bridge, so only that rank receives a real
    gradient; the others are driven with explicitly zeroed buffers. Summing
    (real + zeros) and then taking each rank's own block is what delivers rank
    r's gradient to the blocks that actually computed rank r's rows. A plain
    "take my slice" backward would leave every non-designated rank's
    transformer blocks with no gradient at all -- silently, since the shapes
    are fine and the run converges on 1/e of the encoder.
    """

    @staticmethod
    def forward(ctx, local, perm, group, encoder_cp):
        ctx.group = group
        ctx.encoder_cp = encoder_cp
        ctx.save_for_backward(perm)
        gathered = local.new_empty((local.shape[0] * encoder_cp,) + tuple(local.shape[1:]))
        torch.distributed.all_gather_into_tensor(gathered, local.contiguous(), group=group)
        return gathered.index_select(0, perm)

    @staticmethod
    def backward(ctx, grad_out):
        (perm,) = ctx.saved_tensors
        # Undo the un-zigzag: grad of the gathered buffer, in rank-major order.
        grad_gathered = grad_out.new_empty(grad_out.shape)
        grad_gathered.index_copy_(0, perm, grad_out.contiguous())
        local_rows = grad_out.shape[0] // ctx.encoder_cp
        grad_local = grad_out.new_empty((local_rows,) + tuple(grad_out.shape[1:]))
        torch.distributed.reduce_scatter_tensor(
            grad_local, grad_gathered, group=ctx.group
        )
        return grad_local, None, None, None


class Qwen35VLVisionEncoder(VisionModule):
    """Megatron-native Qwen3.5-VL vision encoder.

    Processes image / video inputs through:

    1. ``Qwen35VLPatchEmbed``  (Conv3d)
    2. Learned ``nn.Embedding`` position table with bilinear interpolation
    3. 2D Vision RoPE from ``(row, col)`` patch positions
    4. ``TransformerBlock`` × N  with ``PackedSeqParams`` (THD attention)
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
        pg_collection=None,
    ):
        super().__init__(config=config)

        self.hidden_size = config.hidden_size
        self.spatial_merge_size = spatial_merge_size

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

        # --- Transformer blocks ---
        if transformer_layer_spec is None:
            from examples.multimodal_dev.models.qwen35_vl.specs import (
                get_qwen35_vl_vision_spec,
            )
            transformer_layer_spec = get_qwen35_vl_vision_spec()

        # Threading pg_collection through is mandatory once encoder CP exists:
        # without it TransformerBlock falls back to use_mpu_process_groups(), and
        # the vision attention would ring over the DECODER's CP group. At
        # encoder_cp == cp those two rank sets numerically coincide, so that bug
        # is invisible at world=16/pp=2/cp=2/encoder_cp=2 -- the one topology the
        # registered hook test uses.
        self.pg_collection = pg_collection
        block_kwargs = {} if pg_collection is None else {"pg_collection": pg_collection}
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            post_layer_norm=False,
            **block_kwargs,
        )

        # --- Patch merger ---
        self.merger = Qwen35VLPatchMerger(
            config=config,
            hidden_size=config.hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

        # Grid-keyed cache of position indices/weights, merge permutations,
        # RoPE coordinates, and cu_seqlens (plain attribute: not state).
        self._grid_cache = GridCache()

    # ---------------------------------------------------------------
    # Learned position embedding with bilinear interpolation
    # ---------------------------------------------------------------

    def _fast_pos_embed_interpolate(
        self, grid_thw: Tensor,
    ) -> Tensor:
        """Bilinear interpolation of the learned 2D position table.

        Matches HF ``Qwen3VLVisionModel.fast_pos_embed_interpolate``.

        Args:
            grid_thw: ``[num_images, 3]`` (T, H, W) in patch-grid units.

        Returns:
            ``[total_patches, hidden_size]`` position embeddings in
            block-merge order.
        """
        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]
        device = self.pos_embed.weight.device
        n = self.num_grid_per_side

        if not _GRID_CACHE_ENABLED:
            return self._fast_pos_embed_interpolate_uncached(
                grid_thw_list, grid_ts, grid_hs, grid_ws, device, n
            )

        # Bilinear indices/weights and the merge permutation depend only on
        # the grid; fetch them from the grid cache (computed once per grid).
        idx_parts, weight_parts = [], []
        for h, w in zip(grid_hs, grid_ws):
            idx, weight = self._grid_cache.pos(h, w, n, device)
            idx_parts.append(idx)
            weight_parts.append(weight)
        idx_tensor = torch.cat(idx_parts, dim=1)
        weight_tensor = torch.cat(weight_parts, dim=1).to(
            self.pos_embed.weight.dtype
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = (
            pos_embeds[0] + pos_embeds[1]
            + pos_embeds[2] + pos_embeds[3]
        )

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        merge = self.spatial_merge_size
        result = []
        for pe, t, h, w in zip(
            patch_pos_embeds, grid_ts, grid_hs, grid_ws,
        ):
            pe = pe[self._grid_cache.perm(h, w, merge, device)]
            if t > 1:
                pe = pe.repeat(t, 1)
            result.append(pe)

        return torch.cat(result)

    def _fast_pos_embed_interpolate_uncached(
        self, grid_thw_list, grid_ts, grid_hs, grid_ws, device, n
    ) -> Tensor:
        """Original per-grid loop implementation (QWEN35_VL_GRID_CACHE=0)."""
        idx_list: List[List[int]] = [[] for _ in range(4)]
        weight_list: List[List[float]] = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            t, h, w = int(t), int(h), int(w)
            h_idxs = torch.linspace(0, n - 1, h)
            w_idxs = torch.linspace(0, n - 1, w)

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=n - 1)
            w_ceil = (w_floor + 1).clip(max=n - 1)

            dh = h_idxs - h_floor.float()
            dw = w_idxs - w_floor.float()

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
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = (
            pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        )

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

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

        return torch.cat(result)

    # ---------------------------------------------------------------
    # 2D Vision RoPE
    # ---------------------------------------------------------------

    def _compute_rotary_pos_emb(self, grid_thw: Tensor) -> Tensor:
        """Compute 2D Vision RoPE for all patches in block-merge order.

        Matches HF ``Qwen3VLVisionModel.rot_pos_emb``.

        Args:
            grid_thw: ``[num_images, 3]`` (T, H, W) per image.

        Returns:
            Raw sectioned frequencies ``[3, 1, total_patches, head_dim // 2]``
            when ``config.mrope_section`` is set.  Otherwise returns the legacy
            ``[total_patches, head_dim // 2]`` row/column frequency tensor.
        """
        merge = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(int(h), int(w)) for _, h, w in grid_thw_list)
        # The compute device, not grid_thw.device: MDP passes grid_thw as a
        # CPU tensor (this method only needs its Python values). Stubs in
        # tests carry no parameters; they keep following grid_thw.
        pos_embed = getattr(self, "pos_embed", None)
        device = pos_embed.weight.device if pos_embed is not None else grid_thw.device
        # Tests call this method unbound on stub objects without _grid_cache;
        # fall back to the original loop path when the cache is absent.
        grid_cache = getattr(self, "_grid_cache", None)
        if _GRID_CACHE_ENABLED and grid_cache is not None:
            freq_table = grid_cache.freqs(
                self.rot_pos_emb, max_hw, device
            )
            device = freq_table.device
            # (row, col) coordinates depend only on the grid; cached per grid.
            pos_ids = torch.cat(
                [
                    grid_cache.rope(int(t), int(h), int(w), merge, device)
                    for t, h, w in grid_thw_list
                ]
            )
        else:
            freq_table = self.rot_pos_emb(max_hw, device=device)
            # Via the class, not self: tests call this method unbound on stubs.
            pos_ids = Qwen35VLVisionEncoder._rope_pos_ids_uncached(
                grid_thw_list, merge, device
            )

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)

        mrope_section = getattr(self.config, "mrope_section", None)
        if mrope_section is None:
            return embeddings

        sec_t, sec_h, sec_w = (int(section) for section in mrope_section)
        if sec_t != 0 or sec_h + sec_w != embeddings.shape[-1]:
            raise ValueError(
                "Qwen3.5-VL vision RoPE expects mrope_section "
                f"[0, row_dim, col_dim] summing to {embeddings.shape[-1]}, "
                f"got {mrope_section}"
            )

        raw_freqs = embeddings.new_zeros(
            3, 1, embeddings.shape[0], embeddings.shape[1],
        )
        raw_freqs[1, 0, :, :sec_h] = embeddings[:, :sec_h]
        raw_freqs[2, 0, :, sec_h : sec_h + sec_w] = embeddings[
            :, sec_h : sec_h + sec_w
        ]
        return raw_freqs

    @staticmethod
    def _rope_pos_ids_uncached(grid_thw_list, merge, device) -> Tensor:
        """Original per-grid loop (QWEN35_VL_GRID_CACHE=0)."""
        total_tokens = sum(int(t) * int(h) * int(w) for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames = int(num_frames)
            height = int(height)
            width = int(width)
            merged_h = height // merge
            merged_w = width // merge

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

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            pos_ids[offset : offset + coords.shape[0]] = coords
            offset += coords.shape[0]
        return pos_ids

    # ---------------------------------------------------------------
    # Encoder context parallelism
    # ---------------------------------------------------------------

    def _encoder_cp_state(self):
        """``(encoder_cp, encoder_cp_rank, group)`` for this rank.

        Derived from the collection MDP threaded in, never from the MPU: the
        MPU's CP group belongs to the DECODER.
        """
        group = getattr(self.pg_collection, "cp", None) if self.pg_collection else None
        if group is None:
            return 1, 0, None
        size = torch.distributed.get_world_size(group=group)
        if size == 1:
            return 1, 0, None
        return size, torch.distributed.get_rank(group=group), group

    def _encoder_cp_indices(self, grid_thw, encoder_cp, encoder_cp_rank, device):
        """Row index of this rank's shard, and the un-zigzag gather permutation.

        Served from the grid cache when one is present: the result depends only
        on the grid combination, the CP degree and this rank's CP position, but
        the encoder asks for it on every forward and again on every
        whole-encoder replay, and building it walks the entire vision pack in
        Python before two host-to-device copies. Stub objects in the unit tests
        call this method unbound and have no cache, so the uncached path stays.
        """
        grids = tuple(
            tuple(int(v) for v in row) for row in grid_thw.tolist()
        )
        grid_cache = getattr(self, "_grid_cache", None)
        if grid_cache is not None:
            return grid_cache.encoder_cp_indices(
                grids, encoder_cp, encoder_cp_rank, device
            )

        from megatron.core.mdp.encoder_cp_partition import (
            gather_permutation,
            shard_rows,
        )

        frames = []
        for t, h, w in grids:
            frames.extend([h * w] * t)
        rows = []
        for run in shard_rows(frames, encoder_cp, encoder_cp_rank):
            rows.extend(range(run.start, run.start + run.rows))
        shard = torch.tensor(rows, dtype=torch.long, device=device)
        perm = torch.tensor(
            gather_permutation(frames, encoder_cp), dtype=torch.long, device=device
        )
        return shard, perm

    # ---------------------------------------------------------------
    # PackedSeqParams for variable-length attention
    # ---------------------------------------------------------------

    def _build_packed_seq_params(self, grid_thw: Tensor) -> PackedSeqParams:
        """Build ``PackedSeqParams`` from grid dimensions.

        Each temporal frame of each image forms a separate sub-sequence
        in the packed THD layout, matching HF's ``cu_seqlens`` computation.
        The cu_seqlens tensor and max_seqlen depend only on the grid tuple
        and come from the grid cache (avoids a device sync per call).

        Args:
            grid_thw: ``[num_images, 3]``.

        Returns:
            ``PackedSeqParams`` for ``TransformerBlock``.
        """
        if not _GRID_CACHE_ENABLED:
            # This fallback does tensor math on grid_thw; MDP may pass it on
            # the CPU (the cached path needs only its Python values).
            grid_thw = grid_thw.to(self.pos_embed.weight.device)
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            max_seqlen = int((grid_thw[:, 1] * grid_thw[:, 2]).max().item())
            return PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                # TE aborts for qkv_format="thd" under CP unless the padded
                # variants are present. The vision pack has no padding, so the
                # SAME tensor object is passed: TE special-cases that identity
                # and keeps pad_between_seqs=False, so FlashAttention stays
                # eligible and the selected backend does not change.
                cu_seqlens_q_padded=cu_seqlens,
                cu_seqlens_kv_padded=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
            )

        grids = tuple(tuple(int(v) for v in row) for row in grid_thw.tolist())
        cu_seqlens, max_seqlen = self._grid_cache.packed_seq(
            grids, self.pos_embed.weight.device
        )

        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    def forward(
        self,
        pixel_values: Tensor,
        grid_thw: Tensor,
        pixels_are_sharded: bool = False,
    ) -> Tensor:
        """Encode images / video frames.

        Args:
            pixel_values: ``[total_patches, C * T * pH * pW]``
                pre-extracted flat patches in block-merge order -- or, with
                ``pixels_are_sharded`` under encoder CP, only THIS rank's zigzag
                shard of them (``total_patches / encoder_cp`` rows).
            grid_thw: ``[num_images, 3]`` (T, H, W) in patch-grid units, always
                describing the WHOLE chunk.
            pixels_are_sharded: the caller (the MDP runtime) already delivered
                this rank's shard, so the encoder must not slice again. Inert at
                ``encoder_cp=1``, where the shard is the chunk.

        Returns:
            ``[total_merged_patches, out_hidden_size]`` visual embeddings.
        """
        # 1. Patch embedding (Conv3d)
        with _nvtx("patch_embed"):
            hidden_states = self.patch_embed(pixel_values)

        # Bracket the encoder's BACKWARD pass. The bracket opens on the
        # earliest differentiable tensor (patch_embed's output -- pixels never
        # require grad, so there is nothing earlier) and closes on the encoder
        # output below. Backward visits those two nodes last and first
        # respectively, so the range spans exactly the encoder backward. Both
        # arms get it: native (inside the decoder's backward on PP stage 0) and
        # MDP (inside mdp.p5_encoder_backward), which is what makes the two
        # timelines comparable.
        hidden_states, bwd_marked = backward_range_begin(hidden_states)

        # Encoder context parallelism: this rank encodes only its zigzag shard
        # of the chunk. `encoder_cp == 1` short-circuits to the identity, so the
        # non-CP path below is untouched.
        encoder_cp, encoder_cp_rank, cp_group = self._encoder_cp_state()
        shard_index = None
        if encoder_cp > 1:
            shard_index, gather_perm = self._encoder_cp_indices(
                grid_thw, encoder_cp, encoder_cp_rank, hidden_states.device
            )
            if pixels_are_sharded:
                # The runtime sent exactly shard_rows(...) rows in local order,
                # so patch_embed above already ran on the shard alone. All that
                # is left is to check the payload and the layout agree.
                if hidden_states.shape[0] != shard_index.numel():
                    raise RuntimeError(
                        f"pixels_are_sharded: got {hidden_states.shape[0]} patch rows, "
                        f"but this rank's shard of the chunk described by grid_thw is "
                        f"{shard_index.numel()} rows; the delivered payload and the "
                        "layout disagree."
                    )
            else:
                hidden_states = hidden_states.index_select(0, shard_index)

        # 2. Learned position embedding (bilinear interpolation)
        with _nvtx("pos_embed_interpolate"):
            pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
            if shard_index is not None:
                # A per-token additive term, so it is sliced to the same rows --
                # unlike the RoPE table below, which TE indexes by GLOBAL
                # position and must therefore stay whole.
                pos_embeds = pos_embeds.index_select(0, shard_index)
            hidden_states = hidden_states + pos_embeds

        # 3. 2D Vision RoPE
        with _nvtx("rotary_pos_emb"):
            rot_freqs = self._compute_rotary_pos_emb(grid_thw)
            if getattr(self.config, "mrope_section", None) is None:
                emb = torch.cat((rot_freqs, rot_freqs), dim=-1)
                rot_freqs = emb.unsqueeze(1).unsqueeze(1)

        # 4. Transformer blocks with PackedSeqParams
        with _nvtx("build_packed_seq_params"):
            packed_seq_params = self._build_packed_seq_params(grid_thw)
        with _nvtx("transformer_blocks"):
            hidden_states = hidden_states.unsqueeze(1)
            hidden_states = self.decoder(
                hidden_states=hidden_states,
                attention_mask=None,
                rotary_pos_emb=rot_freqs,
                packed_seq_params=packed_seq_params,
            )
            hidden_states = hidden_states.squeeze(1)

        # Gather BEFORE the merger, never after. The merger folds
        # `merge**2 = 4` CONSECUTIVE rows (`view(-1, merge_dim)`), so a
        # rank-local merger would need every zigzag chunk to be a multiple of 4
        # -- 28 of the 137 frames in the shipped mock pool are not, at
        # encoder_cp=2 alone -- and the reshape SUCCEEDS anyway, silently
        # merging patches from different 2x2 spatial blocks. Gathering here
        # keeps the requirement at `h*w % (2*encoder_cp) == 0` and makes this
        # function's output byte-identical to the encoder_cp=1 path.
        if encoder_cp > 1:
            with _nvtx("encoder_cp_gather"):
                hidden_states = _GatherChunkAlongSeq.apply(
                    hidden_states, gather_perm, cp_group, encoder_cp
                )

        # 5. Patch merger
        with _nvtx("patch_merger"):
            hidden_states = self.merger(hidden_states)

        return backward_range_end(hidden_states, "vision_encoder_backward", bwd_marked)
