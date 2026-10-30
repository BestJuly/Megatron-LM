# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-native Kimi-K3 vision tower (MoonViT 3D) and projector.

Matches ``MoonViT3dPretrainedModel`` + ``PatchMergerMLPV2`` from the
``moonshotai/Kimi-K3`` remote code (see ``configuration.py`` for the pinned
revision)::

  patch embed (Conv2d 14x14, as a GEMM on 588-wide rows)
    + interpolated absolute position table (+ sin-cos time rows for t > 1)
    -> TransformerBlock x N  (2-D interleaved RoPE, one THD sequence per item)
    -> final RMSNorm (the block's post_layer_norm)
    -> sd2_tpool: mean over frames, then 2x2 spatial merge
    -> projector: Linear -> GELU(erf) -> Linear -> RMSNorm

Pixel rows arrive in the HF processor's raster order: ``navit_patchify`` emits
patches ``(t, h, w)``-major, and each 588-wide row is channel-major
``(C, 14, 14)``. Unlike Qwen3.5-VL there is no block-merge reordering; the
merge is a reshape of raster-ordered rows, as in HF.

Each item's ``t*h*w`` rows form one attention sequence (all frames attend to
each other), whereas Qwen3.5-VL attends frame by frame.
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from examples.multimodal_dev.models.kimi_k3_vl.configuration import VISION_KWARGS
from examples.multimodal_dev.observability import backward_range_begin, backward_range_end
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


def _grid_list(grid_thw) -> List[Tuple[int, int, int]]:
    """``grid_thw`` (tensor or sequence) as a list of Python int triples."""
    rows = grid_thw.tolist() if isinstance(grid_thw, Tensor) else grid_thw
    return [(int(t), int(h), int(w)) for t, h, w in rows]


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """``[length, embed_dim]`` sin-cos table, MoonViT's fixed time embedding.

    Bit-for-bit the reference ``get_1d_sincos_pos_embed`` (InternVideo2 style):
    sin over the first half of the channels, cos over the second, computed in
    float32 numpy.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.arange(length, dtype=np.float32).reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class KimiK3VLPatchEmbed(MegatronModule):
    """``Conv2d(in_channels, hidden, patch, patch, bias=False)`` as one GEMM.

    The kernel equals the stride and the input is already one patch per row, so
    the convolution is a linear map of the flattened ``(C, p, p)`` row. The
    weight keeps the Conv2d shape so checkpoints map 1:1 onto HF's
    ``patch_embed.proj.weight``. Replicated across TP ranks.
    """

    def __init__(self, config: TransformerConfig, in_channels: int, patch_size: int):
        super().__init__(config=config)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.proj = torch.nn.Conv2d(
            in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """``[rows, C*p*p]`` -> ``[rows, hidden]``."""
        weight = self.proj.weight
        flat = pixel_values.reshape(pixel_values.shape[0], weight[0].numel()).to(weight.dtype)
        return F.linear(flat, weight.flatten(1))


class KimiK3VLPositionEmbedding(MegatronModule):
    """Learnable 2-D position table + fixed time rows (``divided_fixed``).

    ``weight`` is ``[pos_h, pos_w, hidden]``. For an item of grid ``(t, h, w)``
    the table is bilinear-interpolated to ``(h, w)`` (``align_corners=False``,
    skipped when the size already matches) and, for ``t > 1`` only, repeated
    over frames with ``time_weight[:t]`` added. A single-frame item gets no
    time row, exactly as in HF (whose row 0 is not zero).
    """

    def __init__(
        self,
        config: TransformerConfig,
        height: int,
        width: int,
        num_frames: int,
        interpolation_mode: str = "bilinear",
    ):
        super().__init__(config=config)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.interpolation_mode = interpolation_mode
        self.weight = torch.nn.Parameter(torch.empty(height, width, config.hidden_size))
        if config.perform_initialization:
            torch.nn.init.normal_(self.weight)
        time_weight = torch.from_numpy(
            get_1d_sincos_pos_embed(config.hidden_size, num_frames)
        ).float()
        self.register_buffer("time_weight", time_weight.unsqueeze(1), persistent=False)

    def _table(self, h: int, w: int) -> Tensor:
        if (h, w) == (self.height, self.width):
            return self.weight.flatten(0, 1)
        resized = F.interpolate(
            self.weight.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode=self.interpolation_mode
        )
        return resized.squeeze(0).permute(1, 2, 0).flatten(0, 1)

    def forward(self, grids: Sequence[Tuple[int, int, int]]) -> Tensor:
        """``[sum(t*h*w), hidden]`` position embeddings in raster order."""
        out = []
        for t, h, w in grids:
            if t > self.num_frames:
                raise ValueError(
                    f"Kimi-K3 vision item has t={t} frames; the time embedding covers "
                    f"at most {self.num_frames}"
                )
            table = self._table(h, w)
            if t > 1:
                time_weight = self.time_weight[:t].to(table.device)
                table = (table.unsqueeze(0).float() + time_weight).flatten(0, 1)
            out.append(table)
        return torch.cat([x.float() for x in out]) if len(out) > 1 else out[0].float()


class KimiK3VLRotaryEmbedding(MegatronModule):
    """MoonViT's ``Rope2DPosEmbRepeated`` as MCore interleaved RoPE frequencies.

    HF rotates each adjacent channel pair ``(2i, 2i+1)`` of a head by
    ``cis(angle_i)`` with, for ``j < head_dim / 4``::

        angle_{2j}   = column * theta ** (-4j / head_dim)
        angle_{2j+1} = row    * theta ** (-4j / head_dim)

    MCore's ``rotary_interleaved`` application rotates the same pairs by
    ``freqs[2i] == freqs[2i+1]``, so the per-pair angles are repeated twice
    along the channel axis. Frames of a video item share their frame's table.
    """

    def __init__(self, config: TransformerConfig, head_dim: int, max_side: int, theta: float):
        super().__init__(config=config)
        assert head_dim % 4 == 0, "MoonViT 2-D RoPE needs head_dim divisible by 4"
        self.head_dim = head_dim
        self.max_side = max_side
        self.theta = theta

    def _inv_freq(self, device) -> Tensor:
        dim_range = torch.arange(0, self.head_dim, 4, device=device)[: self.head_dim // 4]
        return 1.0 / (self.theta ** (dim_range.float() / self.head_dim))

    def forward(self, grids: Sequence[Tuple[int, int, int]], device) -> Tensor:
        """``[sum(t*h*w), 1, 1, head_dim]`` fp32 frequencies."""
        inv_freq = self._inv_freq(device)
        out = []
        for t, h, w in grids:
            if not (1 <= h <= self.max_side and 1 <= w <= self.max_side):
                raise ValueError(
                    f"Kimi-K3 vision grid {(t, h, w)} exceeds the 2-D RoPE table "
                    f"({self.max_side} x {self.max_side})"
                )
            rows = torch.arange(h, device=device, dtype=torch.float32)
            cols = torch.arange(w, device=device, dtype=torch.float32)
            col_angles = torch.outer(cols, inv_freq)  # [w, d/4]
            row_angles = torch.outer(rows, inv_freq)  # [h, d/4]
            angles = torch.stack(
                (
                    col_angles.unsqueeze(0).expand(h, w, -1),
                    row_angles.unsqueeze(1).expand(h, w, -1),
                ),
                dim=-1,
            ).reshape(h * w, self.head_dim // 2)
            if t > 1:
                angles = angles.repeat(t, 1)
            out.append(angles)
        angles = torch.cat(out)
        return angles.repeat_interleave(2, dim=-1)[:, None, None, :]


def tpool_patch_merge(hidden_states: Tensor, grids, merge: int) -> Tensor:
    """``sd2_tpool``: average each item over its frames, then merge m x m.

    Returns ``[sum((h/m)*(w/m)), m*m*hidden]`` with the merged window laid out
    ``(kernel_row, kernel_col, channel)``, as HF's
    ``tpool_patch_merger`` + ``PatchMergerMLPV2`` flatten it.
    """
    d = hidden_states.shape[-1]
    out = []
    offset = 0
    for t, h, w in grids:
        rows = t * h * w
        seq = hidden_states[offset : offset + rows]
        offset += rows
        seq = seq.view(t, h // merge, merge, w // merge, merge, d)
        seq = seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        out.append(seq.reshape((h // merge) * (w // merge), merge * merge * d))
    return torch.cat(out) if len(out) > 1 else out[0]


class KimiK3VLProjector(MegatronModule):
    """``PatchMergerMLPV2``: ``Linear -> GELU(erf) -> Linear -> RMSNorm``, no biases."""

    def __init__(
        self, config: TransformerConfig, merge_dim: int, out_hidden_size: int, norm_eps: float
    ):
        super().__init__(config=config)
        self.linear_fc1 = build_module(
            ColumnParallelLinear,
            merge_dim,
            merge_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )
        self.linear_fc2 = build_module(
            RowParallelLinear,
            merge_dim,
            out_hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
        )
        self.post_norm = TENorm(config=config, hidden_size=out_hidden_size, eps=norm_eps)

    def forward(self, merged: Tensor) -> Tensor:
        """``[rows, merge_dim]`` -> ``[rows, out_hidden_size]``."""
        x, _ = self.linear_fc1(merged)
        x = F.gelu(x, approximate="none")
        x, _ = self.linear_fc2(x)
        return self.post_norm(x)


class KimiK3VLVisionEncoder(VisionModule):
    """MoonViT 3D tower + projector; output width is the decoder hidden size.

    Args:
        config: Vision ``TransformerConfig`` (``get_kimi_k3_vl_vision_config``).
        transformer_layer_spec: ``ModuleSpec`` for the ViT layers.
        out_hidden_size: Decoder hidden size (projector output).
        vision_kwargs: Override of ``configuration.VISION_KWARGS``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec = None,
        out_hidden_size: int = VISION_KWARGS["out_hidden_size"],
        vision_kwargs: dict = None,
    ):
        super().__init__(config=config)
        kw = dict(VISION_KWARGS)
        kw.update(vision_kwargs or {})
        self.spatial_merge_size = kw["spatial_merge_size"]

        self.patch_embed = KimiK3VLPatchEmbed(config, kw["in_channels"], kw["patch_size"])
        self.pos_emb = KimiK3VLPositionEmbedding(
            config, kw["pos_emb_height"], kw["pos_emb_width"], kw["pos_emb_time"]
        )
        self.rotary_pos_emb = KimiK3VLRotaryEmbedding(
            config, config.kv_channels, kw["rope_max_side"], kw["rope_theta"]
        )

        if transformer_layer_spec is None:
            from examples.multimodal_dev.models.kimi_k3_vl.specs import get_kimi_k3_vl_vision_spec

            transformer_layer_spec = get_kimi_k3_vl_vision_spec()
        # post_layer_norm=True builds the block's final RMSNorm, MoonViT's
        # encoder.final_layernorm.
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            post_layer_norm=True,
        )

        merge = self.spatial_merge_size
        self.projector = KimiK3VLProjector(
            config,
            merge_dim=config.hidden_size * merge * merge,
            out_hidden_size=out_hidden_size,
            norm_eps=kw["projector_norm_eps"],
        )

    def check_grids(self, grids) -> None:
        """Reject grids MoonViT cannot encode (before any kernel launches)."""
        merge = self.spatial_merge_size
        for t, h, w in grids:
            if t < 1 or h < 1 or w < 1 or h % merge or w % merge:
                raise ValueError(
                    f"Kimi-K3 vision grid {(t, h, w)} must be positive with h, w "
                    f"divisible by {merge}"
                )

    @staticmethod
    def build_packed_seq_params(grids, device) -> PackedSeqParams:
        """One THD sequence per item, covering all of its frames."""
        lengths = [t * h * w for t, h, w in grids]
        cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int64), 0).to(torch.int32)
        cu = cu.to(device, non_blocking=True)
        max_seqlen = max(lengths)
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

    def encode_tower(self, pixel_values: Tensor, grid_thw) -> Tensor:
        """ViT output after the final norm, before tpool/projector: ``[rows, hidden]``."""
        grids = _grid_list(grid_thw)
        self.check_grids(grids)
        hidden_states = self.patch_embed(pixel_values)
        hidden_states, bwd_marked = backward_range_begin(hidden_states)
        hidden_states = hidden_states + self.pos_emb(grids).to(hidden_states.dtype)
        rotary = self.rotary_pos_emb(grids, hidden_states.device)
        packed_seq_params = self.build_packed_seq_params(grids, hidden_states.device)
        hidden_states = self.decoder(
            hidden_states=hidden_states.unsqueeze(1),
            attention_mask=None,
            rotary_pos_emb=rotary,
            packed_seq_params=packed_seq_params,
        ).squeeze(1)
        return hidden_states, bwd_marked

    def forward(self, pixel_values: Tensor, grid_thw) -> Tensor:
        """Encode vision items.

        Args:
            pixel_values: ``[sum(t*h*w), C*14*14]`` raster-ordered patch rows.
            grid_thw: ``[num_items, 3]`` ``(t, h, w)`` in patch units (tensor on
                any device, or a sequence of triples).

        Returns:
            ``[sum((h/2)*(w/2)), out_hidden_size]`` decoder-ready embeddings.
        """
        grids = _grid_list(grid_thw)
        if not grids:
            # A text-only microbatch: nothing to encode, nothing to scatter.
            weight = self.projector.linear_fc2.weight
            return weight.new_zeros(0, weight.shape[0])
        hidden_states, bwd_marked = self.encode_tower(pixel_values, grids)
        merged = tpool_patch_merge(hidden_states, grids, self.spatial_merge_size)
        out = self.projector(merged)
        return backward_range_end(out, "vision_encoder_backward", bwd_marked)
