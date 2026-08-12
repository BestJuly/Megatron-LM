# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Grid-keyed caches for the Qwen3.5-VL vision encoder's position math.

The per-grid index/weight/coordinate tensors used by the encoder's position
embedding interpolation, spatial-merge reorder, and 2D RoPE depend only on
``(t, h, w)`` (plus the position-table side ``n`` and the merge factor) —
never on learned weights. Image grids repeat heavily across a training run,
so these are computed once per distinct grid, vectorized, and cached on
device. This removes the per-grid Python loops (linspace/clip/tolist/repeat
chains) that made the encoder forward CPU-launch-bound.

Only ``torch`` is imported, so numerical parity against the original loop
implementations can be tested standalone.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor


def pos_embed_indices(
    h: int, w: int, n: int, device: torch.device
) -> Tuple[Tensor, Tensor]:
    """Bilinear indices/weights of one (h, w) grid over an n x n table.

    Returns ``idx [4, h*w] long`` and ``weight [4, h*w] float32``, matching
    the four (floor/ceil x floor/ceil) corners of the original
    ``_fast_pos_embed_interpolate`` loop body.
    """
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

    idx = torch.stack(
        [
            (base_h[None].T + w_floor[None]).flatten(),
            (base_h[None].T + w_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_floor[None]).flatten(),
            (base_h_ceil[None].T + w_ceil[None]).flatten(),
        ]
    ).long()
    weight = torch.stack(
        [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
    )
    return idx.to(device), weight.to(device)


def merge_permutation(h: int, w: int, merge: int, device: torch.device) -> Tensor:
    """Row permutation of one frame's ``h*w`` patches into block-merge order.

    ``out = frame_rows[perm]`` reproduces the original
    ``view(h//m, m, w//m, m, -1).permute(0, 2, 1, 3, 4).flatten(0, 3)``
    reorder; multi-frame grids apply the same permutation per frame.
    """
    perm = (
        torch.arange(h * w, device=device)
        .view(h // merge, merge, w // merge, merge)
        .permute(0, 2, 1, 3)
        .reshape(-1)
    )
    return perm


def rope_coords(t: int, h: int, w: int, merge: int, device: torch.device) -> Tensor:
    """(row, col) position ids of one grid in block-merge order, ``[t*h*w, 2]``.

    Matches the per-grid body of the original ``_compute_rotary_pos_emb``.
    """
    merged_h = h // merge
    merged_w = w // merge

    block_rows = torch.arange(merged_h, device=device)
    block_cols = torch.arange(merged_w, device=device)
    intra = torch.arange(merge, device=device)

    row_idx = block_rows[:, None, None, None] * merge + intra[None, None, :, None]
    col_idx = block_cols[None, :, None, None] * merge + intra[None, None, None, :]
    row_idx = row_idx.expand(merged_h, merged_w, merge, merge).reshape(-1)
    col_idx = col_idx.expand(merged_h, merged_w, merge, merge).reshape(-1)

    coords = torch.stack((row_idx, col_idx), dim=-1)
    if t > 1:
        coords = coords.repeat(t, 1)
    return coords


class GridCache:
    """Per-encoder-instance cache of grid-derived tensors, keyed by grid."""

    def __init__(self):
        self._pos: Dict[tuple, Tuple[Tensor, Tensor]] = {}
        self._perm: Dict[tuple, Tensor] = {}
        self._rope: Dict[tuple, Tensor] = {}
        self._psp: Dict[tuple, Tuple[Tensor, int]] = {}
        self._freq: Dict[tuple, Tensor] = {}

    def freqs(self, rot_pos_emb_module, max_hw: int, device) -> Tensor:
        """Frequency lookup table (depends only on the module's fixed dim/theta)."""
        key = (max_hw, str(device))
        if key not in self._freq:
            self._freq[key] = rot_pos_emb_module(max_hw, device=device)
        return self._freq[key]

    def pos(self, h: int, w: int, n: int, device) -> Tuple[Tensor, Tensor]:
        key = (h, w, n, str(device))
        if key not in self._pos:
            self._pos[key] = pos_embed_indices(h, w, n, device)
        return self._pos[key]

    def perm(self, h: int, w: int, merge: int, device) -> Tensor:
        key = (h, w, merge, str(device))
        if key not in self._perm:
            self._perm[key] = merge_permutation(h, w, merge, device)
        return self._perm[key]

    def rope(self, t: int, h: int, w: int, merge: int, device) -> Tensor:
        key = (t, h, w, merge, str(device))
        if key not in self._rope:
            self._rope[key] = rope_coords(t, h, w, merge, device)
        return self._rope[key]

    def packed_seq(self, grids: tuple, device) -> Tuple[Tensor, int]:
        """cu_seqlens tensor and max frame seqlen for one grid tuple."""
        key = (grids, str(device))
        if key not in self._psp:
            seqlens = [int(h) * int(w) for t, h, w in grids for _ in range(int(t))]
            cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
            torch.cumsum(
                torch.tensor(seqlens, dtype=torch.int32, device=device), 0,
                out=cu[1:],
            )
            self._psp[key] = (cu, max(seqlens))
        return self._psp[key]
