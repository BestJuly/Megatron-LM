# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Per-image LPT (Longest Processing Time first) BalanceData.

Distributes vision encoder work across encoder-DP ranks at *per-image*
granularity. Each image is treated as an independent unit of work with
FLOPs proxied by attention (O(num_patches^2 * hidden)) + FFN
(O(num_patches * hidden^2)) cost.

Algorithm:
    1. Flatten all images globally across the iter's microbatches into
       (sample_idx, img_idx_within_sample, flops) triples.
    2. Sort descending by FLOPs.
    3. Greedy LPT: pop the min-loaded rank from a min-heap, assign the
       current image to it, push the updated load back. Worst-case bound
       is ``(4/3 - 1/(3m)) * OPT`` (Graham 1969).

Module is torch-free at import time so its pure-Python planning helpers can
run on a login node without GPU / torch dependencies.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Sequence, Tuple


def compute_image_flops(img_size: int, patch: int, hidden: int) -> int:
    """Compute the FLOPs proxy for a single image in the vision encoder.

    The proxy combines quadratic self-attention cost with linear FFN cost:

    .. code-block:: text

        num_patches = (img_size / patch) ** 2
        attention   = num_patches ** 2 * hidden       # self-attention
        ffn         = num_patches * hidden ** 2       # FFN
        total       = attention + ffn

    Args:
        img_size: Square image edge length in pixels (e.g., 256, 512, 1024).
        patch: ViT patch edge length in pixels (e.g., 16).
        hidden: Vision encoder hidden dim (e.g., 1280 for Qwen3-VL).

    Returns:
        Total FLOPs proxy as a positive integer.
    """
    if img_size <= 0 or patch <= 0 or hidden <= 0:
        raise ValueError(
            f"compute_image_flops: all args must be positive, got "
            f"img_size={img_size}, patch={patch}, hidden={hidden}"
        )
    if img_size % patch != 0:
        raise ValueError(
            f"compute_image_flops: img_size ({img_size}) must be divisible " f"by patch ({patch})"
        )
    num_patches = (img_size // patch) ** 2
    attention = num_patches * num_patches * hidden
    ffn = num_patches * hidden * hidden
    return attention + ffn


def compute_patch_grid_flops(t_patches: int, h_patches: int, w_patches: int, hidden: int) -> int:
    """Compute the FLOPs proxy from the post-Conv3D ViT patch grid."""
    if t_patches <= 0 or h_patches <= 0 or w_patches <= 0 or hidden <= 0:
        raise ValueError(
            "compute_patch_grid_flops: all args must be positive, got "
            f"t={t_patches}, h={h_patches}, w={w_patches}, hidden={hidden}"
        )
    num_patches = int(t_patches) * int(h_patches) * int(w_patches)
    attention = num_patches * num_patches * int(hidden)
    ffn = num_patches * int(hidden) * int(hidden)
    return attention + ffn


def lpt_flops_from_grid_rows(
    image_grid_thw_rows: Sequence[Sequence[int]], *, hidden: int
) -> List[int]:
    """Return per-image LPT costs from ``image_grid_thw`` rows.

    ``image_grid_thw`` stores the patch grid after ViT Conv3D patchification.
    With Qwen-style Conv3D, the temporal kernel/stride and spatial patch size
    are already reflected in ``(t, h, w)``. The ``raw_patch`` proxy therefore
    uses ``t * h * w`` directly instead of reconstructing a square image.
    """
    costs: List[int] = []
    for row in image_grid_thw_rows:
        if len(row) != 3:
            raise ValueError("lpt_flops_from_grid_rows requires [t, h, w] rows, got " f"{row!r}")
        t_p, h_p, w_p = int(row[0]), int(row[1]), int(row[2])
        costs.append(compute_patch_grid_flops(t_p, h_p, w_p, hidden))
    return costs


def balance_per_image_lpt_by_flops(
    raw_batch_flops: Sequence[Sequence[int]], num_ranks: int
) -> Dict[int, List[Tuple[int, int]]]:
    """Distribute images across ranks when per-image costs are precomputed."""
    if num_ranks <= 0:
        raise ValueError(
            f"balance_per_image_lpt_by_flops: num_ranks must be positive, got " f"{num_ranks}"
        )

    flat: List[Tuple[int, int, int]] = []
    for sample_idx, sample in enumerate(raw_batch_flops):
        for img_idx, flops in enumerate(sample):
            flops = int(flops)
            if flops < 0:
                raise ValueError(
                    "balance_per_image_lpt_by_flops: flops must be " f"non-negative, got {flops}"
                )
            flat.append((flops, sample_idx, img_idx))

    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    heap: List[Tuple[int, int]] = [(0, r) for r in range(num_ranks)]
    heapq.heapify(heap)
    assignment: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(num_ranks)}
    for flops, sample_idx, img_idx in flat:
        load, rank = heapq.heappop(heap)
        assignment[rank].append((sample_idx, img_idx))
        heapq.heappush(heap, (load + flops, rank))

    return assignment


def balance_per_image_lpt(
    raw_batch: Sequence[Sequence[Tuple[int, int, int]]], num_ranks: int
) -> Dict[int, List[Tuple[int, int]]]:
    """Distribute images across ``num_ranks`` via greedy min-heap LPT.

    Args:
        raw_batch: Per-sample list of image descriptors. Each sample is a
            sequence of ``(img_size, patch, hidden)`` triples (one per
            image in the sample). Sample index = position in ``raw_batch``.
        num_ranks: Number of encoder-DP ranks to distribute work across.

    Returns:
        ``rank_assignment``: dict ``{rank: [(sample_idx, img_idx), ...]}``
        mapping each rank to the list of (sample_idx, img_idx_within_sample)
        pairs that rank should run the encoder forward on. Order within
        each rank's list is descending-FLOPs (LPT assignment order).

    Notes:
        * Empty samples (no images) contribute nothing.
        * Empty raw_batch -> every rank gets an empty list.
        * Worst-case load imbalance bound: ``(4/3 - 1/(3m)) * OPT`` -
          Graham 1969 LPT theorem.
    """
    if num_ranks <= 0:
        raise ValueError(f"balance_per_image_lpt: num_ranks must be positive, got " f"{num_ranks}")

    # Step 1 - compute image costs.
    raw_batch_flops: List[List[int]] = []
    for sample_idx, sample in enumerate(raw_batch):
        sample_flops: List[int] = []
        for img_idx, descriptor in enumerate(sample):
            if len(descriptor) != 3:
                raise ValueError(
                    f"balance_per_image_lpt: each image descriptor must be "
                    f"(img_size, patch, hidden); sample {sample_idx} img "
                    f"{img_idx} got {descriptor!r}"
                )
            img_size, patch, hidden = descriptor
            flops = compute_image_flops(img_size, patch, hidden)
            sample_flops.append(flops)
        raw_batch_flops.append(sample_flops)

    return balance_per_image_lpt_by_flops(raw_batch_flops, num_ranks)
