# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic image-owner assignment helpers for MDP metadata loading."""

from __future__ import annotations

from examples.multimodal_dev.balance_data import (
    balance_per_image_lpt_by_flops,
    lpt_flops_from_grid_rows,
)


def balance_window_lpt(costs_by_item, num_ranks):
    """Assign image jobs to ranks with largest-processing-time first.

    Args:
        costs_by_item: Nested sequence of image costs. Outer index is the
            packed item/sample index; inner index is the image index within it.
        num_ranks: Number of MDP owner ranks.

    Returns:
        ``(assignment, loads)`` where assignment maps rank to
        ``(item_idx, image_idx)`` pairs.
    """
    if int(num_ranks) <= 0:
        raise ValueError(f"num_ranks must be positive, got {num_ranks}")

    assignment = balance_per_image_lpt_by_flops(costs_by_item, int(num_ranks))
    loads = [
        sum(float(costs_by_item[item_idx][image_idx]) for item_idx, image_idx in items)
        for items in assignment.values()
    ]

    return assignment, loads


def assign_images_lpt(costs_by_item, num_ranks, *, across_items):
    """Assign images either per item or across the complete item window."""
    if across_items:
        assignment, _ = balance_window_lpt(costs_by_item, num_ranks)
        return assignment

    assignment = {rank: [] for rank in range(int(num_ranks))}
    for item_idx, costs in enumerate(costs_by_item):
        item_assignment, _ = balance_window_lpt([costs], num_ranks)
        for rank, items in item_assignment.items():
            assignment[rank].extend(
                (int(item_idx), int(image_idx)) for _single_item_idx, image_idx in items
            )
    return assignment


def image_costs_from_grid(rows, *, hidden_size):
    """Return the image-cost proxy used by MDP owner planning."""
    return lpt_flops_from_grid_rows(rows, hidden=int(hidden_size))


def vision_rows_from_grid(grid_thw, spatial_merge_size=2):
    t, h, w = [int(x) for x in grid_thw]
    merge = max(int(spatial_merge_size), 1)
    return (t * h * w) // (merge * merge)
