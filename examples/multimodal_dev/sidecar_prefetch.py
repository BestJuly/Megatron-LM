# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small helpers for MDP sidecar microbatch prefetch windows."""

from __future__ import annotations


def validate_fused_vision_window(fused_window: bool, max_sequence_length: int) -> bool:
    """Validate the fused-window switch against its raw-patch cap."""
    if bool(fused_window) and int(max_sequence_length) == 0:
        raise ValueError(
            "--mdp-fused-vision-window requires a non-zero "
            "--mdp-vision-encoder-max-sequence-length"
        )
    return bool(fused_window)


def sidecar_prefetch_window_count(
    fused_window: bool, *, current_microbatch: int | None, num_microbatches: int | None
) -> int:
    """Return how many microbatches this sidecar hook call should precompute.

    With the fused window off, every hook call prepares exactly one
    microbatch. With it on, the first hook call of the optimization step
    prepares the full microbatch window and the rest prepare none.
    """
    if not fused_window or current_microbatch is None or num_microbatches is None:
        return 1
    return max(1, int(num_microbatches)) if int(current_microbatch) == 0 else 0


def vision_length_pack_plan(lengths: list[int], max_rows: int) -> list[list[int]]:
    """Pack image jobs by vision sequence length using first-fit decreasing."""
    if max_rows <= 0:
        return [list(range(len(lengths)))] if lengths else []
    jobs = sorted(
        ((idx, int(length)) for idx, length in enumerate(lengths)),
        key=lambda job: job[1],
        reverse=True,
    )
    packs: list[list[int]] = []
    remaining: list[int] = []
    for idx, length in jobs:
        placed = False
        for pack_idx, capacity in enumerate(remaining):
            if length <= capacity:
                packs[pack_idx].append(idx)
                remaining[pack_idx] -= length
                placed = True
                break
        if not placed:
            packs.append([idx])
            remaining.append(int(max_rows) - length)
    return packs


def image_vision_pack_plan(
    lengths_by_microbatch: list[list[int]], max_sequence_length: int
) -> list[list[int]]:
    """Pack flattened window images by vision sequence length."""
    return vision_length_pack_plan(
        [int(length) for lengths in lengths_by_microbatch for length in lengths],
        int(max_sequence_length),
    )
