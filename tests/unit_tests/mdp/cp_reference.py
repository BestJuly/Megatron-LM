# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Token-level reference for the decoder CP input layouts (tests only).

Written the way the decoder slices a packed THD buffer, independently of
``megatron.core.mdp.cp_partition``: TE's zigzag ``thd_get_partitioned_indices``
per sequence, and ``get_cp_slice_for_thd``'s contiguous row ranges.
"""


def local_positions(cu, total, cp, rank, mode):
    """Global packed positions CP rank ``rank`` owns, in its local token order."""
    if mode == "contiguous":
        local = total // cp
        return list(range(rank * local, (rank + 1) * local))
    out = []
    for start, end in zip(cu[:-1], cu[1:]):
        chunk = (end - start) // (2 * cp)
        for c in (rank, 2 * cp - 1 - rank):
            out.extend(range(start + c * chunk, start + (c + 1) * chunk))
    return out


def owner_by_position(cu, total, cp, mode):
    """The CP rank owning every packed position ``0..total-1``."""
    owner = [None] * total
    for rank in range(cp):
        for position in local_positions(cu, total, cp, rank, mode):
            owner[position] = rank
    return owner
