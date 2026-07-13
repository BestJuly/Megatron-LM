# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MDP rank-layout and process-group helpers.

The PP x CP MDP path uses Megatron's ``RankGenerator`` for production rank
lists.  The torch-free mirror below keeps that layout independently testable
on login nodes.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def get_parallel_order(args) -> str:
    """Return the rank order used by Megatron's model-parallel initializer."""
    if bool(getattr(args, "use_tp_pp_dp_mapping", False)):
        return "tp-cp-ep-pp-dp"
    return "tp-cp-ep-dp-pp"


def _prefix_product(values: Sequence[int], init: int = 1) -> List[int]:
    products = [int(init)]
    running = int(init)
    for value in values:
        running *= int(value)
        products.append(running)
    return products


def _decompose_index(index: int, shape: Sequence[int]) -> List[int]:
    stride = _prefix_product(shape)
    return [
        (int(index) // int(divisor)) % int(size)
        for size, divisor in zip(shape, stride)
    ]


def _inner_product(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(int(a) * int(b) for a, b in zip(left, right))


def _generate_masked_rank_groups(
    *,
    world_size: int,
    parallel_size: Sequence[int],
    mask: Sequence[bool],
) -> List[List[int]]:
    """Torch-free copy of Megatron's orthogonal rank-group arithmetic."""
    global_stride = _prefix_product(parallel_size)
    masked_shape = [size for size, selected in zip(parallel_size, mask) if selected]
    unmasked_shape = [size for size, selected in zip(parallel_size, mask) if not selected]
    masked_stride = [stride for stride, selected in zip(global_stride, mask) if selected]
    unmasked_stride = [stride for stride, selected in zip(global_stride, mask) if not selected]

    group_size = _prefix_product(masked_shape)[-1]
    if group_size <= 0:
        raise ValueError("rank-group mask produced an empty group")
    num_groups = int(world_size) // int(group_size)

    groups: List[List[int]] = []
    for group_index in range(num_groups):
        group_idx = _decompose_index(group_index, unmasked_shape)
        ranks = []
        for rank_in_group in range(group_size):
            rank_idx = _decompose_index(rank_in_group, masked_shape)
            ranks.append(
                _inner_product(rank_idx, masked_stride)
                + _inner_product(group_idx, unmasked_stride)
            )
        groups.append(ranks)
    return groups


def _normalize_parallel_order(order: str, sizes: dict) -> List[str]:
    tokens = order.split("-")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"parallel order contains duplicates: {order!r}")
    for name, size in sizes.items():
        if name not in tokens:
            if int(size) != 1:
                raise ValueError(
                    f"parallel order {order!r} omits non-unit dimension "
                    f"{name}={size}"
                )
            tokens.append(name)
    for token in tokens:
        if token not in sizes:
            raise ValueError(f"unknown parallel dimension {token!r}")
    return tokens


def compute_pp_cp_inner_dp_layout(
    world_size: int,
    tp_size: int,
    cp_size: int,
    pp_size: int,
    order: str = "tp-cp-ep-dp-pp",
) -> List[List[int]]:
    """Compute PP x CP InnerDP groups at fixed TP and outer-DP coordinates."""
    for name, value in (
        ("world_size", world_size),
        ("tp_size", tp_size),
        ("cp_size", cp_size),
        ("pp_size", pp_size),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be positive int, got {value!r}")

    inner_size = int(cp_size) * int(pp_size)
    model_size = int(tp_size) * inner_size
    if int(world_size) % model_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"tp_size * cp_size * pp_size ({model_size})"
        )

    sizes = {
        "tp": int(tp_size),
        "cp": int(cp_size),
        "ep": 1,
        "dp": int(world_size) // model_size,
        "pp": int(pp_size),
    }
    tokens = _normalize_parallel_order(order, sizes)
    groups = _generate_masked_rank_groups(
        world_size=int(world_size),
        parallel_size=[sizes[token] for token in tokens],
        mask=[token in ("pp", "cp") for token in tokens],
    )

    covered = sorted(rank for group in groups for rank in group)
    if covered != list(range(int(world_size))):
        raise RuntimeError(
            "PPxCP InnerDP layout must cover every rank exactly once; got "
            f"{covered[:16]}... len={len(covered)} world={world_size}"
        )
    for group in groups:
        if len(group) != inner_size:
            raise RuntimeError(
                f"PPxCP InnerDP group {group} has size {len(group)}, "
                f"expected {inner_size}"
            )
    return groups


def find_pp_cp_inner_dp_group_for_rank(
    rank: int,
    pp_cp_groups: Sequence[Sequence[int]],
) -> Tuple[List[int], int]:
    """Return ``(group, local_index)`` for ``rank`` in a PP x CP sheet."""
    for group in pp_cp_groups:
        if rank in group:
            return list(group), group.index(rank)
    raise ValueError(
        f"rank {rank} is not in any PPxCP group {list(pp_cp_groups)}"
    )


def get_pp_cp_local_rank(args, pp_size: int, cp_size: int) -> int:
    """Return this process's local rank in its deterministic PP x CP group."""
    groups = compute_pp_cp_inner_dp_layout(
        world_size=int(getattr(args, "world_size", 1)),
        tp_size=int(getattr(args, "tensor_model_parallel_size", 1)),
        cp_size=int(cp_size),
        pp_size=int(pp_size),
        order=get_parallel_order(args),
    )
    _, local_rank = find_pp_cp_inner_dp_group_for_rank(
        int(getattr(args, "rank", 0)), groups
    )
    return int(local_rank)


def _new_group_with_current_device(torch_module, **kwargs):
    """Bind NCCL groups to the current CUDA device when the API supports it."""
    backend = kwargs.get("backend")
    backend_name = str(backend).lower() if backend is not None else None
    if torch_module.cuda.is_available() and backend_name in (None, "nccl"):
        kwargs.setdefault(
            "device_id",
            torch_module.device("cuda", torch_module.cuda.current_device()),
        )
    return torch_module.distributed.new_group(**kwargs)


def build_pp_cp_inner_dp_group(
    *,
    pp_cp_groups: Sequence[Sequence[int]],
    this_rank: int,
    backend=None,
):
    """Build all PP x CP groups in a common order and return this rank's group."""
    import torch

    local_group = None
    local_ranks = None
    local_index = None
    for group in pp_cp_groups:
        ranks = [int(rank) for rank in group]
        process_group = _new_group_with_current_device(
            torch, ranks=ranks, backend=backend
        )
        if int(this_rank) in ranks:
            local_group = process_group
            local_ranks = ranks
            local_index = ranks.index(int(this_rank))
    if local_group is None or local_ranks is None or local_index is None:
        raise RuntimeError(
            f"rank {this_rank} did not match any PPxCP group {list(pp_cp_groups)}"
        )
    return local_group, local_ranks, local_index


def build_local_process_group(
    *,
    rank_groups: Sequence[Sequence[int]],
    this_rank: int,
    backend=None,
):
    """Build same-ordered process groups and return this rank's group."""
    import torch

    local_group = None
    local_ranks = None
    local_index = None
    for group in rank_groups:
        ranks = [int(rank) for rank in group]
        process_group = _new_group_with_current_device(
            torch, ranks=ranks, backend=backend
        )
        if int(this_rank) in ranks:
            local_group = process_group
            local_ranks = ranks
            local_index = ranks.index(int(this_rank))
    if local_group is None or local_ranks is None or local_index is None:
        raise RuntimeError(
            f"rank {this_rank} did not match any rank group {list(rank_groups)}"
        )
    return local_group, local_ranks, local_index
