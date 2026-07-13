# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Sequence-axis ModalityBridge for MDP vision embeddings.

Companion to ``balance_data.py``: after BalanceData has scheduled which
images each rank owns, and after each rank has run the vision encoder
locally on its assigned images, ModalityBridge gathers the resulting
embeddings across the active MDP group for assembly into the LLM input
sequence.

* **Sequence-axis only.** Output flows into the LLM's existing
  CP-along-seq path; no hidden-axis scatter is introduced.
* **All_gather metadata first.** Each rank's per-image embedding count
  varies post-LPT, so we metadata-gather the shapes before the data
  all_gather to size the receive buffers.
* **Canonical-order reconstruction.** The (sample_idx, img_idx) pairs
  from ``balance_per_image_lpt`` carry the source position; we sort
  the gathered embeddings back to per-sample image-token-block order
  for the downstream ``_scatter_vision_embeddings`` consumer.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.training.utils import get_nvtx_range

_nvtx_range = get_nvtx_range()


def get_mdp_images_to_language_group(model):
    """Return the active MDP images_to_language process group for *model*."""
    return getattr(model, "_mdp_inner_dp_group", None)


def _gather_padded_sequence_parallel_region(padded_local, group):
    """Gather a 2-D padded token tensor with Megatron's AG/RS autograd pair.

    This is the same mathematical pattern Megatron uses for sequence
    parallelism: forward gathers and concatenates along dim 0; backward
    reduce-scatters the dim-0 gradient. The local wrapper exists only to pin
    the MDP contract: ``padded_local`` is ``[max_count, hidden]`` and the
    process group is the active InnerDP group, not necessarily the tensor
    parallel group.
    """
    if padded_local.dim() != 2:
        raise ValueError(
            "_gather_padded_sequence_parallel_region expects a 2-D tensor "
            f"[max_count, hidden], got {tuple(padded_local.shape)}"
        )

    with _nvtx_range("MDPBridge/forward_all_gather"):
        gathered = gather_from_sequence_parallel_region(
            padded_local.contiguous(), tensor_parallel_output_grad=True, group=group
        )
        return gathered


# ---------------------------------------------------------------------------
# Pure-python order helpers (torch-free, unit-testable)
# ---------------------------------------------------------------------------


def build_global_ordering(
    rank_assignment: Dict[int, List[Tuple[int, int]]]
) -> List[Tuple[int, int, int, int]]:
    """Flatten ``rank_assignment`` into a global (sample_idx, img_idx)
    sequence in *gather order*.

    The all_gather over an encoder-DP group produces, on each receiving
    rank, a concatenation ``[rank0_payload | rank1_payload | ... |
    rankN_payload]``. ``build_global_ordering`` returns the matching
    sequence of ``(rank, local_idx, sample_idx, img_idx)`` tuples so the
    consumer can map each post-gather slot back to its source.

    Returns:
        A list of length sum(len(v) for v in rank_assignment.values()),
        each entry is ``(rank, local_idx_within_rank, sample_idx,
        img_idx_within_sample)``.
    """
    ordering: List[Tuple[int, int, int, int]] = []
    for rank in sorted(rank_assignment.keys()):
        for local_idx, (sample_idx, img_idx) in enumerate(rank_assignment[rank]):
            ordering.append((rank, local_idx, sample_idx, img_idx))
    return ordering


def reconstruct_canonical_order(
    gathered: Sequence[Any], rank_assignment: Dict[int, List[Tuple[int, int]]]
) -> List[Any]:
    """Reorder gathered per-image payloads into canonical per-sample
    image-token-block order.

    "Canonical order" = sort by (sample_idx, img_idx_within_sample). This
    matches the order expected by ``MultimodalModel._scatter_vision_embeddings``:
    image tokens are scattered into the language input in the order
    they appear in the raw_batch.

    Args:
        gathered: Sequence of per-image payloads (e.g., embedding
            tensors) in *gather order*; element ``i`` corresponds to
            ``build_global_ordering(rank_assignment)[i]``.
        rank_assignment: The dict returned by
            :func:`balance_data.balance_per_image_lpt`.

    Returns:
        Same payloads sorted so that element ``i`` corresponds to the
        i-th (sample_idx, img_idx) pair in canonical order.
    """
    ordering = build_global_ordering(rank_assignment)
    if len(ordering) != len(gathered):
        raise ValueError(
            f"reconstruct_canonical_order: ordering length "
            f"{len(ordering)} != gathered length {len(gathered)}"
        )
    # Pair each payload with its (sample_idx, img_idx) tag, sort, drop tag.
    tagged = [
        (sample_idx, img_idx, payload)
        for (rank, local_idx, sample_idx, img_idx), payload in zip(ordering, gathered)
    ]
    tagged.sort(key=lambda t: (t[0], t[1]))
    return [payload for (_s, _i, payload) in tagged]


def gather_order_is_canonical(rank_assignment: Dict[int, List[Tuple[int, int]]]) -> bool:
    """Return True when rank-major gather order already matches image order."""
    last = None
    for rank in sorted(rank_assignment.keys()):
        for sample_idx, img_idx in rank_assignment[rank]:
            current = (int(sample_idx), int(img_idx))
            if last is not None and current < last:
                return False
            last = current
    return True


def _iter_assignment_row_counts(
    rank_assignment: Dict[int, List[Tuple[int, int]]], global_per_image_row_counts: Sequence[int]
) -> List[Tuple[int, int, int, int]]:
    """Return gather-order row counts from assignment metadata.

    The second assignment field is the image index into
    ``global_per_image_row_counts``. For single microbatches this is the local
    image index while the first field remains the sample/order tag.
    """
    rows: List[Tuple[int, int, int, int]] = []
    for rank, local_idx, sample_idx, img_idx in build_global_ordering(rank_assignment):
        img_idx = int(img_idx)
        if img_idx < 0 or img_idx >= len(global_per_image_row_counts):
            raise ValueError(
                "MDP row-count metadata: image index "
                f"{img_idx} from sample/order tag {sample_idx} is outside "
                "global_per_image_row_counts length "
                f"{len(global_per_image_row_counts)}."
            )
        count = int(global_per_image_row_counts[img_idx])
        if count < 0:
            raise ValueError(
                "MDP row-count metadata: negative row " f"count {count} for image index {img_idx}."
            )
        rows.append((int(rank), int(local_idx), img_idx, count))
    return rows


def gather_order_row_counts_from_assignment(
    rank_assignment: Dict[int, List[Tuple[int, int]]], global_per_image_row_counts: Sequence[int]
) -> List[int]:
    """Build gather-order row counts without a per-rank count collective."""
    return [
        count
        for _rank, _local_idx, _img_idx, count in _iter_assignment_row_counts(
            rank_assignment, global_per_image_row_counts
        )
    ]


def per_rank_row_counts_from_assignment(
    rank_assignment: Dict[int, List[Tuple[int, int]]],
    global_per_image_row_counts: Sequence[int],
    world_size: int,
) -> List[int]:
    """Return total embedding rows contributed by each gather rank.

    This is the per-rank companion to
    :func:`gather_order_row_counts_from_assignment`. It lets the data gather
    compute its pad target from BalanceData metadata instead of doing a
    fixed-size shape collective first.
    """
    per_rank = [0 for _ in range(int(world_size))]
    for rank, _local_idx, _img_idx, count in _iter_assignment_row_counts(
        rank_assignment, global_per_image_row_counts
    ):
        if rank < 0 or rank >= int(world_size):
            raise ValueError(
                "per_rank_row_counts_from_assignment: rank "
                f"{rank} outside world_size={world_size}."
            )
        per_rank[rank] += count
    return per_rank


def _reorder_by_gather_order_row_counts(
    gathered_embeddings,
    rank_assignment: Dict[int, List[Tuple[int, int]]],
    gather_order_row_counts: Sequence[int],
    *,
    error_prefix: str,
):
    expected_total_rows = sum(int(rc) for rc in gather_order_row_counts)
    actual_total_rows = int(gathered_embeddings.shape[0])
    if expected_total_rows != actual_total_rows:
        raise ValueError(
            f"{error_prefix}: row-count sum ({expected_total_rows}) does "
            f"not match flat gathered tensor row count ({actual_total_rows})."
        )
    if gather_order_is_canonical(rank_assignment):
        return gathered_embeddings

    chunks: List[Any] = []
    offset = 0
    for row_count in gather_order_row_counts:
        row_count = int(row_count)
        chunks.append(gathered_embeddings[offset : offset + row_count])
        offset += row_count
    canonical_chunks = reconstruct_canonical_order(chunks, rank_assignment)
    return torch.cat(canonical_chunks, dim=0)


# ---------------------------------------------------------------------------
# CP-local image-row helpers
# ---------------------------------------------------------------------------


def cp_local_image_positions_and_row_ids_from_cpu_metadata(
    *, image_positions, input_shape, cp_size: int, cp_rank: int, cu_seqlens_padded=None
):
    """Build a CP-local image scatter plan from loader-side CPU metadata."""
    if input_shape is None:
        raise RuntimeError("CPU metadata image plan requires input_shape")
    shape = tuple(int(x) for x in input_shape)
    if len(shape) == 1:
        batch = 1
        seq_len = int(shape[0])
    elif len(shape) == 2:
        batch = int(shape[0])
        seq_len = int(shape[1])
    else:
        raise RuntimeError(
            "CPU metadata image plan expects input_shape [S] or [B,S], " f"got {shape}"
        )
    if batch <= 0 or seq_len < 0:
        raise RuntimeError(f"invalid input_shape for CPU image plan: {shape}")

    positions_full = [int(pos) for pos in (image_positions or [])]
    positions_are_sorted = all(
        positions_full[idx] <= positions_full[idx + 1] for idx in range(len(positions_full) - 1)
    )
    if positions_are_sorted:
        sorted_positions = positions_full
        sorted_row_ids = None
    else:
        sorted_pos_rows = sorted((pos, row) for row, pos in enumerate(positions_full))
        sorted_positions = [pos for pos, _row in sorted_pos_rows]
        sorted_row_ids = [row for _pos, row in sorted_pos_rows]
    chunk_count = 2 * int(cp_size)
    selected_chunks = (int(cp_rank), chunk_count - int(cp_rank) - 1)
    local_positions: List[int] = []
    local_row_ids: List[int] = []

    def append_image_rows_in_range(begin: int, end: int, local_base: int) -> None:
        idx = bisect_left(sorted_positions, int(begin))
        while idx < len(sorted_positions):
            full_pos = sorted_positions[idx]
            if full_pos >= int(end):
                break
            local_positions.append(int(local_base) + int(full_pos) - int(begin))
            row_id = idx if sorted_row_ids is None else sorted_row_ids[idx]
            local_row_ids.append(int(row_id))
            idx += 1

    if cu_seqlens_padded is not None:
        if batch != 1:
            raise RuntimeError(
                "CPU THD metadata image plan expects packed B=1 input_ids, "
                f"got input_shape={shape}"
            )
        cu = [int(x) for x in cu_seqlens_padded]
        local_offset = 0
        for start, end in zip(cu[:-1], cu[1:]):
            start = max(0, min(int(start), seq_len))
            end = max(start, min(int(end), seq_len))
            length = end - start
            if length == 0:
                continue
            if length % chunk_count != 0:
                raise RuntimeError(
                    "CPU THD metadata image plan requires each padded "
                    f"sequence length to divide 2*cp_size={chunk_count}; "
                    f"got segment length {length} from {start}->{end}"
                )
            chunk = length // chunk_count
            for chunk_id in selected_chunks:
                begin = start + int(chunk_id) * chunk
                append_image_rows_in_range(begin, begin + chunk, local_offset)
                local_offset += chunk
    else:
        if seq_len % chunk_count != 0:
            raise RuntimeError(
                "CPU BSHD metadata image plan requires seq_len to divide "
                f"2*cp_size={chunk_count}; got seq_len={seq_len}"
            )
        chunk = seq_len // chunk_count
        local_seq_len = 2 * chunk
        for b_idx in range(batch):
            local_batch_base = b_idx * local_seq_len
            full_batch_base = b_idx * seq_len
            for local_chunk_idx, chunk_id in enumerate(selected_chunks):
                begin = full_batch_base + int(chunk_id) * chunk
                append_image_rows_in_range(
                    begin, begin + chunk, local_batch_base + local_chunk_idx * chunk
                )

    return (
        torch.tensor(local_positions, dtype=torch.int64),
        torch.tensor(local_row_ids, dtype=torch.int64),
        len(positions_full),
    )


def select_vision_rows_for_cp_rank(
    vision_embeddings, cp_local_row_ids, *, validate_indices: bool = True
):
    """Return vision rows consumed by one CP-local sequence shard."""
    if vision_embeddings.dim() != 2:
        raise RuntimeError(
            "select_vision_rows_for_cp_rank: vision_embeddings must be "
            f"2-D [rows, hidden]; got shape {tuple(vision_embeddings.shape)}"
        )
    row_ids = (
        cp_local_row_ids.to(dtype=torch.int64, device=vision_embeddings.device)
        .contiguous()
        .view(-1)
    )
    if row_ids.numel() == 0:
        return torch.empty(
            (0, vision_embeddings.shape[1]),
            dtype=vision_embeddings.dtype,
            device=vision_embeddings.device,
        )
    if validate_indices:
        min_id = int(row_ids.min().item())
        max_id = int(row_ids.max().item())
        if min_id < 0 or max_id >= int(vision_embeddings.shape[0]):
            raise RuntimeError(
                "select_vision_rows_for_cp_rank: cp_local_row_ids out of "
                "range for vision_embeddings rows="
                f"{vision_embeddings.shape[0]} (min={min_id}, max={max_id})"
            )
    return vision_embeddings.index_select(0, row_ids)


def scatter_vision_rows_at_positions(text_embeddings, vision_embeddings, image_positions):
    """Scatter vision rows into ``[S,B,H]`` embeddings by flat token indices.

    This is equivalent to the model's historical ``masked_scatter`` over
    ``[B,S,H]`` but avoids materializing an expanded boolean mask and uses the
    already-known image-token positions.
    """
    if text_embeddings.dim() != 3:
        raise RuntimeError(
            "scatter_vision_rows_at_positions: text_embeddings must be "
            f"3-D [S,B,H]; got shape {tuple(text_embeddings.shape)}"
        )
    if vision_embeddings.dim() != 2:
        raise RuntimeError(
            "scatter_vision_rows_at_positions: vision_embeddings must be "
            f"2-D [rows,H]; got shape {tuple(vision_embeddings.shape)}"
        )
    positions = (
        image_positions.to(dtype=torch.int64, device=text_embeddings.device).contiguous().view(-1)
    )
    needed = int(positions.numel())
    if int(vision_embeddings.shape[0]) < needed:
        raise RuntimeError(
            "scatter_vision_rows_at_positions: not enough vision rows for "
            f"image positions ({vision_embeddings.shape[0]} < {needed})"
        )
    combined = text_embeddings.transpose(0, 1).contiguous()
    flat = combined.view(-1, combined.shape[-1])
    source = vision_embeddings[:needed].to(dtype=flat.dtype, device=flat.device)
    flat = flat.index_copy(0, positions, source)
    return flat.view_as(combined).transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Torch-dependent gather helpers
# ---------------------------------------------------------------------------


def gather_to_inner_dp_zero(
    local_embeddings,
    rank_assignment: Dict[int, List[Tuple[int, int]]],
    encoder_dp_group,
    global_per_image_row_counts=None,
    local_zero_dep=None,
    return_zero_dependency_only: bool = False,
):
    """All-gather per-image embeddings across an InnerDP process group.

    Native NCCL ``all_gather`` requires every tensor in ``tensor_list`` to
    match the input tensor shape, while per-image LPT produces unequal
    per-rank token counts. This function uses a pad + trim pattern:

    1. Determine per-rank counts from BalanceData metadata when available,
       otherwise all-gather the per-rank counts (fixed-size int64 tensor).
    2. Compute ``max_count = max(counts)``.
    3. Pad ``local_embeddings`` to ``[max_count, hidden]`` with zeros.
    4. All-gather the padded tensor (now equal shape on every rank).
    5. Trim each rank's slice back to its actual ``counts[r]``.
    6. Concatenate in rank order (gather order).

    Args:
        local_embeddings: ``Tensor[local_image_tokens, hidden]`` on this
            rank. Tokens for the local rank's images concatenated along
            the sequence axis. ``local_image_tokens`` MAY be zero.
        rank_assignment: dict from ``balance_per_image_lpt``; reserved
            for metadata-derived count checks.
        encoder_dp_group: NCCL ``ProcessGroup`` over the CP ranks.
        global_per_image_row_counts: optional CPU/list metadata with the
            projected vision rows for every image in the flattened
            microbatch. When present, avoids the shape all_gather.
        local_zero_dep: optional scalar zero dependency attached to padding
            rows so empty local shards can still keep trainable local modules
            in the autograd graph.
        return_zero_dependency_only: when True, return a scalar zero
            dependency on the gathered tensor immediately after the
            collective. Non-consuming PP stages need the collective autograd
            edge, but not canonical gathered rows.
    Returns:
        Tensor ``[total_image_tokens, hidden]`` on every group member, or a
        scalar zero dependency when ``return_zero_dependency_only`` is True.
    """
    world_size = dist.get_world_size(group=encoder_dp_group)
    device = local_embeddings.device
    dtype = local_embeddings.dtype

    # Determine hidden dim - must be consistent across ranks. We use
    # the local rank's shape when local_embeddings has >= 1 image; when
    # empty (shape [0]), we still need a hidden dim for the padded
    # buffer. Fall back to local_embeddings.shape[-1] (works for both
    # [N, H] and [0, H]); raise if local_embeddings is rank-1.
    if local_embeddings.dim() != 2:
        raise ValueError(
            "gather_to_inner_dp_zero: local_embeddings must be 2-D "
            f"[N, hidden]; got shape {tuple(local_embeddings.shape)}"
        )
    hidden = local_embeddings.shape[1]

    # Step 1 - collect per-rank token counts and hidden sizes. Metadata-only
    # MDP batches already carry per-image row counts, so production runs
    # can avoid this small shape all_gather. The fallback keeps the defensive
    # hidden-size validation for paths without metadata.
    if global_per_image_row_counts is not None:
        counts = per_rank_row_counts_from_assignment(
            rank_assignment, global_per_image_row_counts, world_size
        )
        group_rank = dist.get_rank(group=encoder_dp_group)
        expected_local_rows = int(counts[int(group_rank)])
        actual_local_rows = int(local_embeddings.shape[0])
        if actual_local_rows != expected_local_rows:
            raise RuntimeError(
                "gather_to_inner_dp_zero: metadata row count mismatch on "
                f"group_rank={group_rank}: local_embeddings has "
                f"{actual_local_rows} rows but BalanceData metadata expects "
                f"{expected_local_rows}. counts={counts}"
            )
    else:
        local_shape = torch.tensor(
            [local_embeddings.shape[0], hidden], dtype=torch.int64, device=device
        )
        shape_list = [torch.zeros(2, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(shape_list, local_shape, group=encoder_dp_group)
        counts = [int(t[0].item()) for t in shape_list]
        hidden_sizes = [int(t[1].item()) for t in shape_list]
        if len(set(hidden_sizes)) != 1:
            raise RuntimeError(
                "gather_to_inner_dp_zero: hidden size mismatch across gather "
                f"group: hidden_sizes={hidden_sizes}, counts={counts}. Empty "
                "local image shards must use the projected language hidden size."
            )

    # Step 2 - pad target.
    max_count = max(counts) if counts else 0
    if max_count == 0:
        # All ranks have zero images this iter (degenerate but legal -
        # e.g., text-only batch); return an empty [0, hidden] tensor on
        # every rank.
        empty = torch.empty((0, hidden), dtype=dtype, device=device)
        if return_zero_dependency_only:
            zero = local_embeddings.reshape(-1)[:0].sum() * 0.0
            if local_zero_dep is not None:
                zero = zero + local_zero_dep.to(dtype=dtype) * 0.0
            return zero
        if local_zero_dep is not None:
            empty = empty + local_zero_dep.to(dtype=dtype) * 0.0
        return empty

    # Step 3 - pad ``local_embeddings`` to [max_count, hidden].
    local_n = local_embeddings.shape[0]
    if local_n < max_count:
        pad = torch.zeros((max_count - local_n, hidden), dtype=dtype, device=device)
        if local_zero_dep is not None:
            pad = pad + local_zero_dep.to(dtype=pad.dtype)
        padded_local = torch.cat([local_embeddings, pad], dim=0).contiguous()
    else:
        # local_n == max_count already; no pad needed.
        padded_local = local_embeddings.contiguous()
        if local_zero_dep is not None:
            padded_local = padded_local + local_zero_dep.to(dtype=padded_local.dtype)

    # Step 4 - all_gather the equal-shape padded tensor with explicit
    # autograd support.
    padded_cat = _gather_padded_sequence_parallel_region(padded_local, encoder_dp_group)
    if return_zero_dependency_only:
        return padded_cat.reshape(-1)[:1].sum() * 0.0

    padded_out = [padded_cat.narrow(0, r * max_count, max_count) for r in range(world_size)]

    # Step 5 - trim each rank's slice back to its actual count.
    trimmed = [padded_out[r][: counts[r]] for r in range(world_size)]

    # Step 6 - concatenate in rank order (gather order).
    gathered = torch.cat(trimmed, dim=0)
    return gathered


# ---------------------------------------------------------------------------
# Canonical-order reorder for gathered embeddings (variable rows per image)
# ---------------------------------------------------------------------------


def _local_row_count_list(local_per_image_row_counts) -> List[int]:
    return [
        int(x)
        for x in local_per_image_row_counts.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
        .view(-1)
        .tolist()
    ]


def _metadata_gather_order_row_counts(
    rank_assignment: Dict[int, List[Tuple[int, int]]],
    global_per_image_row_counts,
    local_per_image_row_counts,
    my_rank_in_group: int,
) -> List[int]:
    expected_local = []
    for sample_idx, img_idx in rank_assignment.get(my_rank_in_group, []) or []:
        img_idx = int(img_idx)
        if img_idx < 0 or img_idx >= len(global_per_image_row_counts):
            raise ValueError(
                "reorder_gathered_embeddings: image index "
                f"{img_idx} from sample/order tag {sample_idx} is outside "
                "global_per_image_row_counts length "
                f"{len(global_per_image_row_counts)}."
            )
        expected_local.append(int(global_per_image_row_counts[img_idx]))

    if local_per_image_row_counts is not None:
        actual_local = _local_row_count_list(local_per_image_row_counts)
        if actual_local != expected_local:
            raise ValueError(
                "reorder_gathered_embeddings: local row counts "
                f"{actual_local} do not match metadata row counts "
                f"{expected_local} for rank {my_rank_in_group}."
            )

    return gather_order_row_counts_from_assignment(rank_assignment, global_per_image_row_counts)


def _all_gather_rank_image_counts(n_local: int, world_size: int, device, group):
    local_n_tensor = torch.tensor([n_local], dtype=torch.int64, device=device)
    n_list = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(n_list, local_n_tensor, group=group)
    return [int(t.item()) for t in n_list]


def _fallback_gather_order_row_counts(
    local_per_image_row_counts, n_local: int, n_per_rank: Sequence[int], device, group
) -> List[int]:
    max_n = max(n_per_rank) if n_per_rank else 0
    if max_n == 0:
        return []
    if local_per_image_row_counts.dim() != 1:
        raise ValueError(
            "reorder_gathered_embeddings: local_per_image_row_counts must "
            f"be 1-D; got shape {tuple(local_per_image_row_counts.shape)}"
        )

    local_counts_i64 = local_per_image_row_counts.to(dtype=torch.int64, device=device)
    if n_local < max_n:
        pad = torch.zeros((max_n - n_local,), dtype=torch.int64, device=device)
        padded_counts = torch.cat([local_counts_i64, pad], dim=0)
    else:
        padded_counts = local_counts_i64

    world_size = len(n_per_rank)
    padded_out = [
        torch.zeros((max_n,), dtype=torch.int64, device=device) for _ in range(world_size)
    ]
    dist.all_gather(padded_out, padded_counts, group=group)

    gather_order_row_counts: List[int] = []
    for rank, n_rank in enumerate(n_per_rank):
        for idx in range(int(n_rank)):
            gather_order_row_counts.append(int(padded_out[rank][idx].item()))
    return gather_order_row_counts


def reorder_gathered_embeddings(
    gathered_embeddings,
    local_per_image_row_counts,
    rank_assignment: Dict[int, List[Tuple[int, int]]],
    group,
    global_per_image_row_counts=None,
):
    """Reorder rank-major gathered image embeddings into canonical image order."""
    total_imgs = sum(len(v) for v in rank_assignment.values())
    if total_imgs <= 1:
        return gathered_embeddings

    if gathered_embeddings.dim() != 2:
        raise ValueError(
            "reorder_gathered_embeddings: gathered_embeddings must be "
            f"2-D [total_rows, hidden]; got shape {tuple(gathered_embeddings.shape)}"
        )

    world_size = dist.get_world_size(group=group)
    device = gathered_embeddings.device
    my_rank_in_group = dist.get_rank(group=group)
    expected_n_local = len(rank_assignment.get(my_rank_in_group, []) or [])

    if local_per_image_row_counts is None:
        if global_per_image_row_counts is None:
            raise ValueError(
                "reorder_gathered_embeddings requires "
                "local_per_image_row_counts unless global metadata row "
                "counts are provided."
            )
        n_local = expected_n_local
    else:
        n_local = (
            int(local_per_image_row_counts.shape[0]) if local_per_image_row_counts.dim() >= 1 else 0
        )

    if n_local != expected_n_local:
        raise ValueError(
            "reorder_gathered_embeddings: local_per_image_row_counts has "
            f"length {n_local} but rank_assignment[{my_rank_in_group}] "
            f"says {expected_n_local} images on this rank. Caller must "
            "pass row counts in LPT order matching the per-rank assignment."
        )

    if global_per_image_row_counts is not None:
        gather_order_row_counts = _metadata_gather_order_row_counts(
            rank_assignment,
            global_per_image_row_counts,
            local_per_image_row_counts,
            my_rank_in_group,
        )
    else:
        n_per_rank = _all_gather_rank_image_counts(n_local, world_size, device, group)
        for rank, actual in enumerate(n_per_rank):
            expected = len(rank_assignment.get(rank, []) or [])
            if expected != actual:
                raise ValueError(
                    "reorder_gathered_embeddings: rank_assignment[rank="
                    f"{rank}] has {expected} images, but all_gather says "
                    f"rank {rank} contributed {actual} images."
                )
        gather_order_row_counts = _fallback_gather_order_row_counts(
            local_per_image_row_counts, n_local, n_per_rank, device, group
        )
        if len(gather_order_row_counts) != total_imgs:
            raise ValueError(
                "reorder_gathered_embeddings: gathered "
                f"{len(gather_order_row_counts)} row-count entries but "
                f"rank_assignment says total_imgs={total_imgs}."
            )

    return _reorder_by_gather_order_row_counts(
        gathered_embeddings,
        rank_assignment,
        gather_order_row_counts,
        error_prefix="reorder_gathered_embeddings",
    )
