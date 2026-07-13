# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MDP batch transport and BalanceData helpers for ``forward_step``."""

import json
from typing import Any, Dict, Iterator

import torch

from examples.multimodal_dev.data.energon_vision_balance import vision_rows_from_grid
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from megatron.core.utils import unwrap_model
from megatron.training import get_args

_MAX_JSON_BROADCAST_BYTES = 64 * 1024 * 1024


def _arg(name, default=None):
    return getattr(get_args(), name, default)


def _int_arg(name, default: int) -> int:
    return int(_arg(name, default))


def _bool_arg(name, default: bool = False) -> bool:
    return bool(_arg(name, default))


def _dtype_to_id(dtype):
    dtype_id = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int64: 3,
        torch.int32: 4,
        torch.bool: 5,
        torch.uint8: 6,
    }.get(dtype)
    if dtype_id is None:
        raise TypeError(f"MDP batch broadcast does not support dtype {dtype!r}")
    return dtype_id


def _id_to_dtype(id_val):
    dtype = {
        0: torch.float32,
        1: torch.float16,
        2: torch.bfloat16,
        3: torch.int64,
        4: torch.int32,
        5: torch.bool,
        6: torch.uint8,
    }.get(id_val)
    if dtype is None:
        raise TypeError(f"MDP batch broadcast received unknown dtype id {id_val!r}")
    return dtype


def _broadcast_tensor_sync(tensor, *, src, group):
    work = torch.distributed.broadcast(tensor, src, group=group, async_op=True)
    work.wait()
    if torch.cuda.is_available() and torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.current_stream().synchronize()


def _broadcast_json_payload_from_rank(payload, *, src, group, device="cuda"):
    """Broadcast a small JSON-serializable metadata payload via tensors.

    Use an explicit tensor length + payload protocol so metadata and tensor
    broadcasts complete in the same deterministic order on every TP rank.
    """
    is_src = torch.distributed.get_rank() == int(src)
    if is_src:
        payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        length_value = len(payload_bytes)
    else:
        payload_bytes = b""
        length_value = 0

    length_tensor = torch.tensor([int(length_value)], dtype=torch.int64, device=device)
    _broadcast_tensor_sync(length_tensor, src=src, group=group)

    length = int(length_tensor.item())
    if length < 0 or length > _MAX_JSON_BROADCAST_BYTES:
        raise RuntimeError(
            "MDP JSON broadcast received an invalid payload length "
            f"{length} bytes from source rank {src}; "
            f"limit is {_MAX_JSON_BROADCAST_BYTES} bytes"
        )
    if is_src:
        payload_tensor = torch.tensor(list(payload_bytes), dtype=torch.uint8, device=device)
    else:
        payload_tensor = torch.empty(length, dtype=torch.uint8, device=device)
    if length > 0:
        _broadcast_tensor_sync(payload_tensor, src=src, group=group)

    if is_src:
        return payload
    raw = bytes(payload_tensor.cpu().tolist()).decode("utf-8")
    return json.loads(raw) if raw else None


def _mdp_image_row_counts_from_grid_rows(rows):
    """Return projected vision rows per image from CPU ``image_grid_thw`` rows."""
    merge_size = _int_arg("vision_spatial_merge_size", 2)
    return [vision_rows_from_grid(row, spatial_merge_size=merge_size) for row in (rows or [])]


def _normalize_assignment(assignment):
    if assignment is None:
        return None
    normalized = {}
    for rank, values in assignment.items():
        normalized[int(rank)] = [
            (int(sample_idx), int(image_idx)) for sample_idx, image_idx in values
        ]
    return normalized


def _set_mdp_rank_assignment(models, assignment, row_counts=None):
    for model in models:
        object.__setattr__(model, "_mdp_rank_assignment", assignment)
        object.__setattr__(model, "_mdp_rank_assignment_row_counts", row_counts)


def _extract_balancedata_grid_rows(data):
    if not isinstance(data, dict):
        return None
    grid = data.get("image_grid_thw")
    if grid is None or not torch.is_tensor(grid):
        return None
    # Preserve loader-side metadata before broadcast_data_batch moves
    # tensors to CUDA. This avoids GPU->CPU readback in BalanceData.
    if grid.is_cuda:
        return None
    if grid.dim() == 3:
        grid = grid.flatten(0, 1)
    if grid.dim() != 2 or grid.shape[-1] != 3:
        return None
    return [(int(row[0]), int(row[1]), int(row[2])) for row in grid.detach().cpu().tolist()]


def _tensor_cpu_int_list(tensor):
    if tensor is None or not torch.is_tensor(tensor) or tensor.is_cuda:
        return None
    return [int(x) for x in tensor.detach().contiguous().view(-1).tolist()]


def _metadata_int_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.is_cuda:
            return None
        value = value.detach().reshape(-1).tolist()
    return [int(item) for item in value]


def _extract_mdp_cp_local_plan(data):
    if not isinstance(data, dict):
        return None
    input_ids = data.get("input_ids")
    if input_ids is None:
        input_ids = data.get("tokens")
    if input_ids is None or not torch.is_tensor(input_ids) or input_ids.is_cuda:
        return None
    image_token_id = _int_arg("image_token_id", 248056)
    ids = input_ids.detach()
    if ids.dim() == 1:
        input_shape = (1, int(ids.shape[0]))
    elif ids.dim() == 2:
        input_shape = (int(ids.shape[0]), int(ids.shape[1]))
    else:
        return None
    image_positions = ids.reshape(-1).eq(image_token_id).nonzero(as_tuple=False).view(-1)
    return {
        "input_shape": input_shape,
        "image_positions": [int(x) for x in image_positions.tolist()],
        "cu_seqlens": _tensor_cpu_int_list(data.get("cu_seqlens")),
        "cu_seqlens_padded": _tensor_cpu_int_list(data.get("cu_seqlens_padded")),
    }


def _extract_loader_prepartition_metadata(data):
    if not isinstance(data, dict):
        return None, None, None

    assignment_tensor = data.get("_mdp_prepartitioned_assignment")
    assignment = None
    if torch.is_tensor(assignment_tensor) and not assignment_tensor.is_cuda:
        assignment = {}
        for row in assignment_tensor.detach().reshape(-1, 3).tolist():
            rank, sample_idx, image_idx = [int(value) for value in row]
            assignment.setdefault(rank, []).append((sample_idx, image_idx))
    elif isinstance(assignment_tensor, dict):
        assignment = {
            int(rank): [(int(sample_idx), int(image_idx)) for sample_idx, image_idx in values]
            for rank, values in assignment_tensor.items()
        }

    row_counts = _metadata_int_list(data.get("_mdp_prepartitioned_row_counts"))
    local_raw_counts = _metadata_int_list(data.get("_mdp_prepartitioned_local_raw_counts"))
    return assignment, row_counts, local_raw_counts


def _strip_mdp_image_descriptors(data):
    if not isinstance(data, dict):
        return data
    descriptor_key = "_mdp_image_descriptors_json"
    if descriptor_key not in data and "_mdp_image_descriptors" not in data:
        return data

    data = dict(data)
    data.pop(descriptor_key, None)
    data.pop("_mdp_image_descriptors", None)
    return data


def _strip_loader_prepartition_python_metadata(data):
    if not isinstance(data, dict):
        return data
    data = dict(data)
    data.pop("_mdp_prepartitioned_assignment", None)
    data.pop("_mdp_prepartitioned_row_counts", None)
    data.pop("_mdp_prepartitioned_local_raw_counts", None)
    return data


def _loader_prepartition_enabled() -> bool:
    if not _bool_arg("mdp_encoder_mode", True):
        return False
    return _arg("mdp_inner_dp_scope", "cp") in ("cp", "pp_cp")


def _numel_from_shape(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


def _batch_tensor_metadata(data):
    metadata = []
    for key, value in (data or {}).items():
        if isinstance(value, torch.Tensor):
            metadata.append(
                (key, tuple(int(dim) for dim in value.shape), _dtype_to_id(value.dtype))
            )
        else:
            metadata.append((key, None, None))
    return metadata


def broadcast_data_batch_from_rank(data, *, src, group, device="cuda", broadcast_device=None):
    """Broadcast a batch dict from an arbitrary process-group source rank."""
    batch, _side_metadata = _broadcast_data_batch_and_side_metadata_from_rank(
        data,
        side_metadata=None,
        src=src,
        group=group,
        device=device,
        broadcast_device=broadcast_device,
    )
    return batch


def _broadcast_data_batch_and_side_metadata_from_rank(
    data, *, side_metadata, src, group, device="cuda", broadcast_device=None
):
    """Broadcast a batch dict and small side metadata from one source rank."""
    broadcast_device = broadcast_device or device
    if data is None:
        data = {}

    if (
        not torch.distributed.is_initialized()
        or group is None
        or torch.distributed.get_world_size(group=group) <= 1
    ):
        return {
            key: (value.to(device).contiguous() if isinstance(value, torch.Tensor) else None)
            for key, value in data.items()
        }, side_metadata

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    is_src = rank == int(src)
    if is_src:
        metadata_payload = (side_metadata, _batch_tensor_metadata(data))
    else:
        metadata_payload = None
    metadata_payload = _broadcast_json_payload_from_rank(
        metadata_payload, src=src, group=group, device=broadcast_device
    )
    if not isinstance(metadata_payload, (tuple, list)) or len(metadata_payload) != 2:
        raise RuntimeError(
            "MDP batch broadcast received invalid metadata payload: " f"{metadata_payload!r}"
        )
    side_metadata, metadata = metadata_payload
    metadata = metadata or []

    result = {}
    tensor_entries = []
    for key, shape, dtype_id in metadata:
        if shape is None:
            result[key] = None
            continue
        tensor_entries.append((key, tuple(int(dim) for dim in shape), int(dtype_id)))

    dtype_order = []
    entries_by_dtype = {}
    for entry in tensor_entries:
        dtype_id = entry[2]
        if dtype_id not in entries_by_dtype:
            dtype_order.append(dtype_id)
            entries_by_dtype[dtype_id] = []
        entries_by_dtype[dtype_id].append(entry)

    for dtype_id in dtype_order:
        entries = entries_by_dtype[dtype_id]
        dtype = _id_to_dtype(dtype_id)
        total_numel = sum(_numel_from_shape(shape) for _key, shape, _ in entries)
        if is_src:
            flat_parts = []
            for key, _shape, _ in entries:
                tensor = data.get(key, None) if data else None
                flat_parts.append(tensor.to(broadcast_device).contiguous().view(-1))
            if len(flat_parts) == 1:
                flat_buffer = flat_parts[0]
            elif total_numel == 0:
                flat_buffer = torch.empty(0, dtype=dtype, device=broadcast_device)
            else:
                flat_buffer = torch.cat(flat_parts, dim=0)
        else:
            flat_buffer = torch.empty(total_numel, dtype=dtype, device=broadcast_device)
        _broadcast_tensor_sync(flat_buffer, src=src, group=group)

        offset = 0
        for key, shape, _ in entries:
            numel = _numel_from_shape(shape)
            value = flat_buffer.narrow(0, offset, numel).view(shape)
            if torch.device(value.device) != torch.device(device):
                value = value.to(device=device, non_blocking=False)
            result[key] = value.contiguous()
            offset += numel

    return result, side_metadata


def _rows_from_grid_metadata(image_grid_thw, image_grid_thw_rows):
    if image_grid_thw_rows is not None:
        return [(int(row[0]), int(row[1]), int(row[2])) for row in image_grid_thw_rows]
    if image_grid_thw is None:
        return []
    return [
        (int(row[0]), int(row[1]), int(row[2])) for row in image_grid_thw.detach().cpu().tolist()
    ]


def apply_mdp_prepartition(
    model,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_grid_thw_rows=None,
    prepartitioned_assignment=None,
    prepartitioned_row_counts=None,
    prepartitioned_image_grid_thw=None,
):
    """Apply loader-prepartition metadata to the local MDP image shard.

    Reads ``model._mdp_enabled`` (set by
    ``pretrain_multimodal.model_provider``). When MDP is enabled:

    1. Stash the full assignment on ``model._mdp_rank_assignment`` for
       the downstream gather (``modality_bridge.gather_to_inner_dp_zero``).
    2. Return the image shard materialized by the loader for this rank.

    When MDP is disabled, returns the inputs unchanged (no-op).

    Args:
        model: the possibly wrapped model with ``_mdp_*`` attributes set
            by ``model_provider``.
        pixel_values: ``[total_patches, pixel_dim]`` tensor.
        image_grid_thw: ``[num_images, 3]`` tensor of
            ``(t_patches, h_patches, w_patches)`` per image.

    Returns:
        ``(pixel_values, image_grid_thw)`` for this rank's loader assignment.
        If MDP is disabled, the inputs are returned unchanged.
    """
    # The MDP attributes live on the inner model, while this function may
    # receive a DDP wrapper. Reads can fall through the wrapper, but writes
    # must target the inner model so ``base.py`` sees the rank assignment.
    inner_model = unwrap_model(model)

    # Lazy attribute lookup keeps the off-path cheap and import-safe.
    if not bool(getattr(inner_model, "_mdp_enabled", False)):
        # Set _mdp_rank_assignment to None so the gather collapses to a
        # single-rank-owns-all path. Write to both objects for wrapper and
        # inner-model consumers.
        _set_mdp_rank_assignment((inner_model, model), None, None)
        return pixel_values, image_grid_thw

    # The loader assignment is already expressed in the active InnerDP
    # group's local-rank coordinates (CP-only or PP x CP) and normalized in
    # ``fetch_and_broadcast``.
    if prepartitioned_assignment is not None:
        assignment = prepartitioned_assignment
        row_counts = (
            [int(value) for value in prepartitioned_row_counts]
            if prepartitioned_row_counts is not None
            else _mdp_image_row_counts_from_grid_rows(
                _rows_from_grid_metadata(image_grid_thw, image_grid_thw_rows)
            )
        )
        if prepartitioned_image_grid_thw is not None:
            image_grid_thw = prepartitioned_image_grid_thw
        _set_mdp_rank_assignment((inner_model, model), assignment, row_counts)
        return pixel_values, image_grid_thw

    raise RuntimeError("MDP requires loader-prepartition metadata.")


def fetch_and_broadcast(data_iterator: Iterator[Dict[str, Any]]):
    """Read one batch on TP-rank-0 and broadcast to all TP ranks."""
    device = "cuda"
    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()
    tp_world_size = (
        int(get_tensor_model_parallel_world_size()) if model_parallel_is_initialized() else 1
    )
    mdp_encoder_enabled = _bool_arg("mdp_encoder_mode", True)

    balancedata_grid_rows = None
    cp_local_plan = None
    prepartition_assignment = None
    prepartition_row_counts = None
    prepartition_local_raw_counts = None
    has_data_local = False
    if get_tensor_model_parallel_rank() == 0:
        if mdp_encoder_enabled and _loader_prepartition_enabled():
            try:
                data = next(data_iterator)
                # CP-local prepartition materializes only this rank's owner
                # images in the loader. JSON-only lazy descriptors must not
                # be transported into this immediate model-forward path.
                data = _strip_mdp_image_descriptors(data)
                balancedata_grid_rows = _extract_balancedata_grid_rows(data)
                cp_local_plan = _extract_mdp_cp_local_plan(data)
                (
                    prepartition_assignment,
                    prepartition_row_counts,
                    prepartition_local_raw_counts,
                ) = _extract_loader_prepartition_metadata(data)
                data = _strip_loader_prepartition_python_metadata(data)
                has_data_local = True
            except StopIteration:
                data = None
                has_data_local = False
        elif mdp_encoder_enabled:
            raise RuntimeError("MDP data loading requires loader-prepartition metadata.")
        else:
            try:
                data = next(data_iterator)
                has_data_local = True
            except StopIteration:
                data = None
                has_data_local = False
    else:
        data = None

    metadata_payload = (
        bool(has_data_local),
        balancedata_grid_rows,
        cp_local_plan,
        prepartition_assignment,
        prepartition_row_counts,
        prepartition_local_raw_counts,
    )
    if tp_world_size > 1:
        batch, metadata_payload = _broadcast_data_batch_and_side_metadata_from_rank(
            data,
            side_metadata=(metadata_payload if get_tensor_model_parallel_rank() == 0 else None),
            src=src,
            group=group,
            device=device,
            broadcast_device=device,
        )
    else:
        batch = broadcast_data_batch_from_rank(
            data, src=src, group=group, device=device, broadcast_device=device
        )
    (
        has_data_present,
        balancedata_grid_rows,
        cp_local_plan,
        prepartition_assignment,
        prepartition_row_counts,
        prepartition_local_raw_counts,
    ) = metadata_payload
    if not bool(has_data_present):
        return None

    prepartition_assignment = _normalize_assignment(prepartition_assignment)

    _normalize_collated_multimodal_tensors(batch)

    batch["_balancedata_image_grid_thw_rows"] = balancedata_grid_rows
    batch["_mdp_cp_local_plan"] = cp_local_plan
    if prepartition_assignment is not None:
        batch["_mdp_prepartitioned_assignment"] = prepartition_assignment
    if prepartition_row_counts is not None:
        batch["_mdp_prepartitioned_row_counts"] = prepartition_row_counts
    if prepartition_local_raw_counts is not None:
        # Preserve the loader's raw-patch counts for downstream consumers
        # without changing CP ownership semantics.
        batch["_mdp_prepartitioned_local_raw_counts"] = prepartition_local_raw_counts
    return batch


def _normalize_collated_multimodal_tensors(batch: Dict[str, torch.Tensor]) -> None:
    """Normalize tensor ranks produced by ``default_collate`` in-place."""
    if "position_ids" in batch and batch["position_ids"] is not None:
        p = batch["position_ids"]
        if p.dim() == 3 and p.shape[1] == 3:
            batch["position_ids"] = p.permute(1, 0, 2).contiguous()

    if "pixel_values" in batch and batch["pixel_values"] is not None:
        pv = batch["pixel_values"]
        if pv.dim() == 3:
            bsz, patches, dim = pv.shape
            batch["pixel_values"] = pv.reshape(bsz * patches, dim)

    if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
        grid = batch["image_grid_thw"]
        if grid.dim() == 3:
            # default_collate produces (B, N, 3). Encoder expects (B*N, 3).
            batch["image_grid_thw"] = grid.flatten(0, 1)

    if (
        "_mdp_prepartitioned_image_grid_thw" in batch
        and batch["_mdp_prepartitioned_image_grid_thw"] is not None
    ):
        grid = batch["_mdp_prepartitioned_image_grid_thw"]
        if grid.dim() == 3:
            batch["_mdp_prepartitioned_image_grid_thw"] = grid.flatten(0, 1)
