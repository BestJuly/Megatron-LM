# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Forward step, TP broadcast, and loss for multimodal_dev training."""

import math
from functools import partial
from itertools import accumulate
from typing import Any, Dict, Iterator

import torch
import torch.nn.functional as F

from examples.multimodal_dev.mdp_batch import (
    apply_mdp_prepartition,
    broadcast_data_batch_from_rank,
    fetch_and_broadcast,
)
from megatron.core import mpu
from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from megatron.core.utils import get_thd_batch_on_this_cp_rank
from megatron.training import get_args

# -------------------------------------------------------------------
# dtype <-> int mapping for cross-rank broadcast
# -------------------------------------------------------------------

_DTYPE_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.bool: 5,
}
_ID_MAP = {v: k for k, v in _DTYPE_MAP.items()}


def _dtype_to_id(dtype):
    return _DTYPE_MAP.get(dtype, 0)


def _id_to_dtype(id_val):
    return _ID_MAP.get(id_val, torch.float32)


# -------------------------------------------------------------------
# Tensor broadcast helper
# -------------------------------------------------------------------

def _broadcast_tensor(tensor, src, group, device):
    """Broadcast a single tensor from *src* to all ranks in *group*."""
    encoded_ndim = torch.tensor(
        [tensor.dim() + 1 if tensor is not None else 0],
        dtype=torch.long,
        device=device,
    )
    torch.distributed.broadcast(encoded_ndim, src, group=group)

    if encoded_ndim.item() == 0:
        return None
    ndim = int(encoded_ndim.item()) - 1

    if tensor is not None:
        shape_tensor = torch.tensor(
            list(tensor.shape), dtype=torch.long, device=device,
        )
        dtype_id = torch.tensor(
            [_dtype_to_id(tensor.dtype)],
            dtype=torch.long,
            device=device,
        )
    else:
        shape_tensor = torch.zeros(
            ndim, dtype=torch.long, device=device,
        )
        dtype_id = torch.zeros(1, dtype=torch.long, device=device)

    torch.distributed.broadcast(shape_tensor, src, group=group)
    torch.distributed.broadcast(dtype_id, src, group=group)

    dtype = _id_to_dtype(dtype_id.item())
    shape = tuple(shape_tensor.tolist())

    if tensor is None:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    torch.distributed.broadcast(tensor, src, group=group)
    return tensor


# -------------------------------------------------------------------
# Batch broadcast across TP ranks
# -------------------------------------------------------------------

def broadcast_data_batch(data, device="cuda"):
    """Broadcast a data-batch dict from TP rank 0 to all TP ranks."""
    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()

    if data is None:
        data = {}

    if get_tensor_model_parallel_rank() == 0:
        keys = list(data.keys())
        key_str = ",".join(keys)
        key_bytes = key_str.encode("utf-8")
        key_len = torch.tensor(
            [len(key_bytes)], dtype=torch.long, device=device,
        )
    else:
        key_len = torch.zeros(1, dtype=torch.long, device=device)
        keys = []

    torch.distributed.broadcast(key_len, src, group=group)

    if get_tensor_model_parallel_rank() == 0:
        key_tensor = torch.tensor(
            list(key_bytes), dtype=torch.uint8, device=device,
        )
    else:
        key_tensor = torch.zeros(
            key_len.item(), dtype=torch.uint8, device=device,
        )

    torch.distributed.broadcast(key_tensor, src, group=group)

    if get_tensor_model_parallel_rank() != 0:
        key_str = bytes(key_tensor.cpu().tolist()).decode("utf-8")
        keys = key_str.split(",") if key_str else []

    result = {}
    for key in keys:
        tensor = data.get(key, None) if data else None
        if tensor is not None and isinstance(tensor, torch.Tensor):
            tensor = tensor.to(device)
        result[key] = _broadcast_tensor(
            tensor if isinstance(tensor, torch.Tensor) else None,
            src, group, device,
        )

    return result


# -------------------------------------------------------------------
# THD (packed sequence) helpers
# -------------------------------------------------------------------

def _build_packed_seq_params(
    seq_lengths: torch.Tensor, device: torch.device,
) -> PackedSeqParams:
    """Build ``PackedSeqParams`` from per-sample valid sequence lengths.

    Args:
        seq_lengths: ``[B]`` valid token counts per sample.
        device: Target device for cu_seqlens tensors.

    Returns:
        A ``PackedSeqParams`` instance with ``qkv_format='thd'``.
    """
    if not isinstance(seq_lengths, torch.Tensor):
        seq_lengths = torch.tensor(seq_lengths)
    lengths_t = seq_lengths.to(device=device, dtype=torch.int32)
    cu_seqlens = torch.zeros(
        lengths_t.numel() + 1, dtype=torch.int32, device=device,
    )
    torch.cumsum(lengths_t, dim=0, out=cu_seqlens[1:])
    max_seqlen = int(lengths_t.max().item())
    return _build_packed_seq_params_from_cu_seqlens(
        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
    )


def _build_packed_seq_params_from_cu_seqlens(
    cu_seqlens: torch.Tensor, max_seqlen: int,
) -> PackedSeqParams:
    """Build ``PackedSeqParams`` from packed cumulative sequence lengths.

    ``cu_seqlens`` must already be on the target compute device.
    """
    cs = cu_seqlens.to(dtype=torch.int32)
    total_tokens = int(cs[-1].item())
    return PackedSeqParams(
        cu_seqlens_q=cs,
        cu_seqlens_kv=cs,
        cu_seqlens_q_padded=cs,
        cu_seqlens_kv_padded=cs,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
        total_tokens=total_tokens,
    )


def _prepare_prepacked_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an Energon-packed dict and restore ``PackedSeqParams``."""
    cu_seqlens = batch.pop("cu_seqlens")
    cu_seqlens_padded = batch.pop("cu_seqlens_padded")
    max_seqlen = batch.pop("max_seqlen")
    if cu_seqlens.dim() == 2:
        if cu_seqlens.shape[0] != 1:
            raise ValueError("packed Energon batches require micro-batch-size 1")
        cu_seqlens = cu_seqlens[0]
    if cu_seqlens_padded.dim() == 2:
        if cu_seqlens_padded.shape[0] != 1:
            raise ValueError("packed Energon batches require micro-batch-size 1")
        cu_seqlens_padded = cu_seqlens_padded[0]
    if cu_seqlens.dim() != 1 or cu_seqlens_padded.dim() != 1:
        raise ValueError("packed sequence boundaries must be one-dimensional")

    for key in ("input_ids", "labels", "loss_mask"):
        tensor = batch.get(key)
        if tensor is not None and tensor.dim() == 1:
            batch[key] = tensor.unsqueeze(0)
    position_ids = batch.get("position_ids")
    if position_ids is not None and position_ids.dim() == 2:
        batch["position_ids"] = position_ids.unsqueeze(1)

    sequence_length = int(batch["input_ids"].shape[-1])
    if int(cu_seqlens_padded[-1].item()) != sequence_length:
        raise ValueError(
            "packed sequence boundary does not match input length: "
            f"boundary={int(cu_seqlens_padded[-1].item())}, "
            f"input={sequence_length}"
        )
    max_seqlen_value = int(max_seqlen.reshape(-1)[0].item())
    logical_cu_seqlens = cu_seqlens.to(dtype=torch.int32)
    physical_cu_seqlens = cu_seqlens_padded.to(dtype=torch.int32)
    batch["packed_seq_params"] = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical_cu_seqlens,
        cu_seqlens_kv=logical_cu_seqlens,
        cu_seqlens_q_padded=physical_cu_seqlens,
        cu_seqlens_kv_padded=physical_cu_seqlens,
        max_seqlen_q=max_seqlen_value,
        max_seqlen_kv=max_seqlen_value,
        total_tokens=sequence_length,
    )
    return batch


def _mdp_prepartition_kwargs(batch):
    """Collect loader-side LPT metadata for ``apply_mdp_prepartition``."""
    return {
        "prepartitioned_assignment": batch.get("_mdp_prepartitioned_assignment"),
        "prepartitioned_row_counts": batch.get("_mdp_prepartitioned_row_counts"),
        "prepartitioned_image_grid_thw": batch.get(
            "_mdp_prepartitioned_image_grid_thw"
        ),
    }


def _get_mdp_prepartitioned_batch(data_iterator):
    """Fetch the b436 loader-prepartitioned THD batch for CP-local MDP."""
    batch = fetch_and_broadcast(data_iterator)
    if batch is None:
        return None
    if "input_ids" not in batch and batch.get("tokens") is not None:
        batch["input_ids"] = batch["tokens"]
    batch.pop("local_cp_size", None)

    # Keep one THD normalization owner. PR #2's helper preserves logical q/kv
    # boundaries while using physical padded boundaries for residual-stream
    # shape; the MDP path must not reconstruct either view independently.
    return _prepare_prepacked_batch(batch)


def pack_or_pad_batch(batch: list[Dict[str, Any]], use_packed_sequence: bool=False, seq_length: int=None, device = "cuda") -> list[Dict[str, Any]]:
    """Pack or pad a ``[B, S]`` batch into ``[1, T]`` THD format."""
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()

    # SP is an explicit runtime option; TP>1 does not imply SP is enabled.
    try:
        has_sp = bool(get_args().sequence_parallel)
    except Exception:
        has_sp = False

    if cp_size > 1:
        divisible_by = (tp_size * cp_size * 2) if has_sp else (cp_size * 2)
    else:
        divisible_by = tp_size if has_sp else 1
    # NOTE: don't consider fp8 padding now
    
    if use_packed_sequence:
        input_ids_list, labels_list, loss_mask_list, pixel_values_list, image_grid_thw_list = [], [], [], [], []
        seqlens_list, seqlens_padded_list = [], []

        # NOTE: for attention_mask, we don't use attention mask
        #       for position_ids, let model handle it itself
        #       we don't cut input id, althrough it exceeds seq_length

        packed_batch = dict()

        for sample in batch:
            seqlen = sample["input_ids"].shape[0]
            assert sample["labels"].shape == sample["input_ids"].shape == sample["loss_mask"].shape, "labels, input_ids, and loss_mask must have the same shape"
            target_len = math.ceil(seqlen / divisible_by) * divisible_by
            input_ids = F.pad(sample["input_ids"], (0, target_len - seqlen), value=0)
            labels = F.pad(sample["labels"], (0, target_len - seqlen), value=-100)
            loss_mask = F.pad(sample["loss_mask"], (0, target_len - seqlen), value=0)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            loss_mask_list.append(loss_mask)
            seqlens_list.append(seqlen)
            seqlens_padded_list.append(target_len)
            pixel_values_list.append(sample["pixel_values"])
            image_grid_thw_list.append(sample["image_grid_thw"])

        cu_seqlens = list(accumulate(seqlens_list, initial=0))
        cu_seqlens_padded = list(accumulate(seqlens_padded_list, initial=0))

        packed_batch["input_ids"] = torch.concat(input_ids_list, dim=0).unsqueeze(0)
        packed_batch["labels"] = torch.concat(labels_list, dim=0).unsqueeze(0)
        packed_batch["loss_mask"] = torch.concat(loss_mask_list, dim=0).unsqueeze(0)

        # TODO, maybe pixel_values's seqlens needs to be recorded. 
        packed_batch["pixel_values"] = torch.concat(pixel_values_list)
        packed_batch["image_grid_thw"] = torch.concat(image_grid_thw_list)
        
        # broadcast to all tp ranks
        packed_batch = broadcast_data_batch(packed_batch, device=device)

        packed_batch["packed_seq_params"] = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            cu_seqlens_kv=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            cu_seqlens_q_padded=torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=device),
            cu_seqlens_kv_padded=torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=device),
            max_seqlen_q=max(seqlens_padded_list),
            max_seqlen_kv=max(seqlens_padded_list),
            total_tokens=cu_seqlens_padded[-1],
        )
        return packed_batch
    else:
        assert seq_length is not None, "seq_length must be provided when use_packed_sequence is False"
        max_seqlens = max([x["input_ids"].shape[0] for x in batch])
        target_seqlens = min(max_seqlens, seq_length)
        # Round target seqlen up to the parallelism alignment factor so the
        # batched tensor is divisible for CP (+SP) splitting downstream.
        if divisible_by > 1:
            target_seqlens = math.ceil(target_seqlens / divisible_by) * divisible_by
        padded_batch = dict()
        
        for sample in batch:
            sample["input_ids"] = F.pad(sample["input_ids"], (0, target_seqlens - sample["input_ids"].shape[0]), value=0)
            sample["labels"] = F.pad(sample["labels"], (0, target_seqlens - sample["labels"].shape[0]), value=-100)
            sample["loss_mask"] = F.pad(sample["loss_mask"], (0, target_seqlens - sample["loss_mask"].shape[0]), value=0)

        padded_batch["input_ids"] = torch.concat([x["input_ids"].unsqueeze(0) for x in batch], dim=0)
        padded_batch["labels"] = torch.concat([x["labels"].unsqueeze(0) for x in batch], dim=0)
        padded_batch["loss_mask"] = torch.concat([x["loss_mask"].unsqueeze(0) for x in batch], dim=0)
        padded_batch["pixel_values"] = torch.concat([x["pixel_values"] for x in batch])
        padded_batch["image_grid_thw"] = torch.concat([x["image_grid_thw"] for x in batch])
        # broadcast to all tp ranks
        padded_batch = broadcast_data_batch(padded_batch, device=device)
        return padded_batch


def _shard_qwen3_packed_batch_for_cp(batch, cp_size, cp_rank):
    """Apply MCore's canonical data-side THD shard for plain Qwen3 GPT.

    Vision tensors are not sequence-aligned and must not be passed to
    ``get_thd_batch_on_this_cp_rank``.  Position IDs use the same THD
    partition indices but may have an extra MRoPE channel dimension, so
    they are indexed separately along their last dimension.
    """
    packed_seq_params = batch["packed_seq_params"]
    cu_seqlens = packed_seq_params.cu_seqlens_q
    cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
    max_seqlen = torch.tensor(
        [packed_seq_params.max_seqlen_q],
        dtype=torch.int32,
        device=cu_seqlens.device,
    )

    sequence_batch = {"tokens": batch["input_ids"]}
    for key in ("labels", "loss_mask", "attention_mask"):
        if batch.get(key) is not None:
            sequence_batch[key] = batch[key]

    sequence_batch, packed_seq_params = get_thd_batch_on_this_cp_rank(
        sequence_batch,
        cu_seqlens,
        cu_seqlens_padded,
        max_seqlen,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )

    sharded_batch = dict(batch)
    sharded_batch["input_ids"] = sequence_batch.pop("tokens")
    sharded_batch.update(sequence_batch)
    sharded_batch["packed_seq_params"] = packed_seq_params

    position_ids = batch.get("position_ids")
    if position_ids is not None:
        indices = get_thd_partitioned_indices(
            cu_seqlens_padded,
            int(cu_seqlens_padded[-1].item()),
            cp_size,
            cp_rank,
        ).long()
        sharded_batch["position_ids"] = position_ids.index_select(-1, indices)

    # ``forward_step`` uses this marker to avoid sharding the loss mask a
    # second time after the plain GPT model returns its already-local output.
    sharded_batch["_data_side_cp_sharded"] = True
    return sharded_batch


# -------------------------------------------------------------------
# get_batch
# -------------------------------------------------------------------

def get_batch(data_iterator: Iterator[Dict[str, Any]]):
    """Get a batch from *data_iterator* and broadcast across TP ranks."""
    device = "cuda"
    args = get_args()

    if (
        bool(getattr(args, "mdp_encoder_mode", True))
        and (
            get_context_parallel_world_size() > 1
            or getattr(args, "mdp_inner_dp_scope", "cp") == "pp_cp"
        )
    ):
        return _get_mdp_prepartitioned_batch(data_iterator)

    if get_tensor_model_parallel_rank() == 0:
        try:
            data = next(data_iterator)
            has_data = torch.tensor(
                [1], dtype=torch.uint8, device=device,
            )
        except StopIteration:
            has_data = torch.tensor(
                [0], dtype=torch.uint8, device=device,
            )
            data = None
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device=device)
        data = None

    src = get_tensor_model_parallel_src_rank()
    group = get_tensor_model_parallel_group()
    torch.distributed.broadcast(has_data, src, group=group)

    if has_data.item() == 0:
        return None

    if get_tensor_model_parallel_rank() == 0:
        prepacked = torch.tensor(
            [int(isinstance(data, dict) and "cu_seqlens" in data)],
            dtype=torch.uint8,
            device=device,
        )
    else:
        prepacked = torch.empty(1, dtype=torch.uint8, device=device)
    torch.distributed.broadcast(prepacked, src, group=group)

    if prepacked.item():
        batch = _prepare_prepacked_batch(
            broadcast_data_batch(data, device=device)
        )
    else:
        batch = pack_or_pad_batch(
            data, args.use_packed_sequence, args.seq_length, device=device
        )

    # Fix shapes produced by default_collate.
    if "position_ids" in batch and batch["position_ids"] is not None:
        p = batch["position_ids"]
        if p.dim() == 3 and p.shape[1] == 3:
            batch["position_ids"] = p.permute(1, 0, 2).contiguous()

    if "pixel_values" in batch and batch["pixel_values"] is not None:
        pv = batch["pixel_values"]
        if pv.dim() == 3:
            B, P, D = pv.shape
            batch["pixel_values"] = pv.reshape(B * P, D)

    if (
        "image_grid_thw" in batch
        and batch["image_grid_thw"] is not None
    ):
        g = batch["image_grid_thw"]
        if g.dim() == 3:
            batch["image_grid_thw"] = g.squeeze(1)

    cp_size = get_context_parallel_world_size()
    if (
        getattr(args, "model_arch", None) == "qwen3"
        and batch.get("packed_seq_params") is not None
        and cp_size > 1
    ):
        batch = _shard_qwen3_packed_batch_for_cp(
            batch,
            cp_size=cp_size,
            cp_rank=get_context_parallel_rank(),
        )

    return batch


# -------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------

def loss_func(loss_mask, output_tensor):
    """Compute masked language model loss."""
    losses = output_tensor.float()
    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat(
        [total_loss.clone().detach().view(1), total_tokens.view(1)],
    )

    return (total_loss, total_tokens, {"lm loss": reporting_loss})


# -------------------------------------------------------------------
# Forward step
# -------------------------------------------------------------------

def _wrapped_model_method(model, name):
    while model is not None:
        method = getattr(model, name, None)
        if method is not None:
            return method
        model = getattr(model, "module", None)
    return None


def _wrapped_model_attr(model, name, default=None):
    while model is not None:
        if hasattr(model, name):
            return getattr(model, name)
        model = getattr(model, "module", None)
    return default


def _pack_sidecar_packed_seq_params(batch):
    """Replace ``PackedSeqParams`` with tensors safe for PP broadcast."""
    packed = dict(batch)
    packed_seq_params = packed.pop("packed_seq_params", None)
    if packed_seq_params is None:
        return packed

    cu_seqlens_q = packed_seq_params.cu_seqlens_q
    device = cu_seqlens_q.device
    total_tokens = getattr(packed_seq_params, "total_tokens", None)
    if total_tokens is None:
        physical_boundaries = packed_seq_params.cu_seqlens_q_padded
        if physical_boundaries is None:
            physical_boundaries = cu_seqlens_q
        total_tokens = int(physical_boundaries[-1].item())
    packed.update(
        {
            "_pp_cp_sidecar_cu_seqlens_q": cu_seqlens_q,
            "_pp_cp_sidecar_cu_seqlens_kv": packed_seq_params.cu_seqlens_kv,
            "_pp_cp_sidecar_cu_seqlens_q_padded": (
                packed_seq_params.cu_seqlens_q_padded
            ),
            "_pp_cp_sidecar_cu_seqlens_kv_padded": (
                packed_seq_params.cu_seqlens_kv_padded
            ),
            "_pp_cp_sidecar_max_seqlen_q": torch.tensor(
                [int(packed_seq_params.max_seqlen_q)],
                dtype=torch.int64,
                device=device,
            ),
            "_pp_cp_sidecar_max_seqlen_kv": torch.tensor(
                [int(packed_seq_params.max_seqlen_kv)],
                dtype=torch.int64,
                device=device,
            ),
            "_pp_cp_sidecar_total_tokens": torch.tensor(
                [int(total_tokens)],
                dtype=torch.int64,
                device=device,
            ),
        }
    )
    return packed


def _unpack_sidecar_packed_seq_params(batch):
    """Restore ``PackedSeqParams`` after the PP metadata broadcast."""
    unpacked = dict(batch)
    cu_seqlens_q = unpacked.pop("_pp_cp_sidecar_cu_seqlens_q", None)
    if cu_seqlens_q is None:
        return unpacked
    cu_seqlens_kv = unpacked.pop("_pp_cp_sidecar_cu_seqlens_kv", None)
    cu_seqlens_q_padded = unpacked.pop(
        "_pp_cp_sidecar_cu_seqlens_q_padded", None
    )
    cu_seqlens_kv_padded = unpacked.pop(
        "_pp_cp_sidecar_cu_seqlens_kv_padded", None
    )
    max_seqlen_q = unpacked.pop("_pp_cp_sidecar_max_seqlen_q")
    max_seqlen_kv = unpacked.pop("_pp_cp_sidecar_max_seqlen_kv")
    total_tokens = unpacked.pop("_pp_cp_sidecar_total_tokens")
    unpacked["packed_seq_params"] = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=(
            cu_seqlens_q if cu_seqlens_kv is None else cu_seqlens_kv
        ),
        cu_seqlens_q_padded=(
            cu_seqlens_q
            if cu_seqlens_q_padded is None
            else cu_seqlens_q_padded
        ),
        cu_seqlens_kv_padded=(
            cu_seqlens_q
            if cu_seqlens_kv_padded is None
            else cu_seqlens_kv_padded
        ),
        max_seqlen_q=int(max_seqlen_q.reshape(-1)[0].item()),
        max_seqlen_kv=int(max_seqlen_kv.reshape(-1)[0].item()),
        total_tokens=int(total_tokens.reshape(-1)[0].item()),
    )
    return unpacked


def _drop_sidecar_vision_payload(batch):
    """Remove PP0-owned vision tensors while retaining language metadata."""
    trimmed = dict(batch)
    for key in (
        "pixel_values",
        "_balancedata_image_grid_thw_rows",
        "_mdp_prepartitioned_image_grid_thw",
        "_mdp_prepartitioned_assignment",
        "_mdp_prepartitioned_row_counts",
        "_mdp_prepartitioned_local_raw_counts",
        "_mdp_image_descriptors",
        "_mdp_image_descriptors_json",
    ):
        trimmed.pop(key, None)
    return trimmed


def _stage_forward_view_from_full_batch(batch, vp_stage=None, config=None):
    """Return only the batch fields consumed by this pipeline stage."""
    if mpu.is_pipeline_first_stage(
        ignore_virtual=False, vp_stage=vp_stage
    ) or mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
        return batch
    if config is not None and mtp_on_this_rank(
        config, ignore_virtual=False, vp_stage=vp_stage
    ):
        return batch
    return {
        key: batch[key]
        for key in ("position_ids", "attention_mask", "packed_seq_params")
        if batch.get(key) is not None
    }


def _build_pp_batch_sidecar_cache(
    *, data_iterator, model, vp_stage=None, forward_only=False
):
    """Broadcast one MDP-off packed batch from PP0 before pipeline P2P."""
    config = _wrapped_model_attr(model, "config")
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_src = mpu.get_pipeline_model_parallel_first_rank()
    is_pp_first = mpu.is_pipeline_first_stage(
        ignore_virtual=False, vp_stage=vp_stage
    )

    if is_pp_first:
        if data_iterator is None:
            raise RuntimeError(
                "MDP-off PP batch sidecar requires a data iterator on PP0"
            )
        batch = get_batch(data_iterator)
        if batch is None:
            raise RuntimeError(
                "MDP-off PP batch sidecar received no batch on PP0"
            )
        broadcast_batch = _drop_sidecar_vision_payload(
            _pack_sidecar_packed_seq_params(batch)
        )
    else:
        batch = None
        broadcast_batch = None

    received_batch = broadcast_data_batch_from_rank(
        broadcast_batch,
        src=pp_src,
        group=pp_group,
        device="cuda",
    )
    if not is_pp_first:
        batch = _unpack_sidecar_packed_seq_params(received_batch)

    forward_batch = dict(
        _stage_forward_view_from_full_batch(batch, vp_stage, config=config)
    )
    forward_batch["_mdp_pp_cp_sidecar_applied"] = True
    return {
        "batch": forward_batch,
        "vision_embeddings": None,
        "forward_only": bool(forward_only),
    }


def build_mdp_pp_cp_sidecar_cache(
    *,
    data_iterator,
    model,
    vp_stage=None,
    forward_only=False,
):
    """Consume and encode one loader-prepartitioned PP x CP microbatch."""
    if bool(_wrapped_model_attr(model, "_pp_cp_batch_sidecar", False)):
        return _build_pp_batch_sidecar_cache(
            data_iterator=data_iterator,
            model=model,
            vp_stage=vp_stage,
            forward_only=forward_only,
        )

    batch = get_batch(data_iterator)
    if batch is None:
        raise RuntimeError(
            "PP x CP vision sidecar received no batch before pipeline forward"
        )

    pixel_values = batch.get("pixel_values")
    global_image_grid_thw = batch.get("image_grid_thw")
    image_grid_thw = global_image_grid_thw
    if (
        pixel_values is not None
        and pixel_values.is_floating_point()
        and pixel_values.dtype == torch.float32
    ):
        pixel_values = pixel_values.bfloat16()

    pixel_values, image_grid_thw = apply_mdp_prepartition(
        model=model,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        image_grid_thw_rows=batch.get("_balancedata_image_grid_thw_rows"),
        prepartitioned_assignment=batch.get("_mdp_prepartitioned_assignment"),
        prepartitioned_row_counts=batch.get("_mdp_prepartitioned_row_counts"),
        prepartitioned_image_grid_thw=batch.get(
            "_mdp_prepartitioned_image_grid_thw"
        ),
    )
    compute_vision = _wrapped_model_method(
        model, "mdp_pp_cp_sidecar_compute_vision"
    )
    if compute_vision is None:
        raise RuntimeError(
            "PP x CP vision sidecar requires mdp_pp_cp_sidecar_compute_vision"
        )
    context = torch.no_grad() if forward_only else torch.enable_grad()
    with context:
        vision_embeddings = compute_vision(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mdp_cp_local_plan=batch.get("_mdp_cp_local_plan"),
        )

    forward_batch = dict(batch)
    forward_batch["pixel_values"] = None
    # The owner-local grid is only a vision-encoder input. The language
    # forward retains the global grid for MRoPE/position-id fallback.
    forward_batch["image_grid_thw"] = global_image_grid_thw
    forward_batch["_mdp_pp_cp_sidecar_applied"] = True
    return {
        "batch": forward_batch,
        "vision_embeddings": vision_embeddings,
        "forward_only": bool(forward_only),
    }


def _pop_mdp_pp_cp_sidecar_cache(model):
    pop_cache = _wrapped_model_method(model, "mdp_pp_cp_sidecar_pop_cache")
    if pop_cache is None:
        return None
    cache = pop_cache()
    if cache is None:
        return None
    activate_cache = _wrapped_model_method(
        model, "mdp_pp_cp_sidecar_activate_cache"
    )
    if activate_cache is None:
        raise RuntimeError(
            "PP x CP vision sidecar cache requires an activation callback"
        )
    activate_cache(cache)
    return cache


def forward_step(data_iterator, model, return_schedule_plan=False):
    """Forward step for multimodal_dev training."""
    args = get_args()
    sidecar_cache = _pop_mdp_pp_cp_sidecar_cache(model)
    batch = (
        sidecar_cache["batch"]
        if sidecar_cache is not None
        else get_batch(data_iterator)
    )

    if batch is None:
        return None, None

    pixel_values = batch.get("pixel_values", None)
    image_grid_thw = batch.get("image_grid_thw", None)
    if (
        pixel_values is not None
        and pixel_values.is_floating_point()
        and pixel_values.dtype == torch.float32
    ):
        pixel_values = pixel_values.bfloat16()

    if (
        bool(getattr(args, "mdp_encoder_mode", True))
        and not bool(batch.get("_mdp_pp_cp_sidecar_applied", False))
    ):
        pixel_values, image_grid_thw = apply_mdp_prepartition(
            model=model,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_grid_thw_rows=batch.get("_balancedata_image_grid_thw_rows"),
            **_mdp_prepartition_kwargs(batch),
        )

    model_kwargs = dict(
        input_ids=batch.get("input_ids"),
        position_ids=batch.get("position_ids"),
        attention_mask=batch.get("attention_mask", None),
        labels=batch.get("labels", None),
        loss_mask=batch.get("loss_mask", None),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        packed_seq_params=batch.get("packed_seq_params", None),
        mdp_cp_local_plan=batch.get("_mdp_cp_local_plan"),
    )
    if return_schedule_plan:
        args = get_args()
        if not args.overlap_moe_expert_parallel_comm:
            raise RuntimeError(
                "overlap_moe_expert_parallel_comm must be enabled "
                "to return the schedule plan"
            )
        if (
            getattr(args, "model_arch", None) in {"qwen35_vl", "qwen3vl"}
            and model_kwargs["packed_seq_params"] is not None
            and get_context_parallel_world_size() > 1
        ):
            raise RuntimeError(
                "Qwen VL packed THD with context parallelism does not "
                "support overlap_moe_expert_parallel_comm: deferred MRoPE "
                "execution cannot use the regular forward CP override"
            )
        output_tensor = model.build_schedule_plan(**model_kwargs)
    else:
        # We don't provide position_ids, now. Let model handle it itself.
        output_tensor = model(**model_kwargs)

    loss_mask = batch.get("loss_mask", None)
    if loss_mask is None and batch.get("input_ids") is not None:
        loss_mask = torch.ones_like(
            batch["input_ids"], dtype=torch.float,
        )

    # CP-split loss_mask to match the model output (which is CP-split
    # inside MultimodalModel.forward / Qwen35VLModel.forward).
    # THD: use the same TE-based per-sample partition index as the model.
    # BSHD: use the matching zigzag split.
    cp_size = get_context_parallel_world_size()
    if (
        cp_size > 1
        and loss_mask is not None
        and not batch.get("_data_side_cp_sharded", False)
    ):
        from examples.multimodal_dev.models.base import (
            _cp_split_tensor,
            _thd_cp_partition_index,
        )

        cp_rank = get_context_parallel_rank()
        psp = batch.get("packed_seq_params", None)
        if psp is not None:
            idx = _thd_cp_partition_index(
                psp.cu_seqlens_q_padded,
                loss_mask.shape[1], cp_size, cp_rank,
            )
            loss_mask = loss_mask.index_select(1, idx)
        else:
            loss_mask = _cp_split_tensor(
                loss_mask, seq_dim=1,
                cp_size=cp_size, cp_rank=cp_rank,
            )

    return output_tensor, partial(loss_func, loss_mask)
