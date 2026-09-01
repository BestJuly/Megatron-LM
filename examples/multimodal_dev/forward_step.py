# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward step, TP broadcast, and loss for multimodal_dev training."""

import math
from functools import partial
from itertools import accumulate
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn.functional as F

from examples.multimodal_dev.observability import nvtx_phase
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams, build_static_thd_metadata
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.utils import get_attr_wrapped_model
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
    ndim = torch.tensor(
        [len(tensor.shape) if tensor is not None else 0], dtype=torch.long, device=device
    )
    torch.distributed.broadcast(ndim, src, group=group)

    if ndim.item() == 0:
        return None

    if tensor is not None:
        shape_tensor = torch.tensor(list(tensor.shape), dtype=torch.long, device=device)
        dtype_id = torch.tensor([_dtype_to_id(tensor.dtype)], dtype=torch.long, device=device)
    else:
        shape_tensor = torch.zeros(ndim.item(), dtype=torch.long, device=device)
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

    # Single-member TP group: every rank is the source; ~4 broadcasts per
    # field (ndim/shape/dtype/payload) would be pure launch overhead. Keep
    # only the device move. Pinned sources move without the implicit
    # per-copy device sync of pageable H2D (same bytes, same stream order).
    if torch.distributed.get_world_size(group=group) == 1:
        return {
            key: (
                value.to(device, non_blocking=value.is_pinned())
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in data.items()
        }

    if get_tensor_model_parallel_rank() == 0:
        keys = list(data.keys())
        key_str = ",".join(keys)
        key_bytes = key_str.encode("utf-8")
        key_len = torch.tensor([len(key_bytes)], dtype=torch.long, device=device)
    else:
        key_len = torch.zeros(1, dtype=torch.long, device=device)
        keys = []

    torch.distributed.broadcast(key_len, src, group=group)

    if get_tensor_model_parallel_rank() == 0:
        key_tensor = torch.tensor(list(key_bytes), dtype=torch.uint8, device=device)
    else:
        key_tensor = torch.zeros(key_len.item(), dtype=torch.uint8, device=device)

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
            tensor if isinstance(tensor, torch.Tensor) else None, src, group, device
        )

    return result


# -------------------------------------------------------------------
# THD (packed sequence) helpers
# -------------------------------------------------------------------


def accumulate_flops_stats(packed_seq_params, real_cu_seqlens=None) -> None:
    """Feed one micro-batch's real ``cu_seqlens`` into the FLOPs accumulators.

    Called from the forward step -- once per micro-batch, on every rank of the
    model-parallel group, on the main compute stream -- so the logged TFLOP/s
    reflects the tokens actually packed instead of the BSHD closed form
    ``micro_batch_size * seq_length`` (meaningless here: the packed multimodal
    datasets ignore ``--seq-length`` entirely).
    ``consume_seqlen_stats_in_iteration`` divides the world all-reduce by
    ``TP * CP * PP``, which matches this call pattern.

    Deliberately NOT called from the collate path: under
    ``--mdp-overlap-window-capture`` the collate for iteration ``i+1`` runs on a
    background thread and a side CUDA stream during iteration ``i``, which would
    both mis-attribute the tokens by one iteration and enqueue the accumulation
    off the main stream. The forward step consumes the captured window only
    after the main stream has waited on the capture event, so it is ordered.

    ``cu_seqlens_q`` is the REAL (unpadded) cumulative length vector;
    ``cu_seqlens_q_padded`` carries the collate/CP alignment padding and is
    intentionally not used. The reduction stays on device (no ``.item()``), so
    no host sync is added to the hot path.

    ``real_cu_seqlens`` overrides ``cu_seqlens_q``. Under ``--thd-static-packing``
    the tail policy is always ``append_dummy_seq``, so the static pad is
    represented as an ordinary extra sequence and therefore lands in
    ``cu_seqlens_q`` itself. Accumulating that would overstate ``sum(L)`` and
    ``sum(L^2)`` by the padding fraction -- exactly the shape of a false speedup,
    since it appears only on the padded side. The collator emits the pre-tail
    vector as ``flops_cu_seqlens`` in that case; without static packing there is
    no tail and no override.

    Imported lazily: ``megatron.training.training`` pulls in the whole training
    stack, and this module is also imported by unit tests that never build it.
    """
    if packed_seq_params is None:
        # BSHD path: leave the accumulator untouched so
        # ``num_floating_point_operations`` keeps its closed-form defaults.
        return
    cu_seqlens = (
        real_cu_seqlens
        if real_cu_seqlens is not None
        else getattr(packed_seq_params, "cu_seqlens_q", None)
    )
    if cu_seqlens is None:
        return
    try:
        from megatron.training.training import update_seqlen_stats_from_cu_seqlens
    except ImportError:  # pragma: no cover - training stack unavailable
        return
    update_seqlen_stats_from_cu_seqlens(cu_seqlens)


def accumulate_vision_flops_stats_from_grids(grid_thw) -> None:
    """Report one micro-batch's vision work from a ``[N, 3]`` ``grid_thw``.

    Used by the native (in-model encoder) path, where ``image_grid_thw`` is
    broadcast to every PP stage. Stays on device: the two reductions are fused
    kernels, no ``.item()``.
    """
    if grid_thw is None or grid_thw.numel() == 0:
        return
    t, h, w = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    frame = (h * w).to(torch.float64)
    t = t.to(torch.float64)
    _update_vision_stats((t * frame).sum(), (t * frame * frame).sum())


def accumulate_vision_flops_stats_from_items(vision_items) -> None:
    """Report one micro-batch's vision work from MDP's captured vision records.

    The MDP replay record carries ``grid_thw`` as plain Python tuples on every
    rank (only the pixel payload is owner-sharded), so the two sums are computed
    on the host and added to the device accumulator as scalars. Byte-for-byte
    the same quantities the native path reports for the same data.
    """
    rows = 0
    attn_sq = 0
    for item in vision_items:
        t, h, w = (int(v) for v in item.grid_thw)
        frame = h * w
        rows += t * frame
        attn_sq += t * frame * frame
    if rows:
        _update_vision_stats(float(rows), float(attn_sq))


def _accumulate_workload_stats(
    model, packed_seq_params, *, vision_items=None, image_grid_thw=None, real_cu_seqlens=None
) -> None:
    """Report one micro-batch's decoder and vision work exactly once per rank.

    With virtual pipeline parallelism, every model chunk on a physical rank
    invokes the forward step for the same micro-batch. The global consumer
    de-duplicates replicated statistics by ``TP * CP * PP``, so letting every
    virtual chunk report would overcount by the VPP size. Chunk zero is the
    canonical reporter; non-VPP models expose ``vp_stage=None`` and retain the
    original behavior.
    """
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    if vp_stage not in (None, 0):
        return

    accumulate_flops_stats(packed_seq_params, real_cu_seqlens=real_cu_seqlens)
    if vision_items is not None:
        accumulate_vision_flops_stats_from_items(vision_items)
    else:
        accumulate_vision_flops_stats_from_grids(image_grid_thw)


def _update_vision_stats(patch_rows, attn_squared_sum) -> None:
    """Forward vision work to the training-loop accumulator (lazy import)."""
    try:
        from megatron.training.training import update_vision_stats
    except ImportError:  # pragma: no cover - training stack unavailable
        return
    update_vision_stats(patch_rows, attn_squared_sum)


def _build_packed_seq_params(seq_lengths: torch.Tensor, device: torch.device) -> PackedSeqParams:
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
    cu_seqlens = torch.zeros(lengths_t.numel() + 1, dtype=torch.int32, device=device)
    torch.cumsum(lengths_t, dim=0, out=cu_seqlens[1:])
    max_seqlen = int(lengths_t.max().item())
    return _build_packed_seq_params_from_cu_seqlens(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


def _build_packed_seq_params_from_cu_seqlens(
    cu_seqlens: torch.Tensor, max_seqlen: int
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


# Columns of ``vision_item_meta``: sample_index, image_ordinal, grid t/h/w,
# payload_row_start, sample_padded_start, sample_padded_len.
VISION_ITEM_META_COLUMNS = 8


def build_vision_sidecar(
    batch: list[Dict[str, Any]],
    cu_seqlens_padded: list[int],
    image_token_id: int,
    spatial_merge_size: int,
) -> Dict[str, torch.Tensor]:
    """Build the per-item vision sidecar for a THD-packed batch.

    For every vision item, records ``(sample_index, image_ordinal, t, h, w,
    payload_row_start, sample_padded_start, sample_padded_len)`` plus that
    item's decoder image-token positions in the packed ``[1, T]`` layout,
    ordered by ``(sample_index, image_ordinal)``. Both outputs are plain
    integer tensors so they survive the TP broadcast.

    The last two columns are the enclosing sample's span in
    ``cu_seqlens_padded``. They are what lets MDP decide, in pure integer host
    arithmetic and without a device round-trip, which context-parallel rank
    owns each of the item's decoder rows once the decoder is CP-sharded (see
    :mod:`megatron.core.mdp.cp_partition`). They are free here because
    ``cu_seqlens_padded`` is already a host list at this point.

    Consistency guards (fail the batch rather than silently degrade):

    * pixel data and grid metadata are all-or-nothing per sample;
    * per-sample pixel rows equal ``sum(t*h*w)`` over its grids;
    * per-sample image-token slots equal ``sum(t*(h/m)*(w/m))``, so truncation
      can never leave a cut image block;
    * every item's token slots are contiguous in the sample.
    """
    meta_rows = []
    position_chunks = []
    payload_row_start = 0
    merge = spatial_merge_size
    for sample_index, sample in enumerate(batch):
        grids = sample.get("image_grid_thw")
        pixels = sample.get("pixel_values")
        num_items = 0 if grids is None else int(grids.shape[0])
        pixel_rows = 0 if pixels is None else int(pixels.shape[0])
        if (num_items == 0) != (pixel_rows == 0):
            raise ValueError(
                f"sample {sample_index}: pixel data and grid metadata must either both "
                f"exist or both be absent (items={num_items}, pixel_rows={pixel_rows})"
            )
        input_ids = sample["input_ids"]
        image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        expected_rows = 0
        expected_slots = 0
        item_slot_counts = []
        for ordinal in range(num_items):
            t, h, w = (int(v) for v in grids[ordinal])
            if h % merge != 0 or w % merge != 0:
                raise ValueError(
                    f"sample {sample_index} item {ordinal}: grid ({t},{h},{w}) not "
                    f"divisible by spatial_merge_size={merge}"
                )
            expected_rows += t * h * w
            item_slot_counts.append(t * (h // merge) * (w // merge))
            expected_slots += item_slot_counts[-1]
        if expected_rows != pixel_rows:
            raise ValueError(
                f"sample {sample_index}: pixel rows {pixel_rows} != sum(t*h*w) "
                f"{expected_rows} over its grids"
            )
        if int(image_positions.numel()) != expected_slots:
            raise ValueError(
                f"sample {sample_index}: {int(image_positions.numel())} image-token "
                f"slots != sum(t*(h/m)*(w/m)) {expected_slots}; truncation must never "
                "cut an image-token block"
            )
        slot_cursor = 0
        for ordinal in range(num_items):
            t, h, w = (int(v) for v in grids[ordinal])
            slots = item_slot_counts[ordinal]
            item_positions = image_positions[slot_cursor : slot_cursor + slots]
            slot_cursor += slots
            if slots and int(item_positions[-1] - item_positions[0]) != slots - 1:
                raise ValueError(
                    f"sample {sample_index} item {ordinal}: image-token slots are not "
                    "contiguous"
                )
            meta_rows.append(
                [
                    sample_index,
                    ordinal,
                    t,
                    h,
                    w,
                    payload_row_start,
                    cu_seqlens_padded[sample_index],
                    cu_seqlens_padded[sample_index + 1] - cu_seqlens_padded[sample_index],
                ]
            )
            payload_row_start += t * h * w
            position_chunks.append(item_positions.to(torch.int64) + cu_seqlens_padded[sample_index])
    if meta_rows:
        return {
            "vision_item_meta": torch.tensor(meta_rows, dtype=torch.int64),
            "vision_decoder_positions": torch.cat(position_chunks),
        }
    return {
        "vision_item_meta": torch.empty(0, VISION_ITEM_META_COLUMNS, dtype=torch.int64),
        "vision_decoder_positions": torch.empty(0, dtype=torch.int64),
    }


def pack_or_pad_batch(
    batch: Optional[list[Dict[str, Any]]],
    use_packed_sequence: bool = False,
    seq_length: Optional[int] = None,
    device="cuda",
    pad_to_multiple: Optional[int] = None,
    with_vision_sidecar: bool = False,
) -> Dict[str, Any]:
    """Pack or pad a ``[B, S]`` batch into ``[1, T]`` THD or ``[B, S]`` BSHD.

    Must be invoked on every TP rank. On the TP source rank ``batch`` is
    the per-sample dict list from the dataset; on other TP ranks ``batch``
    may be ``None`` (the function relies on the trailing TP broadcast to
    distribute results). All metadata needed to reconstruct
    ``PackedSeqParams`` (``cu_seqlens``, ``cu_seqlens_padded``,
    ``max_seqlen``, ``total_tokens``) is broadcast alongside the data, so
    every rank can build an identical ``PackedSeqParams`` on its own.
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    is_src = mpu.get_tensor_model_parallel_rank() == 0

    # SP is an explicit runtime option; TP>1 does not imply SP is enabled.
    # get_args() itself raises in test contexts where megatron globals are
    # not initialised.
    try:
        has_sp = bool(getattr(get_args(), "sequence_parallel", False))
    except AssertionError:
        has_sp = False

    if cp_size > 1:
        divisible_by = (tp_size * cp_size * 2) if has_sp else (cp_size * 2)
    else:
        divisible_by = tp_size if has_sp else 1
    if pad_to_multiple is not None:
        divisible_by = max(divisible_by, pad_to_multiple)

    if use_packed_sequence:
        packed_batch: Dict[str, Any] = {}

        # --thd-static-packing: emit a fixed-shape THD batch. Every microbatch
        # becomes exactly `max_seqlen_per_dp_cp_rank * cp_size` rows with
        # cu_seqlens* of `thd_max_packed_sequences + 1` entries, which is the
        # contract MCore's THD CUDA-graph machinery expects from a packing
        # scheduler. This collator works in GLOBAL (pre-CP-slice) coordinates:
        # CP slicing happens later in models/base.py.
        static_target_T = None
        static_max_num_seqs = None
        try:
            static_args = get_args()
        except AssertionError:
            static_args = None
        if static_args is not None and getattr(static_args, "thd_static_packing", False):
            static_target_T = int(static_args.max_seqlen_per_dp_cp_rank) * cp_size
            static_max_num_seqs = int(static_args.thd_max_packed_sequences)
            # The tail policy is always append_dummy_seq here, matching what
            # --sequence-packing-scheduler produces; TransformerConfig rejects
            # thd_static_packing + extend_last, which is unusable at any CP size
            # (it leaves cu_seqlens_q ending at the real token count while the
            # tensors are padded to target_T, and TE then returns a shorter
            # attention output than the padded input -- a view mismatch in
            # Attention._apply_output_gate).
            #
            # The cost is that the pad tail becomes an ordinary sequence in
            # cu_seqlens_q, which would inflate the FLOPs accumulator; the
            # pre-tail vector is therefore emitted separately (see
            # accumulate_flops_stats).

        # Owner-sharded pixel reading: during MDP window capture of a
        # microbatch owned by another worker, skip pixel
        # materialization + H2D wholesale. All text tensors and vision item
        # metadata (grid_thw, sidecar) are still built from input_ids/grids,
        # so every offset stays valid. False outside a sharded MDP capture.
        from megatron.core.mdp.window import pixel_capture_suppressed

        suppress_pixels = pixel_capture_suppressed()

        # MDP capture fast path (TP=1): build each packed field directly in
        # one pinned buffer (no per-sample F.pad + concat churn) and move it
        # with a non-blocking copy. Pageable H2D copies each carry an implicit
        # device sync that serializes the window-capture (prefetch) thread;
        # pinned + non_blocking removes the sync and the staging pass. Output
        # bytes are identical to the generic path. torch's caching host
        # allocator recycles the pinned blocks and event-tracks their reuse.
        try:
            use_pinned = (
                bool(getattr(get_args(), "mdp_enable", False)) and tp_size == 1
            )
        except AssertionError:
            use_pinned = False

        if is_src:
            assert batch is not None, "source TP rank must provide a batch"
            input_ids_list, labels_list, loss_mask_list = [], [], []
            pixel_values_list, image_grid_thw_list = [], []
            seqlens_list, seqlens_padded_list = [], []

            for sample in batch:
                seqlen = sample["input_ids"].shape[0]
                assert (
                    sample["labels"].shape == sample["input_ids"].shape == sample["loss_mask"].shape
                ), "labels, input_ids, and loss_mask must have the same shape"
                target_len = math.ceil(seqlen / divisible_by) * divisible_by
                if not use_pinned:
                    input_ids_list.append(
                        F.pad(sample["input_ids"], (0, target_len - seqlen), value=0)
                    )
                    labels_list.append(
                        F.pad(sample["labels"], (0, target_len - seqlen), value=-100)
                    )
                    loss_mask_list.append(
                        F.pad(sample["loss_mask"], (0, target_len - seqlen), value=0)
                    )
                seqlens_list.append(seqlen)
                seqlens_padded_list.append(target_len)
                if not suppress_pixels:
                    pixel_values_list.append(sample["pixel_values"])
                image_grid_thw_list.append(sample["image_grid_thw"])

            cu_seqlens = list(accumulate(seqlens_list, initial=0))
            cu_seqlens_padded = list(accumulate(seqlens_padded_list, initial=0))

            # padding_mask: True at collate-padded positions within each packed
            # sample. Real tokens occupy [cu_seqlens_padded[i], +seqlens_list[i]);
            # the tail up to cu_seqlens_padded[i+1] is padding. Consumed by MoE
            # routing in megatron.core to exclude padded tokens from aux loss,
            # z-loss, and expert-bias accumulation.
            total_tokens_padded = cu_seqlens_padded[-1]
            # Physical row count of the emitted tensors. Under static packing it
            # is the fixed target, so the tail beyond the pack is padding too.
            physical_T = total_tokens_padded
            if static_target_T is not None:
                assert total_tokens_padded <= static_target_T, (
                    f"Packed THD length ({total_tokens_padded}) exceeds the static "
                    f"target ({static_target_T}). Increase "
                    "--max-seqlen-per-dp-cp-rank, or lower the number of samples per "
                    "microbatch (--micro-batch-size, or the greedy token budget)."
                )
                physical_T = static_target_T
            padding_mask_thd = torch.zeros(
                physical_T, dtype=torch.bool, pin_memory=use_pinned
            )
            for i, real_seqlen in enumerate(seqlens_list):
                pad_start = cu_seqlens_padded[i] + real_seqlen
                pad_end = cu_seqlens_padded[i + 1]
                if pad_end > pad_start:
                    padding_mask_thd[pad_start:pad_end] = True
            if physical_T > total_tokens_padded:
                padding_mask_thd[total_tokens_padded:] = True

            if use_pinned:
                # Single padded buffer per field; pad regions filled with the
                # same values F.pad used, sample slices copied in place. Sized to
                # physical_T so static packing costs no second copy.
                def _packed_field(key, fill):
                    out = torch.empty(
                        physical_T, dtype=batch[0][key].dtype, pin_memory=True
                    )
                    out.fill_(fill)
                    for i, sample in enumerate(batch):
                        start = cu_seqlens_padded[i]
                        out[start : start + seqlens_list[i]].copy_(sample[key])
                    return out

                input_ids_list = [_packed_field("input_ids", 0)]
                labels_list = [_packed_field("labels", -100)]
                loss_mask_list = [_packed_field("loss_mask", 0)]

            if with_vision_sidecar:
                try:
                    args = get_args()
                    sidecar_image_token_id = getattr(args, "image_token_id", 248056)
                    sidecar_merge = getattr(args, "vision_spatial_merge_size", None) or 2
                except AssertionError:
                    sidecar_image_token_id = 248056
                    sidecar_merge = 2
                packed_batch.update(
                    build_vision_sidecar(
                        batch,
                        cu_seqlens_padded,
                        image_token_id=sidecar_image_token_id,
                        spatial_merge_size=sidecar_merge,
                    )
                )

            if use_pinned:
                # The fields already live in single pinned buffers; a concat
                # of a one-element list would copy them into fresh pageable
                # memory and forfeit the non-blocking upload.
                packed_batch["input_ids"] = input_ids_list[0].unsqueeze(0)
                packed_batch["labels"] = labels_list[0].unsqueeze(0)
                packed_batch["loss_mask"] = loss_mask_list[0].unsqueeze(0)
            else:
                def _concat_field(pieces, fill):
                    packed = torch.concat(pieces, dim=0)
                    tail = physical_T - packed.shape[0]
                    if tail:
                        packed = F.pad(packed, (0, tail), value=fill)
                    return packed.unsqueeze(0)

                packed_batch["input_ids"] = _concat_field(input_ids_list, 0)
                packed_batch["labels"] = _concat_field(labels_list, -100)
                packed_batch["loss_mask"] = _concat_field(loss_mask_list, 0)
            packed_batch["padding_mask"] = padding_mask_thd.unsqueeze(0)
            if not suppress_pixels:
                if use_pinned and pixel_values_list:
                    total_rows = sum(int(p.shape[0]) for p in pixel_values_list)
                    pixels = torch.empty(
                        (total_rows,) + tuple(pixel_values_list[0].shape[1:]),
                        dtype=pixel_values_list[0].dtype,
                        pin_memory=True,
                    )
                    torch.cat(pixel_values_list, out=pixels)
                    packed_batch["pixel_values"] = pixels
                else:
                    packed_batch["pixel_values"] = torch.concat(pixel_values_list)
            grid_thw = torch.concat(image_grid_thw_list)
            packed_batch["image_grid_thw"] = (
                grid_thw.pin_memory() if use_pinned else grid_thw
            )
            # cu_seqlens / cu_seqlens_padded need to reach non-source TP ranks
            # so each rank can build an identical PackedSeqParams.
            if use_pinned:
                packed_batch["cu_seqlens"] = torch.tensor(
                    cu_seqlens, dtype=torch.int32
                ).pin_memory()
                packed_batch["cu_seqlens_padded"] = torch.tensor(
                    cu_seqlens_padded, dtype=torch.int32
                ).pin_memory()
            else:
                packed_batch["cu_seqlens"] = torch.tensor(
                    cu_seqlens, dtype=torch.int32, device=device
                )
                packed_batch["cu_seqlens_padded"] = torch.tensor(
                    cu_seqlens_padded, dtype=torch.int32, device=device
                )

        # The vision sidecar is consumed on the CPU by the MDP adapter; with a
        # single-member TP group there is no broadcast, so skip the GPU round
        # trip (H2D here + D2H in the adapter) entirely.
        sidecar_cpu = {}
        if is_src and use_pinned:
            for key in ("vision_item_meta", "vision_decoder_positions"):
                if key in packed_batch:
                    sidecar_cpu[key] = packed_batch.pop(key)

        packed_batch = broadcast_data_batch(packed_batch, device=device)
        packed_batch.update(sidecar_cpu)

        cu_seqlens_t = packed_batch.pop("cu_seqlens")
        cu_seqlens_padded_t = packed_batch.pop("cu_seqlens_padded")
        if is_src and use_pinned:
            # Known on the host already; reading them back from the device
            # would force a sync against the in-flight non-blocking copies.
            max_seqlen_q = max(seqlens_padded_list) if seqlens_padded_list else 0
            total_tokens = cu_seqlens_padded[-1]
        else:
            # Derive max_seqlen / total_tokens from the (broadcast) cu_seqlens —
            # no extra collective needed.
            max_seqlen_q = int((cu_seqlens_padded_t[1:] - cu_seqlens_padded_t[:-1]).max().item())
            total_tokens = int(cu_seqlens_padded_t[-1].item())

        pad_between_seqs = None
        if static_target_T is not None:
            cu_seqlens_t, cu_seqlens_padded_t, real_cu_seqlens_t = build_static_thd_metadata(
                cu_seqlens_t,
                cu_seqlens_padded_t,
                target_len=static_target_T,
                max_num_seqs=static_max_num_seqs,
                cp_size=cp_size,
            )
            # max_seqlen must be the padded static value: the tail belongs to a
            # sequence now, and a stale (shorter) max silently produces wrong
            # attention rather than a crash.
            max_seqlen_q = static_target_T
            total_tokens = static_target_T
            # Must be batch-independent (that is the point of static shapes), so
            # derive it from the alignment rather than from this batch's
            # vectors: with divisible_by == 1 no sample is ever padded, so
            # cu_seqlens and cu_seqlens_padded coincide and there is provably no
            # gap between sequences. Saying True there is not free -- it makes
            # FlashAttention ineligible and, when the fused cuDNN backend is not
            # selected either, drops TE onto its unfused O(T^2) attention, which
            # OOMs at these lengths.
            pad_between_seqs = divisible_by > 1
            if real_cu_seqlens_t is not None:
                # append_dummy_seq put the tail into cu_seqlens_q itself, which
                # would inflate sum(L) and sum(L^2) in the FLOPs accumulator.
                packed_batch["flops_cu_seqlens"] = real_cu_seqlens_t

        packed_batch["packed_seq_params"] = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_t,
            cu_seqlens_kv=cu_seqlens_t,
            cu_seqlens_q_padded=cu_seqlens_padded_t,
            cu_seqlens_kv_padded=cu_seqlens_padded_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
            total_tokens=total_tokens,
            pad_between_seqs=pad_between_seqs,
        )
        return packed_batch

    # ---------- padded (BSHD) branch ----------
    assert seq_length is not None, "seq_length must be provided when use_packed_sequence is False"
    padded_batch: Dict[str, Any] = {}

    if is_src:
        assert batch is not None, "source TP rank must provide a batch"
        max_seqlens = max(x["input_ids"].shape[0] for x in batch)
        target_seqlens = min(max_seqlens, seq_length)
        # Round target seqlen up to the parallelism alignment factor so the
        # batched tensor is divisible for CP (+SP) splitting downstream.
        if divisible_by > 1:
            target_seqlens = math.ceil(target_seqlens / divisible_by) * divisible_by

        # Capture real lengths before in-place padding so we can build a
        # padding_mask for MoE routing (True at collate-padded positions).
        real_seqlens = [s["input_ids"].shape[0] for s in batch]

        for sample in batch:
            sample["input_ids"] = F.pad(
                sample["input_ids"], (0, target_seqlens - sample["input_ids"].shape[0]), value=0
            )
            sample["labels"] = F.pad(
                sample["labels"], (0, target_seqlens - sample["labels"].shape[0]), value=-100
            )
            sample["loss_mask"] = F.pad(
                sample["loss_mask"], (0, target_seqlens - sample["loss_mask"].shape[0]), value=0
            )

        padded_batch["input_ids"] = torch.concat(
            [x["input_ids"].unsqueeze(0) for x in batch], dim=0
        )
        padded_batch["labels"] = torch.concat([x["labels"].unsqueeze(0) for x in batch], dim=0)
        padded_batch["loss_mask"] = torch.concat(
            [x["loss_mask"].unsqueeze(0) for x in batch], dim=0
        )
        # Keep None as the known-no-padding fast path for MoE routing.
        has_padding = any(real_seqlen < target_seqlens for real_seqlen in real_seqlens)
        if has_padding:
            positions = torch.arange(target_seqlens).unsqueeze(0)
            padded_batch["padding_mask"] = positions >= torch.tensor(real_seqlens).unsqueeze(1)
        padded_batch["pixel_values"] = torch.concat([x["pixel_values"] for x in batch])
        padded_batch["image_grid_thw"] = torch.concat([x["image_grid_thw"] for x in batch])

    return broadcast_data_batch(padded_batch, device=device)


# -------------------------------------------------------------------
# get_batch
# -------------------------------------------------------------------


def get_batch(data_iterator: Iterator[list[Dict[str, Any]]]):
    """Get a batch from *data_iterator* and broadcast across TP ranks."""
    device = "cuda"
    args = get_args()

    group = get_tensor_model_parallel_group()
    # Single-member TP group: skip the device flag tensor and the broadcast
    # entirely. Behavior-identical, and it keeps the MDP window-capture
    # prefetch thread free of NCCL calls (--mdp-overlap-window-capture).
    if torch.distributed.get_world_size(group=group) == 1:
        try:
            data = next(data_iterator)
        except StopIteration:
            return None
    else:
        if get_tensor_model_parallel_rank() == 0:
            try:
                data = next(data_iterator)
                has_data = torch.tensor([1], dtype=torch.uint8, device=device)
            except StopIteration:
                has_data = torch.tensor([0], dtype=torch.uint8, device=device)
                data = None
        else:
            has_data = torch.empty(1, dtype=torch.uint8, device=device)
            data = None

        src = get_tensor_model_parallel_src_rank()
        torch.distributed.broadcast(has_data, src, group=group)

        if has_data.item() == 0:
            return None

    # Because broadcast will not broadcast packed_seq_params, we move it into pack_or_pad_batch
    batch = pack_or_pad_batch(
        data,
        args.use_packed_sequence,
        args.seq_length,
        device=device,
        with_vision_sidecar=getattr(args, "mdp_enable", False),
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

    if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
        g = batch["image_grid_thw"]
        if g.dim() == 3:
            batch["image_grid_thw"] = g.squeeze(1)

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
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    return (total_loss, total_tokens, {"lm loss": reporting_loss})


# -------------------------------------------------------------------
# Forward step
# -------------------------------------------------------------------


def mdp_forward_step(runtime, data_iterator, model, return_schedule_plan: bool = False):
    """Forward step over an MDP replay record.

    The iterator yields immutable ``MdpMicrobatchRecord`` objects captured in
    P1. Pixels never reach the decoder: the first PP stage receives the
    pre-encoded detached leaf from endpoint storage instead.  The EP-overlap
    path builds a decoder-only schedule plan from that same leaf.
    """
    record = next(data_iterator)
    batch = dict(record.model_payload)

    _accumulate_workload_stats(
        model,
        record.decoder_packed_seq_params,
        vision_items=record.vision_items,
        real_cu_seqlens=batch.get("flops_cu_seqlens"),
    )

    vision_embeddings = None
    # `record.text_only` is a whole-microbatch property; whether THIS rank holds
    # vision rows is a per-(microbatch, cp_rank) fact only the plan knows. Under
    # decoder CP an endpoint with no rows for a vision-bearing microbatch is a
    # normal state, not a routing failure.
    if is_pipeline_first_stage() and runtime.expects_leaf(record.microbatch_id):
        vision_embeddings = runtime.storage.get_leaf(record.microbatch_id)
        if vision_embeddings is None:
            raise RuntimeError(
                f"MDP: microbatch {record.microbatch_id} has vision rows for this "
                "endpoint but no leaf in endpoint storage; P3 embedding routing "
                "did not complete"
            )

    model_inputs = dict(
        input_ids=batch["input_ids"],
        position_ids=batch.get("position_ids"),
        attention_mask=batch.get("attention_mask", None),
        labels=batch.get("labels", None),
        loss_mask=batch.get("loss_mask", None),
        padding_mask=batch.get("padding_mask", None),
        pixel_values=None,
        image_grid_thw=batch.get("image_grid_thw", None),
        packed_seq_params=record.decoder_packed_seq_params,
        vision_embeddings=vision_embeddings,
    )
    if return_schedule_plan:
        assert get_args().overlap_moe_expert_parallel_comm, (
            "overlap_moe_expert_parallel_comm must be enabled to return a schedule plan"
        )
        output_tensor = model.build_schedule_plan(**model_inputs)
    else:
        output_tensor = model(**model_inputs)

    loss_mask = batch.get("loss_mask", None)
    if loss_mask is None:
        loss_mask = torch.ones_like(batch["input_ids"], dtype=torch.float)
    if is_pipeline_last_stage():
        from examples.multimodal_dev.models.base import MultimodalModel

        loss_mask = MultimodalModel.cp_split_loss_mask(
            loss_mask, record.decoder_packed_seq_params
        )
    return output_tensor, partial(loss_func, loss_mask)


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward step for multimodal_dev training.

    ``return_schedule_plan`` is requested only by MCore's native decoder EP
    communication-overlap scheduler.  The default path remains an eager model
    forward.
    """
    from megatron.core.mdp import integration as mdp_integration

    mdp_runtime = mdp_integration.get_runtime()
    if mdp_runtime is not None:
        return mdp_forward_step(
            mdp_runtime, data_iterator, model, return_schedule_plan=return_schedule_plan
        )

    # Native counterpart of mdp.p1_get_batch: the dataset fetch + THD pack
    # + TP broadcast. MDP hoists this out of the schedule into window capture,
    # so a like-for-like timeline comparison needs it named on both sides.
    with nvtx_phase("get_batch"):
        batch = get_batch(data_iterator)

    if batch is None:
        return None, None

    _accumulate_workload_stats(
        model,
        batch.get("packed_seq_params", None),
        image_grid_thw=batch.get("image_grid_thw", None),
        real_cu_seqlens=batch.get("flops_cu_seqlens"),
    )

    # ``pixel_values`` is the heavy vision tensor and is only consumed
    # on the first PP stage; drop it elsewhere.  ``image_grid_thw`` is
    # small and is needed on every PP stage by ``compute_position_ids``
    # (MRoPE freqs are computed per-stage from position_ids).
    is_first = is_pipeline_first_stage()
    is_last = is_pipeline_last_stage()

    pixel_values = batch.get("pixel_values", None) if is_first else None
    image_grid_thw = batch.get("image_grid_thw", None)
    # A text-only microbatch collates to zero pixel rows; take the text path.
    if pixel_values is not None and pixel_values.shape[0] == 0:
        pixel_values = None
        image_grid_thw = None
    if (
        pixel_values is not None
        and pixel_values.is_floating_point()
        and pixel_values.dtype == torch.float32
    ):
        pixel_values = pixel_values.bfloat16()

    model_inputs = dict(
        input_ids=batch["input_ids"],
        position_ids=batch.get("position_ids"),
        attention_mask=batch.get("attention_mask", None),
        labels=batch.get("labels", None),
        loss_mask=batch.get("loss_mask", None),
        padding_mask=batch.get("padding_mask", None),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        packed_seq_params=batch.get("packed_seq_params", None),
    )
    if return_schedule_plan:
        assert get_args().overlap_moe_expert_parallel_comm, (
            "overlap_moe_expert_parallel_comm must be enabled to return a schedule plan"
        )
        output_tensor = model.build_schedule_plan(**model_inputs)
    else:
        output_tensor = model(**model_inputs)

    loss_mask = batch.get("loss_mask", None)
    if loss_mask is None:
        loss_mask = torch.ones_like(batch["input_ids"], dtype=torch.float)

    # Slice loss_mask the same way the model sliced its inputs, so the
    # mask aligns with the CP-shard output.  Delegated to MultimodalModel
    # so the slicing rule lives in one place.  The PP scheduler only
    # invokes the loss closure on the last PP stage, so on non-last
    # stages the mask is left untouched.
    if is_last:
        from examples.multimodal_dev.models.base import MultimodalModel

        loss_mask = MultimodalModel.cp_split_loss_mask(
            loss_mask, batch.get("packed_seq_params", None)
        )

    return output_tensor, partial(loss_func, loss_mask)
