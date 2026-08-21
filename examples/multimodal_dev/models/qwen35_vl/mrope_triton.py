# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels for Qwen3.5-VL THD MRoPE position IDs."""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _thd_mrope_position_ids_kernel(
    input_ids_ptr,
    cu_seqlens_ptr,
    cu_seqlens_padded_ptr,
    image_grid_thw_ptr,
    position_ids_ptr,
    deltas_ptr,
    TOTAL_T: tl.constexpr,
    NUM_IMAGES: tl.constexpr,
    IMAGE_TOKEN_ID: tl.constexpr,
    VISION_START_TOKEN_ID: tl.constexpr,
    SPATIAL_MERGE_SIZE: tl.constexpr,
    HAS_IMAGES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate all three MRoPE axes for one packed segment."""
    segment_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    segment_start = tl.load(cu_seqlens_padded_ptr + segment_idx)
    segment_end = tl.load(cu_seqlens_padded_ptr + segment_idx + 1)
    physical_len = segment_end - segment_start
    valid_len = tl.load(cu_seqlens_ptr + segment_idx + 1) - tl.load(cu_seqlens_ptr + segment_idx)
    physical_mask = offsets < physical_len
    valid_mask = offsets < valid_len
    token_offsets = segment_start + offsets

    # THD padding follows the BSHD convention and receives position 1.
    position_t = tl.where(valid_mask, offsets, 1)
    position_h = position_t
    position_w = position_t

    if HAS_IMAGES:
        tokens = tl.load(input_ids_ptr + token_offsets, mask=valid_mask, other=-1)
        previous_tokens = tl.load(
            input_ids_ptr + token_offsets - 1, mask=valid_mask & (offsets > 0), other=-1
        )
        image_starts = (tokens == IMAGE_TOKEN_ID) & (previous_tokens == VISION_START_TOKEN_ID)

        # image_grid_thw is ordered globally across packed segments. Count the
        # image starts before this segment entirely on device so no host scalar
        # extraction is required.
        prefix_offsets = tl.arange(0, BLOCK_SIZE)
        prefix_mask = prefix_offsets < segment_start
        prefix_tokens = tl.load(
            input_ids_ptr + prefix_offsets, mask=prefix_mask & (prefix_offsets < TOTAL_T), other=-1
        )
        prefix_previous_tokens = tl.load(
            input_ids_ptr + prefix_offsets - 1,
            mask=prefix_mask & (prefix_offsets > 0) & (prefix_offsets < TOTAL_T),
            other=-1,
        )
        global_image_base = tl.sum(
            (
                (prefix_tokens == IMAGE_TOKEN_ID)
                & (prefix_previous_tokens == VISION_START_TOKEN_ID)
            ).to(tl.int32),
            axis=0,
        )

        local_image_count = tl.cumsum(image_starts.to(tl.int32), axis=0)
        image_rank = global_image_base + local_image_count - 1
        has_current_image = local_image_count > 0
        safe_image_rank = tl.maximum(image_rank, 0)
        grid_mask = valid_mask & has_current_image & (safe_image_rank < NUM_IMAGES)
        grid_offset = safe_image_rank * 3
        grid_t = tl.load(image_grid_thw_ptr + grid_offset, mask=grid_mask, other=1)
        grid_h = tl.load(image_grid_thw_ptr + grid_offset + 1, mask=grid_mask, other=1)
        grid_w = tl.load(image_grid_thw_ptr + grid_offset + 2, mask=grid_mask, other=1)
        grid_h = grid_h // SPATIAL_MERGE_SIZE
        grid_w = grid_w // SPATIAL_MERGE_SIZE
        image_token_count = grid_t * grid_h * grid_w
        image_extent = tl.maximum(grid_t, tl.maximum(grid_h, grid_w))

        marker_offsets = tl.where(image_starts, offsets, -1)
        current_image_start = tl.associative_scan(marker_offsets, axis=0, combine_fn=_maximum)
        image_delta = image_extent - image_token_count
        cumulative_shift = tl.cumsum(tl.where(image_starts, image_delta, 0), axis=0)

        relative_image_offset = offsets - current_image_start
        inside_image = (
            valid_mask
            & has_current_image
            & (relative_image_offset >= 0)
            & (relative_image_offset < image_token_count)
        )
        shift_before_image = cumulative_shift - image_delta
        image_base = current_image_start + shift_before_image
        spatial_size = grid_h * grid_w
        image_t = relative_image_offset // spatial_size
        image_h = (relative_image_offset % spatial_size) // grid_w
        image_w = relative_image_offset % grid_w

        text_position = offsets + cumulative_shift
        position_t = tl.where(inside_image, image_base + image_t, text_position)
        position_h = tl.where(inside_image, image_base + image_h, text_position)
        position_w = tl.where(inside_image, image_base + image_w, text_position)
        position_t = tl.where(valid_mask, position_t, 1)
        position_h = tl.where(valid_mask, position_h, 1)
        position_w = tl.where(valid_mask, position_w, 1)

    tl.store(
        position_ids_ptr + token_offsets, position_t, mask=physical_mask & (token_offsets < TOTAL_T)
    )
    tl.store(
        position_ids_ptr + TOTAL_T + token_offsets,
        position_h,
        mask=physical_mask & (token_offsets < TOTAL_T),
    )
    tl.store(
        position_ids_ptr + 2 * TOTAL_T + token_offsets,
        position_w,
        mask=physical_mask & (token_offsets < TOTAL_T),
    )

    max_position = tl.max(
        tl.where(valid_mask, tl.maximum(position_t, tl.maximum(position_h, position_w)), -1), axis=0
    )
    delta = tl.where(valid_len > 0, max_position + 1 - valid_len, 0)
    tl.store(deltas_ptr + segment_idx, delta)


def get_fused_thd_mrope_unavailable_reason(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
) -> str | None:
    """Return why the fused image/text THD path cannot run, if applicable."""
    if not HAVE_TRITON:
        return "Triton is not available"
    if video_grid_thw is not None:
        return "video MRoPE is not supported by the fused THD kernel"
    tensors = [input_ids, cu_seqlens, cu_seqlens_padded]
    if image_grid_thw is not None:
        tensors.append(image_grid_thw)
    if any(not tensor.is_cuda for tensor in tensors):
        return "fused THD MRoPE requires CUDA tensors"
    if any(tensor.device != input_ids.device for tensor in tensors[1:]):
        return "fused THD MRoPE requires all tensors on the same CUDA device"
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        return f"input_ids must have shape [1, T], got {tuple(input_ids.shape)}"
    if input_ids.dtype not in (torch.int32, torch.int64):
        return f"input_ids dtype {input_ids.dtype} is not supported"
    if cu_seqlens.dim() != 1 or cu_seqlens_padded.dim() != 1:
        return "cu_seqlens and cu_seqlens_padded must be one-dimensional"
    if cu_seqlens.shape != cu_seqlens_padded.shape:
        return "cu_seqlens and cu_seqlens_padded must have identical shapes"
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        return f"cu_seqlens dtype {cu_seqlens.dtype} is not supported"
    if cu_seqlens_padded.dtype not in (torch.int32, torch.int64):
        return f"cu_seqlens_padded dtype {cu_seqlens_padded.dtype} is not supported"
    if image_grid_thw is not None:
        if image_grid_thw.dim() != 2 or image_grid_thw.shape[1] != 3:
            return f"image_grid_thw must have shape [N, 3], got {tuple(image_grid_thw.shape)}"
        if image_grid_thw.dtype not in (torch.int32, torch.int64):
            return f"image_grid_thw dtype {image_grid_thw.dtype} is not supported"
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "fused THD MRoPE requires contiguous tensors"
    if input_ids.shape[1] == 0:
        return "empty THD inputs are not supported"
    if input_ids.shape[1] > 65536:
        return "THD inputs longer than 65536 tokens are not supported"
    return None


def fused_thd_mrope_position_ids(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    vision_start_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate THD MRoPE position IDs without host synchronization."""
    reason = get_fused_thd_mrope_unavailable_reason(
        input_ids, cu_seqlens, cu_seqlens_padded, image_grid_thw, video_grid_thw=None
    )
    if reason is not None:
        raise ValueError(reason)

    total_tokens = input_ids.shape[1]
    num_segments = cu_seqlens.numel() - 1
    block_size = triton.next_power_of_2(total_tokens)
    position_ids = torch.empty((3, 1, total_tokens), dtype=input_ids.dtype, device=input_ids.device)
    deltas = torch.empty((num_segments, 1), dtype=torch.int64, device=input_ids.device)
    grid_ptr = input_ids if image_grid_thw is None else image_grid_thw
    num_images = 0 if image_grid_thw is None else image_grid_thw.shape[0]

    with torch.cuda.device(input_ids.device):
        _thd_mrope_position_ids_kernel[(num_segments,)](
            input_ids,
            cu_seqlens,
            cu_seqlens_padded,
            grid_ptr,
            position_ids,
            deltas,
            TOTAL_T=total_tokens,
            NUM_IMAGES=num_images,
            IMAGE_TOKEN_ID=image_token_id,
            VISION_START_TOKEN_ID=vision_start_token_id,
            SPATIAL_MERGE_SIZE=spatial_merge_size,
            HAS_IMAGES=num_images > 0,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    return position_ids, deltas
