# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels for Qwen3.5-VL THD MRoPE position IDs.

The position of a token inside a packed segment depends on every image
that starts before it in the same segment, so the computation is a scan.
The scan is evaluated over fixed-width tiles with the carry kept in
registers, which keeps both the register footprint and the total work
independent of the packed length:

* ``_thd_mrope_image_count_kernel`` counts image starts per segment.
  Its exclusive prefix sum gives each segment its base row in the
  globally ordered ``image_grid_thw``.
* ``_thd_mrope_position_ids_kernel`` walks each segment tile by tile and
  emits all three MRoPE axes plus the per-segment delta.

Both kernels touch every token once, so the total work is ``O(T)`` rather
than ``O(num_segments * T)``. Only the tile width and the "this batch has
images" branch are ``tl.constexpr``; the packed length and the image
count are runtime scalars so that variable-length batches reuse a single
compiled variant instead of re-triggering JIT compilation.
"""

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


# Tile width of the intra-segment scan. Fixed so that the compiled variant
# does not depend on the packed length and so that register pressure stays
# flat as sequences grow.
_MROPE_POSITION_BLOCK = 1024

# Guard against int32 overflow in the flattened ``[3, 1, T]`` output index
# (``2 * total_tokens + token_offset``), with ample headroom.
_MAX_THD_TOKENS = 1 << 24


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit(do_not_specialize=["image_token_id", "vision_start_token_id"])
def _thd_mrope_image_count_kernel(
    input_ids_ptr,
    cu_seqlens_ptr,
    cu_seqlens_padded_ptr,
    segment_image_counts_ptr,
    image_token_id,
    vision_start_token_id,
    BLOCK: tl.constexpr,
):
    """Count the image starts contained in one packed segment."""
    segment_idx = tl.program_id(0)
    # Tokens live in the padded layout; only the first ``valid_len`` of them
    # are real.
    segment_start = tl.load(cu_seqlens_padded_ptr + segment_idx).to(tl.int32)
    valid_len = tl.load(cu_seqlens_ptr + segment_idx + 1).to(tl.int32) - tl.load(
        cu_seqlens_ptr + segment_idx
    ).to(tl.int32)

    count = 0
    for tile_start in tl.range(0, valid_len, BLOCK):
        offsets = tile_start + tl.arange(0, BLOCK)
        valid_mask = offsets < valid_len
        token_offsets = segment_start + offsets
        tokens = tl.load(input_ids_ptr + token_offsets, mask=valid_mask, other=-1)
        # A segment never inherits a vision-start token from its predecessor,
        # so offset 0 can never be an image start.
        previous_tokens = tl.load(
            input_ids_ptr + token_offsets - 1, mask=valid_mask & (offsets > 0), other=-1
        )
        image_starts = (tokens == image_token_id) & (previous_tokens == vision_start_token_id)
        count += tl.sum(image_starts.to(tl.int32), axis=0)

    tl.store(segment_image_counts_ptr + segment_idx, count)


# Triton would otherwise specialize integer arguments on "== 1" and
# "divisible by 16", which reintroduces per-shape recompilation for the
# packed length and the image count.
@triton.jit(
    do_not_specialize=[
        "total_tokens",
        "num_images",
        "image_token_id",
        "vision_start_token_id",
        "spatial_merge_size",
    ]
)
def _thd_mrope_position_ids_kernel(
    input_ids_ptr,
    cu_seqlens_ptr,
    cu_seqlens_padded_ptr,
    image_grid_thw_ptr,
    image_count_prefix_ptr,
    position_ids_ptr,
    deltas_ptr,
    total_tokens,
    num_images,
    image_token_id,
    vision_start_token_id,
    spatial_merge_size,
    HAS_IMAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generate all three MRoPE axes for one packed segment."""
    segment_idx = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)

    # Everything is computed in int32 regardless of the int32/int64 index
    # tensors, so that the loop-carried scan state has one stable type.
    segment_start = tl.load(cu_seqlens_padded_ptr + segment_idx).to(tl.int32)
    segment_end = tl.load(cu_seqlens_padded_ptr + segment_idx + 1).to(tl.int32)
    physical_len = segment_end - segment_start
    valid_len = tl.load(cu_seqlens_ptr + segment_idx + 1).to(tl.int32) - tl.load(
        cu_seqlens_ptr + segment_idx
    ).to(tl.int32)

    # ``image_grid_thw`` is ordered globally across packed segments; the
    # exclusive prefix of per-segment image counts is this segment's base row.
    global_image_base = 0
    if HAS_IMAGES:
        previous_segment = tl.maximum(segment_idx - 1, 0)
        inclusive_prefix = tl.load(image_count_prefix_ptr + previous_segment).to(tl.int32)
        global_image_base = tl.where(segment_idx > 0, inclusive_prefix, 0)

    # Scan carry across tiles: image starts seen so far, accumulated
    # position shift, offset of the most recent image start, and the running
    # maximum position used for the segment delta.
    carry_count = 0
    carry_shift = 0
    carry_image_start = -1
    carry_max_position = -1

    for tile_start in tl.range(0, physical_len, BLOCK):
        offsets = tile_start + lanes
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
            image_starts = (tokens == image_token_id) & (
                previous_tokens == vision_start_token_id
            )

            image_count = carry_count + tl.cumsum(image_starts.to(tl.int32), axis=0)
            image_rank = global_image_base + image_count - 1
            has_current_image = image_count > 0
            safe_image_rank = tl.maximum(image_rank, 0)
            grid_mask = valid_mask & has_current_image & (safe_image_rank < num_images)
            grid_offset = safe_image_rank * 3
            grid_t = tl.load(image_grid_thw_ptr + grid_offset, mask=grid_mask, other=1).to(
                tl.int32
            )
            grid_h = tl.load(image_grid_thw_ptr + grid_offset + 1, mask=grid_mask, other=1).to(
                tl.int32
            )
            grid_w = tl.load(image_grid_thw_ptr + grid_offset + 2, mask=grid_mask, other=1).to(
                tl.int32
            )
            grid_h = grid_h // spatial_merge_size
            grid_w = grid_w // spatial_merge_size
            image_token_count = grid_t * grid_h * grid_w
            image_extent = tl.maximum(grid_t, tl.maximum(grid_h, grid_w))

            marker_offsets = tl.where(image_starts, offsets, -1)
            current_image_start = tl.maximum(
                tl.associative_scan(marker_offsets, axis=0, combine_fn=_maximum),
                carry_image_start,
            )
            image_delta = image_extent - image_token_count
            cumulative_shift = carry_shift + tl.cumsum(
                tl.where(image_starts, image_delta, 0), axis=0
            )

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

            # Carry the tile totals into the next tile. The image count and
            # the shift are read from the last lane: the shift in particular
            # is not monotonic (an image can be shorter than its spatial
            # extent), so it cannot be recovered with a max reduction.
            # Lanes past the segment end contribute nothing to either scan.
            carry_count = tl.sum(tl.where(lanes == BLOCK - 1, image_count, 0), axis=0)
            carry_shift = tl.sum(tl.where(lanes == BLOCK - 1, cumulative_shift, 0), axis=0)
            carry_image_start = tl.max(current_image_start, axis=0)

        store_mask = physical_mask & (token_offsets < total_tokens)
        out_dtype = position_ids_ptr.dtype.element_ty
        tl.store(position_ids_ptr + token_offsets, position_t.to(out_dtype), mask=store_mask)
        tl.store(
            position_ids_ptr + total_tokens + token_offsets,
            position_h.to(out_dtype),
            mask=store_mask,
        )
        tl.store(
            position_ids_ptr + 2 * total_tokens + token_offsets,
            position_w.to(out_dtype),
            mask=store_mask,
        )

        tile_max = tl.max(
            tl.where(valid_mask, tl.maximum(position_t, tl.maximum(position_h, position_w)), -1),
            axis=0,
        )
        carry_max_position = tl.maximum(carry_max_position, tile_max)

    delta = tl.where(valid_len > 0, carry_max_position + 1 - valid_len, 0)
    tl.store(deltas_ptr + segment_idx, delta.to(deltas_ptr.dtype.element_ty))


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
    if input_ids.shape[1] > _MAX_THD_TOKENS:
        return f"THD inputs longer than {_MAX_THD_TOKENS} tokens are not supported"
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
    position_ids = torch.empty((3, 1, total_tokens), dtype=input_ids.dtype, device=input_ids.device)
    deltas = torch.empty((num_segments, 1), dtype=torch.int64, device=input_ids.device)
    num_images = 0 if image_grid_thw is None else image_grid_thw.shape[0]
    has_images = num_images > 0

    with torch.cuda.device(input_ids.device):
        if has_images:
            segment_image_counts = torch.empty(
                num_segments, dtype=torch.int32, device=input_ids.device
            )
            _thd_mrope_image_count_kernel[(num_segments,)](
                input_ids,
                cu_seqlens,
                cu_seqlens_padded,
                segment_image_counts,
                image_token_id,
                vision_start_token_id,
                BLOCK=_MROPE_POSITION_BLOCK,
                num_warps=4,
            )
            # Inclusive prefix over a ``[num_segments]`` tensor; the kernel
            # reads element ``segment_idx - 1`` to get the exclusive base.
            image_count_prefix = torch.cumsum(segment_image_counts, 0, dtype=torch.int32)
            grid_ptr = image_grid_thw
        else:
            # Unused by the kernel, but Triton still needs valid pointers.
            image_count_prefix = cu_seqlens
            grid_ptr = cu_seqlens

        _thd_mrope_position_ids_kernel[(num_segments,)](
            input_ids,
            cu_seqlens,
            cu_seqlens_padded,
            grid_ptr,
            image_count_prefix,
            position_ids,
            deltas,
            total_tokens,
            num_images,
            image_token_id,
            vision_start_token_id,
            spatial_merge_size,
            HAS_IMAGES=has_images,
            BLOCK=_MROPE_POSITION_BLOCK,
            num_warps=4,
        )
    return position_ids, deltas
