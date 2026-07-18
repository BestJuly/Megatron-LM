# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused multi-microbatch vision caches for the MDP pipeline sidecar.

Owner selection is deliberately absent from this module. The Energon loader
must attach a complete, deterministic assignment before it materializes image
descriptors. This module only repacks that already assigned work into bounded
raw-patch windows and preserves the original text-microbatch order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from examples.multimodal_dev.data.energon_mdp import grid_rows_from_batch
from examples.multimodal_dev.sidecar_prefetch import image_vision_pack_plan

_VISION_PAYLOAD_KEYS = (
    "pixel_values",
    "_balancedata_image_grid_thw_rows",
    "_mdp_prepartitioned_image_grid_thw",
    "_mdp_prepartitioned_assignment",
    "_mdp_prepartitioned_row_counts",
    "_mdp_prepartitioned_local_raw_counts",
    "_mdp_image_descriptors",
    "_mdp_image_descriptors_json",
)


@dataclass(frozen=True)
class _WindowMetadata:
    row_counts: list[int]
    raw_counts_by_microbatch: list[list[int]]
    assignment: dict[int, list[tuple[int, int]]]
    local_images: dict[int, tuple[torch.Tensor, torch.Tensor]]
    image_to_microbatch: dict[int, int]
    group_rank: int


@dataclass
class _PackBackwardState:
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    assignment: dict[int, list[tuple[int, int]]]
    row_counts: list[int]
    image_indices: list[int]
    remaining_microbatches: int
    backward_mode: str
    retained_output: torch.Tensor | None
    grads: dict[int, torch.Tensor] = field(default_factory=dict)
    done: bool = False


@dataclass(frozen=True)
class _BackwardEntry:
    state: _PackBackwardState
    image_index: int | None
    leaf: torch.Tensor | None


def _wrapped_attr(model, name: str, default=None):
    while model is not None:
        if hasattr(model, name):
            return getattr(model, name)
        model = getattr(model, "module", None)
    return default


def _drop_vision_payload(batch: dict) -> dict:
    batch = dict(batch)
    for key in _VISION_PAYLOAD_KEYS:
        batch.pop(key, None)
    batch["_mdp_pp_cp_sidecar_applied"] = True
    return batch


def _int_values(value) -> list[int]:
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [int(item) for item in (value or [])]


def _raw_patch_counts(grid_rows: Sequence[Sequence[int]]) -> list[int]:
    return [int(row[0]) * int(row[1]) * int(row[2]) for row in grid_rows]


def _assignment_rows(batch: dict) -> list[list[int]]:
    assignment = batch.get("_mdp_prepartitioned_assignment")
    if isinstance(assignment, dict):
        return [
            [int(owner), int(document), int(image_index)]
            for owner in sorted(assignment)
            for document, image_index in assignment[owner]
        ]
    if not torch.is_tensor(assignment):
        raise RuntimeError(
            "MDP fused vision window requires loader-prepartitioned assignment metadata"
        )
    assignment = assignment.detach().to(dtype=torch.int64, device="cpu")
    if assignment.numel() == 0:
        return []
    if assignment.dim() != 2 or assignment.shape[1] != 3:
        raise RuntimeError(
            "MDP fused vision assignment must have shape [images,3], "
            f"got {tuple(assignment.shape)}"
        )
    return [[int(item) for item in row] for row in assignment.tolist()]


def _split_pixels(pixel_values, counts: Sequence[int]) -> list[torch.Tensor]:
    if not counts:
        if torch.is_tensor(pixel_values) and pixel_values.shape[0] != 0:
            raise RuntimeError("MDP fused vision has pixels without local image metadata")
        return []
    if not torch.is_tensor(pixel_values):
        raise RuntimeError("MDP fused vision local assignment is missing pixel_values")
    chunks = []
    offset = 0
    for count in counts:
        chunks.append(pixel_values.narrow(0, offset, int(count)))
        offset += int(count)
    if offset != int(pixel_values.shape[0]):
        raise RuntimeError(
            "MDP fused vision pixel/grid mismatch: "
            f"metadata has {offset} raw patches, pixel_values has "
            f"{int(pixel_values.shape[0])}"
        )
    return chunks


def _group_rank_and_size(model) -> tuple[int, int]:
    group = _wrapped_attr(model, "_mdp_inner_dp_group")
    if group is None or not torch.distributed.is_initialized():
        return 0, 1
    return (
        int(torch.distributed.get_rank(group=group)),
        int(torch.distributed.get_world_size(group=group)),
    )


def _window_metadata(batches: Sequence[dict], model) -> _WindowMetadata:
    """Validate and flatten loader-owned assignments without rebalancing."""
    group_rank, group_size = _group_rank_and_size(model)
    row_counts_by_microbatch = []
    raw_counts_by_microbatch = []
    assignment = {rank: [] for rank in range(group_size)}
    local_images = {}
    image_to_microbatch = {}
    image_offset = 0

    for microbatch, batch in enumerate(batches):
        row_counts = _int_values(batch.get("_mdp_prepartitioned_row_counts"))
        grid_rows = grid_rows_from_batch(batch)
        raw_counts = _raw_patch_counts(grid_rows)
        if len(row_counts) != len(raw_counts):
            raise RuntimeError(
                "MDP fused vision global grid/row-count mismatch: "
                f"raw_counts={len(raw_counts)} row_counts={len(row_counts)}"
            )

        rows = _assignment_rows(batch)
        seen = set()
        local_image_indices = []
        for owner, _document, image_index in rows:
            if owner < 0 or owner >= group_size:
                raise RuntimeError(
                    f"MDP fused vision owner {owner} is outside group size {group_size}"
                )
            if image_index < 0 or image_index >= len(row_counts):
                raise RuntimeError(
                    "MDP fused vision assignment image index is outside the microbatch: "
                    f"index={image_index}, images={len(row_counts)}"
                )
            if image_index in seen:
                raise RuntimeError(f"MDP fused vision assignment repeats image {image_index}")
            seen.add(image_index)
            global_index = image_offset + image_index
            assignment[owner].append((microbatch, global_index))
            image_to_microbatch[global_index] = microbatch
            if owner == group_rank:
                local_image_indices.append(image_index)
        if seen != set(range(len(row_counts))):
            missing = sorted(set(range(len(row_counts))) - seen)
            raise RuntimeError(
                f"MDP fused vision assignment does not cover every image; missing={missing}"
            )

        local_grid = batch.get("_mdp_prepartitioned_image_grid_thw")
        if not torch.is_tensor(local_grid):
            raise RuntimeError("MDP fused vision assignment is missing local image_grid_thw")
        local_grid_rows = list(local_grid.unbind(0)) if local_grid.numel() else []
        local_raw_counts = _int_values(batch.get("_mdp_prepartitioned_local_raw_counts"))
        if not local_raw_counts:
            local_raw_counts = _raw_patch_counts(
                [[int(item) for item in row.detach().cpu().tolist()] for row in local_grid_rows]
            )
        local_pixel_chunks = _split_pixels(batch.get("pixel_values"), local_raw_counts)
        if not (len(local_image_indices) == len(local_grid_rows) == len(local_pixel_chunks)):
            raise RuntimeError(
                "MDP fused vision local assignment/materialization mismatch: "
                f"assignment={len(local_image_indices)} grids={len(local_grid_rows)} "
                f"pixel_chunks={len(local_pixel_chunks)}"
            )
        for local_index, image_index in enumerate(local_image_indices):
            local_images[image_offset + image_index] = (
                local_pixel_chunks[local_index],
                local_grid_rows[local_index],
            )

        row_counts_by_microbatch.append(row_counts)
        raw_counts_by_microbatch.append(raw_counts)
        image_offset += len(row_counts)

    return _WindowMetadata(
        row_counts=[count for counts in row_counts_by_microbatch for count in counts],
        raw_counts_by_microbatch=raw_counts_by_microbatch,
        assignment=assignment,
        local_images=local_images,
        image_to_microbatch=image_to_microbatch,
        group_rank=group_rank,
    )


def _empty_pixels_and_grid(batches: Sequence[dict]):
    pixel_template = next(
        (
            batch.get("pixel_values")
            for batch in batches
            if torch.is_tensor(batch.get("pixel_values"))
        ),
        None,
    )
    grid_template = next(
        (
            batch.get("_mdp_prepartitioned_image_grid_thw")
            for batch in batches
            if torch.is_tensor(batch.get("_mdp_prepartitioned_image_grid_thw"))
        ),
        None,
    )
    if pixel_template is None or pixel_template.dim() != 2:
        raise RuntimeError("MDP fused vision could not infer the local pixel shape")
    if grid_template is None:
        raise RuntimeError("MDP fused vision could not infer the local grid shape")
    return (
        pixel_template.new_empty((0, int(pixel_template.shape[1]))),
        grid_template.new_empty((0, 3)),
    )


def _split_embeddings(output, image_indices: Sequence[int], row_counts: Sequence[int]):
    chunks = {}
    offset = 0
    for image_index in image_indices:
        rows = int(row_counts[int(image_index)])
        chunks[int(image_index)] = output.narrow(0, offset, rows)
        offset += rows
    if offset != int(output.shape[0]):
        raise RuntimeError(
            "MDP fused vision output/row-count mismatch: "
            f"metadata has {offset} rows, output has {int(output.shape[0])}"
        )
    return chunks


def _empty_embeddings(model, *, reference=None):
    config = _wrapped_attr(model, "config")
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError("MDP fused vision could not infer language hidden size")
    if torch.is_tensor(reference):
        return reference.new_empty((0, hidden_size))
    parameter = next(model.parameters())
    return parameter.new_empty((0, hidden_size))


def build_fused_vision_caches(
    model,
    batches: Sequence[dict],
    *,
    max_sequence_length: int,
    backward_mode: str,
    forward_only: bool,
) -> list[dict]:
    """Build ordered forward caches from a loader-planned microbatch window."""
    if backward_mode not in ("retain", "recompute"):
        raise ValueError("MDP fused vision backward must be retain or recompute")
    if not batches:
        raise ValueError("MDP fused vision window requires at least one microbatch")

    from examples.multimodal_dev.mdp_batch import apply_mdp_prepartition

    metadata = _window_metadata(batches, model)
    backward_enabled = torch.is_grad_enabled() and not bool(forward_only)
    pre_process = bool(_wrapped_attr(model, "pre_process", False))
    compute_vision = _wrapped_attr(model, "mdp_pp_cp_sidecar_compute_vision")
    if compute_vision is None:
        raise RuntimeError("MDP fused vision requires mdp_pp_cp_sidecar_compute_vision")

    total_images = len(metadata.row_counts)
    if total_images == 0:
        caches = []
        for batch in batches:
            if pre_process:
                embeddings = (
                    _empty_embeddings(model, reference=batch.get("pixel_values"))
                    .detach()
                    .requires_grad_(backward_enabled)
                )
            else:
                parameter = next(model.parameters())
                embeddings = parameter.reshape(-1)[:1].sum() * 0.0
            caches.append(
                {
                    "batch": _drop_vision_payload(batch),
                    "vision_embeddings": embeddings,
                    "fused_backward_entries": [] if backward_enabled else None,
                    "forward_only": bool(forward_only),
                }
            )
        return caches

    pack_plan = image_vision_pack_plan(metadata.raw_counts_by_microbatch, int(max_sequence_length))
    per_microbatch_chunks = [[] for _ in batches]
    per_microbatch_entries = [[] for _ in batches]
    first_pack_pixels = None
    empty_pixels, empty_grid = _empty_pixels_and_grid(batches)

    for pack_indices in pack_plan:
        canonical_indices = sorted(
            [int(index) for index in pack_indices],
            key=lambda index: (int(metadata.image_to_microbatch[index]), index),
        )
        pack_set = set(canonical_indices)
        pack_assignment = {
            int(rank): [
                (int(microbatch), int(image_index))
                for microbatch, image_index in items
                if int(image_index) in pack_set
            ]
            for rank, items in metadata.assignment.items()
        }

        local_pixels = []
        local_grids = []
        for _microbatch, image_index in pack_assignment.get(metadata.group_rank, []):
            pixels, grid = metadata.local_images[int(image_index)]
            local_pixels.append(pixels)
            local_grids.append(grid)
        pack_pixels = torch.cat(local_pixels, dim=0).contiguous() if local_pixels else empty_pixels
        pack_grid = torch.stack(local_grids, dim=0).contiguous() if local_grids else empty_grid
        pack_pixels, pack_grid = apply_mdp_prepartition(
            model=model,
            pixel_values=pack_pixels,
            image_grid_thw=pack_grid,
            prepartitioned_assignment=pack_assignment,
            prepartitioned_row_counts=metadata.row_counts,
            prepartitioned_image_grid_thw=pack_grid,
        )
        if first_pack_pixels is None:
            first_pack_pixels = pack_pixels

        grad_context = (
            torch.enable_grad()
            if backward_enabled and backward_mode == "retain"
            else torch.no_grad()
        )
        with grad_context:
            pack_embeddings = compute_vision(
                pixel_values=pack_pixels, image_grid_thw=pack_grid, mdp_cp_local_plan=None
            )

        state = None
        if backward_enabled:
            state = _PackBackwardState(
                pixel_values=pack_pixels.detach(),
                image_grid_thw=pack_grid.detach(),
                assignment=pack_assignment,
                row_counts=list(metadata.row_counts),
                image_indices=list(canonical_indices),
                remaining_microbatches=len(
                    {int(metadata.image_to_microbatch[index]) for index in canonical_indices}
                ),
                backward_mode=backward_mode,
                retained_output=(pack_embeddings if backward_mode == "retain" else None),
            )

        if pre_process:
            image_chunks = _split_embeddings(
                pack_embeddings, canonical_indices, metadata.row_counts
            )
            for image_index in canonical_indices:
                microbatch = int(metadata.image_to_microbatch[image_index])
                leaf = image_chunks[image_index].detach().requires_grad_(backward_enabled)
                per_microbatch_chunks[microbatch].append((image_index, leaf))
                if backward_enabled:
                    assert state is not None
                    per_microbatch_entries[microbatch].append(
                        _BackwardEntry(state=state, image_index=image_index, leaf=leaf)
                    )
        elif backward_enabled:
            assert state is not None
            for microbatch in sorted(
                {int(metadata.image_to_microbatch[index]) for index in canonical_indices}
            ):
                per_microbatch_entries[microbatch].append(
                    _BackwardEntry(state=state, image_index=None, leaf=None)
                )

    caches = []
    for microbatch, batch in enumerate(batches):
        if pre_process:
            chunks = [
                leaf
                for _image_index, leaf in sorted(
                    per_microbatch_chunks[microbatch], key=lambda item: item[0]
                )
            ]
            embeddings = (
                torch.cat(chunks, dim=0).contiguous()
                if chunks
                else _empty_embeddings(model, reference=batch.get("pixel_values"))
                .detach()
                .requires_grad_(backward_enabled)
            )
        else:
            assert first_pack_pixels is not None
            embeddings = first_pack_pixels.reshape(-1)[:1].sum() * 0.0
        caches.append(
            {
                "batch": _drop_vision_payload(batch),
                "vision_embeddings": embeddings,
                "fused_backward_entries": (
                    per_microbatch_entries[microbatch] if backward_enabled else None
                ),
                "forward_only": bool(forward_only),
            }
        )
    return caches


def _fused_grad_tensor(output, state: _PackBackwardState):
    if output.dim() != 2:
        raise RuntimeError(
            f"MDP fused backward expected [rows,hidden] output, got {tuple(output.shape)}"
        )
    chunks = []
    for image_index in state.image_indices:
        grad = state.grads.get(int(image_index))
        if grad is None:
            rows = int(state.row_counts[int(image_index)])
            grad = output.new_zeros((rows, output.shape[1]))
        chunks.append(grad.to(device=output.device, dtype=output.dtype))
    return torch.cat(chunks, dim=0) if chunks else output.new_zeros(output.shape)


def _backward_pack_output(model, state: _PackBackwardState, output) -> None:
    if bool(_wrapped_attr(model, "pre_process", False)):
        torch.autograd.backward(output, grad_tensors=_fused_grad_tensor(output, state))
    elif torch.is_tensor(output) and output.requires_grad:
        # Join the bridge backward collective without contributing gradients.
        (output.reshape(-1)[:1].sum() * 0.0).backward()


def _recompute_backward(model, state: _PackBackwardState) -> None:
    from examples.multimodal_dev.mdp_batch import apply_mdp_prepartition

    pixels, grid = apply_mdp_prepartition(
        model=model,
        pixel_values=state.pixel_values.detach(),
        image_grid_thw=state.image_grid_thw.detach(),
        prepartitioned_assignment=state.assignment,
        prepartitioned_row_counts=state.row_counts,
        prepartitioned_image_grid_thw=state.image_grid_thw.detach(),
    )
    compute_vision = _wrapped_attr(model, "mdp_pp_cp_sidecar_compute_vision")
    with torch.enable_grad():
        output = compute_vision(pixel_values=pixels, image_grid_thw=grid, mdp_cp_local_plan=None)
        _backward_pack_output(model, state, output)


def _retained_backward(model, state: _PackBackwardState) -> None:
    output = state.retained_output
    state.retained_output = None
    if output is None:
        raise RuntimeError("MDP fused retain backward is missing its graph output")
    _backward_pack_output(model, state, output)


def fused_vision_post_backward(model, entries: Sequence[_BackwardEntry]) -> None:
    """Accumulate one text microbatch's leaves and finish ready vision packs."""
    seen_states: dict[int, _PackBackwardState] = {}
    for entry in entries:
        state = entry.state
        seen_states[id(state)] = state
        leaf = entry.leaf
        image_index = entry.image_index
        if leaf is not None and image_index is not None:
            grad = leaf.grad
            state.grads[int(image_index)] = (
                torch.zeros_like(leaf) if grad is None else grad.detach()
            )

    for state in seen_states.values():
        state.remaining_microbatches -= 1
        if state.remaining_microbatches > 0 or state.done:
            continue
        state.done = True
        if state.backward_mode == "retain":
            _retained_backward(model, state)
        else:
            _recompute_backward(model, state)
