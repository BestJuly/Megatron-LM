# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Ordered fused-window MDP planning/prefetch for descriptor-bearing Energon batches.

This iterator keeps the text microbatch stream ordered. It only rewrites
rank-local vision payloads from metadata after looking at a small window of
already-packed batches.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from examples.multimodal_dev.data.energon_vision_balance import (
    assign_images_lpt,
    image_costs_from_grid,
    vision_rows_from_grid,
)
from examples.multimodal_dev.mdp_image_materialize import (
    decode_image_descriptors,
    materialize_descriptor,
)
from examples.multimodal_dev.sidecar_prefetch import validate_fused_vision_window


def loader_prepartition_window_size(args, *, loader_prepartition: bool, inner_scope: str) -> int:
    """Resolve the deterministic MDP planning-window size."""
    if not loader_prepartition or inner_scope not in ("cp", "pp_cp"):
        return 1
    max_sequence_length = int(getattr(args, "mdp_vision_encoder_max_sequence_length", 0) or 0)
    if not validate_fused_vision_window(
        getattr(args, "mdp_fused_vision_window", False), max_sequence_length
    ):
        return 1
    dp_size = max(1, int(getattr(args, "data_parallel_size", 1)))
    micro_batch_size = max(1, int(getattr(args, "micro_batch_size", 1)))
    global_batch_size = int(getattr(args, "global_batch_size", micro_batch_size * dp_size))
    return max(1, global_batch_size // (micro_batch_size * dp_size))


def grid_rows_from_batch(batch: dict) -> list[list[int]]:
    """Return ``image_grid_thw`` as ``[[t, h, w], ...]`` rows from a batch."""
    grid = batch.get("image_grid_thw")
    if not torch.is_tensor(grid) or grid.numel() == 0:
        return []
    if grid.dim() == 3:
        grid = grid.flatten(0, 1)
    if grid.dim() != 2 or grid.shape[1] != 3:
        raise RuntimeError(f"MDP expects image_grid_thw as [N,3], got {tuple(grid.shape)}")
    return [[int(x) for x in row] for row in grid.detach().cpu().tolist()]


def _descriptors(batch: dict) -> list[dict]:
    raw = batch.get("_mdp_image_descriptors")
    if raw is None:
        raw = decode_image_descriptors(batch.get("_mdp_image_descriptors_json"))
    return [dict(desc) for desc in (raw or [])]


def _image_cu(batch: dict) -> list[int]:
    image_cu = batch.get("image_cu_seqlens")
    if torch.is_tensor(image_cu):
        return [int(x) for x in image_cu.detach().cpu().reshape(-1).tolist()]
    return []


def _doc_index_for_image(image_cu: Sequence[int], image_idx: int) -> int:
    if len(image_cu) < 2:
        return 0
    return max(0, int(bisect_right(image_cu, int(image_idx)) - 1))


def _assignment_tensor_for_batch(assignment, item_idx: int, batch: dict) -> torch.Tensor:
    rows = []
    image_cu = _image_cu(batch)
    for rank in sorted(assignment):
        for assigned_item_idx, image_idx in assignment[rank]:
            if int(assigned_item_idx) != int(item_idx):
                continue
            rows.append([int(rank), _doc_index_for_image(image_cu, int(image_idx)), int(image_idx)])
    if not rows:
        return torch.zeros(0, 3, dtype=torch.int32)
    return torch.tensor(rows, dtype=torch.int32)


class MDPWindowMaterializingIterator:
    """Prefetch ordered windows and materialize only this rank's images.

    The wrapped iterator yields descriptor-bearing Energon batches. This
    wrapper reads ``lookahead_microbatches`` consecutive batches, computes
    deterministic LPT owner assignments, submits CPU materialization work for
    each returned batch, and yields the prepared batches in the original order.
    Non-fused vision balances each microbatch independently; fused vision can
    balance across the complete lookahead window.

    The base iterator is consumed only by this object and only in order. The
    thread pool handles descriptor materialization; it does not run CUDA work or
    call collectives. Window boundaries depend only on FIFO position and
    ``lookahead_microbatches``, so corresponding PP/CP ranks produce the same
    plans when their source streams are identical.
    """

    def __init__(
        self,
        source: Iterable[dict],
        *,
        lookahead_microbatches: int,
        prefetch_windows: int,
        rank: int,
        world: int,
        pixel_dim: int,
        patch_size: int,
        spatial_merge_size: int,
        lpt_hidden_size: int = 2048,
        balance_across_microbatches: bool = False,
        materialize_workers: int = 1,
    ):
        self._source = iter(source)
        self._lookahead = max(1, int(lookahead_microbatches))
        if int(prefetch_windows) < 1:
            raise ValueError("--mdp-loader-prepartition-prefetch-windows must be positive")
        self._prefetch_windows = int(prefetch_windows)
        self._rank = int(rank)
        self._world = max(1, int(world))
        self._pixel_dim = int(pixel_dim)
        self._patch_size = int(patch_size)
        self._spatial_merge_size = int(spatial_merge_size)
        self._lpt_hidden_size = int(lpt_hidden_size)
        self._balance_across_microbatches = bool(balance_across_microbatches)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(materialize_workers)))
        self._pending_windows: deque[list[Future]] = deque()
        self._ready_slots: deque[dict] = deque()
        self._source_exhausted = False
        self._closed = False
        try:
            self._fill_pending()
        except Exception:
            self.close()
            raise

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        while not self._ready_slots:
            self._activate_next_window()
        batch = self._ready_slots.popleft()
        if self._source_exhausted and not self._pending_windows and not self._ready_slots:
            self.close()
        return batch

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _fill_pending(self) -> None:
        while not self._source_exhausted and len(self._pending_windows) < self._prefetch_windows:
            window = self._fetch_window()
            if not window:
                self._source_exhausted = True
                break
            self._pending_windows.append(self._schedule_window(window))

    def _fetch_window(self) -> list[dict]:
        window: list[dict] = []
        for _ in range(self._lookahead):
            try:
                window.append(next(self._source))
            except StopIteration:
                self._source_exhausted = True
                break
        return window

    def _activate_next_window(self) -> None:
        self._fill_pending()
        if not self._pending_windows:
            self.close()
            raise StopIteration
        futures = self._pending_windows.popleft()
        self._fill_pending()
        try:
            self._ready_slots.extend(future.result() for future in futures)
        except Exception:
            self.close()
            raise

    def _schedule_window(self, window: Sequence[dict]) -> list[Future]:
        grids_by_item = [grid_rows_from_batch(batch) for batch in window]
        row_counts_by_item = [
            [int(vision_rows_from_grid(row, self._spatial_merge_size)) for row in rows]
            for rows in grids_by_item
        ]
        costs_by_item = [
            image_costs_from_grid(rows, hidden_size=self._lpt_hidden_size) for rows in grids_by_item
        ]
        assignment = assign_images_lpt(
            costs_by_item, self._world, across_items=self._balance_across_microbatches
        )

        futures = []
        for item_idx, batch in enumerate(window):
            futures.append(
                self._executor.submit(
                    self._prepare_batch,
                    dict(batch),
                    item_idx=item_idx,
                    assignment=assignment,
                    grids=grids_by_item[item_idx],
                    row_counts=row_counts_by_item[item_idx],
                )
            )
        return futures

    def _prepare_batch(
        self,
        batch: dict,
        *,
        item_idx: int,
        assignment,
        grids: Sequence[Sequence[int]],
        row_counts: Sequence[int],
    ) -> dict:
        descriptors = _descriptors(batch)
        if len(descriptors) != len(grids):
            raise RuntimeError(
                "planning prefetch descriptor/grid mismatch: "
                f"descriptors={len(descriptors)} grids={len(grids)}"
            )

        local_jobs = []
        local_raw_counts = []
        for assigned_item_idx, image_idx in assignment.get(self._rank, []):
            if int(assigned_item_idx) != int(item_idx):
                continue
            image_idx = int(image_idx)
            grid = [int(x) for x in grids[image_idx]]
            local_jobs.append((descriptors[image_idx], grid))
            local_raw_counts.append(int(grid[0]) * int(grid[1]) * int(grid[2]))

        local_pixels = []
        local_grids = []
        for descriptor, grid in local_jobs:
            patches = materialize_descriptor(
                descriptor, grid, pixel_dim=self._pixel_dim, patch_size=self._patch_size
            )
            local_pixels.append(patches.to(torch.bfloat16))
            local_grids.append(torch.tensor(grid, dtype=torch.long))

        batch["pixel_values"] = (
            torch.cat(local_pixels, dim=0)
            if local_pixels
            else torch.zeros(0, self._pixel_dim, dtype=torch.bfloat16)
        )
        batch["_mdp_prepartitioned_image_grid_thw"] = (
            torch.stack(local_grids, dim=0) if local_grids else torch.zeros(0, 3, dtype=torch.long)
        )
        batch["_mdp_prepartitioned_assignment"] = _assignment_tensor_for_batch(
            assignment, int(item_idx), batch
        )
        batch["_mdp_prepartitioned_row_counts"] = torch.tensor(
            [int(x) for x in row_counts], dtype=torch.int32
        )
        batch["_mdp_prepartitioned_local_raw_counts"] = torch.tensor(
            local_raw_counts, dtype=torch.int32
        )
        batch.pop("_mdp_image_descriptors", None)
        batch.pop("_mdp_image_descriptors_json", None)
        return batch
