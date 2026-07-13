# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from examples.multimodal_dev.data.dataset_utils import (
    RawSample,
    append_image_or_descriptor,
    materialize_image_descriptor as _materialize_image_descriptor,
)


def materialize_image_descriptor(desc, grid_thw, *, pixel_dim: int, patch_size: int):
    return _materialize_image_descriptor(
        desc,
        grid_thw,
        pixel_dim=pixel_dim,
        patch_size=patch_size,
    )


class FineVisionMaxBackend:
    """HuggingFaceM4/FineVisionMax parquet backend."""

    DEFAULT_SUBSETS = ("full",)

    def __init__(
        self,
        root: str,
        max_samples: Optional[int] = None,
        subsets: Optional[Sequence[str]] = None,
        split: str = "train",
    ):
        import pyarrow.parquet as pq

        self.root = Path(root)
        self._pq = pq
        self._parquet_files: Dict[str, object] = {}
        self._row_group_starts: Dict[str, Tuple[int, ...]] = {}
        self._row_group_cache: Dict[Tuple[str, int], object] = {}
        self._index: List[Tuple[str, int]] = []

        subset_names = list(subsets) if subsets is not None else list(self.DEFAULT_SUBSETS)
        target = max_samples if max_samples is not None else 10**18
        for subset in subset_names:
            subset_dir = self.root / subset
            search_dir = subset_dir if subset_dir.is_dir() else self.root
            for pq_path in sorted(search_dir.glob("*.parquet")):
                if len(self._index) >= target:
                    break
                pq_file = self._get_parquet_file(str(pq_path))
                for row_idx in range(pq_file.metadata.num_rows):
                    self._index.append((str(pq_path), row_idx))
                    if len(self._index) >= target:
                        break
            if len(self._index) >= target:
                break

    def __len__(self):
        return len(self._index)

    def _get_parquet_file(self, path: str):
        pq_file = self._parquet_files.get(path)
        if pq_file is None:
            pq_file = self._pq.ParquetFile(path)
            self._parquet_files[path] = pq_file
        return pq_file

    def _get_row_group_starts(self, path: str) -> Tuple[int, ...]:
        starts = self._row_group_starts.get(path)
        if starts is not None:
            return starts
        pq_file = self._get_parquet_file(path)
        offset = 0
        values = []
        for group_idx in range(pq_file.num_row_groups):
            values.append(offset)
            offset += pq_file.metadata.row_group(group_idx).num_rows
        starts = tuple(values)
        self._row_group_starts[path] = starts
        return starts

    def _locate_row_group(self, path: str, row_idx: int) -> Tuple[int, int]:
        starts = self._get_row_group_starts(path)
        group_idx = bisect_right(starts, int(row_idx)) - 1
        if group_idx < 0:
            raise IndexError(f"negative row index: {row_idx}")
        local_idx = int(row_idx) - starts[group_idx]
        return group_idx, local_idx

    def _read_row(self, path: str, row_idx: int):
        group_idx, local_idx = self._locate_row_group(path, row_idx)
        key = (path, group_idx)
        table = self._row_group_cache.get(key)
        if table is None:
            table = self._get_parquet_file(path).read_row_group(
                group_idx, columns=["images", "texts"]
            )
            self._row_group_cache[key] = table
            if len(self._row_group_cache) > 4:
                oldest = next(iter(self._row_group_cache))
                self._row_group_cache.pop(oldest, None)
        return table, local_idx

    def __getitem__(self, idx) -> RawSample:
        pq_path, row_idx = self._index[int(idx) % len(self._index)]
        table, local_idx = self._read_row(pq_path, row_idx)
        images = table.column("images")[local_idx].as_py() or []
        texts = table.column("texts")[local_idx].as_py() or []

        pil_images: List[Image.Image] = []
        image_descriptors: List[Dict] = []
        for image_idx, image in enumerate(images):
            if not isinstance(image, dict):
                continue
            data = image.get("bytes")
            if data is None:
                continue
            descriptor = {
                "kind": "parquet_list_image",
                "parquet_path": pq_path,
                "row_idx": int(row_idx),
                "column": "images",
                "image_idx": int(image_idx),
            }
            append_image_or_descriptor(
                pil_images,
                image_descriptors,
                data,
                descriptor,
                materializer="examples.multimodal_dev.data.finevision_max",
            )

        parts = []
        for turn in texts:
            if not isinstance(turn, dict):
                continue
            user = turn.get("user") or ""
            assistant = turn.get("assistant") or ""
            if user:
                parts.append(f"user: {user}")
            if assistant:
                parts.append(f"assistant: {assistant}")
        return RawSample(
            images=pil_images,
            text="\n".join(parts),
            image_descriptors=image_descriptors,
        )
