# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from examples.multimodal_dev.data.dataset_utils import RawSample, append_image_or_descriptor
from examples.multimodal_dev.data.dataset_utils import (
    materialize_image_descriptor as _materialize_image_descriptor,
)


def materialize_image_descriptor(desc, grid_thw, *, pixel_dim: int, patch_size: int):
    return _materialize_image_descriptor(desc, grid_thw, pixel_dim=pixel_dim, patch_size=patch_size)


class PixmoDocsBackend:
    """allenai/pixmo-docs: high-resolution single-image rows with dense Q/A."""

    DEFAULT_SUBSETS = ("charts", "diagrams", "tables", "other")

    def __init__(
        self,
        root: str,
        max_samples: Optional[int] = None,
        subsets: Optional[Sequence[str]] = None,
        split: str = "train",
    ):
        import pyarrow.parquet as pq

        self.root = Path(root)
        if subsets is None:
            subsets = [
                s
                for s in self.DEFAULT_SUBSETS
                if (self.root / s).is_dir() and any((self.root / s).glob(f"{split}-*.parquet"))
            ]
        self.subsets = list(subsets)
        self._tables: dict = {}
        self._index: List[Tuple[str, int]] = []

        target = max_samples if max_samples is not None else 10**18
        for sub in self.subsets:
            sub_dir = self.root / sub
            if not sub_dir.is_dir():
                continue
            for pq_path in sorted(sub_dir.glob(f"{split}-*.parquet")):
                if len(self._index) >= target:
                    break
                t = pq.read_table(str(pq_path), columns=["image", "questions"])
                self._tables[str(pq_path)] = t
                for i in range(t.num_rows):
                    self._index.append((str(pq_path), i))
                    if len(self._index) >= target:
                        break
            if len(self._index) >= target:
                break

        if max_samples is not None and max_samples < len(self._index):
            rng = random.Random(0xC0FFEE)
            self._index = rng.sample(self._index, max_samples)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx) -> RawSample:
        pq_path, row_idx = self._index[idx]
        t = self._tables[pq_path]
        img_struct = t.column("image")[row_idx].as_py()
        qa_struct = t.column("questions")[row_idx].as_py() or {}

        pil_images: List[Image.Image] = []
        image_descriptors: List[Dict] = []
        b = img_struct.get("bytes") if isinstance(img_struct, dict) else None
        if b is not None:
            descriptor = {
                "kind": "parquet_column_image",
                "parquet_path": pq_path,
                "row_idx": int(row_idx),
                "column": "image",
            }
            append_image_or_descriptor(
                pil_images,
                image_descriptors,
                b,
                descriptor,
                materializer="examples.multimodal_dev.data.pixmo_docs",
            )

        qs = qa_struct.get("question", []) or []
        ans = qa_struct.get("answer", []) or []
        text_parts = [f"Q: {q}\nA: {a}" for q, a in zip(qs, ans)]
        text = "\n\n".join(text_parts) if text_parts else ""
        return RawSample(images=pil_images, text=text, image_descriptors=image_descriptors)
