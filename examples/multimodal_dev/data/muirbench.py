# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

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


class MuirbenchBackend:
    """MUIRBENCH/MUIRBENCH - every sample has a list ``image_list`` of inline-bytes
    images, plus question, options, answer."""

    def __init__(self, root: str, max_samples: Optional[int] = None):
        import pyarrow.parquet as pq
        self.root = Path(root)
        self._tables = []
        self._table_paths = []
        self._index = []
        for pq_path in sorted((self.root / "data").glob("*.parquet")):
            t = pq.read_table(str(pq_path))
            self._tables.append(t)
            self._table_paths.append(str(pq_path))
            for i in range(t.num_rows):
                self._index.append((len(self._tables) - 1, i))
        if max_samples is not None and max_samples < len(self._index):
            rng = random.Random(0xC0FFEE)
            self._index = rng.sample(self._index, max_samples)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx) -> RawSample:
        ti, ri = self._index[idx]
        t = self._tables[ti]
        imgs_struct = t.column("image_list")[ri].as_py()
        question = t.column("question")[ri].as_py()
        options = t.column("options")[ri].as_py() or []
        answer = t.column("answer")[ri].as_py()
        pil_images = []
        image_descriptors: List[Dict] = []
        for image_idx, s in enumerate(imgs_struct):
            b = s.get("bytes")
            if b is None:
                continue
            descriptor = {
                "kind": "parquet_list_image",
                "parquet_path": self._table_paths[ti],
                "row_idx": int(ri),
                "column": "image_list",
                "image_idx": int(image_idx),
            }
            append_image_or_descriptor(
                pil_images,
                image_descriptors,
                b,
                descriptor,
                materializer="examples.multimodal_dev.data.muirbench",
            )
        text = (
            f"Question: {question}\n"
            f"Options: {' | '.join(map(str, options))}\n"
            f"Answer: {answer}"
        )
        return RawSample(
            images=pil_images,
            text=text,
            image_descriptors=image_descriptors,
        )
