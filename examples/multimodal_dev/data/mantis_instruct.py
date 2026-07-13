# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from examples.multimodal_dev.data.dataset_utils import RawSample, append_image_or_descriptor
from examples.multimodal_dev.data.dataset_utils import (
    materialize_image_descriptor as _materialize_image_descriptor,
)


def materialize_image_descriptor(desc, grid_thw, *, pixel_dim: int, patch_size: int):
    return _materialize_image_descriptor(desc, grid_thw, pixel_dim=pixel_dim, patch_size=patch_size)


class MantisInstructBackend:
    """Mantis-Instruct (TIGER-Lab/Mantis-Instruct).

    Schema per parquet row:
      id           : str
      images       : list[{bytes: bytes|None, path: str}]  - bytes can be None;
                     when None the actual image lives in ``<subset>/train_images.zip``
                     keyed by ``path``.
      conversation : list[{role: 'user'|'assistant', content: str}]
      source       : str

    Some subsets store images inline (bytes != None); others use the zip.
    We support both transparently.
    """

    # Subsets we KEEP by default (verified multi-image, n_images >= 2 median).
    # nlvr2 (2), imagecode (10), lrv_multi (1-8 med 3), multi_vqa (2-6 med 4),
    # spot-the-diff (2), birds-to-words (2). iconqa and coinstruct are
    # 1-image-only - excluded.
    DEFAULT_MULTI_IMAGE_SUBSETS = (
        "nlvr2",
        "imagecode",
        "lrv_multi",
        "multi_vqa",
        "spot-the-diff",
        "birds-to-words",
    )

    def __init__(
        self,
        root: str,
        subsets: Optional[Sequence[str]] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        import pyarrow.parquet as pq

        self.root = Path(root)
        if subsets is None:
            # Default: only the verified multi-image subsets that exist on
            # disk (gracefully skip the ones not yet downloaded).
            subsets = [
                s
                for s in self.DEFAULT_MULTI_IMAGE_SUBSETS
                if (self.root / s).is_dir() and any((self.root / s).glob(f"{split}-*.parquet"))
            ]
        self.subsets = list(subsets)
        # Build a flat index: list of (subset, parquet_path, row_idx).
        self._index: List[Tuple[str, str, int]] = []
        # Keep zip paths only.  DataLoader workers are forked after dataset
        # construction; inheriting an already-open ZipFile can make reads race
        # on a shared fd and silently drop images.  Open lazily per worker PID.
        self._zip_paths: dict = {}
        self._zip_handles: dict = {}
        # Keep parquet tables in memory (Mantis subsets we use are small;
        # the heavy data lives in the zips).
        self._tables: dict = {}

        for sub in self.subsets:
            for pq_path in sorted((self.root / sub).glob(f"{split}-*.parquet")):
                t = pq.read_table(str(pq_path))
                self._tables[str(pq_path)] = t
                for i in range(t.num_rows):
                    self._index.append((sub, str(pq_path), i))
            zp = self.root / sub / f"{split}_images.zip"
            if zp.exists():
                self._zip_paths[sub] = str(zp)

        if max_samples is not None and max_samples < len(self._index):
            # Stable subsample: fixed seed so repeated runs see the same set.
            rng = random.Random(0xC0FFEE)
            self._index = rng.sample(self._index, max_samples)

    def __len__(self):
        return len(self._index)

    def _zip_for_subset(self, sub: str):
        zip_path = self._zip_paths.get(sub)
        if not zip_path:
            return None
        key = (os.getpid(), sub)
        zf = self._zip_handles.get(key)
        if zf is None:
            zf = zipfile.ZipFile(zip_path, "r")
            self._zip_handles[key] = zf
        return zf

    def __getitem__(self, idx) -> RawSample:
        sub, pq_path, row_idx = self._index[idx]
        t = self._tables[pq_path]
        # Pull only the needed columns for one row (cheap with arrow).
        row_images = t.column("images")[row_idx].as_py()
        row_conv = t.column("conversation")[row_idx].as_py()

        pil_images: List[Image.Image] = []
        image_descriptors: List[Dict] = []
        for image_idx, img_struct in enumerate(row_images):
            b = img_struct.get("bytes")
            p = img_struct.get("path")
            descriptor = {
                "kind": "parquet_list_image",
                "parquet_path": pq_path,
                "row_idx": int(row_idx),
                "column": "images",
                "image_idx": int(image_idx),
            }
            zf = self._zip_for_subset(sub)
            if b is None and zf is not None and p:
                # The zip sometimes nests entries under the subset prefix.
                candidates = [p, f"{sub}/{p}"]
                for candidate in candidates:
                    try:
                        b = zf.read(candidate)
                    except KeyError:
                        continue
                    descriptor = {
                        "kind": "zip_image",
                        "zip_path": zf.filename,
                        "path": p,
                        "candidates": candidates,
                    }
                    break
            if b is None:
                continue
            append_image_or_descriptor(
                pil_images,
                image_descriptors,
                b,
                descriptor,
                materializer="examples.multimodal_dev.data.mantis_instruct",
            )

        text_parts = []
        for turn in row_conv:
            text_parts.append(f"{turn.get('role','user')}: {turn.get('content','')}")
        text = "\n".join(text_parts)
        return RawSample(images=pil_images, text=text, image_descriptors=image_descriptors)
