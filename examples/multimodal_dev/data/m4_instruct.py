# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image

from examples.multimodal_dev.data.dataset_utils import RawSample, append_image_or_descriptor
from examples.multimodal_dev.data.dataset_utils import (
    materialize_image_descriptor as _materialize_image_descriptor,
)


def materialize_image_descriptor(desc, grid_thw, *, pixel_dim: int, patch_size: int):
    return _materialize_image_descriptor(desc, grid_thw, pixel_dim=pixel_dim, patch_size=patch_size)


class M4InstructBackend:
    """lmms-lab/M4-Instruct-Data: heavy-tail multi-image conversations."""

    _SKIP_SOURCES = ("dreamsim",)
    _SOURCE_ZIP_ALIASES = {"RAVEN": "RAVEN_train_images"}

    def __init__(
        self,
        root: str,
        max_samples: Optional[int] = None,
        subsets: Optional[Sequence[str]] = None,
        annotations_filename: str = "m4_instruct_annotations.json",
    ):
        self.root = Path(root)
        ann_path = self.root / annotations_filename
        if not ann_path.exists():
            raise FileNotFoundError(f"M4 annotations not found: {ann_path}")

        with open(ann_path, "r") as f:
            all_samples = json.load(f)

        available_zips = {
            p.stem
            for p in self.root.glob("*.zip")
            if not any(p.stem.startswith(skip) for skip in self._SKIP_SOURCES)
        }
        wanted = None if subsets is None else set(subsets)

        self._zip_paths: dict = {}
        self._zip_handles: dict = {}
        for stem in available_zips:
            self._zip_paths[stem] = str(self.root / f"{stem}.zip")

        target = max_samples if max_samples is not None else 10**18
        self._samples: List[dict] = []
        for s in all_samples:
            meta = s.get("metadata") or {}
            ds_name = meta.get("dataset")
            if wanted is not None and ds_name not in wanted:
                continue
            zip_stem = self._SOURCE_ZIP_ALIASES.get(ds_name, ds_name)
            if zip_stem not in self._zip_paths:
                imgs = s.get("image") or []
                if imgs:
                    first = imgs[0]
                    prefix = first.split("/", 1)[0]
                    cand = self._SOURCE_ZIP_ALIASES.get(prefix, prefix)
                    if cand in self._zip_paths:
                        zip_stem = cand
                    else:
                        continue
                else:
                    continue
            s["_resolved_zip_stem"] = zip_stem
            self._samples.append(s)
            if len(self._samples) >= target:
                break

        del all_samples

        if max_samples is not None and max_samples < len(self._samples):
            rng = random.Random(0xC0FFEE)
            self._samples = rng.sample(self._samples, max_samples)

        self._zipfile_mod = zipfile

    def __len__(self):
        return len(self._samples)

    def _get_zip(self, stem: str):
        zh = self._zip_handles.get(stem)
        if zh is None:
            path = self._zip_paths.get(stem)
            if path is None:
                return None
            zh = self._zipfile_mod.ZipFile(path, "r")
            self._zip_handles[stem] = zh
        return zh

    def __getitem__(self, idx) -> RawSample:
        sample = self._samples[idx]
        zip_stem = sample.get("_resolved_zip_stem")
        zh = self._get_zip(zip_stem) if zip_stem else None

        pil_images: List[Image.Image] = []
        image_descriptors: List[Dict] = []
        for p in sample.get("image", []) or []:
            if zh is None:
                continue
            data = None
            candidates = [p, f"{zip_stem}/{p}", p.split("/", 1)[-1]]
            for candidate in candidates:
                try:
                    data = zh.read(candidate)
                    break
                except KeyError:
                    continue
            if data is None:
                continue
            descriptor = {
                "kind": "zip_image",
                "zip_path": self._zip_paths[zip_stem],
                "path": p,
                "candidates": candidates,
            }
            append_image_or_descriptor(
                pil_images,
                image_descriptors,
                data,
                descriptor,
                materializer="examples.multimodal_dev.data.m4_instruct",
            )

        convs = sample.get("conversations") or []
        text_parts = []
        for turn in convs:
            val = turn.get("value") or ""
            role = turn.get("from") or "human"
            text_parts.append(f"{role}: {val}")
        text = "\n".join(text_parts)
        return RawSample(images=pil_images, text=text, image_descriptors=image_descriptors)
