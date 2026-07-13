# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for multimodal dataset backends."""

from __future__ import annotations

import io
import threading
import zipfile
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from megatron.training import get_args

_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


@dataclass
class RawSample:
    images: List[Image.Image]
    text: str
    image_descriptors: Optional[List[Dict]] = None


_ZIP_FILE_CACHE: Dict[str, zipfile.ZipFile] = {}
_ZIP_FILE_LOCKS: Dict[str, threading.Lock] = {}
_ZIP_FILE_CACHE_LOCK = threading.Lock()

_PARQUET_FILE_CACHE: Dict[str, Any] = {}
_PARQUET_FILE_LOCKS: Dict[str, threading.Lock] = {}
_PARQUET_FILE_CACHE_LOCK = threading.Lock()
_PARQUET_ROW_GROUP_STARTS_CACHE: Dict[int, Sequence[int]] = {}


def metadata_only_batch_enabled() -> bool:
    return bool(getattr(get_args(), "mdp_encoder_mode", False))


def append_image_or_descriptor(
    pil_images: List[Image.Image],
    image_descriptors: List[Dict],
    image_bytes: bytes,
    descriptor: Dict,
    *,
    materializer: Optional[str] = None,
) -> None:
    desc = dict(descriptor)
    if materializer is not None:
        desc["materializer"] = materializer
    if metadata_only_batch_enabled():
        with Image.open(io.BytesIO(image_bytes)) as im:
            desc["width"] = int(im.size[0])
            desc["height"] = int(im.size[1])
    else:
        with Image.open(io.BytesIO(image_bytes)) as im:
            pil_images.append(im.convert("RGB"))
    image_descriptors.append(desc)


def _get_parquet_file(path: str):
    import pyarrow.parquet as pq

    path = str(path)
    with _PARQUET_FILE_CACHE_LOCK:
        pq_file = _PARQUET_FILE_CACHE.get(path)
        if pq_file is None:
            pq_file = pq.ParquetFile(path)
            _PARQUET_FILE_CACHE[path] = pq_file
            _PARQUET_FILE_LOCKS[path] = threading.Lock()
        return pq_file


def _get_parquet_lock(path: str) -> threading.Lock:
    path = str(path)
    with _PARQUET_FILE_CACHE_LOCK:
        lock = _PARQUET_FILE_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _PARQUET_FILE_LOCKS[path] = lock
        return lock


def _get_row_group_starts(pq_file) -> Sequence[int]:
    cache_key = id(pq_file)
    cached = _PARQUET_ROW_GROUP_STARTS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    offset = 0
    values = []
    for idx in range(pq_file.num_row_groups):
        values.append(offset)
        offset += pq_file.metadata.row_group(idx).num_rows
    starts = tuple(values)
    _PARQUET_ROW_GROUP_STARTS_CACHE[cache_key] = starts
    return starts


def _locate_row_group(pq_file, row_idx: int):
    starts = _get_row_group_starts(pq_file)
    if not starts:
        raise ValueError("parquet file has no row groups")
    group_idx = bisect_right(starts, int(row_idx)) - 1
    if group_idx < 0:
        raise IndexError(f"negative parquet row index: {row_idx}")

    group_rows = pq_file.metadata.row_group(group_idx).num_rows
    local_idx = int(row_idx) - starts[group_idx]
    if local_idx >= group_rows:
        raise IndexError(f"parquet row index out of range: row_idx={row_idx} group={group_idx}")
    return group_idx, local_idx


def _load_parquet_image_value(desc: Dict[str, Any]):
    pq_path = desc["parquet_path"]
    column = desc["column"]
    row_idx = int(desc["row_idx"])
    pq_file = _get_parquet_file(pq_path)
    with _get_parquet_lock(pq_path):
        row_group_idx, local_idx = _locate_row_group(pq_file, row_idx)
        table = pq_file.read_row_group(row_group_idx, columns=[column])
    return table.column(column)[local_idx].as_py()


def _get_zip_file(path: str) -> zipfile.ZipFile:
    path = str(path)
    with _ZIP_FILE_CACHE_LOCK:
        zf = _ZIP_FILE_CACHE.get(path)
        if zf is None:
            zf = zipfile.ZipFile(path, "r")
            _ZIP_FILE_CACHE[path] = zf
            _ZIP_FILE_LOCKS[path] = threading.Lock()
        return zf


def _get_zip_lock(path: str) -> threading.Lock:
    path = str(path)
    with _ZIP_FILE_CACHE_LOCK:
        lock = _ZIP_FILE_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _ZIP_FILE_LOCKS[path] = lock
        return lock


def load_image_from_descriptor(desc: Dict[str, Any]) -> Image.Image:
    kind = desc.get("kind")
    if kind == "zip_image":
        zip_path = desc["zip_path"]
        candidates = desc.get("candidates") or [desc.get("path")]
        data = None
        zf = _get_zip_file(zip_path)
        with _get_zip_lock(zip_path):
            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    data = zf.read(candidate)
                    break
                except KeyError:
                    continue
        if data is None:
            raise FileNotFoundError(f"image not found in zip descriptor: {desc}")
        return Image.open(io.BytesIO(data)).convert("RGB")

    if kind in ("parquet_list_image", "parquet_column_image"):
        value = _load_parquet_image_value(desc)
        if kind == "parquet_list_image":
            value = value[int(desc["image_idx"])]
        if not isinstance(value, dict):
            raise ValueError(f"unsupported parquet image value for descriptor: {desc}")
        data = value.get("bytes")
        if data is None:
            raise ValueError(f"parquet image descriptor has no bytes: {desc}")
        return Image.open(io.BytesIO(data)).convert("RGB")

    raise ValueError(f"unsupported image descriptor kind: {kind!r}")


def preprocess_image_to_patches(
    im: Image.Image,
    grid_thw: Sequence[int],
    *,
    patch_size: int,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    grid_t, h_p, w_p = [int(x) for x in grid_thw]
    height = h_p * int(patch_size)
    width = w_p * int(patch_size)
    if im.size != (width, height):
        im = im.resize((width, height), Image.Resampling.BICUBIC)

    arr = (np.asarray(im, dtype=np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    frame = torch.from_numpy(arr).permute(2, 0, 1).float()
    frames = frame.unsqueeze(0).expand(grid_t * int(temporal_patch_size), -1, -1, -1).contiguous()

    p = int(patch_size)
    m = int(spatial_merge_size)
    if h_p % m != 0 or w_p % m != 0:
        raise ValueError(
            "image grid must be divisible by spatial_merge_size: "
            f"grid=({grid_t}, {h_p}, {w_p}) merge={m}"
        )
    patches = frames.reshape(grid_t, int(temporal_patch_size), 3, h_p // m, m, p, w_p // m, m, p)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    return patches.reshape(grid_t * h_p * w_p, 3 * int(temporal_patch_size) * p * p).to(
        torch.float32
    )


def materialize_image_descriptor(
    desc: Dict[str, Any], grid_thw: Sequence[int], *, pixel_dim: int, patch_size: int
) -> torch.Tensor:
    patches = preprocess_image_to_patches(
        load_image_from_descriptor(desc),
        grid_thw,
        patch_size=int(patch_size),
        temporal_patch_size=int(desc.get("temporal_patch_size", 2)),
        spatial_merge_size=int(desc.get("spatial_merge_size", 2)),
    )
    if int(patches.shape[-1]) != int(pixel_dim):
        raise ValueError(
            "materialized image pixel_dim mismatch: "
            f"got {patches.shape[-1]}, expected {pixel_dim}"
        )
    return patches
