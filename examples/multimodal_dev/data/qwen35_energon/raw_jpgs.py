# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Materializer for Qwen3.5-VL samples that store raw JPEG payloads."""

from __future__ import annotations

import io
import pickle
import tarfile

from PIL import Image

from examples.multimodal_dev.data.dataset_utils import preprocess_image_to_patches


class _BytesOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"global objects are not allowed in .jpgs payloads: {module}.{name}"
        )


def _load_image_bytes_payload(payload: bytes) -> list[bytes]:
    images = _BytesOnlyUnpickler(io.BytesIO(payload)).load()
    if not isinstance(images, (list, tuple)) or not all(
        isinstance(image, (bytes, bytearray)) for image in images
    ):
        raise ValueError(".jpgs payload must contain a list of image byte strings")
    return [bytes(image) for image in images]


def _image_bytes_from_tar(desc):
    tar_path = desc.get("tar_path")
    key = desc.get("key")
    if not tar_path or not key:
        return None
    with tarfile.open(str(tar_path)) as tar:
        member = tar.extractfile(f"{key}.jpgs")
        if member is None:
            return None
        images = _load_image_bytes_payload(member.read())
    return images[int(desc.get("image_idx", 0))]


def materialize_image_descriptor(desc, grid_thw, *, pixel_dim: int, patch_size: int):
    image_bytes = (
        desc.get("_raw_image_bytes")
        or desc.get("image_bytes")
        or desc.get("jpg")
        or desc.get("bytes")
        or _image_bytes_from_tar(desc)
    )
    if image_bytes is None:
        raise ValueError("raw_jpgs descriptor is missing image bytes")
    with Image.open(io.BytesIO(bytes(image_bytes))) as image:
        patches = preprocess_image_to_patches(
            image.convert("RGB"),
            grid_thw,
            patch_size=int(patch_size),
            temporal_patch_size=int(desc.get("temporal_patch_size", 2)),
            spatial_merge_size=int(desc.get("spatial_merge_size", 2)),
        )
    if int(patches.shape[-1]) != int(pixel_dim):
        raise ValueError(
            "raw_jpgs materialized pixel_dim mismatch: "
            f"got {patches.shape[-1]}, expected {pixel_dim}"
        )
    return patches
