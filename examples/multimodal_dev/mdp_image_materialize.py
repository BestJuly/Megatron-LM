# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP image descriptor serialization and loader-side materialization.

Dataset-specific storage details such as mock generation, zip/parquet reads, or
video decode belong in the corresponding ``examples.multimodal_dev.data`` file.
"""

from __future__ import annotations

import base64
import importlib
import json
from typing import Any, Dict, List, Optional, Sequence

import torch

_BYTES_KEY = "__mdp_bytes_b64__"


def _json_safe_descriptor_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {_BYTES_KEY: base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _json_safe_descriptor_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_descriptor_value(item) for item in value]
    return value


def _restore_descriptor_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {_BYTES_KEY}:
            return base64.b64decode(str(value[_BYTES_KEY]).encode("ascii"))
        return {key: _restore_descriptor_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_descriptor_value(item) for item in value]
    return value


def encode_image_descriptors(descriptors: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(
        [_json_safe_descriptor_value(desc) for desc in descriptors],
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_image_descriptors(value: Any) -> Optional[List[Dict[str, Any]]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, bytes):
        value = json.loads(value.decode("utf-8"))
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            return []
        if len(value) == 1 and isinstance(value[0], (str, bytes)):
            return decode_image_descriptors(value[0])
        if all(isinstance(item, dict) for item in value):
            return [_restore_descriptor_value(item) for item in value]
    return None


def _materializer_from_descriptor(desc: Dict[str, Any]):
    module_name = desc.get("materializer")
    if not module_name:
        raise ValueError("MDP image descriptor is missing a 'materializer' module: " f"{desc!r}")
    module = importlib.import_module(str(module_name))
    fn = getattr(module, "materialize_image_descriptor", None)
    if fn is None:
        raise AttributeError(
            f"MDP descriptor materializer module {module_name!r} does not "
            "define materialize_image_descriptor()."
        )
    return fn


def materialize_descriptor(
    desc: Dict[str, Any], grid_thw: Sequence[int], *, pixel_dim: int, patch_size: int
) -> torch.Tensor:
    patches = _materializer_from_descriptor(desc)(
        desc, grid_thw, pixel_dim=int(pixel_dim), patch_size=int(patch_size)
    )
    if not torch.is_tensor(patches):
        raise TypeError(
            "MDP descriptor materializer must return a torch.Tensor, got "
            f"{type(patches).__name__}"
        )
    if patches.dim() != 2:
        raise ValueError(
            "MDP descriptor materializer must return [patches, pixel_dim], "
            f"got shape {tuple(patches.shape)}"
        )
    if int(patches.shape[-1]) != int(pixel_dim):
        raise ValueError(
            "MDP materialized pixel_dim mismatch: " f"got {patches.shape[-1]}, expected {pixel_dim}"
        )
    return patches.to(dtype=torch.float32)
