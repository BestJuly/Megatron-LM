# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native Kimi-K3 image / video preprocessing (no ``trust_remote_code``).

A port of ``KimiK3VisionProcessor.preprocess`` + ``media_utils`` at the pinned
HF revision (``configuration.KIMI_K3_HF_REVISION``) with
``preprocessor_config.json``'s ``media_proc_cfg``:

1. ``navit_resize_image``: scale down (never up) so the image fits the patch
   budget (``in_patch_limit`` patches overall, ``patch_limit_on_one_side`` per
   side), then pad right/bottom to a multiple of ``patch * merge`` = 28 px;
2. bicubic resize, chessboard fill of transparent pixels *after* the resize,
   zero padding;
3. ``(x / 255 - 0.5) / 0.5`` in float32;
4. ``navit_patchify``: ``[t*h*w, 3*14*14]`` rows in ``(t, h, w)`` raster order,
   each row channel-major.

The prompt text for one image is
``<|media_begin|>image WxH<|media_content|><|media_pad|><|media_end|>`` with
the *original* size; the single ``<|media_pad|>`` stands for
``(h/2)*(w/2)`` merged rows and is expanded by the dataset.

Video: the HF processor at this revision accepts images only, while MoonViT
accepts items of up to 4 frames (``temporal_merge_kernel_size``) and averages
them after the tower. :func:`preprocess_video_frames` resizes every frame of a
clip with the per-frame budget (``in_patch_limit_each_frame``) and groups the
frames into items of at most 4. It has no HF reference to match against.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

MEDIA_PROC_CFG = {
    "in_patch_limit": 65536,
    "patch_size": 14,
    "image_mean": (0.5, 0.5, 0.5),
    "image_std": (0.5, 0.5, 0.5),
    "merge_kernel_size": 2,
    "fixed_output_tokens": None,
    "patch_limit_on_one_side": 512,
    "in_patch_limit_each_frame": 16384,
    "in_patch_limit_video": 655360,
    "temporal_merge_kernel_size": 4,
    "chessboard_square_size": 8,
    "chessboard_square_on_top_left": True,
    "chessboard_white_value": 255,
    "chessboard_gray_value": 180,
}


@dataclass(frozen=True)
class ResizePlan:
    """Output of :func:`navit_resize`."""

    new_width: int
    new_height: int
    pad_width: int
    pad_height: int
    num_tokens: int


def navit_resize(
    width: int,
    height: int,
    patch_size: int = MEDIA_PROC_CFG["patch_size"],
    merge_kernel_size: int = MEDIA_PROC_CFG["merge_kernel_size"],
    in_patch_limit: int = MEDIA_PROC_CFG["in_patch_limit"],
    patch_limit_on_one_side: int = MEDIA_PROC_CFG["patch_limit_on_one_side"],
) -> ResizePlan:
    """``media_utils.navit_resize_image`` (``fixed_output_tokens=None``)."""
    s1 = math.sqrt(
        in_patch_limit / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)
    factor = merge_kernel_size * patch_size
    pad_h = (factor - new_h % factor) % factor
    pad_w = (factor - new_w % factor) % factor
    token_h = (new_h + pad_h) // factor
    token_w = (new_w + pad_w) // factor
    if token_h * merge_kernel_size > patch_limit_on_one_side or (
        token_w * merge_kernel_size > patch_limit_on_one_side
    ):
        raise ValueError(
            f"image {width}x{height} exceeds the {patch_limit_on_one_side}-patch side limit "
            "after padding"
        )
    return ResizePlan(new_w, new_h, pad_w, pad_h, token_h * token_w)


@lru_cache(maxsize=256)
def _chessboard(height: int, width: int) -> np.ndarray:
    cfg = MEDIA_PROC_CFG
    square = cfg["chessboard_square_size"]
    bg = np.ones((height, width, 3), dtype=np.uint8) * cfg["chessboard_white_value"]
    parity = 1 if cfg["chessboard_square_on_top_left"] else 0
    for y in range(0, height, square):
        for x in range(0, width, square):
            if (y // square + x // square) % 2 == parity:
                bg[y : y + square, x : x + square] = cfg["chessboard_gray_value"]
    return bg


def fill_transparent_background(image: Image.Image) -> Image.Image:
    """``fill_transparent_bg_with`` with the chessboard pattern."""
    if image.mode == "RGB":
        return image
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB")
    img = np.array(image.convert("RGBA"))
    bg = _chessboard(img.shape[0], img.shape[1])
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    alpha_3d = np.stack([alpha] * 3, axis=2)
    result = (alpha_3d * img[:, :, :3] + (1 - alpha_3d) * bg).astype(np.uint8)
    return Image.fromarray(result)


def _resize_pad(image: Image.Image, plan: ResizePlan) -> np.ndarray:
    resized = image.resize((plan.new_width, plan.new_height), resample=Image.Resampling.BICUBIC)
    array = np.asarray(fill_transparent_background(resized))
    return np.pad(array, ((0, plan.pad_height), (0, plan.pad_width), (0, 0)), mode="constant")


def _normalize(array: np.ndarray) -> np.ndarray:
    mean = np.array(MEDIA_PROC_CFG["image_mean"])
    std_inv = 1.0 / np.array(MEDIA_PROC_CFG["image_std"])
    x = (array / 255.0).astype(np.float32)
    x -= mean
    x *= std_inv
    return x


def patchify(frames: np.ndarray, patch_size: int = MEDIA_PROC_CFG["patch_size"]):
    """``navit_patchify``: ``[T, H, W, C]`` -> ``([T*h*w, C*p*p], (T, h, w))``."""
    t, h, w, c = frames.shape
    patches = frames.reshape(t, h // patch_size, patch_size, w // patch_size, patch_size, c)
    patches = patches.transpose(0, 1, 3, 5, 2, 4).reshape(-1, c * patch_size * patch_size)
    return patches, (t, h // patch_size, w // patch_size)


def make_image_prompt(width: int, height: int) -> str:
    """``KimiK3VisionProcessor.make_image_prompt`` (original image size)."""
    return f"<|media_begin|>image {width}x{height}<|media_content|><|media_pad|><|media_end|>"


def preprocess_image(image: Image.Image) -> Dict:
    """One image -> ``pixel_values [rows, 588]``, ``grid_thw (1, h, w)``, prompt."""
    plan = navit_resize(image.width, image.height)
    patches, grid = patchify(_normalize(_resize_pad(image, plan))[None])
    assert grid[1] * grid[2] // 4 == plan.num_tokens
    return {
        "pixel_values": torch.from_numpy(np.ascontiguousarray(patches)),
        "grid_thw": grid,
        "num_tokens": plan.num_tokens,
        "prompt": make_image_prompt(image.width, image.height),
    }


def preprocess_images(images: Sequence[Image.Image]) -> Dict:
    """Several images, concatenated as one sample's vision payload."""
    items = [preprocess_image(im) for im in images]
    return {
        "pixel_values": torch.cat([it["pixel_values"] for it in items]),
        "image_grid_thw": torch.tensor([it["grid_thw"] for it in items], dtype=torch.long),
        "num_tokens": [it["num_tokens"] for it in items],
        "prompts": [it["prompt"] for it in items],
    }


def preprocess_video_frames(frames: Sequence[Image.Image]) -> List[Dict]:
    """A clip -> items of at most ``temporal_merge_kernel_size`` frames each.

    All frames share the first frame's resize plan (per-frame patch budget), so
    every item of the clip has the same ``(h, w)``.
    """
    cfg = MEDIA_PROC_CFG
    first = frames[0]
    plan = navit_resize(first.width, first.height, in_patch_limit=cfg["in_patch_limit_each_frame"])
    arrays = [_normalize(_resize_pad(f, plan)) for f in frames]
    k = cfg["temporal_merge_kernel_size"]
    items = []
    for start in range(0, len(arrays), k):
        patches, grid = patchify(np.stack(arrays[start : start + k]))
        items.append(
            {
                "pixel_values": torch.from_numpy(np.ascontiguousarray(patches)),
                "grid_thw": grid,
                "num_tokens": plan.num_tokens,
            }
        )
    return items
