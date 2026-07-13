# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from examples.multimodal_dev.data.dataset_utils import RawSample, preprocess_image_to_patches


def _load_video_sequence(path: str, num_frames: int):
    from nvidia.dali import fn, pipeline_def, types

    device_id = 0
    if torch.cuda.is_available():
        device_id = int(torch.cuda.current_device())

    @pipeline_def(batch_size=1, num_threads=1, device_id=device_id)
    def _video_pipe():
        return fn.readers.video(
            device="gpu",
            filenames=[path],
            sequence_length=int(num_frames),
            shard_id=0,
            num_shards=1,
            random_shuffle=False,
            initial_fill=1,
            normalized=False,
            image_type=types.RGB,
            dtype=types.UINT8,
            skip_vfr_check=True,
        )

    pipe = _video_pipe()
    pipe.build()
    return pipe.run()[0].as_cpu().as_array()[0]


def materialize_image_descriptor(
    desc: Dict,
    grid_thw: Sequence[int],
    *,
    pixel_dim: int,
    patch_size: int,
):
    if desc.get("kind") != "video_frame":
        raise ValueError(f"OmniStar materializer cannot handle descriptor: {desc!r}")
    frame_idx = int(desc.get("frame_idx", 0))
    num_frames = max(frame_idx + 1, int(desc.get("num_frames", frame_idx + 1)))
    frames = _load_video_sequence(desc["video_path"], num_frames)
    if frame_idx >= int(frames.shape[0]):
        raise IndexError(
            f"video frame index out of range: {frame_idx} >= {frames.shape[0]}"
        )
    patches = preprocess_image_to_patches(
        Image.fromarray(frames[frame_idx]).convert("RGB"),
        grid_thw,
        patch_size=int(patch_size),
    )
    if int(patches.shape[-1]) != int(pixel_dim):
        raise ValueError(
            "OmniStar materialized pixel_dim mismatch: "
            f"got {patches.shape[-1]}, expected {pixel_dim}"
        )
    return patches


class OmniStarRngBackend:
    """yzy666/OmniStar-RNG annotated video-frame backend."""

    def __init__(
        self,
        root: str,
        max_samples: Optional[int] = None,
        min_frames: int = 30,
        max_frames: int = 30,
        annotations_filename: str = "RNG_task_train.jsonl",
    ):
        self.root = Path(root)
        self._samples: List[dict] = []
        ann_path = self.root / annotations_filename
        target = max_samples if max_samples is not None else 10**18
        max_frames = max(1, int(max_frames))
        with open(ann_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                frame_paths = sample.get("image") or []
                if len(frame_paths) < int(min_frames):
                    continue
                sample = dict(sample)
                sample["image"] = frame_paths[:max_frames]
                sample["height_list"] = list(sample.get("height_list") or [])[:max_frames]
                sample["width_list"] = list(sample.get("width_list") or [])[:max_frames]
                self._samples.append(sample)
                if len(self._samples) >= target:
                    break

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx) -> RawSample:
        sample = self._samples[int(idx) % len(self._samples)]
        frame_paths = list(sample.get("image") or [])
        heights = list(sample.get("height_list") or [])
        widths = list(sample.get("width_list") or [])

        video_id = None
        if frame_paths:
            video_id = Path(frame_paths[0]).parent.name
        video_path = str(self.root / "videos" / f"{video_id}.mp4") if video_id else ""

        descriptors: List[Dict] = []
        total_frames = len(frame_paths)
        for frame_idx, rel_path in enumerate(frame_paths):
            width = int(widths[frame_idx]) if frame_idx < len(widths) else 0
            height = int(heights[frame_idx]) if frame_idx < len(heights) else 0
            if width <= 0 or height <= 0:
                continue
            descriptors.append(
                {
                    "kind": "video_frame",
                    "materializer": "examples.multimodal_dev.data.omnistar_rng",
                    "video_path": video_path,
                    "frame_idx": int(frame_idx),
                    "num_frames": int(total_frames),
                    "width": width,
                    "height": height,
                    "path": rel_path,
                }
            )

        text_parts = []
        for turn in sample.get("conversations") or []:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from") or "human"
            value = turn.get("value") or ""
            text_parts.append(f"{role}: {value}")

        return RawSample(
            images=[],
            text="\n".join(text_parts),
            image_descriptors=descriptors,
        )
