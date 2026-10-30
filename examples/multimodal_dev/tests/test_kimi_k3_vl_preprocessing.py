# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native Kimi-K3 image preprocessing vs the HF ``KimiK3VisionProcessor``.

``models/kimi_k3_vl/preprocessing.py`` ports the pinned HF processor so that
training does not need ``trust_remote_code``. On images of several aspect
ratios -- including one with transparency (chessboard fill after resize) and
one that hits ``patch_limit_on_one_side`` -- both must produce identical
``grid_thw``, identical token counts and prompts, and pixel rows within 1e-6.

Needs the transformers 4.56 overlay (see tests/_kimi_hf.py); runs on CPU.
"""

import copy
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from examples.multimodal_dev.models.kimi_k3_vl import preprocessing as native
from examples.multimodal_dev.tests._kimi_hf import hf_reference_dir, import_hf_module


def _images():
    rng = np.random.default_rng(0)

    def rgb(w, h):
        return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")

    rgba = rng.integers(0, 256, (256, 256, 4), dtype=np.uint8)
    rgba[:, :128, 3] = 0  # transparent left half
    return {
        "landscape_640x480": rgb(640, 480),
        "tall_300x900": rgb(300, 900),
        "wide_1000x200": rgb(1000, 200),
        "rgba_transparent_256": Image.fromarray(rgba, "RGBA"),
        "side_limit_8000x300": rgb(8000, 300),
        "tiny_30x20": rgb(30, 20),
    }


@pytest.fixture(scope="module")
def hf_processor():
    proc_mod = import_hf_module("kimi_k3_vision_processing")
    with open(os.path.join(hf_reference_dir(), "preprocessor_config.json")) as f:
        cfg = json.load(f)["media_proc_cfg"]
    return proc_mod.KimiK3VisionProcessor(media_proc_cfg=cfg)


@pytest.mark.parametrize("name", list(_images()))
def test_native_preprocessing_matches_hf(hf_processor, name):
    image = _images()[name]
    hf_out = hf_processor.preprocess([{"type": "image", "image": copy.copy(image)}])
    hf_tokens = hf_processor.media_tokens_calculator({"type": "image", "image": copy.copy(image)})
    hf_prompt = hf_processor.make_image_prompt(image.width, image.height)

    ours = native.preprocess_image(image)
    assert tuple(int(v) for v in hf_out["grid_thws"][0]) == tuple(ours["grid_thw"])
    assert ours["num_tokens"] == int(hf_tokens)
    assert ours["prompt"] == hf_prompt
    hf_pixels = torch.as_tensor(hf_out["pixel_values"]).reshape(ours["pixel_values"].shape[0], -1)
    assert hf_pixels.shape == ours["pixel_values"].shape
    assert torch.allclose(ours["pixel_values"].float(), hf_pixels.float(), atol=1e-6, rtol=0)
