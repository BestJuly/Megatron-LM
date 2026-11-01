# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP collation under decoder TP: pixels stay on the TP source.

Under MDP the TP source is the only pixel owner of its TP group (rank map
``pixel_owners``), and pixels reach the encoder through the planning group's
all-to-all, never through the decoder. ``pack_or_pad_batch`` therefore keeps
them out of its TP broadcast: the source keeps its own copy, peers receive
everything else (text tensors, vision sidecar) and no pixels.

Run with::

    torchrun --nproc_per_node=2 -m pytest -q \\
        examples/multimodal_dev/tests/test_mdp_tp_collate.py
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset
from examples.multimodal_dev.forward_step import pack_or_pad_batch
from examples.multimodal_dev.models.qwen35_vl.mdp_adapter import build_collate_spec
from megatron.core import parallel_state
from megatron.core.mdp import window as mdp_window
from tests.unit_tests.test_utilities import Utils

_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    _WORLD < 2 or _WORLD % 2, reason="needs an even torchrun world for TP=2"
)

IMAGE_TOKEN_ID = 248056
SPEC = build_collate_spec(SimpleNamespace(image_token_id=IMAGE_TOKEN_ID))
SIDECAR_FIELDS = (
    "input_ids",
    "labels",
    "loss_mask",
    "vision_item_meta",
    "vision_decoder_positions",
)


@pytest.fixture(scope="module", autouse=True)
def _init_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    yield
    Utils.destroy_model_parallel()


def _multimodal_batch():
    dataset = MdpThdMockDataset(num_samples=32)
    indices = [i for i, (grids, _) in enumerate(dataset.scenarios) if grids][:3]
    return [dict(dataset[i]) for i in indices]


def _pack(owner_context=None):
    is_src = parallel_state.get_tensor_model_parallel_rank() == 0
    batch = _multimodal_batch() if is_src else None
    mdp_window._PIXEL_OWNERSHIP.value = owner_context
    try:
        return pack_or_pad_batch(
            batch, use_packed_sequence=True, with_vision_sidecar=True, vision_collate_spec=SPEC
        )
    finally:
        mdp_window._PIXEL_OWNERSHIP.value = None


def _assert_identical_across_tp(packed):
    group = parallel_state.get_tensor_model_parallel_group()
    for field in SIDECAR_FIELDS:
        local = packed[field].cuda().contiguous()
        copies = [torch.empty_like(local) for _ in range(2)]
        torch.distributed.all_gather(copies, local, group=group)
        assert torch.equal(copies[0], copies[1]), field


def test_owner_keeps_pixels_and_peer_receives_none():
    packed = _pack()
    _assert_identical_across_tp(packed)
    if parallel_state.get_tensor_model_parallel_rank() == 0:
        expected = torch.cat([s["pixel_values"] for s in _multimodal_batch()])
        assert packed["pixel_values"].is_cuda
        assert torch.equal(packed["pixel_values"].cpu(), expected)
    else:
        assert "pixel_values" not in packed


def test_non_owner_source_materializes_no_pixels():
    # The capture context says another worker owns this microbatch: neither
    # the source nor its peer may hold pixels, yet the sidecar still arrives.
    packed = _pack(owner_context=(1, 0))
    _assert_identical_across_tp(packed)
    assert "pixel_values" not in packed
