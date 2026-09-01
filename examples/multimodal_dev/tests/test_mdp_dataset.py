# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dual-THD dataset/collator contract tests for MDP (design doc section 8.7/8.8).

Covers the MdpThdMockDataset scenarios (multi-image, variable grid, video,
interleaved text, text-only), the vision sidecar produced by
``pack_or_pad_batch(with_vision_sidecar=True)``, true-vs-padded ``cu_seqlens``,
and the negative consistency guards.

Run with::

    torchrun --nproc_per_node=1 -m pytest -q \\
        examples/multimodal_dev/tests/test_mdp_dataset.py
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset, item_sentinel
from examples.multimodal_dev.forward_step import (
    VISION_ITEM_META_COLUMNS,
    build_vision_sidecar,
    pack_or_pad_batch,
)
from tests.unit_tests.test_utilities import Utils

IMAGE_TOKEN_ID = 248056
MERGE = 2


@pytest.fixture(scope="module", autouse=True)
def _init_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _batch(indices, dataset=None):
    dataset = dataset or MdpThdMockDataset(num_samples=32)
    return [dataset[i] for i in indices]


def test_dataset_is_deterministic():
    a = MdpThdMockDataset(num_samples=8)[1]
    b = MdpThdMockDataset(num_samples=8)[1]
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_scenarios_cover_required_cases():
    dataset = MdpThdMockDataset(num_samples=8)
    per_sample_items = [int(dataset[i]["image_grid_thw"].shape[0]) for i in range(5)]
    assert 0 in per_sample_items, "text-only sample required"
    assert any(n >= 2 for n in per_sample_items), "multi-image sample required"
    grids = torch.cat([dataset[i]["image_grid_thw"] for i in range(5)])
    assert (grids[:, 0] > 1).any(), "multi-frame video item required"
    assert len({(int(h), int(w)) for _, h, w in grids}) > 1, "variable grids required"


def test_sidecar_metadata_and_sentinels():
    dataset = MdpThdMockDataset(num_samples=32)
    # Three multimodal samples plus a text-only one, located in the scenario
    # pool rather than assumed at a fixed index.
    text_only = next(i for i, (grids, _) in enumerate(dataset.scenarios) if not grids)
    multimodal = [i for i, (grids, _) in enumerate(dataset.scenarios) if grids][:3]
    indices = sorted(multimodal + [text_only])
    text_only_position = indices.index(text_only)
    batch = _batch(indices, dataset)
    packed = pack_or_pad_batch(
        [dict(s) for s in batch], use_packed_sequence=True, with_vision_sidecar=True
    )
    meta = packed["vision_item_meta"].cpu()
    positions = packed["vision_decoder_positions"].cpu()
    input_ids = packed["input_ids"][0].cpu()
    pixels = packed["pixel_values"].cpu()

    expected_items = sum(int(s["image_grid_thw"].shape[0]) for s in batch)
    assert meta.shape == (expected_items, VISION_ITEM_META_COLUMNS)
    # Ordered by (sample_index, image_ordinal).
    order = [(int(r[0]), int(r[1])) for r in meta]
    assert order == sorted(order)
    assert text_only_position not in {
        s for s, _ in order
    }, "text-only sample must contribute no items"

    position_cursor = 0
    for row in meta:
        sample_index, ordinal, t, h, w, payload_row_start = (int(v) for v in row)
        payload_rows = t * h * w
        output_rows = t * (h // MERGE) * (w // MERGE)
        # Sentinel round-trip: the packed pixel slice is exactly this item.
        sentinel = float(item_sentinel(indices[sample_index], ordinal))
        slice_ = pixels[payload_row_start : payload_row_start + payload_rows]
        assert (slice_ == sentinel).all(), (sample_index, ordinal)
        # Decoder positions land on image tokens, one per merged row.
        item_positions = positions[position_cursor : position_cursor + output_rows]
        position_cursor += output_rows
        assert (input_ids[item_positions] == IMAGE_TOKEN_ID).all()
    assert position_cursor == positions.numel()
    # Every image token in the pack is claimed exactly once.
    assert positions.numel() == int((input_ids == IMAGE_TOKEN_ID).sum())
    assert positions.unique().numel() == positions.numel()


def test_true_and_padded_cu_seqlens_differ_under_alignment():
    batch = _batch([0, 1, 3])
    packed = pack_or_pad_batch(
        [dict(s) for s in batch],
        use_packed_sequence=True,
        pad_to_multiple=16,
        with_vision_sidecar=True,
    )
    params = packed["packed_seq_params"]
    true_cu = params.cu_seqlens_q.cpu()
    padded_cu = params.cu_seqlens_q_padded.cpu()
    assert not torch.equal(true_cu, padded_cu)
    # padding_mask marks exactly the collate-padded tail positions.
    mask = packed["padding_mask"][0].cpu()
    expected_padding = int(padded_cu[-1]) - int(
        sum(true_cu[i + 1] - true_cu[i] for i in range(len(true_cu) - 1))
    )
    assert int(mask.sum()) == expected_padding
    # Sidecar decoder positions must follow the padded physical layout.
    meta = packed["vision_item_meta"].cpu()
    positions = packed["vision_decoder_positions"].cpu()
    input_ids = packed["input_ids"][0].cpu()
    assert (input_ids[positions] == IMAGE_TOKEN_ID).all()
    # The sample-span columns must mirror cu_seqlens_q_padded exactly: MDP
    # derives each row's context-parallel owner from them without ever
    # touching the device vector again.
    for row in meta.tolist():
        sample_index = int(row[0])
        assert int(row[6]) == int(padded_cu[sample_index])
        assert int(row[7]) == int(padded_cu[sample_index + 1] - padded_cu[sample_index])


def test_sidecar_positions_respect_interleaved_text():
    # A multi-image scenario carries text between its images: consecutive
    # position blocks must be non-adjacent.
    dataset = MdpThdMockDataset(num_samples=8)
    index, (grids, _) = next(
        (i, s) for i, s in enumerate(dataset.scenarios) if len(s[0]) >= 2
    )
    batch = _batch([index], dataset)
    packed = pack_or_pad_batch(
        [dict(s) for s in batch], use_packed_sequence=True, with_vision_sidecar=True
    )
    meta = packed["vision_item_meta"].cpu()
    positions = packed["vision_decoder_positions"].cpu()
    assert meta.shape[0] == len(grids)
    t0, h0, w0 = (int(v) for v in meta[0][2:5])
    first_rows = t0 * (h0 // MERGE) * (w0 // MERGE)
    first_end = int(positions[first_rows - 1])
    second_start = int(positions[first_rows])
    assert second_start > first_end + 1, "interleaved text must separate the items"


def test_text_only_batch_produces_empty_sidecar():
    dataset = MdpThdMockDataset(num_samples=8)
    text_only = next(i for i, (grids, _) in enumerate(dataset.scenarios) if not grids)
    batch = _batch([text_only], dataset)
    packed = pack_or_pad_batch(
        [dict(s) for s in batch], use_packed_sequence=True, with_vision_sidecar=True
    )
    assert packed["vision_item_meta"].shape == (0, VISION_ITEM_META_COLUMNS)
    assert packed["vision_decoder_positions"].numel() == 0
    assert packed["pixel_values"].shape[0] == 0


# ------------------------- negative guards -------------------------


def _sidecar(batch):
    lengths = [s["input_ids"].shape[0] for s in batch]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return build_vision_sidecar(
        batch, cu, image_token_id=IMAGE_TOKEN_ID, spatial_merge_size=MERGE
    )


def test_guard_pixel_grid_all_or_nothing():
    sample = dict(MdpThdMockDataset(num_samples=8)[0])
    sample["pixel_values"] = sample["pixel_values"][:0]
    with pytest.raises(ValueError, match="both exist or both be absent"):
        _sidecar([sample])


def test_guard_pixel_rows_match_grids():
    sample = dict(MdpThdMockDataset(num_samples=8)[0])
    sample["pixel_values"] = sample["pixel_values"][:-1]
    with pytest.raises(ValueError, match="sum\\(t\\*h\\*w\\)"):
        _sidecar([sample])


def test_guard_truncation_never_cuts_an_image_block():
    sample = dict(MdpThdMockDataset(num_samples=8)[0])
    # Truncate the sample inside its image block: slot count now mismatches.
    image_positions = (sample["input_ids"] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    cut = int(image_positions[-1])  # drop the final image token
    for key in ("input_ids", "labels", "loss_mask"):
        sample[key] = sample[key][:cut]
    with pytest.raises(ValueError, match="truncation must never cut"):
        _sidecar([sample])


def test_guard_grid_divisible_by_merge():
    sample = dict(MdpThdMockDataset(num_samples=8)[0])
    sample["image_grid_thw"] = torch.tensor([[1, 3, 8]], dtype=torch.long)
    with pytest.raises(ValueError, match="divisible by spatial_merge_size"):
        _sidecar([sample])
