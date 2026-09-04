# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-compute tests for the encoder chunk payload alignment padding.

``Qwen35VLMdpAdapter.encode`` pads the packed patch rows up to the FP8
recipe's alignment with one synthetic grid item and strips that item's output
rows before handing the chunk back, so MDP core's exact-row-count invariants
never see the padding. No CUDA, no distributed world: the encoder is a stub
that records what it was called with.
"""

from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
from megatron.core.fp8_utils import get_fp8_align_size

_MERGE = 2


class _StubEncoder(torch.nn.Module):
    """Stands in for Qwen35VLVisionEncoder: one output row per merge window."""

    def __init__(self, fp8, fp8_recipe):
        super().__init__()
        self.config = SimpleNamespace(fp8=fp8, fp8_recipe=fp8_recipe)
        self.calls = []

    def forward(self, payload, grid_thw):
        self.calls.append((payload.clone(), grid_thw.clone()))
        rows = int(sum(t * (h // _MERGE) * (w // _MERGE) for t, h, w in grid_thw.tolist()))
        # Row index in the output, so the strip can be checked positionally.
        return torch.arange(rows, dtype=torch.float32).unsqueeze(1)


def _layout(*grids):
    return SimpleNamespace(segments=[SimpleNamespace(grid_thw=grid) for grid in grids])


def _encode(fp8, fp8_recipe, grids):
    adapter = Qwen35VLMdpAdapter(out_hidden_size=16)
    encoder = _StubEncoder(fp8, fp8_recipe)
    rows = sum(t * h * w for t, h, w in grids)
    payload = torch.ones(rows, 4)
    output = adapter.encode(encoder, payload, _layout(*grids))
    return encoder, output, rows


@pytest.mark.parametrize("fp8_recipe", ["tensorwise", "blockwise", "mxfp8"])
def test_fp8_payload_is_padded_to_the_recipe_alignment_and_output_is_stripped(fp8_recipe):
    align = get_fp8_align_size(fp8_recipe)
    # 1*6*2 + 1*2*2 = 16 rows: aligned for 16-row recipes, 16 short for MXFP8's
    # 32. Add a (1, 2, 2) item so every recipe has something to pad.
    grids = [(1, 6, 2), (1, 2, 2), (1, 2, 2)]
    encoder, output, real_rows = _encode("hybrid", fp8_recipe, grids)
    assert real_rows % align != 0, "the scenario must actually need padding"

    (payload_seen, grid_seen), = encoder.calls
    pad_rows = (align - real_rows % align) % align
    # The encoder saw an aligned payload...
    assert payload_seen.shape[0] == real_rows + pad_rows
    assert payload_seen.shape[0] % align == 0
    # ...whose padding rows are zero and sit after every real row...
    assert torch.equal(payload_seen[:real_rows], torch.ones(real_rows, 4))
    assert torch.equal(payload_seen[real_rows:], torch.zeros(pad_rows, 4))
    # ...described by exactly one extra (k, merge, merge) grid item.
    assert grid_seen.shape[0] == len(grids) + 1
    assert grid_seen[-1].tolist() == [pad_rows // (_MERGE * _MERGE), _MERGE, _MERGE]

    # The chunk handed back has exactly the real items' merged rows, in order:
    # the synthetic item's output rows -- the last ones -- are gone.
    expected_rows = sum(t * (h // _MERGE) * (w // _MERGE) for t, h, w in grids)
    assert output.shape[0] == expected_rows
    assert torch.equal(output.squeeze(1), torch.arange(expected_rows, dtype=torch.float32))


def test_aligned_fp8_payload_and_bf16_payload_are_passed_through_untouched():
    grids = [(1, 8, 4)]  # 32 rows: aligned for every recipe
    encoder, output, real_rows = _encode("hybrid", "mxfp8", grids)
    (payload_seen, grid_seen), = encoder.calls
    assert payload_seen.shape[0] == real_rows
    assert grid_seen.shape[0] == 1
    assert output.shape[0] == real_rows // (_MERGE * _MERGE)

    # bf16 encoder: an unaligned payload is not padded, since nothing is quantized.
    encoder, output, real_rows = _encode(None, "delayed", [(1, 2, 2)])
    (payload_seen, grid_seen), = encoder.calls
    assert payload_seen.shape[0] == real_rows == 4
    assert grid_seen.shape[0] == 1
    assert output.shape[0] == 1
