# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Vision scatter under decoder CP: split-then-local-scatter == scatter-then-split.

The native multimodal path used to embed the full sequence, scatter every
vision row, and CP-split afterwards. ``MultimodalModel`` now splits first and
each CP rank scatters only the rows whose image-token slots it owns -- taken from
the full encoder output on the native path (``_cp_local_vision_rows``) or
delivered as this rank's shard by MDP (``megatron.core.mdp.cp_partition``,
cut from host decoder positions alone). Both must reproduce the old order
bit for bit, in both decoder CP input layouts.

Rows carry distinct values, so a misrouted row cannot alias into a pass. The
packs cover an image block inside one chunk, one straddling a chunk boundary,
one spanning both zigzag halves of a sample, interleaved text, a sample with no
image, and (contiguous) a block crossing the CP boundary between two samples.

Run with::

    torchrun --nproc_per_node=4 -m pytest -q examples/multimodal_dev/tests/test_cp_vision_scatter.py
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.base import MultimodalModel, cp_partition_index
from megatron.core import parallel_state
from megatron.core.mdp.cp_partition import cp_item_runs
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.test_utilities import Utils

_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(_WORLD < 2, reason="needs a torchrun world")

IMAGE = 7
HIDDEN = 8


class _Splitter(MultimodalModel):
    """Just the split/selection methods: no decoder, no encoder."""

    def __init__(self, mode):  # pylint: disable=super-init-not-called
        torch.nn.Module.__init__(self)
        self.image_token_id = IMAGE
        self.config = type("Cfg", (), {"cp_partition_mode": mode})()


def _mdp_shards(positions, cu, cp, mode):
    """Per-CP-rank vision rows as MDP cuts them from host positions alone.

    Every image block (a maximal run of consecutive image slots inside one
    sample) is one item; ``cp_item_runs`` cuts it into the runs each rank owns.
    """
    shards = [[] for _ in range(cp)]
    row = 0
    while row < len(positions):
        start = positions[row]
        seq = max(i for i in range(len(cu) - 1) if cu[i] <= start)
        rows = 1
        while (
            row + rows < len(positions)
            and positions[row + rows] == start + rows
            and start + rows < cu[seq + 1]
        ):
            rows += 1
        for run in cp_item_runs(
            start,
            rows,
            sample_padded_start=cu[seq],
            sample_padded_len=cu[seq + 1] - cu[seq],
            total_rows=cu[-1],
            cp_size=cp,
            mode=mode,
        ):
            first = row + run.item_row_start
            shards[run.cp_rank].extend(range(first, first + run.item_rows))
        row += rows
    return shards


def _pack(cp):
    """Three samples of padded lengths multiple of 2*cp, image blocks laid out
    to straddle chunk and CP boundaries."""
    unit = 2 * cp
    lengths = [8 * unit, 4 * unit, 6 * unit]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    total = cu[-1]
    ids = torch.full((1, total), 3, dtype=torch.long)
    chunk0 = lengths[0] // unit  # zigzag chunk size of sample 0
    blocks = [
        (1, 1 + chunk0 // 2),  # inside one chunk
        (chunk0 - 1, chunk0 + 2),  # straddles a chunk boundary
        (lengths[0] - 3, lengths[0] - 1),  # last chunk: rank 0's second zigzag half
        (cu[2] + 1, cu[3] - 1),  # sample 2 end to end (sample 1 has no image)
    ]
    for start, end in blocks:
        ids[0, start:end] = IMAGE
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        cu_seqlens_q_padded=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        cu_seqlens_kv_padded=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        total_tokens=total,
    )
    return ids.cuda(), params, cu


@pytest.fixture(params=[2, 4], ids=["cp2", "cp4"])
def cp_size(request):
    cp = request.param
    if _WORLD % cp:
        pytest.skip(f"world {_WORLD} not divisible by cp={cp}")
    Utils.initialize_model_parallel(context_parallel_size=cp)
    yield cp
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("mode", ["zigzag", "contiguous"])
def test_split_then_local_scatter_is_bitwise_equal(cp_size, mode):
    cp_rank = parallel_state.get_context_parallel_rank()
    ids, params, cu = _pack(cp_size)
    params.cp_partition_mode = mode
    total = ids.shape[1]
    num_rows = int((ids == IMAGE).sum())
    vision = torch.arange(num_rows * HIDDEN, dtype=torch.float32, device="cuda").view(
        num_rows, HIDDEN
    )
    text = -torch.arange(total * HIDDEN, dtype=torch.float32, device="cuda").view(total, HIDDEN)

    # Oracle: the old native order -- scatter into the full sequence, then split.
    full = text.clone()
    full[(ids[0] == IMAGE)] = vision
    idx = cp_partition_index(params, total, cp_size, cp_rank)
    expected = full.index_select(0, idx)

    splitter = _Splitter(mode)
    cp_index = splitter._cp_index(params, total, ids.device)
    assert torch.equal(cp_index, idx)
    local_ids = splitter._cp_split_sequence(ids, 1, cp_index)
    local_text = text.index_select(0, idx)

    def scatter(rows):
        out = local_text.clone()
        out[local_ids[0] == IMAGE] = rows
        return out

    # Native path: select this rank's rows from the full encoder output.
    native_rows = splitter._cp_local_vision_rows(ids, vision, cp_index)
    assert torch.equal(scatter(native_rows), expected)

    # MDP path: the shard cut from host positions alone.
    positions = (ids[0] == IMAGE).nonzero().view(-1).tolist()
    shards = _mdp_shards(positions, cu, cp_size, mode)
    mdp_rows = vision.index_select(
        0, torch.tensor(shards[cp_rank], dtype=torch.long, device="cuda")
    )
    assert torch.equal(mdp_rows, native_rows)
    assert torch.equal(scatter(mdp_rows), expected)

    # The shards partition the rows, and the layout really splits them.
    sizes = [len(shard) for shard in shards]
    assert sum(sizes) == num_rows and sum(1 for n in sizes if n) > 1


def test_rank_without_image_tokens_gets_an_empty_shard():
    if _WORLD % 2:
        pytest.skip("needs an even world")
    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        cp_rank = parallel_state.get_context_parallel_rank()
        ids, params, cu = _pack(2)
        params.cp_partition_mode = "contiguous"
        ids[0, ids.shape[1] // 2 :] = 3  # no image in rank 1's contiguous half
        positions = (ids[0] == IMAGE).nonzero().view(-1).tolist()
        shards = _mdp_shards(positions, cu, 2, "contiguous")
        assert shards[1] == [] and len(shards[0]) == len(positions)
        splitter = _Splitter("contiguous")
        rows = splitter._cp_local_vision_rows(
            ids,
            torch.zeros(len(positions), HIDDEN, device="cuda"),
            splitter._cp_index(params, ids.shape[1], ids.device),
        )
        assert rows.shape[0] == len(shards[cp_rank])
    finally:
        Utils.destroy_model_parallel()


def test_layout_mismatch_between_collator_and_decoder_fails_loudly():
    if _WORLD % 2:
        pytest.skip("needs an even world")
    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        ids, params, _ = _pack(2)
        params.cp_partition_mode = "zigzag"
        with pytest.raises(RuntimeError, match="cp_partition_mode"):
            _Splitter("contiguous")._cp_index(params, ids.shape[1], ids.device)
    finally:
        Utils.destroy_model_parallel()
