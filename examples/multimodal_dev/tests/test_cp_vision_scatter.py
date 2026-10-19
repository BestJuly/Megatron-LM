# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Vision-embedding scatter under decoder context parallelism.

No other test in this repository exercises the vision scatter at CP>1:
``test_cp_correctness.py`` builds a plain ``GPTModel``, and
``test_cp_thd_correctness.py`` installs a vision encoder that raises if called
and strips every image token out of ``input_ids`` before packing. That left the
one property MDP's decoder-CP routing rests on completely uncovered.

The property: MDP delivers each endpoint only its own rows, so it must scatter
*after* the CP split, while the native path scatters the full vision output into
the full sequence and splits afterwards. Those two orderings must produce the
identical ``decoder_input`` — ``masked_scatter`` is pure data movement and
``index_select`` is a gather, so composing them either way copies the same values
to the same places, provided the delivered rows are ordered by rank-local
position. This test is what makes that "provided" checked rather than assumed.

Run with::

    PYTHONPATH=. torchrun --nproc-per-node 8 -m pytest -q \\
        examples/multimodal_dev/tests/test_cp_vision_scatter.py
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.forward_step import build_vision_sidecar, pack_or_pad_batch
from examples.multimodal_dev.models.base import MultimodalModel, _thd_cp_partition_index
from megatron.core.mdp.cp_partition import split_item
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.parallel_state import get_context_parallel_rank
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(_WORLD < 2, reason="needs a torchrun world of at least 2")

IMAGE_TOKEN_ID = 7
MERGE = 2
HIDDEN = 128
VOCAB = 512


class _UnusedVisionEncoder(MegatronModule):
    """Satisfies the constructor; this test supplies embeddings directly."""

    def forward(self, pixel_values, image_grid_thw):
        """Never called: every call here passes vision_embeddings explicitly."""
        raise RuntimeError("this test drives the scatter, not the encoder")


def _config(cp_size):
    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        ffn_hidden_size=2 * HIDDEN,
        num_attention_heads=4,
        num_query_groups=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        context_parallel_size=cp_size,
        sequence_parallel=False,
    )


def _model(cp_size, max_seq_len):
    config = _config(cp_size)
    model = MultimodalModel(
        language_config=config,
        language_spec=get_gpt_layer_with_transformer_engine_spec(),
        vision_encoder=_UnusedVisionEncoder(config),
        vocab_size=VOCAB,
        max_sequence_length=max_seq_len,
        image_token_id=IMAGE_TOKEN_ID,
        position_embedding_type="rope",
        parallel_output=False,
    )
    return model.cuda()


def _samples(grids_per_sample, sample_len, seed=17):
    """Per-sample dicts whose image-token blocks match their grids exactly."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = []
    for sample_index, grids in enumerate(grids_per_sample):
        ids = torch.randint(
            IMAGE_TOKEN_ID + 1, VOCAB, (sample_len,), generator=generator
        )
        cursor = 1  # leave a text token in front so blocks are not at row 0
        grid_rows = []
        for (t, h, w) in grids:
            slots = t * (h // MERGE) * (w // MERGE)
            ids[cursor : cursor + slots] = IMAGE_TOKEN_ID
            cursor += slots + 1  # interleave one text token between blocks
            grid_rows.append([t, h, w])
        pixels = torch.zeros(sum(t * h * w for t, h, w in grids), 1)
        samples.append(
            {
                "input_ids": ids.cuda(),
                "labels": ids.clone().cuda(),
                "loss_mask": torch.ones(sample_len).cuda(),
                "pixel_values": pixels.cuda(),
                "image_grid_thw": torch.tensor(
                    grid_rows, dtype=torch.long
                ).cuda()
                if grid_rows
                else torch.empty(0, 3, dtype=torch.long).cuda(),
            }
        )
    return samples


def _pack(samples):
    packed = pack_or_pad_batch(
        [dict(s) for s in samples], use_packed_sequence=True, device="cuda"
    )
    cu_padded = packed["packed_seq_params"].cu_seqlens_q_padded.tolist()
    sidecar = build_vision_sidecar(
        samples, cu_padded, image_token_id=IMAGE_TOKEN_ID, spatial_merge_size=MERGE
    )
    return packed, sidecar, cu_padded


def _full_vision_rows(sidecar):
    """Deterministic, per-row-distinct embeddings so a misroute cannot alias."""
    total = int(sidecar["vision_decoder_positions"].numel())
    base = torch.arange(total, dtype=torch.float32, device="cuda").unsqueeze(1)
    return (base * 1000 + torch.arange(HIDDEN, device="cuda").unsqueeze(0)).bfloat16()


def _shard_for_rank(sidecar, full_rows, cp_size, cp_rank):
    """The rows MDP's planner would deliver to one endpoint, in leaf order.

    Mirrors MdpPlanner: split each item, keep the runs owned by this rank, and
    order them by rank-local row. Built independently here so the test is a
    check on the planner's ordering rule, not a restatement of it.
    """
    meta = sidecar["vision_item_meta"].tolist()
    positions = sidecar["vision_decoder_positions"].tolist()
    entries = []
    row_cursor = 0
    for row in meta:
        _, _, t, h, w, _, sample_start, sample_len = (int(v) for v in row)
        output_rows = t * (h // MERGE) * (w // MERGE)
        offset = positions[row_cursor] - sample_start
        for interval in split_item(
            offset_in_sample=offset,
            output_rows=output_rows,
            sample_padded_start=sample_start,
            sample_padded_len=sample_len,
            cp_size=cp_size,
        ):
            if interval.cp_rank != cp_rank:
                continue
            start = row_cursor + interval.item_row_start
            entries.append(
                (interval.local_row_start, full_rows[start : start + interval.rows])
            )
        row_cursor += output_rows
    entries.sort(key=lambda e: e[0])
    if not entries:
        return torch.empty(0, HIDDEN, dtype=full_rows.dtype, device=full_rows.device)
    return torch.cat([rows for _, rows in entries])


def _prepare(model, packed, vision_embeddings):
    return model._prepare_decoder_inputs(
        input_ids=packed["input_ids"],
        position_ids=None,
        attention_mask=None,
        labels=packed["labels"],
        loss_mask=packed["loss_mask"],
        padding_mask=packed.get("padding_mask"),
        pixel_values=None,
        image_grid_thw=None,
        decoder_input=None,
        packed_seq_params=packed["packed_seq_params"],
        vision_embeddings=vision_embeddings,
    )


# Item layouts chosen to hit the cases that matter: a block inside one chunk, a
# block straddling a chunk boundary, a block spanning both halves of a sample,
# and a sample with no vision at all.
_LAYOUTS = [
    [[(1, 4, 4)], []],
    [[(1, 8, 8)], [(1, 4, 4), (1, 4, 4)]],
    [[(2, 8, 8)], [(1, 4, 4)]],
]


@pytest.mark.parametrize("grids_per_sample", _LAYOUTS)
def test_split_then_scatter_matches_scatter_then_split(grids_per_sample):
    """The linchpin: the two orderings must agree bit for bit."""
    cp_size = _WORLD
    sample_len = 128
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=cp_size
    )
    model_parallel_cuda_manual_seed(1234)
    try:
        model = _model(cp_size, sample_len * len(grids_per_sample))
        samples = _samples(grids_per_sample, sample_len)
        packed, sidecar, _ = _pack(samples)
        full_rows = _full_vision_rows(sidecar)
        cp_rank = get_context_parallel_rank()

        # Reference: scatter the FULL vision output into the FULL sequence, then
        # take this rank's shard -- exactly what the native path does.
        with torch.no_grad():
            text = model.language_model.embedding(
                input_ids=packed["input_ids"], position_ids=None
            )
            scattered = model._scatter_vision_embeddings(
                packed["input_ids"], text, full_rows
            )
            index = _thd_cp_partition_index(
                packed["packed_seq_params"].cu_seqlens_q_padded,
                scattered.shape[0],
                cp_size,
                cp_rank,
            )
            reference = scattered.index_select(0, index)

            # Under test: hand the model only this rank's rows and let it split
            # first and scatter into the rank-local stream.
            shard = _shard_for_rank(sidecar, full_rows, cp_size, cp_rank)
            produced = _prepare(model, packed, shard)["decoder_input"]

        assert produced.shape == reference.shape
        assert torch.equal(produced, reference), (
            f"cp_rank={cp_rank}: split-then-scatter diverged from "
            f"scatter-then-split at {int((produced != reference).sum())} elements"
        )
    finally:
        Utils.destroy_model_parallel()


def test_an_endpoint_with_no_vision_rows_is_not_an_error():
    """An empty shard is a designed-for state, not a missing-source failure."""
    cp_size = _WORLD
    sample_len = 128
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=cp_size
    )
    model_parallel_cuda_manual_seed(1234)
    try:
        model = _model(cp_size, sample_len)
        # A single small block near the start of the sample lands inside one
        # chunk, so at least one rank owns none of it.
        samples = _samples([[(1, 4, 4)]], sample_len)
        packed, sidecar, _ = _pack(samples)
        full_rows = _full_vision_rows(sidecar)
        cp_rank = get_context_parallel_rank()
        shard = _shard_for_rank(sidecar, full_rows, cp_size, cp_rank)

        with torch.no_grad():
            prepared = _prepare(model, packed, shard if shard.numel() else None)
        assert prepared["decoder_input"] is not None

        owners = torch.zeros(cp_size, dtype=torch.int64, device="cuda")
        owners[cp_rank] = 1 if shard.numel() else 0
        torch.distributed.all_reduce(owners)
        assert int(owners.sum()) >= 1, "some rank must own the item's rows"
        assert int(owners.sum()) < cp_size, (
            "this layout is meant to leave at least one rank empty; if every "
            "rank owns rows the empty-shard path is not being exercised"
        )
    finally:
        Utils.destroy_model_parallel()
