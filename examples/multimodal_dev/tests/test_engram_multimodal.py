# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram (n-gram memory / Qwen PLE) on the multimodal entry.

* The decoder-only pipeline token prefetch is not applied: every multimodal pipeline stage
  collates its own batch and the MDP schedule captures and replays the iterator.
* Packed (THD) batches with context parallelism are rejected explicitly.
* The hashed n-gram ids of an image-bearing sequence equal HF ``qwen4_exp``: image
  placeholders are hashed as-is, collate padding as EOS, packed documents reset at their
  padded boundaries. The HF comparison needs a ``transformers`` build with ``qwen4_exp``
  on ``PYTHONPATH`` and is skipped otherwise.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.arguments import add_multimodal_args, validate_engram_args
from examples.multimodal_dev.models.base import engram_padding_token_id, mask_padding_for_decoder
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.engram import Engram, _row_cu_seqlens
from megatron.core.models.engram.hashing import build_ngram_hashes
from megatron.core.packed_seq_params import PackedSeqParams

# Official Qwen3.8-Flash-Next ids (config.json).
VOCAB = 248320
EOS = 248044
VISION_START, VISION_END, IMAGE = 248053, 248054, 248056
COLLATE_PAD = 0  # multimodal_dev pads input_ids with 0
NGRAM_BASE = 1009  # small tables; the hashing does not depend on the table scale
HEADS = 2
SEED = 1234


def _parse(monkeypatch, *extra):
    from megatron.training.arguments import parse_args

    monkeypatch.setattr(sys, "argv", ["pretrain_multimodal.py", "--model-arch", "qwen35_vl", *extra])
    return parse_args(extra_args_provider=add_multimodal_args)


_ENGRAM_CLI = ("--engram-vocab-sizes", "1009", "1009", "--engram-layer-ids", "1")


# ------------------------------------------------------------- token prefetch
def test_multimodal_entry_disables_engram_token_prefetch(monkeypatch):
    from megatron.training.arguments import parse_args

    monkeypatch.setattr(sys, "argv", ["pretrain_hybrid.py", *_ENGRAM_CLI])
    assert parse_args().engram_token_prefetch is True  # decoder-only default
    args = _parse(monkeypatch, *_ENGRAM_CLI)
    assert args.engram_enabled and args.engram_token_prefetch is False


def test_multimodal_iterator_is_not_wrapped(monkeypatch):
    from megatron.training import training

    calls = []
    monkeypatch.setattr(
        training, "prepare_tokens_for_pipeline", lambda *a, **k: calls.append(a) or "wrapped"
    )
    monkeypatch.setattr(training.mpu, "get_pipeline_model_parallel_group", lambda: None)
    monkeypatch.setattr(training.mpu, "get_tensor_model_parallel_group", lambda: None)
    data_iterator = iter([[{"input_ids": torch.zeros(4, dtype=torch.long)}]])

    args = _parse(monkeypatch, *_ENGRAM_CLI)
    assert training._maybe_prefetch_engram_tokens(args, data_iterator, 4, 2) is data_iterator
    assert calls == []
    # Guard: the same helper does wrap for a decoder-only configuration.
    args.engram_token_prefetch = True
    assert training._maybe_prefetch_engram_tokens(args, data_iterator, 4, 2) == "wrapped"
    assert len(calls) == 1


# -------------------------------------------------------------- validation
def _validation_args(**overrides):
    base = dict(
        engram_enabled=True,
        engram_token_prefetch=False,
        use_packed_sequence=True,
        context_parallel_size=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_engram_thd_with_context_parallel_is_rejected():
    validate_engram_args(_validation_args())  # THD, CP=1: accepted
    validate_engram_args(_validation_args(use_packed_sequence=False, context_parallel_size=2))
    validate_engram_args(_validation_args(engram_enabled=False, context_parallel_size=2))
    with pytest.raises(RuntimeError, match="context parallelism with packed"):
        validate_engram_args(_validation_args(context_parallel_size=2))
    with pytest.raises(RuntimeError, match="token prefetch must stay disabled"):
        validate_engram_args(_validation_args(engram_token_prefetch=True))


# -------------------------------------------------------- padding token id
def _engram_config():
    return EngramConfig(
        global_vocab_sizes=(NGRAM_BASE, NGRAM_BASE),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=HEADS,
        memory_dim=8,
        kernel_size=4,
        hash_seed=SEED,
        boundary_token_id=EOS,
        variant="qwen",
        unigram_vocab_size=VOCAB,
    )


def test_padding_token_id_is_the_engram_boundary_token():
    engram = Engram.__new__(Engram)
    torch.nn.Module.__init__(engram)
    engram.engram_config = _engram_config()
    assert engram_padding_token_id(torch.nn.Sequential(torch.nn.Linear(2, 2), engram)) == EOS
    assert engram_padding_token_id(torch.nn.Linear(2, 2)) is None  # e.g. Qwen3.5-VL
    ids = torch.tensor([[5, 6, 0, 0]])
    pad = torch.tensor([[False, False, True, True]])
    assert mask_padding_for_decoder(ids, pad, None) is ids
    assert mask_padding_for_decoder(ids, None, EOS) is ids
    assert mask_padding_for_decoder(ids, pad, EOS).tolist() == [[5, 6, EOS, EOS]]


# ------------------------------------------------------------ HF hash parity
def _sample(seed, text_a, n_image, text_b, eos_at=None):
    g = torch.Generator().manual_seed(seed)
    ids = torch.cat(
        [
            torch.randint(10, 150000, (text_a,), generator=g),
            torch.tensor([VISION_START]),
            torch.full((n_image,), IMAGE),
            torch.tensor([VISION_END]),
            torch.randint(10, 150000, (text_b,), generator=g),
        ]
    )
    if eos_at is not None:
        ids[eos_at] = EOS
    return ids


def _megatron_hashes(input_ids, padding_mask, packed_seq_params=None):
    config = _engram_config()
    ids = mask_padding_for_decoder(input_ids, padding_mask, EOS)
    return build_ngram_hashes(
        input_ids=ids,
        tokenizer_remap=None,
        multipliers=torch.tensor(config.multipliers(1), dtype=torch.int64),
        table_sizes=torch.tensor(config.table_sizes(1), dtype=torch.int64),
        max_ngram_order=3,
        num_hash_heads=HEADS,
        boundary_token_id=config.hash_boundary_token_id,
        reset_at_boundary=config.variant_spec.resets_windows_at_boundary_token,
        cu_seqlens=_row_cu_seqlens(packed_seq_params),
    )


def _hf_hashes(input_ids, attention_mask):
    """Per-head n-gram ids from HF ``Qwen4ExpTextNGramEmbedding`` (offsets removed)."""
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    configuration = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfg = configuration.Qwen4ExpTextConfig(
        vocab_size=VOCAB,
        eos_token_id=EOS,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=4,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ngram_size=3,
        heads_per_ngram=HEADS,
        ngram_vocab_size_base=NGRAM_BASE,
        seed=SEED,
        num_experts=4,
        num_experts_per_tok=2,
    )
    ngram = modeling.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx=0)
    captured = []
    ngram.ngram_embedding.register_forward_hook(lambda m, inp, out: captured.append(inp[0]))
    # Qwen4ExpTextModel: ple_input_ids = where(conv_mask, input_ids, eos) (placeholders kept).
    ngram(torch.where(attention_mask, input_ids, EOS), None)
    return captured[0] - ngram.ngram_heads_offsets.view(1, 1, -1)


def test_hashes_match_hf_bshd_image_and_padding():
    row0 = _sample(1, 23, 40, 17, eos_at=70)  # EOS after the image, inside the text
    row1 = _sample(2, 9, 64, 31)
    length = max(row0.numel(), row1.numel())  # 106
    input_ids = torch.full((2, length), COLLATE_PAD, dtype=torch.long)
    input_ids[0, : row0.numel()] = row0
    input_ids[1, : row1.numel()] = row1
    padding_mask = torch.arange(length).unsqueeze(0) >= torch.tensor(
        [[row0.numel()], [row1.numel()]]
    )
    assert padding_mask.any() and (input_ids == IMAGE).sum() == 104

    mega = _megatron_hashes(input_ids, padding_mask)
    hf = _hf_hashes(input_ids, ~padding_mask)
    assert torch.equal(mega, hf)
    # Guard: without the padding rewrite the padded positions hash differently.
    raw = _megatron_hashes(input_ids, None)
    assert not torch.equal(raw[padding_mask], hf[padding_mask])


def test_hashes_match_hf_thd_packed_per_document():
    docs = [_sample(3, 11, 24, 13), _sample(4, 7, 36, 21, eos_at=60)]  # 50, 66 tokens
    real = [d.numel() for d in docs]
    padded = [(n + 7) // 8 * 8 for n in real]  # multimodal_dev per-sample alignment
    packed = torch.full((1, sum(padded)), COLLATE_PAD, dtype=torch.long)
    padding_mask = torch.ones(1, sum(padded), dtype=torch.bool)
    start = 0
    for doc, n_pad in zip(docs, padded):
        packed[0, start : start + doc.numel()] = doc
        padding_mask[0, start : start + doc.numel()] = False
        start += n_pad
    to_cu = lambda lens: torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=to_cu(real),  # multimodal_dev: real lengths
        cu_seqlens_kv=to_cu(real),
        cu_seqlens_q_padded=to_cu(padded),  # ... and the physical layout
        cu_seqlens_kv_padded=to_cu(padded),
    )
    mega = _megatron_hashes(packed, padding_mask, params)
    start = 0
    for doc, n_pad in zip(docs, padded):
        row = torch.full((1, n_pad), COLLATE_PAD, dtype=torch.long)
        row[0, : doc.numel()] = doc
        mask = torch.zeros(1, n_pad, dtype=torch.bool)
        mask[0, : doc.numel()] = True
        assert torch.equal(mega[:, start : start + n_pad], _hf_hashes(row, mask))
        start += n_pad
