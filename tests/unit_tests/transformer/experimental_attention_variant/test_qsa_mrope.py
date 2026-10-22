# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QSA under mRoPE (multimodal decoders): per-token rotary frequencies.

Under mRoPE every token carries its own T/H/W position, so the rotary frequencies
are one row per token rather than a 1-D table indexed by in-document offset. HF
``Qwen4ExpTextQSAIndexer`` RoPEs each indexer query with its own token's cos/sin and
each pooled block key with the cos/sin of the block's first token.

* HF parity (needs a ``transformers`` build that provides ``qwen4_exp``; run fp32 with
  ``NVIDIA_TF32_OVERRIDE=0``): selected-token sets equal HF for image-bearing 3-D
  position ids, BSHD (batch 2) and packed THD (per document).
* Regression guard: per-token frequencies that carry 1-D positions reproduce the 1-D
  table path bitwise (indexer selection, block keys and attention output).
* Main attention under THD uses each packed token's own row; raw fused-mRoPE
  frequencies are accepted; context parallelism is rejected explicitly.
"""

import types

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    BUDGET,
    HEAD_DIM,
    HIDDEN,
    N_HEADS,
    N_KV_HEADS,
    RATIO,
    ROPE_THETA,
    ROTARY_PERCENT,
    make_config,
)

MROPE_SECTION = [2, 1, 1]  # sums to rotary_dim / 2 = HEAD_DIM * ROTARY_PERCENT / 2


def mrope_position_ids(layout):
    """Qwen-VL style 3-D positions for a ``[("text", n) | ("image", t, h, w)]`` layout.

    Text advances all three axes together; an image grid places T/H/W at
    ``start + (t, h, w)`` and the next segment resumes after its largest position.
    """
    axes = [[], [], []]
    cur = 0
    for seg in layout:
        if seg[0] == "text":
            n = seg[1]
            for a in axes:
                a.extend(range(cur, cur + n))
            cur += n
        else:
            _, t, h, w = seg
            tt, hh, ww = torch.meshgrid(
                torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"
            )
            for a, grid in zip(axes, (tt, hh, ww)):
                a.extend((cur + grid.flatten()).tolist())
            cur += max(t, h, w)
    return torch.tensor(axes, dtype=torch.long)  # [3, s]


LAYOUT_A = [("text", 21), ("image", 1, 8, 10), ("text", 30), ("image", 2, 4, 6), ("text", 33)]
LAYOUT_B = [("text", 7), ("image", 1, 12, 12), ("text", 61)]
LAYOUT_DOC0 = [("text", 9), ("image", 1, 6, 8), ("text", 35)]  # 92 tokens
LAYOUT_DOC1 = [("text", 40), ("image", 1, 10, 6), ("text", 61)]  # 161 tokens


class _Cp2:
    def size(self):
        return 2

    def rank(self):
        return 0


class TestQSAMRoPE:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = make_config()
        self.config.mrope_section = list(MROPE_SECTION)
        self.config.mrope_interleaved = True
        spec = get_qsa_module_spec_for_backend(self.config)
        self.attn = build_module(spec, config=self.config, layer_number=1).cuda()
        with torch.no_grad():
            for norm in (
                self.attn.q_layernorm,
                self.attn.k_layernorm,
                self.attn.indexer.q_layernorm,
                self.attn.indexer.k_layernorm,
            ):
                norm.weight.uniform_(-0.1, 0.1)
        assert self.attn.indexer.per_token_rope
        self.mrope = MultimodalRotaryEmbedding(
            kv_channels=HEAD_DIM,
            rotary_percent=ROTARY_PERCENT,
            rotary_base=ROPE_THETA,
            interleaved_mrope=True,
        )
        self.rope = RotaryEmbedding(
            kv_channels=HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # ----------------------------------------------------------------- helpers
    def _freqs(self, position_ids, raw=False):
        """Per-token mRoPE frequencies ``[s, b, 1, d]`` for ``position_ids [3, b, s]``."""
        return self.mrope(position_ids.cuda(), MROPE_SECTION, return_raw_freqs=raw)

    @staticmethod
    def _hidden(s, b, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(s, b, HIDDEN, generator=g).cuda()

    @staticmethod
    def _psp(doc_lens):
        cu = torch.tensor([0] + list(torch.tensor(doc_lens).cumsum(0)), dtype=torch.int32)
        cu = cu.cuda()
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max(doc_lens),
            max_seqlen_kv=max(doc_lens),
        )

    def _thd_visible(self, selection, doc_lens):
        """Per-document ``[s_d, s_d]`` visible sets from a packed THD selection."""
        r = self.attn.indexer.compress_ratio
        out, s0 = [], 0
        for sd in doc_lens:
            order = selection.order[0, s0 : s0 + sd]  # doc-local block ids
            picked = selection.picked[0, s0 : s0 + sd]
            m_t = selection.own_block[s0 : s0 + sd]
            k = order.shape[-1]
            tok = (order.unsqueeze(-1) * r + torch.arange(r, device=order.device)).reshape(
                sd, k * r
            )
            keep = picked.unsqueeze(-1).expand(-1, -1, r).reshape(sd, k * r)
            buf = torch.zeros(sd, sd + 1, dtype=torch.bool, device=order.device)
            buf.scatter_(1, torch.where(keep, tok, sd), True)
            j = torch.arange(sd, device=order.device)
            tail = (j.unsqueeze(0) >= (m_t * r).unsqueeze(1)) & (j.unsqueeze(0) <= j.unsqueeze(1))
            out.append(buf[:, :sd] | tail)
            s0 += sd
        return out

    def _hf_indexer(self):
        qwen4 = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
        cfg_mod = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
        idx = self.attn.indexer
        cfg = cfg_mod.Qwen4ExpTextConfig(
            hidden_size=HIDDEN,
            num_attention_heads=N_HEADS,
            num_key_value_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            rms_norm_eps=self.config.layernorm_epsilon,
            indexer_n_heads=idx.n_heads,
            indexer_kv_heads=1,
            indexer_head_dim=idx.head_dim,
            indexer_budget=BUDGET,
            indexer_compress_ratio=RATIO,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": ROPE_THETA,
                "partial_rotary_factor": ROTARY_PERCENT,
                "mrope_section": list(MROPE_SECTION),
                "mrope_interleaved": True,
            },
        )
        hf = qwen4.Qwen4ExpTextQSAIndexer(cfg, layer_idx=0).cuda().float()
        with torch.no_grad():
            hf.index_qk_proj.weight.copy_(idx.index_qk_proj.weight)
            hf.q_layernorm.weight.copy_(idx.q_layernorm.weight)
            hf.k_layernorm.weight.copy_(idx.k_layernorm.weight)
        rotary = qwen4.Qwen4ExpTextRotaryEmbedding(cfg).cuda()
        return hf, rotary

    @staticmethod
    def _hf_visible(hf, rotary, hidden_bsh, position_ids):
        """HF selected-token set ``[b, s, s]`` (True = attended)."""
        b, s, _ = hidden_bsh.shape
        cos, sin = rotary(hidden_bsh, position_ids.cuda())
        causal = torch.ones(s, s, dtype=torch.bool, device=hidden_bsh.device).tril()
        mask = causal.view(1, 1, s, s).expand(b, 1, s, s)
        with torch.no_grad():
            return hf(hidden_bsh, (cos, sin), mask, None).squeeze(1)

    # ------------------------------------------------------------- HF parity
    def test_hf_parity_bshd_image_positions(self):
        hf, rotary = self._hf_indexer()
        pos = torch.stack([mrope_position_ids(LAYOUT_A), mrope_position_ids(LAYOUT_B)], dim=1)
        s = pos.shape[-1]
        assert mrope_position_ids(LAYOUT_B).shape[-1] == s
        hidden = self._hidden(s, 2, seed=11)
        with torch.no_grad():
            mega_masked = self.attn.indexer(hidden, self._freqs(pos))  # [b, 1, s, s]
        hf_visible = self._hf_visible(hf, rotary, hidden.permute(1, 0, 2).contiguous(), pos)
        assert torch.equal(~mega_masked.squeeze(1), hf_visible), "BSHD mRoPE selection != HF"
        # The sparse regime is real and the 3-D positions matter for the selection.
        assert int(hf_visible[0, -1].sum()) < s - 1
        with torch.no_grad():
            pos_1d = torch.arange(s).view(1, 1, s).expand(3, 2, s)
            masked_1d = self.attn.indexer(hidden, self._freqs(pos_1d))
        assert not torch.equal(masked_1d, mega_masked), "image positions did not change selection"

    def test_hf_parity_thd_image_positions_per_document(self):
        hf, rotary = self._hf_indexer()
        pos_docs = [mrope_position_ids(LAYOUT_DOC0), mrope_position_ids(LAYOUT_DOC1)]
        doc_lens = [p.shape[-1] for p in pos_docs]
        hidden = self._hidden(sum(doc_lens), 1, seed=13)
        packed_pos = torch.cat(pos_docs, dim=-1).unsqueeze(1)  # [3, 1, t]
        psp = self._psp(doc_lens)
        with torch.no_grad():
            sel = self.attn.indexer.select(
                hidden, self._freqs(packed_pos), cu_seqlens=psp.cu_seqlens_q
            )
        mega = self._thd_visible(sel, doc_lens)
        s0 = 0
        for d, (sd, pos_d) in enumerate(zip(doc_lens, pos_docs)):
            h_d = hidden[s0 : s0 + sd].permute(1, 0, 2).contiguous()
            ref = self._hf_visible(hf, rotary, h_d, pos_d.unsqueeze(1))[0]
            assert torch.equal(mega[d], ref), f"THD mRoPE selection != HF for document {d}"
            s0 += sd

    # ------------------------------------------------------ regression guard
    def test_1d_positions_bitwise_equal_table_path_bshd(self):
        s, b = 300, 2
        hidden = self._hidden(s, b, seed=7)
        table = self.rope(s)  # [s, 1, 1, d]
        per_token = table.expand(s, b, 1, table.shape[-1]).contiguous()
        idx = self.attn.indexer
        with torch.no_grad():
            idx.per_token_rope = False
            ref = idx.select(hidden, table)
            ref_mask = idx(hidden, table)
            idx.per_token_rope = True
            got = idx.select(hidden, per_token)
            got_mask = idx(hidden, per_token)
        for name in ("q", "block_keys", "order", "picked", "own_block"):
            assert torch.equal(getattr(ref, name), getattr(got, name)), name
        assert torch.equal(ref_mask, got_mask)

    def test_1d_positions_bitwise_equal_table_path_thd(self):
        doc_lens = [120, 180]
        hidden = self._hidden(sum(doc_lens), 1, seed=17)
        psp = self._psp(doc_lens)
        table = self.rope(max(doc_lens))
        per_token = torch.cat([table[:sd] for sd in doc_lens], dim=0)
        self.config.qsa_use_sparse_attention = True
        idx = self.attn.indexer
        with torch.no_grad():
            idx.per_token_rope = False
            ref = idx.select(hidden, table, cu_seqlens=psp.cu_seqlens_q)
            ref_out, _ = self.attn(
                hidden, attention_mask=None, rotary_pos_emb=table, packed_seq_params=psp
            )
            idx.per_token_rope = True
            got = idx.select(hidden, per_token, cu_seqlens=psp.cu_seqlens_q)
            got_out, _ = self.attn(
                hidden, attention_mask=None, rotary_pos_emb=per_token, packed_seq_params=psp
            )
        for name in ("q", "block_keys", "order", "picked", "own_block", "block_offsets"):
            assert torch.equal(getattr(ref, name), getattr(got, name)), name
        assert torch.equal(ref_out, got_out)

    # ------------------------------------------------- main attention, layouts
    def test_thd_attention_matches_per_document_bshd(self):
        pos_docs = [mrope_position_ids(LAYOUT_DOC0), mrope_position_ids(LAYOUT_DOC1)]
        doc_lens = [p.shape[-1] for p in pos_docs]
        hidden = self._hidden(sum(doc_lens), 1, seed=19)
        psp = self._psp(doc_lens)
        packed_freqs = self._freqs(torch.cat(pos_docs, dim=-1).unsqueeze(1))
        self.config.qsa_use_sparse_attention = True
        errs, s0 = [], 0
        with torch.no_grad():
            packed, _ = self.attn(
                hidden, attention_mask=None, rotary_pos_emb=packed_freqs, packed_seq_params=psp
            )
            for sd, pos_d in zip(doc_lens, pos_docs):
                solo, _ = self.attn(
                    hidden[s0 : s0 + sd],
                    attention_mask=None,
                    rotary_pos_emb=self._freqs(pos_d.unsqueeze(1)),
                )
                errs.append((packed[s0 : s0 + sd] - solo).abs().max().item())
                s0 += sd
        assert max(errs) < 3e-4, f"THD vs per-document BSHD errs {errs}"

    def test_thd_padded_layout_matches_per_document_bshd(self):
        """Documents padded individually: rows follow cu_seqlens_q_padded, while
        cu_seqlens_q keeps the real lengths (the multimodal_dev packing contract)."""
        pos_docs = [mrope_position_ids(LAYOUT_DOC0), mrope_position_ids(LAYOUT_DOC1)]
        real = [p.shape[-1] for p in pos_docs]  # 92, 161
        padded = [(n + 7) // 8 * 8 for n in real]  # 96, 168
        hidden = self._hidden(sum(padded), 1, seed=37)
        # Padding rows continue their document's text positions.
        packed_pos = torch.cat(
            [
                torch.cat([p, p[:, -1:] + torch.arange(1, n_pad - p.shape[-1] + 1)], dim=-1)
                for p, n_pad in zip(pos_docs, padded)
            ],
            dim=-1,
        ).unsqueeze(1)
        to_cu = lambda lens: torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32
        ).cuda()
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=to_cu(real),
            cu_seqlens_kv=to_cu(real),
            cu_seqlens_q_padded=to_cu(padded),
            cu_seqlens_kv_padded=to_cu(padded),
            max_seqlen_q=max(padded),
            max_seqlen_kv=max(padded),
        )
        self.config.qsa_use_sparse_attention = True
        errs, s0 = [], 0
        with torch.no_grad():
            packed, _ = self.attn(
                hidden,
                attention_mask=None,
                rotary_pos_emb=self._freqs(packed_pos),
                packed_seq_params=psp,
            )
            for n_real, n_pad, pos_d in zip(real, padded, pos_docs):
                solo, _ = self.attn(
                    hidden[s0 : s0 + n_real],
                    attention_mask=None,
                    rotary_pos_emb=self._freqs(pos_d.unsqueeze(1)),
                )
                errs.append((packed[s0 : s0 + n_real] - solo).abs().max().item())
                s0 += n_pad
        assert max(errs) < 3e-4, f"padded THD vs per-document BSHD errs {errs}"

    def test_raw_fused_mrope_freqs_match_materialized(self):
        pos = torch.stack([mrope_position_ids(LAYOUT_A), mrope_position_ids(LAYOUT_B)], dim=1)
        s = pos.shape[-1]
        hidden = self._hidden(s, 2, seed=23)
        raw, emb = self._freqs(pos, raw=True), self._freqs(pos)
        assert raw.shape[0] == 3 and torch.equal(self.attn._materialize_mrope_freqs(raw), emb)
        self.config.qsa_use_sparse_attention = True
        with torch.no_grad():
            out_raw, _ = self.attn(hidden, attention_mask=None, rotary_pos_emb=raw)
            out_emb, _ = self.attn(hidden, attention_mask=None, rotary_pos_emb=emb)
        assert torch.equal(out_raw, out_emb)

    def test_rejects_wrong_row_count(self):
        hidden = self._hidden(64, 1, seed=29)
        table = self.rope(128)  # a 1-D table is not per-token
        with pytest.raises(ValueError, match="one rotary row per token"):
            self.attn.indexer.select(hidden, table)

    def test_rejects_context_parallel(self):
        idx = self.attn.indexer
        idx.pg_collection = types.SimpleNamespace(cp=_Cp2(), tp=idx.pg_collection.tp)
        hidden = self._hidden(64, 1, seed=31)
        pos = mrope_position_ids([("text", 64)]).unsqueeze(1)
        with pytest.raises(NotImplementedError, match="context parallelism"):
            idx.select(hidden, self._freqs(pos))
