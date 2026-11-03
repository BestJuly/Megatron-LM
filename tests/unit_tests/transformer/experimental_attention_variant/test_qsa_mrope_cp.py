# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QSA under mRoPE with allgather context parallelism (packed THD).

Multimodal decoders keep the per-token mRoPE rows full-length under THD CP: row ``i``
belongs to token ``i`` of the GLOBAL pack. A CP rank's local query therefore reads the
row at its global packed index, and a pooled block key (built after the key all-gather,
in global order) reads its first token's global row. Image-bearing 3-D positions make a
wrong row visible: reading in-document offsets instead (the 1-D table convention) would
reuse document 0's positions for every document.

* CP2 selection equals the single-rank selection at the local rows (order / picked
  exact; projections within bf16 GEMM-shape noise), static and dynamic.
* CP2 attention output, input gradient and CP-summed parameter gradients equal the
  single-rank layer.
* CP2 sparse-KL indexer loss (row-weighted) and gradients equal the single-rank loss.
* BSHD under CP with per-token rows is still rejected.

Run with ``torchrun --nproc-per-node 2`` (or 4).
"""

import gc

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
    qsa_cp_layout_static,
)
from megatron.core.transformer.experimental_attention_variant.qsa_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    HIDDEN,
    ROPE_THETA,
    ROTARY_PERCENT,
    SPARSE_HEAD_DIM,
    make_sparse_config,
    requires_qsa_kernels,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_qsa_mrope import (
    SPARSE_MROPE_SECTION,
    mrope_position_ids,
)

CP = 2
MAX_SEQS = 4
# Documents padded to a multiple of 2 * CP (the per-document zigzag contract); the pack
# fills T. Padding rows continue their document's text positions.
DOCS_A = [
    [("text", 9), ("image", 1, 6, 8), ("text", 35)],  # 92
    [("text", 40), ("image", 1, 10, 6), ("text", 61)],  # 161 -> 164
]
DOCS_B = [
    [("text", 3), ("image", 2, 4, 6), ("text", 17)],  # 68
    [("text", 12)],  # 12
    [("text", 21), ("image", 1, 8, 10), ("text", 95)],  # trimmed so the pack fills T
]


def _pack(docs, total):
    """(per-document padded lengths, packed 3-D positions [3, 1, total])."""
    lens, parts = [], []
    for doc in docs:
        pos = mrope_position_ids(doc)
        n = pos.shape[-1]
        n_pad = (n + 2 * CP - 1) // (2 * CP) * (2 * CP)
        tail = pos[:, -1:] + torch.arange(1, n_pad - n + 1)
        lens.append(n_pad)
        parts.append(torch.cat([pos, tail], dim=-1))
    pos = torch.cat(parts, dim=-1)
    # Trim or extend the last document so the pack fills the buffer.
    if pos.shape[-1] > total:
        lens[-1] -= pos.shape[-1] - total
        pos = pos[:, :total]
    elif pos.shape[-1] < total:
        extra = total - pos.shape[-1]
        lens[-1] += extra
        pos = torch.cat([pos, pos[:, -1:] + torch.arange(1, extra + 1)], dim=-1)
    assert all(n % (2 * CP) == 0 for n in lens), lens
    return lens, pos.unsqueeze(1)


T = 256
PACKINGS = [_pack(DOCS_A, T), _pack(DOCS_B, T)]


def _cu(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    cu += [T] * (MAX_SEQS + 1 - len(cu))
    return torch.tensor(cu, dtype=torch.int32, device="cuda")


@requires_qsa_kernels
class TestQSAMRoPEContextParallel:
    def teardown_method(self, method):
        self.__dict__.clear()
        gc.collect()
        torch.cuda.synchronize()
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()

    def _build(self, loss_coeff=None):
        if Utils.world_size < 2:
            pytest.skip("QSA CP2 tests require at least 2 distributed ranks")
        Utils.initialize_model_parallel(1, 1, context_parallel_size=CP)
        initialize_rng_tracker(
            use_te_rng_tracker=True, use_cudagraphable_rng=True, force_reset=True
        )
        model_parallel_cuda_manual_seed(123)
        config = make_sparse_config()
        config.context_parallel_size = CP
        config.cp_comm_type = "all_gather"
        config.qsa_cp_packing_layout = "per_document"
        config.qsa_indexer_loss_coeff = loss_coeff
        config.mrope_section = list(SPARSE_MROPE_SECTION)
        config.mrope_interleaved = True
        spec = get_qsa_module_spec_for_backend(config)
        attn = build_module(spec, config=config, layer_number=1).cuda()
        with torch.no_grad():
            for norm in (
                attn.q_layernorm,
                attn.k_layernorm,
                attn.indexer.q_layernorm,
                attn.indexer.k_layernorm,
            ):
                norm.weight.uniform_(-0.1, 0.1)
        assert attn.indexer.per_token_rope
        import megatron.core.parallel_state as ps

        self.config, self.attn = config, attn
        self.rank = ps.get_context_parallel_rank()
        self.cp_group = ps.get_context_parallel_group()
        world = torch.distributed.get_world_size()
        groups = [torch.distributed.new_group([r]) for r in range(world)]
        self.own_group = groups[torch.distributed.get_rank()]
        # A size-1 CP group keeps the rows full-length, as the multimodal model does under
        # THD CP (MultimodalModel._thd_mrope_no_cp_override).
        self.mrope = MultimodalRotaryEmbedding(
            kv_channels=SPARSE_HEAD_DIM,
            rotary_percent=ROTARY_PERCENT,
            rotary_base=ROPE_THETA,
            interleaved_mrope=True,
            cp_group=self.own_group,
        )

    def _freqs(self, pos):
        # Full-length per-token rows, as the multimodal model emits them under THD CP.
        return self.mrope(pos.cuda(), list(SPARSE_MROPE_SECTION))

    def _rows(self, cu):
        q, _ = qsa_cp_layout_static(cu, T // CP, CP, self.rank, "per_document")
        return q

    def _single_rank(self, fn):
        pg = self.attn.pg_collection
        old = pg.cp
        pg.cp = self.own_group
        try:
            return fn()
        finally:
            pg.cp = old

    def test_cp2_selection_matches_single_rank(self):
        self._build()
        idx = self.attn.indexer
        g = torch.Generator().manual_seed(5)
        for lengths, pos in PACKINGS:
            cu = _cu(lengths)
            freqs = self._freqs(pos)
            h = torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16()
            rows = self._rows(cu)
            shard = h.index_select(0, rows).contiguous()
            with torch.no_grad():
                sel = idx.select(shard, freqs, cu_seqlens=cu)
                for static in (False, True):
                    idx.static_shapes = static
                    ref = self._single_rank(lambda: idx.select(h, freqs, cu_seqlens=cu))
                    idx.static_shapes = False
                    for name in ("order", "picked"):
                        assert torch.equal(getattr(sel, name), getattr(ref, name)[:, rows]), (
                            lengths,
                            name,
                        )
                    for name in ("own_block", "block_offsets", "token_offsets"):
                        assert torch.equal(getattr(sel, name), getattr(ref, name)[rows]), name
                    torch.testing.assert_close(sel.q, ref.q[rows])
                    n_real = sum(n // self.attn.indexer.compress_ratio for n in lengths)
                    torch.testing.assert_close(sel.block_keys[:n_real], ref.block_keys[:n_real])

    def test_in_document_rows_would_differ(self):
        """The image positions matter: reading in-document offsets selects differently."""
        self._build()
        idx = self.attn.indexer
        lengths, pos = PACKINGS[0]
        cu = _cu(lengths)
        freqs = self._freqs(pos)
        g = torch.Generator().manual_seed(7)
        h = torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16()
        shard = h.index_select(0, self._rows(cu)).contiguous()
        with torch.no_grad():
            right = idx.select(shard, freqs, cu_seqlens=cu)
            idx.per_token_rope = False  # the 1-D convention: in-document offsets
            wrong = idx.select(shard, freqs, cu_seqlens=cu)
            idx.per_token_rope = True
        assert not torch.equal(right.q, wrong.q)

    def _layer(self, hidden, freqs, cu):
        psp = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=T,
            max_seqlen_kv=T,
        )
        out, _ = self.attn(hidden, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=psp)
        return out

    def _fwd_bwd(self, hidden, freqs, cu, grad_out):
        self.attn.zero_grad(set_to_none=True)
        h = hidden.detach().clone().requires_grad_(True)
        out = self._layer(h, freqs, cu)
        out.backward(grad_out)
        grads = {
            n: p.grad.detach().clone()
            for n, p in self.attn.named_parameters()
            if p.grad is not None
        }
        return out.detach(), h.grad.detach(), grads

    def test_cp2_layer_matches_single_rank(self):
        self._build()
        self.attn.train()
        g = torch.Generator().manual_seed(9)
        for lengths, pos in PACKINGS:
            cu = _cu(lengths)
            freqs = self._freqs(pos)
            h = torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16()
            go = torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16()
            rows = self._rows(cu)
            ref_out, ref_dh, ref_grads = self._single_rank(
                lambda: self._fwd_bwd(h, freqs, cu, go)
            )
            out, dh, grads = self._fwd_bwd(
                h.index_select(0, rows).contiguous(),
                freqs,
                cu,
                go.index_select(0, rows).contiguous(),
            )
            torch.testing.assert_close(out, ref_out[rows], rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(dh, ref_dh[rows], rtol=2e-2, atol=2e-2)
            for name, r in ref_grads.items():
                gp = grads[name].clone()
                torch.distributed.all_reduce(gp, group=self.cp_group)
                rel = float((gp.float() - r.float()).norm() / r.float().norm().clamp_min(1e-30))
                assert rel <= 2e-2, f"{name} grad rel-L2 {rel:.3e} ({lengths})"

    @pytest.mark.parametrize("impl", ["fused", "layout"])
    def test_cp2_kl_matches_single_rank(self, impl):
        self._build(loss_coeff=1.0)
        self.config.qsa_indexer_loss_impl = impl
        self.attn.train()
        idx = self.attn.indexer
        params = [p for p in idx.parameters() if p.requires_grad]
        g = torch.Generator().manual_seed(11)
        for lengths, pos in PACKINGS:
            cu = _cu(lengths)
            freqs = self._freqs(pos)
            h = torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16()
            rows = self._rows(cu)
            shard = h.index_select(0, rows).contiguous()

            def run(hidden):
                sel = idx.select(hidden, freqs, cu_seqlens=cu)
                # No stashed main-path q/k: the teacher re-projects and RoPEs them through
                # _rope_qk_cp (the shard's rows at their global mRoPE rows).
                self.attn._indexer_main_qk = None
                loss = self.attn._compute_indexer_loss(hidden, freqs, sel, 1.0, cu_seqlens=cu)
                return loss, sel.picked.any(dim=-1).sum().float()

            self.attn.zero_grad(set_to_none=True)
            ref_loss, _ = self._single_rank(lambda: run(h))
            ref_loss.backward()
            ref_grads = [p.grad.detach().clone() for p in params]

            self.attn.zero_grad(set_to_none=True)
            loss, valid = run(shard)
            tot = valid.clone()
            torch.distributed.all_reduce(tot, group=self.cp_group)
            (loss * valid / tot).backward()
            kl = (loss * valid).detach()
            torch.distributed.all_reduce(kl, group=self.cp_group)
            torch.testing.assert_close(kl / tot, ref_loss.detach(), rtol=1e-4, atol=0)
            for p, r in zip(params, ref_grads):
                gp = p.grad.detach().clone()
                torch.distributed.all_reduce(gp, group=self.cp_group)
                # bf16 summation-order noise (the CP split regroups the sums and every rank
                # rounds its gradient before the all-reduce); measured up to ~1.0e-2.
                rel = float((gp.float() - r.float()).norm() / r.float().norm().clamp_min(1e-30))
                assert rel <= 2e-2, f"indexer grad rel-L2 {rel:.3e} ({impl}, {lengths})"

    def test_bshd_with_per_token_rows_is_rejected(self):
        self._build()
        idx = self.attn.indexer
        pos = mrope_position_ids([("text", 64)]).unsqueeze(1)
        h = torch.randn(32, 1, HIDDEN).cuda().bfloat16()
        with pytest.raises(NotImplementedError, match="packed THD"):
            idx.select(h, self._freqs(pos))
