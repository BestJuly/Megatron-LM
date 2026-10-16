# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""QSA fused indexer loss, CuTe DSL kernels (``ops/cute_qsa_indexer_loss.py``).

* Teacher (query-tile kernel for short documents plus the row kernel, and the row kernel
  alone): normalized targets within 1e-5 of the reference; both kernels agree to fp32
  summation order; bitwise repeatable; the unpooled (tensor-parallel) output max-pools to the
  pooled one.
* The fused loss reuses the main path's q/k, which equal the teacher's own re-projection.

The loss against the reference, its gradients against the fp64 gold and its CUDA-graph
capture are covered by test_qsa_indexer_loss_impl.py.
"""

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_qsa_indexer_loss_impl as base,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    requires_qsa_kernels,
)

pytestmark = requires_qsa_kernels

CASES = base.CASES


class TestQSAIndexerLossCute:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("case", list(CASES))
    def test_teacher(self, case):
        from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_indexer_loss import (  # pylint: disable=line-too-long
            qsa_indexer_teacher,
        )

        attn, hidden, freqs, cu = base.TestQSAIndexerLossImpl._build(None, CASES[case])
        with torch.no_grad():
            sel = attn.indexer.select(hidden, freqs, cu_seqlens=cu)
            q_bt, k_bt = attn._indexer_teacher_qk(hidden, freqs, sel, cu)
            kw = dict(
                compress_ratio=4,
                scale=attn.hidden_size_per_attention_head**-0.5,
                token_offsets=sel.token_offsets,
            )
            res = {}
            for kernel in ("auto", "row"):
                for pool in (True, False):
                    res[kernel, pool] = qsa_indexer_teacher(
                        q_bt, k_bt, sel.order, sel.picked, pool=pool, kernel=kernel, **kw
                    )
            again = qsa_indexer_teacher(q_bt, k_bt, sel.order, sel.picked, **kw)
            ref = attn._indexer_targets(q_bt, k_bt, sel, "reference")
            got = attn._indexer_targets(q_bt, k_bt, sel, "fused")
        assert torch.equal(res["auto", True], again), "teacher not repeatable"
        for kernel in ("auto", "row"):
            assert torch.equal(
                res[kernel, False].amax(-1).masked_fill(~sel.picked, 0.0), res[kernel, True]
            ), kernel
        for pool in (True, False):
            err = (res["auto", pool] - res["row", pool]).abs().max().item()
            base._report(case, f"teacher_auto_vs_row_pool{int(pool)}_maxabs", err)
            assert err <= 1e-5, f"tile vs row teacher {err:.3e}"
        t_err = (got - ref).abs().max().item()
        base._report(case, "fused_targets_maxabs", t_err)
        assert t_err <= 1e-5

    @pytest.mark.parametrize("case", ["thd_multidoc", "bshd_batch2"])
    def test_fused_loss_reuses_identical_qk(self, case):
        """The main path's q/k handed to the teacher equal its own re-projection."""
        attn, hidden, freqs, cu = base.TestQSAIndexerLossImpl._build(None, CASES[case])
        attn.config.qsa_indexer_loss_impl = "fused"
        psp = None
        if cu is not None:
            psp = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=freqs.shape[0],
                max_seqlen_kv=freqs.shape[0],
            )
        seen = []
        orig = attn._compute_indexer_loss

        def spy(hs, fr, sel, coeff, cu_seqlens=None, teacher_qk=None):
            stashed = attn._indexer_main_qk
            assert stashed is not None, "main q/k not stashed"
            seen.append((stashed, attn._indexer_teacher_qk(hs, fr, sel, cu_seqlens)))
            return orig(hs, fr, sel, coeff, cu_seqlens=cu_seqlens, teacher_qk=teacher_qk)

        attn._compute_indexer_loss = spy
        out, _ = attn(hidden, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=psp)
        out.sum().backward()
        assert len(seen) == 1
        (mq, mk), (tq, tk) = seen[0]
        assert torch.equal(mq, tq) and torch.equal(mk, tk)
        assert attn._indexer_main_qk is None
