# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA indexer-loss CuTe DSL kernels (``ops/cute_qsa_indexer_loss.py``).

* Teacher (query-tile kernel for short documents plus the row kernel, and the row kernel
  alone): both kernels agree to fp32 summation order; bitwise repeatable; the unpooled
  (tensor-parallel) output max-pools to the pooled one.
"""

import pytest
import torch

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
        assert torch.equal(res["auto", True], again), "teacher not repeatable"
        for kernel in ("auto", "row"):
            assert torch.equal(
                res[kernel, False].amax(-1).masked_fill(~sel.picked, 0.0), res[kernel, True]
            ), kernel
        for pool in (True, False):
            err = (res["auto", pool] - res["row", pool]).abs().max().item()
            base._report(case, f"teacher_auto_vs_row_pool{int(pool)}_maxabs", err)
            assert err <= 1e-5, f"tile vs row teacher {err:.3e}"
