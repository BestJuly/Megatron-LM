# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA Megatron-module parity vs the reference oracle (QSA-2 dense-mask bridge).

Builds ``QSASelfAttention`` from ``get_qsa_module_spec_for_backend``, copies its
weights into the pure-torch oracle, and checks (a) the indexer's selected-token
mask matches the oracle's batched causal-prefix mask exactly, and (b) the full
attention forward matches the oracle numerically in fp32, with a real sparse
regime (reduced budget so selection actually drops blocks).
"""

import math

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_qsa_module_spec_for_backend,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.qsa_reference import (
    QSAAttentionParams,
    QSAIndexerParams,
    batched_indexer_selected_mask,
    reference_attention_forward,
)

HIDDEN = 256
N_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 32
ROTARY_PERCENT = 0.25  # rotary dim 8
ROPE_THETA = 10000000.0
SEQ = 300
BATCH = 2
BUDGET = 64  # block_topk 16 -> sparse for queries past ~67 tokens
RATIO = 4


def max_rel_err(got, ref):
    """max |got - ref| / max |ref|, in fp32."""
    got, ref = got.float(), ref.float()
    return ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def make_config(head_dim=HEAD_DIM, dtype=torch.float32):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        num_query_groups=N_KV_HEADS,
        kv_channels=head_dim,
        use_cpu_initialization=False,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        layernorm_epsilon=1e-6,
        qk_layernorm=True,
        attention_output_gate=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        experimental_attention_variant="qsa",
        qsa_indexer_budget=BUDGET,
        qsa_indexer_compress_ratio=RATIO,
    )


def copy_weights_to_oracle(attn, config):
    """Extract QSASelfAttention weights into oracle parameter structs."""
    d = HEAD_DIM
    ng = N_KV_HEADS
    hpg = N_HEADS // ng  # query heads per group
    qkv = attn.linear_qkv.weight.detach().float().cpu()  # [(2*hpg + 2) * d * ng, hidden]
    rows_per_group = (2 * hpg + 2) * d
    q_rows, gate_rows, k_rows, v_rows = [], [], [], []
    for g in range(ng):
        base = g * rows_per_group
        q_g = qkv[base : base + hpg * d]
        gate_g = qkv[base + hpg * d : base + 2 * hpg * d]
        k_rows.append(qkv[base + 2 * hpg * d : base + 2 * hpg * d + d])
        v_rows.append(qkv[base + 2 * hpg * d + d : base + rows_per_group])
        # Oracle q_proj packs per-head [query(d), gate(d)] chunks.
        for h in range(hpg):
            q_rows.append(q_g[h * d : (h + 1) * d])
            gate_rows.append(gate_g[h * d : (h + 1) * d])
    q_proj = torch.cat([torch.cat([q_rows[h], gate_rows[h]], dim=0) for h in range(N_HEADS)], dim=0)

    ap = QSAAttentionParams(
        q_proj_weight=q_proj,
        k_proj_weight=torch.cat(k_rows, dim=0),
        v_proj_weight=torch.cat(v_rows, dim=0),
        o_proj_weight=attn.linear_proj.weight.detach().float().cpu(),
        q_norm_weight=attn.q_layernorm.weight.detach().float().cpu(),
        k_norm_weight=attn.k_layernorm.weight.detach().float().cpu(),
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=d,
        rotary_dim=int(d * ROTARY_PERCENT),
        rope_theta=ROPE_THETA,
        rms_norm_eps=config.layernorm_epsilon,
    )
    ip = QSAIndexerParams(
        index_qk_proj_weight=attn.indexer.index_qk_proj.weight.detach().float().cpu(),
        q_norm_weight=attn.indexer.q_layernorm.weight.detach().float().cpu(),
        k_norm_weight=attn.indexer.k_layernorm.weight.detach().float().cpu(),
        n_heads=config.qsa_indexer_n_heads,
        head_dim=config.qsa_indexer_head_dim,
        token_budget=config.qsa_indexer_budget,
        compress_ratio=config.qsa_indexer_compress_ratio,
        rotary_dim=int(d * ROTARY_PERCENT),
        rope_theta=ROPE_THETA,
        rms_norm_eps=config.layernorm_epsilon,
    )
    return ap, ip


class TestQSAAttentionParity:
    # The indexer's trainability is fixed at build time (frozen without the loss), so
    # subclasses that train it set the coefficient here rather than after construction.
    INDEXER_LOSS_COEFF = None

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = make_config()
        self.config.qsa_indexer_loss_coeff = self.INDEXER_LOSS_COEFF
        spec = get_qsa_module_spec_for_backend(self.config)
        assert spec.metainfo == {"fuse_input_layernorm": False}
        self.attn = build_module(spec, config=self.config, layer_number=1).cuda()
        # Randomize norm gamma so parity actually exercises the zero-centered path.
        with torch.no_grad():
            for norm in (
                self.attn.q_layernorm,
                self.attn.k_layernorm,
                self.attn.indexer.q_layernorm,
                self.attn.indexer.k_layernorm,
            ):
                norm.weight.uniform_(-0.1, 0.1)
        self.rope = RotaryEmbedding(
            kv_channels=HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )
        g = torch.Generator().manual_seed(7)
        self.hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=g).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_central_switch_returns_qsa_spec(self):
        spec = get_experimental_attention_variant_module_spec(self.config)
        assert spec.module.__name__ == "QSASelfAttention"

    def test_indexer_mask_matches_oracle_exactly(self):
        _, ip = copy_weights_to_oracle(self.attn, self.config)
        freqs = self.rope(SEQ)
        with torch.no_grad():
            mega_mask = self.attn.indexer(self.hidden, freqs)  # [b, 1, s, s], True = masked
        oracle_visible = batched_indexer_selected_mask(
            self.hidden.permute(1, 0, 2).float().cpu(), ip
        )  # [b, s, s], True = visible
        assert torch.equal(
            ~mega_mask.squeeze(1).cpu(), oracle_visible
        ), "indexer selected-token sets diverge from the oracle"
        # sanity: the sparse regime is real (late queries drop visible blocks)
        n_visible_last = int(oracle_visible[0, -1].sum())
        assert n_visible_last < SEQ - 1

    def test_attention_forward_matches_oracle(self):
        ap, ip = copy_weights_to_oracle(self.attn, self.config)
        freqs = self.rope(SEQ)
        with torch.no_grad():
            out, bias = self.attn(self.hidden, attention_mask=None, rotary_pos_emb=freqs)
        assert bias is None or torch.count_nonzero(bias) == 0

        hidden_bsh = self.hidden.permute(1, 0, 2).float().cpu()
        visible = batched_indexer_selected_mask(hidden_bsh, ip)
        expect = reference_attention_forward(hidden_bsh, ap, visible)  # [b, s, h]
        got = out.permute(1, 0, 2).float().cpu()
        assert torch.allclose(
            got, expect, atol=5e-4, rtol=5e-4
        ), f"max abs err {(got - expect).abs().max().item():.3e}"

    def test_backward_flows_and_indexer_detached(self):
        freqs = self.rope(SEQ)
        hidden = self.hidden.clone().requires_grad_()
        out, _ = self.attn(hidden, attention_mask=None, rotary_pos_emb=freqs)
        out.square().mean().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        # The indexer must receive no gradient from the main path (detached input,
        # non-differentiable selection); its training signal arrives with the KL loss.
        assert self.attn.indexer.index_qk_proj.weight.grad is None
        assert self.attn.linear_qkv.weight.grad is not None

    def test_indexer_trainability_follows_loss_coeff(self):
        # Without the KL loss the indexer has no gradient path, so it is frozen and
        # DDP / the optimizer never track it; the rest of the layer stays trainable.
        expect = bool(self.config.qsa_indexer_loss_coeff)
        assert all(p.requires_grad == expect for p in self.attn.indexer.parameters())
        assert all(
            p.requires_grad for n, p in self.attn.named_parameters() if not n.startswith("indexer.")
        )

    def test_hybrid_stack_builds_qsa_layer(self):
        """The 'Q' hybrid symbol builds a TransformerLayer wrapping QSASelfAttention."""
        from megatron.core.models.hybrid.hybrid_block import HybridStack
        from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_qsa_stack_spec
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.experimental_attention_variant.qsa import QSASelfAttention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        spec = hybrid_qsa_stack_spec(self.config)
        pg = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'pp', 'embd', 'cp', 'dp_cp', 'ep', 'expt_tp', 'expt_dp']
        )
        block = HybridStack(
            self.config,
            spec.submodules,
            layer_type_list=validate_segment_layers("Q-"),
            pp_layer_offset=0,
            pg_collection=pg,
        )
        assert isinstance(block.layers[0], TransformerLayer)
        assert isinstance(block.layers[0].self_attention, QSASelfAttention)

    def test_repeat_forward_bitwise_identical(self):
        freqs = self.rope(SEQ)
        with torch.no_grad():
            a, _ = self.attn(self.hidden, attention_mask=None, rotary_pos_emb=freqs)
            b, _ = self.attn(self.hidden.clone(), attention_mask=None, rotary_pos_emb=freqs)
        assert torch.equal(a, b)


class TestQSATensorParallel:
    """CI-reachable distributed tests (run under a >= 2-rank torchrun harness).

    Verifies the two TP contracts of QSA: the replicated indexer selects the same
    routes on every TP rank (deterministic top-k, D4), and the TP2+SP forward runs
    with the mask produced from sequence-sharded hidden states.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_tp2_sp_route_agreement_and_forward(self):
        if Utils.world_size < 2:
            pytest.skip("QSA TP2+SP parity requires at least 2 distributed ranks")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

        config = make_config()
        config.tensor_model_parallel_size = 2
        config.sequence_parallel = True
        spec = get_qsa_module_spec_for_backend(config)
        attn = build_module(spec, config=config, layer_number=1).cuda()

        import megatron.core.parallel_state as ps

        tp_group = ps.get_tensor_model_parallel_group()
        tp_rank = ps.get_tensor_model_parallel_rank()

        g = torch.Generator().manual_seed(7)
        full = torch.randn(SEQ, BATCH, HIDDEN, generator=g).cuda()  # identical on all ranks
        shard = full.chunk(2, dim=0)[tp_rank].contiguous()
        rope = RotaryEmbedding(
            kv_channels=HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )
        freqs = rope(SEQ)

        with torch.no_grad():
            _, sel = attn.indexer(shard, freqs, return_selection=True)
        # Route agreement: selection must be identical on every TP rank.
        gathered = [torch.empty_like(sel.order) for _ in range(2)]
        torch.distributed.all_gather(gathered, sel.order.contiguous(), group=tp_group)
        assert torch.equal(gathered[0], gathered[1]), "TP ranks disagree on selected routes"

        out, _ = attn(shard, attention_mask=None, rotary_pos_emb=freqs)
        assert out.shape == (SEQ // 2, BATCH, HIDDEN) and torch.isfinite(out.float()).all()

    def test_tp2_kl_teacher_reduce_before_maxpool(self):
        """The Eq.(17) block MaxPool is nonlinear, so the teacher's TP head-sum must
        complete before pooling. With a replicated indexer and a correctly reduced
        teacher, the KL loss gradient must be identical on every TP rank."""
        if Utils.world_size < 2:
            pytest.skip("QSA TP2 KL parity requires at least 2 distributed ranks")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

        config = make_config()
        config.tensor_model_parallel_size = 2
        config.sequence_parallel = True
        config.qsa_indexer_loss_coeff = 0.01
        spec = get_qsa_module_spec_for_backend(config)
        attn = build_module(spec, config=config, layer_number=1).cuda()
        attn.train()

        import megatron.core.parallel_state as ps

        tp_group = ps.get_tensor_model_parallel_group()
        tp_rank = ps.get_tensor_model_parallel_rank()

        g = torch.Generator().manual_seed(7)
        full = torch.randn(SEQ, BATCH, HIDDEN, generator=g).cuda()
        shard = full.chunk(2, dim=0)[tp_rank].contiguous()
        rope = RotaryEmbedding(
            kv_channels=HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )
        # The sharp assertion: the LOSS is bitwise identical across TP ranks. The
        # teacher's local-head softmax differs per rank, so this only holds if the
        # head-sum all-reduce lands before the nonlinear block MaxPool.
        selection = attn.indexer.select(shard, rope(SEQ))
        loss = attn._compute_indexer_loss(shard, rope(SEQ), selection, 0.01)
        gathered_loss = [torch.empty_like(loss) for _ in range(2)]
        torch.distributed.all_gather(gathered_loss, loss.contiguous(), group=tp_group)
        assert torch.equal(gathered_loss[0], gathered_loss[1]), (
            "indexer KL loss diverges across TP ranks — the teacher reduce is "
            "not happening before the block MaxPool"
        )

        out, _ = attn(shard, attention_mask=None, rotary_pos_emb=rope(SEQ))
        out.float().square().mean().backward()
        grad = attn.indexer.index_qk_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        gathered = [torch.empty_like(grad) for _ in range(2)]
        torch.distributed.all_gather(gathered, grad.contiguous(), group=tp_group)
        # Gradients are only allclose, not bitwise: the loss's torch.gather backward
        # is a scatter-add (atomic) outside deterministic mode.
        assert torch.allclose(
            gathered[0], gathered[1], rtol=1e-4, atol=1e-6
        ), "indexer KL gradients diverge across TP ranks beyond scatter-add noise"


class TestQSAIndexerSparseKL(TestQSAAttentionParity):
    """QSA-3: sparse-KL distillation trains the indexer."""

    INDEXER_LOSS_COEFF = 0.01

    def setup_method(self, method):
        super().setup_method(method)
        self.attn.train()
        self.freqs = self.rope(SEQ)

    def _forward_backward(self):
        out, _ = self.attn(self.hidden, attention_mask=None, rotary_pos_emb=self.freqs)
        out.float().square().mean().backward()
        return out

    def test_backward_flows_and_indexer_detached(self):
        # Overrides the parent test: with the KL loss enabled the indexer params DO
        # receive gradients — but only through the loss, never through the trunk
        # (the loss-disabled case is covered by test_no_indexer_grad_when_loss_disabled).
        self._forward_backward()
        assert self.attn.indexer.index_qk_proj.weight.grad is not None

    def test_loss_trains_indexer_and_leaves_forward_unchanged(self):
        # Forward value must be bit-identical with and without the loss attached
        # (DSAIndexerLossAutoScaler is a forward identity).
        with torch.no_grad():
            self.attn.eval()
            ref, _ = self.attn(self.hidden, attention_mask=None, rotary_pos_emb=self.freqs)
        self.attn.train()
        out = self._forward_backward()
        assert torch.equal(out.detach(), ref)
        # The KL loss (and only it) reaches the indexer parameters.
        g = self.attn.indexer.index_qk_proj.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        assert self.attn.linear_qkv.weight.grad is not None

    def test_no_indexer_grad_when_loss_disabled(self):
        self.config.qsa_indexer_loss_coeff = 0.0
        self._forward_backward()
        assert self.attn.indexer.index_qk_proj.weight.grad is None

    def _teacher_top_blocks(self, topk):
        """Dense teacher: per-query top blocks of the max-pooled head-summed probs."""
        ap, _ = copy_weights_to_oracle(self.attn, self.config)
        from tests.unit_tests.transformer.experimental_attention_variant.qsa_reference import (
            apply_partial_rope,
            build_causal_visible,
            build_rope_cos_sin,
            rms_norm,
        )

        h = self.hidden.permute(1, 0, 2).float().cpu()
        B, S, _ = h.shape
        q = rms_norm(
            (h @ ap.q_proj_weight.t()).view(B, S, ap.n_heads, ap.head_dim * 2).chunk(2, dim=-1)[0],
            ap.q_norm_weight,
        )
        kk = rms_norm(
            (h @ ap.k_proj_weight.t()).view(B, S, ap.n_kv_heads, ap.head_dim), ap.k_norm_weight
        )
        pos = torch.arange(S).expand(B, S)
        cos, sin = build_rope_cos_sin(pos, ap.rotary_dim, ap.rope_theta, h.dtype)
        q = apply_partial_rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2))
        kk = apply_partial_rope(kk, cos.unsqueeze(-2), sin.unsqueeze(-2))
        rep = ap.n_heads // ap.n_kv_heads
        kk = kk.repeat_interleave(rep, dim=2)
        attn = torch.einsum("bqhd,bkhd->bhqk", q, kk) * ap.head_dim**-0.5
        causal = build_causal_visible(S)
        attn = attn.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(attn.float(), dim=-1).sum(dim=1)  # [B, S, S]
        n_blocks = S // RATIO
        blk = probs[:, :, : n_blocks * RATIO].view(B, S, n_blocks, RATIO).amax(-1)
        return blk.topk(topk, dim=-1).indices  # [B, S, topk]

    def test_overlap_with_teacher_increases(self):
        k = BUDGET // RATIO  # 16
        teacher_top = self._teacher_top_blocks(k).cuda()
        m_t = (torch.arange(SEQ, device="cuda") + 1) // RATIO
        eval_rows = m_t > 2 * k  # queries where selection is genuinely sparse

        def overlap():
            with torch.no_grad():
                _, sel = self.attn.indexer(self.hidden, self.freqs, return_selection=True)
            hits = (sel.order.unsqueeze(-1) == teacher_top.unsqueeze(-2)).any(-1) & sel.picked
            return (hits.float().sum(-1) / k)[:, eval_rows].mean().item()

        before = overlap()
        opt = torch.optim.Adam(self.attn.indexer.parameters(), lr=3e-3)
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            self._forward_backward()
            # keep only indexer grads for the step
            opt.step()
        after = overlap()
        assert after > before, f"overlap did not improve: {before:.4f} -> {after:.4f}"

    def test_route_and_kl_share_scaled_logits(self):
        """Single-source contract: the KL student logits are the same scaled scores
        that produced the route, and the 1/sqrt(d) scale is part of the KL softmax
        temperature. The scale cancels in the top-k (it is order-preserving), so a
        route-only test cannot catch a missing or wrong scale in the loss."""
        from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
            masked_log_softmax,
        )

        with torch.no_grad():
            _, sel = self.attn.indexer(self.hidden, self.freqs, return_selection=True)
            d_idx = sel.q.shape[-1]
            b, s, k = sel.order.shape
            # Recompute the student scores at the selected blocks exactly like the
            # loss (pass 2) does, from the SAME selection.q / selection.block_keys.
            # fp64 recompute: the selection kernel scores in exact fp32 (ieee dot),
            # while a fp32/TF32 einsum here would inject noise larger than genuine
            # order violations. fp64 pins the comparison to kernel rounding only.
            q_idx = sel.q.permute(1, 0, 2, 3).double()  # [b, s, nh, d]
            bk = sel.block_keys.permute(1, 0, 2).double()  # [b, nb, d]
            bk_sel = torch.gather(
                bk, 1, sel.order.reshape(b, s * k)[..., None].expand(-1, -1, d_idx)
            ).view(b, s, k, d_idx)
            student = torch.einsum("bshd,bskd->bshk", q_idx, bk_sel)
            student = (torch.relu(student).sum(dim=2) / math.sqrt(d_idx)).float()

            # (a) The recomputed scores reproduce the selection's descending order
            # (small tolerance: the kernel scores use a fused ieee dot).
            s0, s1 = student[..., :-1], student[..., 1:]
            both = sel.picked[..., :-1] & sel.picked[..., 1:]
            assert bool(
                (s0 >= s1 - 1e-4)[both].all()
            ), "KL student scores disagree with the route's score order"
            # (b) Temperature: dropping the 1/sqrt(d) scale changes the student
            # distribution even though it cannot change the route.
            lp = masked_log_softmax(student, sel.picked)
            lp_hot = masked_log_softmax(student * math.sqrt(d_idx), sel.picked)
            assert (lp - lp_hot).abs()[sel.picked].max() > 1e-3


class TestQSADDPOverlapGradReduce:
    """With overlap_grad_reduce, DDP records in step 1 which params reported a grad and
    asserts in step 2 that every bucketed param did. Without the KL loss the indexer
    gets no grad, so it must be frozen (excluded from DDP) for training to proceed."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("indexer_loss_coeff", [None, 0.0, 0.01])
    def test_two_steps(self, indexer_loss_coeff):
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

        config = make_config()
        config.qsa_indexer_loss_coeff = indexer_loss_coeff
        spec = get_qsa_module_spec_for_backend(config)
        attn = build_module(spec, config=config, layer_number=1).cuda()
        model = DistributedDataParallel(
            config,
            ddp_config=DistributedDataParallelConfig(
                overlap_grad_reduce=True, grad_reduce_in_fp32=True
            ),
            module=attn,
        )
        model.train()
        indexer_params = {id(p) for p in attn.indexer.parameters()}
        tracked = {id(p) for p in model.params_with_grad}
        assert indexer_params.isdisjoint(tracked) != bool(indexer_loss_coeff)

        freqs = RotaryEmbedding(
            kv_channels=HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )(SEQ)
        hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(7)).cuda()
        for _ in range(2):
            model.zero_grad_buffer()  # step 2 runs the per-param readiness assert
            out, _ = model(hidden, attention_mask=None, rotary_pos_emb=freqs)
            out.float().square().mean().backward()
            model.finish_grad_sync()


class TestQSAIndexerCheckpointStrictLoad:
    """docs/models/qsa.md indexer mapping smoke against the released-checkpoint
    shapes (``index_qk_proj [640, 2560]``, zero-centered norms ``[128]`` — verified
    1:1 against the HF checkpoint by the community implementation)."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sentinel_state_dict_strict_load(self):
        config = make_config()
        config.hidden_size = 2560  # production hidden; indexer defaults 4 heads x 128
        spec = get_qsa_module_spec_for_backend(config)
        indexer = build_module(spec.submodules.indexer, config=config).cuda()

        sd = indexer.state_dict()
        param_keys = sorted(k for k in sd if not k.endswith("_extra_state"))
        assert param_keys == [
            "index_qk_proj.weight",
            "k_layernorm.weight",
            "q_layernorm.weight",
        ], f"indexer state dict drifted from the documented mapping: {param_keys}"
        assert sd["index_qk_proj.weight"].shape == (640, 2560)
        assert sd["q_layernorm.weight"].shape == (128,)
        assert sd["k_layernorm.weight"].shape == (128,)

        # HF tensors map 1:1 (direct copy). Sentinel values prove the load lands.
        sentinel = {
            "index_qk_proj.weight": torch.full((640, 2560), 0.25),
            "q_layernorm.weight": torch.linspace(-0.1, 0.1, 128),
            "k_layernorm.weight": torch.linspace(0.1, -0.1, 128),
        }
        missing, unexpected = indexer.load_state_dict(sentinel, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert all(m.endswith("_extra_state") for m in missing), f"missing keys: {missing}"
        for key, val in sentinel.items():
            got = sd[key].detach().float().cpu()
            assert torch.equal(got, val), f"{key} did not load verbatim"
