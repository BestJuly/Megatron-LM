# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Distributed correctness tests for the gated-residual hyper-connection variant.

Covers GR-3:
1. sequence_parallel gradient semantics: with TP2+SP, the TP all-reduce of the
   replicated GR parameters' gradients reproduces the full-sequence reference.
2. GR + MTP end-to-end on GPTModel (the full chain: input_expand -> n-stream
   decoder -> mhc_multistream -> MTP _concat_embeddings -> nested stack ->
   MTP _postprocess exit contract), forward finite + gradients on every GR
   parameter, at TP1 and TP2+SP.
"""

import os
import sys

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.gated_residual import GatedResidualModule
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version, unwrap_model
from megatron.training.arguments import (
    core_transformer_config_from_args,
    parse_args,
    validate_args,
)
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import HAVE_TE
except ImportError:
    HAVE_TE = False

_SEED = 1234
HIDDEN = 64
STREAMS = 4
LOWRANK = 16


class TestGatedResidualSequenceParallelGrads:
    """TP2+SP: all-reduced GR gradients must equal the full-sequence reference."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(_SEED)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ GPUs")
    def test_sp_grad_allreduce_matches_full_sequence(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=STREAMS,
            hyper_connection_variant="gated_residual",
            hc_lowrank=LOWRANK,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        torch.manual_seed(_SEED)
        module = GatedResidualModule(config, layer_number=1).cuda()
        for name, param in module.named_parameters():
            assert getattr(param, 'sequence_parallel', False), name

        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        seq, batch = 16, 2
        torch.manual_seed(_SEED + 1)  # identical on every rank
        x_full = torch.randn(seq, batch, STREAMS * HIDDEN, device="cuda")

        def run(x):
            module.zero_grad(set_to_none=True)
            mixed, h_res, g_write, residual = module(x.detach().clone().requires_grad_(True))
            out = module.fused_h_res_h_post_bda(
                h_res,
                residual,
                g_write,
                (mixed, None),  # feed mixed back as a stand-in sublayer output
                dropout_prob=0.0,
                training=True,
                fused=False,
            )
            out.float().square().sum().backward()
            return {n: p.grad.detach().clone() for n, p in module.named_parameters()}

        # Full-sequence reference (identical on every rank).
        grads_ref = run(x_full)

        # SP run: each TP rank sees its contiguous sequence shard, then the
        # marked gradients are all-reduced over the TP group — exactly what
        # finalize_model_grads does for sequence_parallel-marked params.
        shard = seq // tp_size
        x_local = x_full[tp_rank * shard : (tp_rank + 1) * shard]
        grads_sp = run(x_local)
        for name in grads_sp:
            torch.distributed.all_reduce(grads_sp[name], group=tp_group)

        for name in grads_ref:
            torch.testing.assert_close(
                grads_sp[name], grads_ref[name], atol=1e-4, rtol=1e-4, msg=name
            )


class TestGatedResidualContextParallelGrads:
    """CP2: gradients reduced over the CP group must equal the full-sequence
    reference.

    GR is a per-token operation, so context parallelism needs no GR-side code:
    each CP rank processes its sequence chunks independently, and the replicated
    GR parameters' gradients are summed by the regular DDP reduction, which runs
    over the DP x CP group (``finalize_model_grads.py`` /
    ``distributed_data_parallel.py`` use
    ``get_data_parallel_group(with_context_parallel=True)``). This test emulates
    that reduction with an explicit all-reduce over the CP group.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, context_parallel_size=2
        )
        model_parallel_cuda_manual_seed(_SEED)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ GPUs")
    def test_cp_grad_allreduce_matches_full_sequence(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=STREAMS,
            hyper_connection_variant="gated_residual",
            hc_lowrank=LOWRANK,
            context_parallel_size=2,
        )
        torch.manual_seed(_SEED)
        module = GatedResidualModule(config, layer_number=1).cuda()

        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = parallel_state.get_context_parallel_rank()
        cp_size = parallel_state.get_context_parallel_world_size()

        seq, batch = 16, 2
        torch.manual_seed(_SEED + 1)  # identical on every rank
        x_full = torch.randn(seq, batch, STREAMS * HIDDEN, device="cuda")

        def run(x):
            module.zero_grad(set_to_none=True)
            mixed, h_res, g_write, residual = module(x.detach().clone().requires_grad_(True))
            out = module.fused_h_res_h_post_bda(
                h_res,
                residual,
                g_write,
                (mixed, None),  # feed mixed back as a stand-in sublayer output
                dropout_prob=0.0,
                training=True,
                fused=False,
            )
            out.float().square().sum().backward()
            return {n: p.grad.detach().clone() for n, p in module.named_parameters()}

        # Full-sequence reference (identical on every rank).
        grads_ref = run(x_full)

        # CP run on this rank's chunks, using the real load-balanced CP layout:
        # the sequence is split into 2*cp chunks and rank r owns chunks r and
        # 2*cp-1-r. A per-token op is split-scheme-agnostic, but mirror the real
        # assignment anyway.
        chunks = x_full.chunk(2 * cp_size, dim=0)
        x_local = torch.cat([chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]], dim=0)
        grads_cp = run(x_local)
        for name in grads_cp:
            torch.distributed.all_reduce(grads_cp[name], group=cp_group)

        for name in grads_ref:
            torch.testing.assert_close(
                grads_cp[name], grads_ref[name], atol=1e-4, rtol=1e-4, msg=name
            )


class TestGatedResidualMTPIntegration:
    """GR + MTP end-to-end on GPTModel (mirrors TestMHCMTPIntegration for mHC)."""

    def setup_method(self, method):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("1.7.0"), reason="TransformerEngine >= 1.7.0 required"
    )
    @pytest.mark.parametrize('tp', [1, 2])
    def test_e2e_forward_backward(self, tp):
        if torch.cuda.device_count() < tp:
            pytest.skip(f"needs {tp}+ GPUs")
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        seq_length, micro_batch_size = 32, 2

        sys.argv = ['test_gated_residual_distributed.py']
        args = parse_args()
        args.num_layers = 2
        args.mtp_num_layers = 2
        args.mtp_loss_scaling_factor = 0.1
        args.vocab_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 256
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = seq_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = tp > 1
        args.context_parallel_size = 1
        args.position_embedding_type = 'rope'
        args.num_experts = None
        args.moe_grouped_gemm = False
        args.train_iters = 1
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.add_bias_linear = False
        args.swiglu = True
        args.bf16 = True
        args.enable_hyper_connections = True
        args.num_residual_streams = 4
        args.hyper_connection_variant = 'gated_residual'
        args.hc_lowrank = LOWRANK
        args.recompute_granularity = None

        validate_args(args)
        set_global_variables(args, False)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)

        def model_provider(
            pre_process=True,
            post_process=True,
            layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
            config=None,
            pg_collection=None,
            vp_stage=None,
            **kwargs,
        ):
            model_parallel_cuda_manual_seed(_SEED)
            a = get_args()
            if config is None:
                config = core_transformer_config_from_args(a)
            layer_spec = layer_spec_fn(
                a.num_experts,
                a.moe_grouped_gemm,
                a.qk_layernorm,
                enable_hyper_connection=config.enable_hyper_connections,
                hyper_connection_variant=config.hyper_connection_variant,
            )
            mtp_spec = get_gpt_mtp_block_spec(
                config=config, spec=layer_spec, use_transformer_engine=True
            )
            return GPTModel(
                config=config,
                transformer_layer_spec=layer_spec,
                mtp_block_spec=mtp_spec,
                vocab_size=a.vocab_size,
                max_sequence_length=a.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=a.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not a.untie_embeddings_and_output_weights,
                position_embedding_type=a.position_embedding_type,
                rotary_percent=a.rotary_percent,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )

        gpt_model, _, _ = setup_model_and_optimizer(ModelType.encoder_or_decoder, model_provider)

        # Structural checks: GR modules everywhere, no learned_output_contract params.
        core_model = unwrap_model(gpt_model[0])
        gr_modules = [
            m for m in core_model.modules() if isinstance(m, GatedResidualModule)
        ]
        # 2 decoder layers x 2 sublayer modules + decoder exit
        # + 2 MTP layers x 2 sublayer modules + MTP exit contract(s).
        assert len(gr_modules) >= 2 * 2 + 1 + 2 * 2 + 1
        for name, _ in core_model.named_parameters():
            assert 'hc_head_' not in name, name

        data = list(range(seq_length))
        tokens = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = (1 + torch.tensor(data, dtype=torch.int64)).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()

        output = gpt_model[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        assert torch.isfinite(output).all(), f"Non-finite output (TP={tp})"

        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        assert "values" in tracker, f"MTP loss not logged (TP={tp})"
        assert torch.isfinite(tracker['values']).all()
        MTPLossLoggingHelper.clean_loss_in_tracker()

        output.mean().backward()
        gr_param_substrings = (
            'hyper_connection',
            'hc_exit_contract',
            'hc_norm',
            'input_mix_weight',
            'block_inject_weight',
        )
        checked = 0
        for name, param in gpt_model[0].named_parameters():
            grad = getattr(param, 'main_grad', None)
            assert grad is not None, f"No gradient for {name}"
            if any(s in name for s in gr_param_substrings):
                checked += 1
                assert torch.isfinite(grad).all(), name
        assert checked > 0
