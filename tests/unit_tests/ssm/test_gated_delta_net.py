# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness
from tests.unit_tests.transformer.test_multi_latent_attention import (
    make_test_packed_seq_params,
    make_test_packed_seq_params_with_padding,
)

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size, sp, cp_size):
        # Initialize parallel and random seed
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.sp_size = tp_size if sp else 1

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        # Initialize model
        self.transformer_config = TransformerConfig(
            hidden_size=256,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=8,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sp,
            context_parallel_size=cp_size,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        gdn_submodules = get_experimental_attention_variant_module_spec(
            config=self.transformer_config
        ).submodules

        self.gdn = GatedDeltaNet(
            self.transformer_config,
            submodules=gdn_submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        )
        self.gdn = self.gdn.cuda().bfloat16()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        gdn = self.gdn

        micro_batch_size = 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        attention_mask = None

        output, bias = gdn(hidden_states, attention_mask)

        assert output.dim() == 3, f"Output too many dimensions ({output.shape=})"
        assert output.shape[0] == seq_length // self.sp_size // self.cp_size, (
            f"Output shape {output.shape[0]=} mismatch with "
            f" {seq_length=} // {self.sp_size=} // {self.cp_size=}."
        )
        assert (
            output.shape[1] == micro_batch_size
        ), f"Output shape {output.shape[1]=} mismatch with {micro_batch_size=}"
        assert (
            output.shape[2] == gdn.config.hidden_size
        ), f"Output shape {output.shape[2]=} mismatch with {gdn.config.hidden_size=}"
        assert (
            output.dtype == hidden_states.dtype
        ), f"Output dtype {output.dtype=} mismatch with {hidden_states.dtype=}"

    def test_jit_compiled_helpers(self):
        import torch._dynamo

        gdn = self.gdn
        batch = 2
        seq_len = 16

        num_v_heads_local = gdn.num_value_heads // gdn.tp_size // gdn.cp_size

        qkv_last_dim = (2 * gdn.qk_dim_local_tp + gdn.v_dim_local_tp) // gdn.cp_size
        qkv = torch.randn(
            batch, seq_len, qkv_last_dim, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        gate = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            gdn.value_head_dim,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        beta = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        alpha = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        # Disable dynamo so coverage.py can trace through the method bodies,
        # which are normally wrapped by @jit_fuser (torch.compile).
        with torch._dynamo.config.patch(disable=True):
            query, key, value, gate_out, beta_out, alpha_out = (
                gdn._prepare_qkv_for_gated_delta_rule(qkv, gate, beta, alpha, batch, seq_len)
            )

        assert query.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert key.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert value.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
        assert query.is_contiguous()
        assert key.is_contiguous()
        assert value.is_contiguous()

        A_log_mock = torch.randn(
            num_v_heads_local, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        dt_bias_mock = torch.randn(
            num_v_heads_local, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )

        with torch._dynamo.config.patch(disable=True):
            g, beta_sig = gdn._compute_g_and_beta(A_log_mock, dt_bias_mock, alpha, beta)

        assert g.dtype == torch.float32
        assert g.shape == alpha.shape
        assert beta_sig.shape == beta.shape

    def test_gpu_forward_thd_correctness(self):
        if self.sp_size > 1:
            pytest.skip("Sequence parallel is not supported for this test case.")

        atol, rtol = 3e-4, 3e-4

        # Input shape
        sequence_length = 32
        micro_batch_size = 4
        cu_seqlens = [0, 32, 64, 96, 128]
        # sbhd input shape: [sequence length, batch size, hidden size]
        sub_sequence_length = sequence_length // self.cp_size
        hidden_states_sbhd = torch.rand(
            (sub_sequence_length, micro_batch_size, self.gdn.config.hidden_size)
        )
        attention_mask_sbhd = None
        hidden_states_sbhd = hidden_states_sbhd.cuda().bfloat16()
        # thd input shape: [sequence length * batch size, 1, hidden size]
        hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
        hidden_states_thd = hidden_states_thd.view(-1, 1, self.gdn.config.hidden_size)
        attention_mask_thd = None
        packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)

        # THD format
        output_thd, _ = self.gdn(
            hidden_states_thd, attention_mask_thd, packed_seq_params=packed_seq_params
        )
        # SBHD format
        output_sbhd, _ = self.gdn(hidden_states_sbhd, attention_mask_sbhd)
        output_sbhd_T = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)

        rank = torch.distributed.get_rank()
        assert output_thd.shape[0] == sub_sequence_length * micro_batch_size
        assert output_thd.shape[1] == 1
        assert output_thd.shape[2] == self.gdn.config.hidden_size
        torch.testing.assert_close(
            output_sbhd_T,
            output_thd,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Output mismatch ({rank=}): {msg}",
        )

    def test_gpu_forward_thd_padding_correctness(self):
        if self.sp_size > 1:
            pytest.skip("Sequence parallel is not supported for this test case.")

        atol, rtol = 3e-4, 3e-4
        sequence_length = 32
        micro_batch_size = 4

        # sbhd input shape: [sequence length, batch size, hidden size]
        sub_sequence_length = sequence_length // self.cp_size
        hidden_states_sbhd = torch.rand(
            (sub_sequence_length, micro_batch_size, self.gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        output_sbhd, _ = self.gdn(hidden_states_sbhd, None)

        # thd input shape: [sequence length * batch size, 1, hidden size]
        hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
        hidden_states_thd = hidden_states_thd.view(-1, 1, self.gdn.config.hidden_size)
        output_bshd = output_sbhd.transpose(0, 1).contiguous()

        rank = torch.distributed.get_rank()

        # A) padded branch: prefer *_padded when available.
        padded_params = make_test_packed_seq_params_with_padding(
            cu_seqlens=[0, 30, 60, 90, 120], cu_seqlens_padded=[0, 32, 64, 96, 128]
        )
        output_thd_padded, _ = self.gdn(hidden_states_thd, None, packed_seq_params=padded_params)
        output_thd2bshd = output_thd_padded.view(*output_bshd.shape)
        torch.testing.assert_close(
            output_bshd[..., :30],
            output_thd2bshd[..., :30],
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"THD padded output mismatch ({rank=}): {msg}",
        )

        # B) no-padded branch: use actual cu_seqlens when it matches total_sequence_length.
        no_padding_params = make_test_packed_seq_params(cu_seqlens=[0, 32, 64, 96, 128])
        output_thd_no_padding, _ = self.gdn(
            hidden_states_thd, None, packed_seq_params=no_padding_params
        )
        assert output_thd_no_padding.shape == output_thd_padded.shape

        # C) padded mismatch branch: if *_padded[-1] mismatches total_sequence_length, should raise.
        padded_mismatch_params = make_test_packed_seq_params_with_padding(
            cu_seqlens=[0, 30, 60, 90, 120], cu_seqlens_padded=[0, 32, 64, 96, 126]
        )
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=padded_mismatch_params)

        # D) actual mismatch branch without *_padded: should raise.
        actual_mismatch_params = make_test_packed_seq_params(cu_seqlens=[0, 32, 64, 96, 129])
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=actual_mismatch_params)


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGDNCuSeqlensResolve:

    @pytest.fixture
    def mock_gdn(self):
        class MockGDN:
            cp_size = 2
            _resolve_cu_seqlens = GatedDeltaNet._resolve_cu_seqlens

        return MockGDN()

    def test_padded_preferred_when_available(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        padded = torch.tensor([0, 504, 1008], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(padded, actual, 1008, "cu_seqlens_q")
        assert torch.equal(result, padded)

    def test_actual_used_when_no_padding(self, mock_gdn):
        actual = torch.tensor([0, 504, 1008], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q")
        assert torch.equal(result, actual)

    def test_raises_when_padding_mismatch(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q")

    def test_raises_when_padded_mismatches_total(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        padded = torch.tensor([0, 504, 1004], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(padded, actual, 1008, "cu_seqlens_q")

    def test_cp1_still_validates_total(self, mock_gdn):
        mock_gdn.cp_size = 1
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q")


@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 2),  # CP
        (2, False, 2),  # TP w/o SP + CP
        (2, True, 2),  # TP w/ SP + CP
    ],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_parallel_gated_delta_net_correctness(tmp_path_dist_ckpt, sequence_packing, tp, sp, cp):
    transformer_config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    if cp:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 5e-4, 5e-4

    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=256,
        micro_batch_size=4,
        sequence_packing=sequence_packing,
    )


def test_flashinfer_prefill_helper_flattens_batch_and_converts_log_g_to_decay():
    from megatron.core.ssm import gated_delta_net as gdn_module

    query = torch.arange(12, dtype=torch.float32).reshape(2, 3, 1, 2)
    key = (query + 100).contiguous()
    value = (query + 200).contiguous()
    g = torch.tensor([[[0.0], [-1.0], [-2.0]], [[-3.0], [-4.0], [-5.0]]])
    beta = torch.linspace(0.1, 0.6, 6, dtype=torch.float32).reshape(2, 3, 1)
    captured = {}

    def fake_flashinfer_kernel(**kwargs):
        captured.update(kwargs)
        return kwargs["q"] + kwargs["v"]

    output = gdn_module._flashinfer_gdn_prefill_forward(
        query, key, value, g, beta, cu_seqlens=None, kernel=fake_flashinfer_kernel
    )

    assert output.shape == query.shape
    assert captured["q"].shape == (6, 1, 2)
    assert captured["k"].shape == (6, 1, 2)
    assert captured["v"].shape == (6, 1, 2)
    assert captured["cu_seqlens"].tolist() == [0, 3, 6]
    torch.testing.assert_close(captured["g"], torch.exp(g.float()).reshape(6, 1))
    torch.testing.assert_close(captured["beta"], beta.float().reshape(6, 1))
    torch.testing.assert_close(output.reshape(6, 1, 2), captured["q"] + captured["v"])


def test_flashinfer_prefill_kernel_omits_gate_log_cumsum_kwarg_by_default():
    from megatron.core.ssm import gated_delta_net as gdn_module

    q = torch.zeros(2, 1, 2)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    g = torch.ones(2, 1)
    beta = torch.ones(2, 1)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    captured = {}

    def fake_flashinfer_kernel(**kwargs):
        captured.update(kwargs)
        return kwargs["v"]

    out = gdn_module._flashinfer_run_prefill_kernel(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        kernel=fake_flashinfer_kernel,
        scale=0.5,
        gate_is_log_cumsum=False,
    )

    assert "gate_is_log_cumsum" not in captured
    torch.testing.assert_close(out, v)


def test_flashinfer_prefill_helper_allocates_forward_context_for_autograd_inputs():
    from megatron.core.ssm import gated_delta_net as gdn_module

    query = torch.ones(1, 64, 1, 2, requires_grad=True)
    key = torch.ones_like(query, requires_grad=True)
    value = torch.ones_like(query, requires_grad=True)
    g = torch.zeros(1, 64, 1, requires_grad=True)
    beta = torch.ones(1, 64, 1, requires_grad=True)
    captured = {}

    def fake_flashinfer_kernel(**kwargs):
        captured.update(kwargs)
        kwargs["output_A"].fill_(1)
        return kwargs["q"] + kwargs["v"]

    output = gdn_module._flashinfer_gdn_prefill_forward(
        query, key, value, g, beta, cu_seqlens=None, kernel=fake_flashinfer_kernel
    )

    assert output.requires_grad
    assert not captured["q"].requires_grad
    assert not captured["k"].requires_grad
    assert not captured["v"].requires_grad
    assert not captured["g"].requires_grad
    assert not captured["beta"].requires_grad
    assert not captured["output_A"].requires_grad
    assert captured["output_A"].shape == (64, 1, 64)
    assert captured["output_A"].dtype == query.dtype


def test_flashinfer_prefill_default_kernel_uses_mcore_owned_source(monkeypatch):
    import sys
    import types

    from megatron.core.ssm import gated_delta_net as gdn_module

    fake_kernel = object()
    pkg = types.ModuleType("mcore_gdn_opt")
    subpkg = types.ModuleType("mcore_gdn_opt.gated_delta_rule")
    forward = types.ModuleType("mcore_gdn_opt.gated_delta_rule.forward")
    forward.chunk_gated_delta_rule_prefill_cute = fake_kernel
    subpkg.forward = forward
    pkg.gated_delta_rule = subpkg
    monkeypatch.setitem(sys.modules, "mcore_gdn_opt", pkg)
    monkeypatch.setitem(sys.modules, "mcore_gdn_opt.gated_delta_rule", subpkg)
    monkeypatch.setitem(sys.modules, "mcore_gdn_opt.gated_delta_rule.forward", forward)

    assert gdn_module._get_flashinfer_gdn_prefill_kernel() is fake_kernel


def test_flashinfer_prefill_is_enabled_by_default(monkeypatch):
    from megatron.core.ssm import gated_delta_net as gdn_module

    monkeypatch.delenv("MCORE_GDN_PREFILL_BACKEND", raising=False)

    assert gdn_module._is_flashinfer_gdn_prefill_enabled()
