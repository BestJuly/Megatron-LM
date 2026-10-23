# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``gdn_kernel_backend``: config validation, backend resolution and GatedDeltaNet dispatch.

The in-tree CuTe DSL backend (``internal``) is the default; ``fla`` and ``cudnn`` stay
selectable. On a Blackwell GPU, a GatedDeltaNet layer in the default configuration must run the
CuTe DSL forward (checked with ``MCORE_GDN_INTERNAL_BACKEND=cute``, which raises instead of
falling back) and match the fla backend.
"""

import logging
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.gated_delta_net import gdn as gdn_module
from megatron.core.ssm.gated_delta_net.common import (
    _warn_internal_gdr_thd_cuda_graph_fallback,
    resolve_gdn_kernel_backend,
)
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet, torch_chunk_gated_delta_rule
from megatron.core.ssm.gated_delta_net.internal_gdn_backend import chunk as internal_chunk
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import CudaGraphModule


def _config(**overrides):
    kwargs = dict(
        hidden_size=128,
        num_layers=1,
        num_attention_heads=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        normalization="RMSNorm",
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        experimental_attention_variant="gdn",
        linear_attention_freq=[1],
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestConfig:
    def test_default_is_internal(self):
        assert _config().gdn_kernel_backend == "internal"

    @pytest.mark.parametrize("backend", ["internal", "fla", "cudnn"])
    def test_backends_are_selectable(self, backend):
        assert _config(gdn_kernel_backend=backend).gdn_kernel_backend == backend

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="internal, fla, cudnn"):
            _config(gdn_kernel_backend="torch")

    def test_cudnn_keeps_its_guards(self):
        with pytest.raises(ValueError, match="incompatible with deterministic_mode"):
            _config(gdn_kernel_backend="cudnn", deterministic_mode=True)
        with pytest.raises(ValueError, match="requires a model with GatedDeltaNet layers"):
            _config(
                gdn_kernel_backend="cudnn",
                experimental_attention_variant=None,
                linear_attention_freq=None,
            )

    def test_default_is_accepted_without_gdn_layers(self):
        # The default must not turn every non-GDN model into a configuration error.
        config = _config(experimental_attention_variant=None, linear_attention_freq=None)
        assert config.gdn_kernel_backend == "internal"

    def test_recompute_h_needs_the_internal_backend(self):
        assert _config(gdn_gdr_recompute_h=True).gdn_gdr_recompute_h
        with pytest.raises(ValueError, match="gdn_gdr_recompute_h only applies"):
            _config(gdn_kernel_backend="fla", gdn_gdr_recompute_h=True)


def _graph_config(backend="internal", thd=True, modules=("attn",), impl="transformer_engine"):
    return SimpleNamespace(
        gdn_kernel_backend=backend,
        sequence_packing_scheduler=None,
        dynamic_context_parallel=False,
        thd_max_packed_sequences=4 if thd else None,
        cuda_graph_impl=impl,
        # TransformerConfig normalizes the module names to CudaGraphModule members.
        cuda_graph_modules=[CudaGraphModule[m] for m in modules],
    )


class TestResolve:
    @pytest.mark.parametrize("backend", ["internal", "fla", "cudnn"])
    def test_eager_keeps_the_configured_backend(self, backend):
        assert resolve_gdn_kernel_backend(_graph_config(backend, impl="none")) == backend

    def test_thd_attention_capture_falls_back_to_fla(self):
        _warn_internal_gdr_thd_cuda_graph_fallback.cache_clear()
        with pytest.warns(UserWarning, match="cannot be captured"):
            assert resolve_gdn_kernel_backend(_graph_config()) == "fla"

    def test_sbhd_capture_and_moe_only_capture_keep_internal(self):
        assert resolve_gdn_kernel_backend(_graph_config(thd=False)) == "internal"
        assert resolve_gdn_kernel_backend(_graph_config(modules=("moe",))) == "internal"

    @pytest.mark.parametrize("backend", ["fla", "cudnn"])
    def test_other_backends_are_never_rewritten(self, backend):
        assert resolve_gdn_kernel_backend(_graph_config(backend)) == backend


def _gdn_shell(config):
    """The attributes ``_GDNBase.__init__`` sets before ``_setup_variant_attrs`` runs."""
    qk = config.linear_key_head_dim * config.linear_num_key_heads
    v = config.linear_value_head_dim * config.linear_num_value_heads
    return SimpleNamespace(
        config=config,
        qk_dim_local_tp=qk,
        v_dim_local_tp=v,
        num_value_heads=config.linear_num_value_heads,
        num_key_heads=config.linear_num_key_heads,
        num_v_heads_local_tp=config.linear_num_value_heads,
        tp_size=1,
    )


class TestDispatch:
    def test_default_binds_the_internal_backend(self):
        shell = _gdn_shell(_config())
        GatedDeltaNet._setup_variant_attrs(shell)
        assert shell.gdn_kernel_backend == "internal"
        rule = shell.gated_delta_rule
        assert isinstance(rule, partial)
        assert rule.func is internal_chunk.chunk_gated_delta_rule
        assert rule.keywords == {"recompute_h": False}

    def test_fla_binds_fla(self):
        shell = _gdn_shell(_config(gdn_kernel_backend="fla"))
        GatedDeltaNet._setup_variant_attrs(shell)
        assert shell.gated_delta_rule is gdn_module.chunk_gated_delta_rule

    def test_deterministic_mode_binds_torch_under_the_default(self):
        shell = _gdn_shell(_config(deterministic_mode=True))
        GatedDeltaNet._setup_variant_attrs(shell)
        assert shell.gated_delta_rule is torch_chunk_gated_delta_rule


def _internal_cute_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import fla  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _internal_cute_available(), reason="needs an SM10x GPU, CuTe DSL and fla")
class TestDefaultRunsCuteDSL:
    """A default-configured GatedDeltaNet runs the CuTe DSL forward and matches fla."""

    def _layer(self, backend):
        from megatron.core import parallel_state
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_experimental_attention_variant_module_spec,
        )
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        config = _config(
            hidden_size=1024,
            num_attention_heads=8,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            gdn_kernel_backend=backend,
        )
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        pg = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        return (
            GatedDeltaNet(
                config,
                submodules=get_experimental_attention_variant_module_spec(config=config).submodules,
                layer_number=1,
                bias=False,
                conv_bias=False,
                conv_init=1.0,
                use_qk_l2norm=True,
                A_init_range=(1, 16),
                pg_collection=pg,
            )
            .cuda()
            .bfloat16()
        )

    def setup_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.destroy_model_parallel()

    def test_cute_forward_runs_and_matches_fla(self, monkeypatch, caplog):
        fla_layer = self._layer("fla")
        cute_layer = self._layer("internal")
        cute_layer.load_state_dict(fla_layer.state_dict())
        torch.manual_seed(0)
        x = torch.randn(2048, 1, 1024, device="cuda", dtype=torch.bfloat16)

        def run(layer):
            h = x.clone().requires_grad_(True)
            out, _ = layer(h, None)
            out.float().square().mean().backward()
            return out.detach(), h.grad

        ref_out, ref_grad = run(fla_layer)
        # "cute" raises instead of falling back, so passing proves the CuTe DSL forward ran.
        # H = 48 value heads: the fused CuTe backward serves H = 64 only, so the backward of
        # this geometry is fla's under "auto"; check the forward under "cute" on its own.
        monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
        with torch.no_grad():
            cute_fwd, _ = cute_layer(x, None)
        from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation

        monkeypatch.setattr(implementation, "_reported_paths", set())
        monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
        with caplog.at_level(logging.INFO):
            out, grad = run(cute_layer)
        paths = " | ".join(r.getMessage() for r in caplog.records)
        print(f"GDN_INTERNAL_PATHS {paths}")
        assert "Internal GDN backend forward: CuTe DSL" in paths
        for name, a, b in (
            ("fwd_cute", cute_fwd, ref_out),
            ("out", out, ref_out),
            ("dx", grad, ref_grad),
        ):
            err = ((a.float() - b.float()).norm() / b.float().norm()).item()
            print(f"GDN_INTERNAL_VS_FLA {name}={err:.3e}")
            assert err < 2e-2, f"{name}: internal vs fla rel-L2 {err:.3e}"
