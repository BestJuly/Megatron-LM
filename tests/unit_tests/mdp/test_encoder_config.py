# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-compute tests for the MDP encoder DDP policy."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.encoder import build_encoder_ddp_config
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.optimizer import MdpChainedOptimizer, build_encoder_optimizer_config
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import ChainedOptimizer


def test_decoder_overlap_is_not_inherited_by_encoder():
    decoder_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        align_param_gather=True,
        grad_reduce_in_fp32=True,
        bucket_size=1234,
    )

    encoder_config = build_encoder_ddp_config(decoder_config)

    assert encoder_config is not decoder_config
    assert decoder_config.overlap_grad_reduce
    assert decoder_config.overlap_param_gather
    assert decoder_config.align_param_gather
    assert not encoder_config.overlap_grad_reduce
    assert not encoder_config.overlap_param_gather
    assert not encoder_config.align_param_gather
    assert encoder_config.use_distributed_optimizer
    assert encoder_config.grad_reduce_in_fp32
    assert encoder_config.bucket_size == 1234


def test_encoder_does_not_inherit_quantized_parameter_buffers():
    decoder = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        fp8_param_gather=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
    )
    encoder = build_encoder_ddp_config(decoder)
    assert not encoder.fp8_param_gather
    assert not encoder.fp4_param_gather
    assert not encoder.reuse_grad_buf_for_mxfp8_param_ag
    assert decoder.fp8_param_gather and decoder.reuse_grad_buf_for_mxfp8_param_ag


def test_encoder_optimizer_projects_only_mxfp8_staging_options():
    decoder = OptimizerConfig(
        bf16=True,
        fp8_recipe="mxfp8",
        lr=0.001,
        clip_grad=1.0,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_param_gather=True,
    )
    encoder = build_encoder_optimizer_config(decoder)
    expected = {
        "fp8_recipe": None,
        "reuse_grad_buf_for_mxfp8_param_ag": False,
        "overlap_param_gather": False,
    }
    for field in fields(decoder):
        assert getattr(encoder, field.name) == expected.get(
            field.name, getattr(decoder, field.name)
        )
    assert encoder.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert not decoder.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert decoder.reuse_grad_buf_for_mxfp8_param_ag


@pytest.mark.parametrize("recipe", [None, "delayed", "tensorwise", "mxfp8"])
def test_non_reusing_optimizer_config_is_unchanged(recipe):
    decoder = OptimizerConfig(bf16=True, fp8_recipe=recipe)
    assert build_encoder_optimizer_config(decoder) is decoder


def test_composite_accepts_only_the_encoder_projection():
    decoder = OptimizerConfig(bf16=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=True)
    encoder = build_encoder_optimizer_config(decoder)
    members = [SimpleNamespace(config=c, model_chunks=[]) for c in (decoder, encoder)]
    composite = MdpChainedOptimizer(members, encoder_member_index=1)
    assert composite.chained_optimizers == members
    assert composite._decoder_chain.chained_optimizers == members[:1]
    assert composite._encoder_member is members[1]
    with pytest.raises(AssertionError):
        ChainedOptimizer(members)
    members[1].config = replace(encoder, lr=0.123)
    with pytest.raises(MdpConfigurationError, match="hyperparameters"):
        MdpChainedOptimizer(members, encoder_member_index=1)


def _run_train_step_until_forward(
    monkeypatch, optimizer, model, *, mdp=True, reuse=True, overlap=True, full_cg=False
):
    """Exercise the real train-step setup, stopping before model execution."""
    import megatron.training.training as training

    args = SimpleNamespace(
        mdp_enable=mdp,
        reuse_grad_buf_for_mxfp8_param_ag=reuse,
        overlap_param_gather=overlap,
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        seq_length=128,
        global_batch_size=1,
        micro_batch_size=1,
        decoder_seq_length=None,
    )

    class ReachedForward(Exception):
        pass

    def stop_at_forward(**_kwargs):
        raise ReachedForward

    with monkeypatch.context() as patch:
        patch.setattr(training, "get_args", lambda: args)
        patch.setattr(training, "get_timers", lambda: None)
        patch.setattr(training, "get_num_microbatches", lambda: 1)
        patch.setattr(training, "has_nvidia_modelopt", False)
        patch.setattr(
            training,
            "get_rerun_state_machine",
            lambda: SimpleNamespace(should_run_forward_backward=lambda _: True),
        )
        patch.setattr(
            training,
            "FullCudaGraphWrapper",
            SimpleNamespace(cuda_graph={"training": object()} if full_cg else {}),
        )
        with pytest.raises(ReachedForward):
            training.train_step(
                None,
                iter(()),
                model,
                optimizer,
                None,
                SimpleNamespace(),
                stop_at_forward,
                iteration=0,
            )


@pytest.mark.parametrize("mdp", [False, True])
@pytest.mark.parametrize(
    "reuse,overlap,hooks,full_cg,should_stage",
    [
        (True, True, True, False, True),
        (False, True, True, False, False),
        (True, False, True, False, False),
        (True, True, False, False, False),
        (True, True, False, True, True),
    ],
)
def test_train_step_staging_preserves_native_behavior(
    monkeypatch, mdp, reuse, overlap, hooks, full_cg, should_stage
):
    import megatron.training.training as training

    calls = []

    class DistOpt:
        def __init__(self, name, reuse, overlap):
            self.config = SimpleNamespace(
                reuse_grad_buf_for_mxfp8_param_ag=reuse, overlap_param_gather=overlap
            )
            self._copy_main_params_to_param_buffer = lambda: calls.append(name)

    monkeypatch.setattr(training, "DistributedOptimizer", DistOpt)
    members = [
        DistOpt("decoder-dense", True, True),
        DistOpt("decoder-expert", True, True),
        DistOpt("encoder", False, False),
        DistOpt("non-reusing", False, True),
        DistOpt("synchronous", True, False),
        SimpleNamespace(),
    ]
    model = SimpleNamespace(
        zero_grad_buffer=lambda: calls.append("zero-buffer"),
        remove_forward_pre_hook_handles=[object()] if hooks else [],
    )
    optimizer = SimpleNamespace(
        chained_optimizers=members, zero_grad=lambda: calls.append("zero-optimizer")
    )
    _run_train_step_until_forward(
        monkeypatch, optimizer, [model], mdp=mdp, reuse=reuse, overlap=overlap, full_cg=full_cg
    )
    expected = []
    if should_stage:
        expected = ["decoder-dense", "decoder-expert"]
        if not mdp:
            # Preserve the legacy traversal even if member flags differ from args.
            expected += ["encoder", "non-reusing", "synchronous"]
    assert calls == ["zero-buffer", "zero-optimizer", *expected]


@pytest.mark.parametrize("overlap", [False, True])
def test_encoder_cannot_change_decoder_deferred_sync_policy(monkeypatch, overlap):
    from megatron.core.optimizer import distrib_optimizer

    calls = []

    class DistOpt:
        def __init__(self, name, config, ddp_overlap):
            self.name, self.config = name, config
            self.ddp_config = SimpleNamespace(overlap_param_gather=ddp_overlap)
            self.model_chunks = []
            self._defer_param_sync = False

        def step_with_ready_grads(self):
            calls.append((self.name, self._defer_param_sync))
            return True

    monkeypatch.setattr(distrib_optimizer, "DistributedOptimizer", DistOpt)
    decoder_config = OptimizerConfig(
        bf16=True,
        fp8_recipe="mxfp8",
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_param_gather=overlap,
    )
    encoder_config = build_encoder_optimizer_config(decoder_config)
    members = [
        DistOpt("dense", decoder_config, overlap),
        DistOpt("expert", decoder_config, overlap),
        DistOpt("encoder", encoder_config, False),
    ]
    composite = MdpChainedOptimizer(members, encoder_member_index=2)
    assert composite._decoder_chain._should_defer_mxfp8_param_sync() is not overlap
    assert composite.step_with_ready_grads()
    assert calls == [("dense", not overlap), ("expert", not overlap), ("encoder", False)]
    assert not any(member._defer_param_sync for member in members)
