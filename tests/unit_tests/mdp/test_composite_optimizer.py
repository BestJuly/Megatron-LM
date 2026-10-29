# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Composite-optimizer tests: WORLD overflow union and atomic update (fp16).

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_composite_optimizer.py

The fault-injection half proves the test catches a broken mechanism: with the
plain ChainedOptimizer, an overflow visible only to one member's grad-stats
subgroup makes ranks disagree about skipping the step; MdpChainedOptimizer
unions the verdict over WORLD before any scaler update.
"""

import math
import os

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.fp8_utils import get_fp8_context, is_mxfp8tensor
from megatron.core.mdp.encoder import build_encoder_ddp_config
from megatron.core.mdp.optimizer import (
    MdpChainedOptimizer,
    build_encoder_optimizer_config,
    build_mdp_composite_optimizer,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1
pytestmark = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        yield
        Utils.destroy_model_parallel()


class _Tiny(torch.nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        torch.manual_seed(seed)
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


_SINGLETONS = {}
_SUBGROUPS = None


def _singleton_group():
    rank = torch.distributed.get_rank()
    if rank not in _SINGLETONS:
        mine = None
        for r in range(torch.distributed.get_world_size()):
            group = torch.distributed.new_group(ranks=[r])
            if r == rank:
                mine = group
        _SINGLETONS[rank] = mine
    return _SINGLETONS[rank]


def _subgroup():
    """Adjacent-pair subgroups: the 'decoder-like' grad-stats domain."""
    global _SUBGROUPS
    if _SUBGROUPS is None:
        rank = torch.distributed.get_rank()
        mine = None
        for base in range(0, torch.distributed.get_world_size(), 2):
            group = torch.distributed.new_group(ranks=[base, base + 1])
            if rank in (base, base + 1):
                mine = group
        _SUBGROUPS = mine
    return _SUBGROUPS


def _pgs(data_group):
    mine = _singleton_group()
    pgs = ProcessGroupCollection()
    pgs.dp = data_group
    pgs.dp_cp = data_group
    pgs.intra_dp_cp = data_group
    pgs.intra_dist_opt = data_group
    pgs.tp = mine
    pgs.pp = mine
    pgs.ep = mine
    pgs.mp = None
    pgs.expt_dp = None
    pgs.tp_ep_pp = None
    pgs.inter_dist_opt = None
    return pgs


def _member(config, optimizer_config, data_group, seed):
    model_config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        fp16=True,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    module = Float16Module(model_config, _Tiny(model_config, seed).cuda())
    ddp = DistributedDataParallel(
        config=model_config,
        ddp_config=config,
        module=module,
        pg_collection=_pgs(data_group),
    )
    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[ddp],
        pg_collection=_pgs(data_group),
        use_gloo_process_groups=False,
    )
    return ddp, optimizer


def _build(composite_cls):
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        fp16=True,
        loss_scale=None,
        initial_loss_scale=2.0**16,
        min_loss_scale=1.0,
        hysteresis=1,  # one overflow halves the scale (no hysteresis absorption)
    )
    subgroup_ddp, subgroup_opt = _member(ddp_config, optimizer_config, _subgroup(), seed=1)
    world_ddp, world_opt = _member(
        ddp_config, optimizer_config, torch.distributed.group.WORLD, seed=2
    )
    if composite_cls is MdpChainedOptimizer:
        composite = build_mdp_composite_optimizer(subgroup_opt, world_opt)
    else:
        composite = ChainedOptimizer([subgroup_opt, world_opt])
    return subgroup_ddp, world_ddp, composite


def _run(composite_cls, inject_rank, *, prepare_only=False):
    subgroup_ddp, world_ddp, composite = _build(composite_cls)
    for ddp in (subgroup_ddp, world_ddp):
        ddp.zero_grad_buffer()
        out = ddp(torch.ones(2, 8, device="cuda", dtype=torch.float16))
        out.float().sum().backward()
        ddp.finish_grad_sync()
    if torch.distributed.get_rank() == inject_rank:
        param = next(subgroup_ddp.module.module.parameters())
        param.main_grad.fill_(float("inf"))
    if prepare_only:
        # Stop at the overflow verdict: actually stepping with divergent
        # verdicts deadlocks in the distributed optimizer's collectives,
        # which is precisely the hazard the WORLD union prevents.
        found_inf = composite.prepare_grads()
        success = not found_inf
    else:
        success, _grad_norm, _ = composite.step()
    scales = [float(s.scale) for s in _scalers(composite)]
    return success, scales


def _scalers(composite):
    scalers, seen = [], set()
    for member in composite.chained_optimizers:
        scaler = getattr(member, "grad_scaler", None)
        if scaler is not None and id(scaler) not in seen:
            seen.add(id(scaler))
            scalers.append(scaler)
    return scalers


def _gather(value: bool):
    world = torch.distributed.get_world_size()
    flag = torch.tensor([1.0 if value else 0.0], device="cuda")
    out = [torch.empty_like(flag) for _ in range(world)]
    torch.distributed.all_gather(out, flag)
    return [bool(v.item()) for v in out]


def test_world_overflow_union_is_atomic():
    # Overflow injected on rank 0 in the member whose grad-stats domain is a
    # 2-rank subgroup: only ranks 0 and 1 see it locally.
    success, scales = _run(MdpChainedOptimizer, inject_rank=0)
    verdicts = _gather(success)
    assert verdicts == [False] * len(verdicts), verdicts
    # Every member scaler on every rank halved from the same global verdict.
    assert all(scale == 2.0**15 for scale in scales), scales
    scale_tensor = torch.tensor(scales, device="cuda")
    gathered = [torch.empty_like(scale_tensor) for _ in range(len(verdicts))]
    torch.distributed.all_gather(gathered, scale_tensor)
    for other in gathered[1:]:
        assert torch.equal(other, gathered[0])


def test_fault_injection_plain_chained_optimizer_diverges():
    # The same scenario through the plain ChainedOptimizer: ranks outside the
    # injected member's grad-stats subgroup see no overflow and take the step.
    # This proves the union test above detects a broken mechanism.
    success, _ = _run(ChainedOptimizer, inject_rank=0, prepare_only=True)
    verdicts = _gather(success)
    assert not verdicts[0], "the detecting rank must see the overflow"
    assert any(verdicts[2:]), (
        "ranks outside the subgroup were expected to (wrongly) take the step; "
        "if this starts failing, ChainedOptimizer gained a global union and "
        "MdpChainedOptimizer may be redundant"
    )


def test_clean_step_succeeds_and_scales_agree():
    success, scales = _run(MdpChainedOptimizer, inject_rank=-1)
    assert all(_gather(success))
    assert all(scale == 2.0**16 for scale in scales)


def test_member_order_is_flat_dense_expert_encoder():
    _, _, composite = _build(MdpChainedOptimizer)
    assert isinstance(composite, MdpChainedOptimizer)
    assert len(composite.chained_optimizers) == 2
    assert not any(
        isinstance(member, ChainedOptimizer) for member in composite.chained_optimizers
    )
    # get_loss_scale asserts the members agree before returning member 0's.
    assert float(composite.get_loss_scale()) == 2.0**16


@pytest.mark.parametrize("overlap", [False, True])
def test_mxfp8_reuse_preserves_both_domains_and_global_clipping(overlap, monkeypatch):
    """Compare three real MXFP8/BF16 updates against independently stepped domains."""
    import transformer_engine.pytorch as te

    from tests.unit_tests.mdp.test_encoder_config import _run_train_step_until_forward

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell")

    common = dict(
        optimizer="adam",
        lr=1e-3,
        clip_grad=0.01,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    decoder_config = OptimizerConfig(
        **common,
        fp8_recipe="mxfp8",
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_param_gather=overlap,
    )
    decoder_ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        grad_reduce_in_fp32=True,
        fp8_param_gather=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_grad_reduce=overlap,
        overlap_param_gather=overlap,
    )

    class Linear(torch.nn.Module):
        def __init__(self, fp8):
            super().__init__()
            self.config = TransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=1,
                bf16=True,
                params_dtype=torch.bfloat16,
                calculate_per_token_loss=True,
                fp8="e4m3" if fp8 else None,
                fp8_recipe="mxfp8",
                fp8_param=fp8,
            )
            torch.manual_seed(1234)
            torch.cuda.manual_seed(1234)
            with get_fp8_context(self.config, is_init=True):
                self.proj = te.Linear(128, 128, bias=False, params_dtype=torch.bfloat16)

        def forward(self, inputs):
            with get_fp8_context(self.config):
                return self.proj(inputs)

    def build(fp8, *, reference=False):
        model = Linear(fp8).cuda()
        if fp8:
            opt_config, ddp_config = decoder_config, decoder_ddp_config
        elif reference:
            # A plain BF16 reference, independent of the projection under test.
            opt_config = OptimizerConfig(**common)
            ddp_config = DistributedDataParallelConfig(
                use_distributed_optimizer=True, grad_reduce_in_fp32=True
            )
        else:
            opt_config = build_encoder_optimizer_config(decoder_config)
            ddp_config = build_encoder_ddp_config(decoder_ddp_config)
        pgs = _pgs(_subgroup() if fp8 else torch.distributed.group.WORLD)
        ddp = DistributedDataParallel(
            config=model.config, ddp_config=ddp_config, module=model, pg_collection=pgs
        )
        opt = get_megatron_optimizer(
            opt_config, [ddp], pg_collection=pgs, use_gloo_process_groups=False
        )
        return ddp, opt

    decoder, decoder_opt = build(True)
    encoder, encoder_opt = build(False)
    ref_decoder, ref_decoder_opt = build(True, reference=True)
    ref_encoder, ref_encoder_opt = build(False, reference=True)
    composite = build_mdp_composite_optimizer(decoder_opt, encoder_opt)
    assert is_mxfp8tensor(decoder.module.proj.weight)
    assert not is_mxfp8tensor(encoder.module.proj.weight)
    assert encoder.module.proj.weight.dtype == torch.bfloat16
    assert composite._decoder_chain.chained_optimizers == decoder_opt.chained_optimizers
    initial_encoder = encoder.module.proj.weight.detach().clone()
    inputs = torch.full(
        (128, 128), (torch.distributed.get_rank() + 1) / 8, device="cuda", dtype=torch.bfloat16
    )
    clipping_exercised = False
    for _ in range(3):
        for ddp, opt in (
            (decoder, decoder_opt),
            (encoder, encoder_opt),
            (ref_decoder, ref_decoder_opt),
            (ref_encoder, ref_encoder_opt),
        ):
            ddp.zero_grad_buffer()
            opt.zero_grad()
        if overlap:
            # Exercise train_step's actual zero-grad and inline staging path.
            _run_train_step_until_forward(monkeypatch, composite, [decoder])
            ref_decoder_opt.prepare_model_params_for_param_sync()
        for ddp in (decoder, encoder, ref_decoder, ref_encoder):
            (100 * ddp(inputs).float().square().mean()).backward()
            ddp.finish_grad_sync()

        refs = (ref_decoder_opt, ref_encoder_opt)
        assert not ref_decoder_opt.prepare_grads()
        assert not ref_encoder_opt.prepare_grads()
        expected_norm = math.sqrt(sum(opt.get_grad_norm() ** 2 for opt in refs))
        coefficient = min(1.0, common["clip_grad"] / (expected_norm + 1e-6))
        clipping_exercised |= coefficient < 1.0
        # Independent global clipping, not per-domain clipping or another MDP chain.
        with torch.no_grad():
            for opt in refs:
                for param in opt.get_parameters():
                    grad = (
                        param.decoupled_grad
                        if opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                        else param.grad
                    )
                    if grad is not None:
                        grad.mul_(coefficient)
                assert opt.step_with_ready_grads()

        success, norm, _ = composite.step()
        assert success
        assert norm == pytest.approx(expected_norm, rel=1e-6)
        for actual, reference in zip(
            decoder_opt.get_parameters(), ref_decoder_opt.get_parameters()
        ):
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            encoder.module.proj.weight, ref_encoder.module.proj.weight, rtol=0, atol=0
        )
        replicas = [
            torch.empty_like(initial_encoder) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(replicas, encoder.module.proj.weight.detach())
        assert all(torch.equal(replicas[0], replica) for replica in replicas[1:])
    assert clipping_exercised
    assert not torch.equal(initial_encoder, encoder.module.proj.weight)
    composite.prepare_model_params_for_param_sync()
