# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Composite-optimizer tests: WORLD overflow union and atomic update (fp16).

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_composite_optimizer.py

The fault-injection half proves the test catches a broken mechanism: with the
plain ChainedOptimizer, an overflow visible only to one member's grad-stats
subgroup makes ranks disagree about skipping the step; MdpChainedOptimizer
unions the verdict over WORLD before any scaler update.
"""

import os

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.mdp.optimizer import MdpChainedOptimizer, build_mdp_composite_optimizer
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
