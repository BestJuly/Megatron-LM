# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Encoder DDP domain tests: WORLD reduction with prescale 1, ZeRO-1 optimizer,
parameter disjointness, and 1/T_global finalization.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_encoder_domain.py
"""

import os

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import (
    assert_parameter_disjointness,
    build_encoder_domain,
    build_encoder_pg_collection,
    finalize_encoder_grads,
)
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1

pytestmark = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


class _TinyEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        torch.manual_seed(42)  # identical replica weights on every rank
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


class _TinyAdapter:
    payload_width = 8
    spatial_merge_size = 2

    def build_encoder(self, model_config, *, pg_collection):
        return _TinyEncoder(model_config)


def _build_domain(*, decoder_overlap=False):
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=1, process_groups=groups
    )
    model_config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=decoder_overlap,
        overlap_param_gather=decoder_overlap,
        align_param_gather=decoder_overlap,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam", lr=1e-3, use_distributed_optimizer=True, clip_grad=1.0
    )
    domain = build_encoder_domain(
        adapter=_TinyAdapter(),
        model_config=model_config,
        mdp_config=MdpConfig(enable=True),
        ddp_config=ddp_config,
        optimizer_config=optimizer_config,
        encoder_pgs=encoder_pgs,
        wrap_mixed_precision=False,
    )
    return domain


def test_decoder_overlap_is_isolated_from_encoder_domain():
    domain = _build_domain(decoder_overlap=True)

    encoder_config = domain.encoder_ddp.ddp_config
    assert not encoder_config.overlap_grad_reduce
    assert not encoder_config.overlap_param_gather
    assert not encoder_config.align_param_gather


def _build_allreduce_ddp():
    """A plain all-reduce DDP over the encoder groups, so the full summed
    gradient is observable on every rank (the ZeRO-1 path reduce-scatters and
    leaves each rank only its shard; its semantics are covered by the
    optimizer-step test)."""
    from megatron.core.distributed import DistributedDataParallel

    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=1, process_groups=groups
    )
    model_config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    return DistributedDataParallel(
        config=model_config,
        ddp_config=ddp_config,
        module=_TinyEncoder(model_config).cuda(),
        pg_collection=encoder_pgs,
    )


def test_world_sum_reduction_with_prescale_one_and_token_scaling():
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    ddp = _build_allreduce_ddp()
    ddp.zero_grad_buffer()

    # Rank-distinct work: gradients must be summed over WORLD (prescale 1),
    # then scaled by 1/clamp(T_global, 1) exactly once.
    x = torch.full((4, 8), float(rank + 1), device="cuda")
    out = ddp(x)
    out.sum().backward()

    param = next(ddp.module.parameters())
    local = param.main_grad.clone()
    expected = local.clone()
    torch.distributed.all_reduce(expected)  # WORLD sum, no pre-division
    tokens = torch.tensor(40.0, device="cuda")
    expected /= 40.0

    finalize_encoder_grads(ddp, globally_reduced_num_tokens=tokens)
    assert torch.allclose(param.main_grad, expected, rtol=1e-6, atol=1e-6)

    # The reduced gradient must be identical on every rank.
    gathered = [torch.empty_like(param.main_grad) for _ in range(world)]
    torch.distributed.all_gather(gathered, param.main_grad)
    for other in gathered[1:]:
        assert torch.equal(other, gathered[0])


def test_zero_token_count_means_no_scaling():
    ddp = _build_allreduce_ddp()
    ddp.zero_grad_buffer()
    out = ddp(torch.ones(2, 8, device="cuda"))
    out.sum().backward()
    param = next(ddp.module.parameters())
    summed = param.main_grad.clone()
    torch.distributed.all_reduce(summed)
    finalize_encoder_grads(
        ddp, globally_reduced_num_tokens=torch.tensor(0.0, device="cuda")
    )
    assert torch.allclose(param.main_grad, summed, rtol=1e-6, atol=1e-6)


def test_optimizer_steps_identically_on_all_ranks():
    domain = _build_domain()
    ddp = domain.encoder_ddp
    ddp.zero_grad_buffer()
    out = ddp(torch.ones(2, 8, device="cuda"))
    out.sum().backward()
    finalize_encoder_grads(
        ddp, globally_reduced_num_tokens=torch.tensor(16.0, device="cuda")
    )
    success, _, _ = domain.encoder_optimizer.step()
    assert success
    param = next(ddp.module.parameters())
    world = torch.distributed.get_world_size()
    gathered = [torch.empty_like(param.data) for _ in range(world)]
    torch.distributed.all_gather(gathered, param.data)
    for other in gathered[1:]:
        assert torch.equal(other, gathered[0])


def test_disjointness_assertion_catches_leaked_parameter():
    domain = _build_domain()

    class _LeakyChunk(torch.nn.Module):
        def __init__(self, leaked):
            super().__init__()
            self.own = torch.nn.Linear(4, 4)
            self.leaked = leaked

    leaked_param_module = domain.encoder_ddp.module
    chunk = _LeakyChunk(leaked_param_module)
    with pytest.raises(MdpConfigurationError, match="contains encoder parameters"):
        assert_parameter_disjointness(domain.encoder_ddp, [chunk])
    # And a clean chunk passes.
    assert_parameter_disjointness(
        domain.encoder_ddp, [torch.nn.Linear(4, 4).cuda()]
    )
