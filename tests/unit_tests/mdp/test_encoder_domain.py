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

# Gate per test rather than per module, so tests that need no world can join
# this file later and stay in the CPU suite.
requires_distributed = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


def _tiny_config(**overrides):
    """The 1-layer, hidden=8 encoder config every test in this file builds on."""
    kwargs = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class _TinyEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        torch.manual_seed(42)  # identical replica weights on every rank
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


class _TinyAdapter:
    """The MdpModelAdapter surface build_encoder_domain consumes; the encoder
    class is a parameter so other tests can build the domain around theirs."""

    payload_width = 8
    spatial_merge_size = 2

    def __init__(self, encoder_class=_TinyEncoder):
        self.encoder_class = encoder_class

    def build_encoder(self, model_config, *, pg_collection):
        return self.encoder_class(model_config)


def _build_encoder_pgs():
    """The encoder process groups for this file's pp=2 topology."""
    rank_map = build_rank_map(
        MdpRankSpec(
            world_size=torch.distributed.get_world_size(), tp=1, pp=2, cp=1, ep=1, encoder_cp=1
        )
    )
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    return build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)


def _build_domain(*, adapter=None, model_config=None, mdp_config=None, decoder_overlap=False):
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=decoder_overlap,
        overlap_param_gather=decoder_overlap,
        align_param_gather=decoder_overlap,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam", lr=1e-3, use_distributed_optimizer=True, clip_grad=1.0
    )
    return build_encoder_domain(
        adapter=adapter or _TinyAdapter(),
        model_config=model_config or _tiny_config(),
        mdp_config=mdp_config or MdpConfig(enable=True),
        ddp_config=ddp_config,
        optimizer_config=optimizer_config,
        encoder_pgs=_build_encoder_pgs(),
        wrap_mixed_precision=False,
    )


@requires_distributed
def test_decoder_overlap_is_isolated_from_encoder_domain():
    encoder_config = _build_domain(decoder_overlap=True).encoder_ddp.ddp_config
    assert not encoder_config.overlap_grad_reduce
    assert not encoder_config.overlap_param_gather
    assert not encoder_config.align_param_gather


def _assert_identical_on_every_rank(tensor):
    """Every rank holds a full encoder replica, so anything the P5/P6 domain
    produces -- the reduced gradient, the stepped weight -- must agree bitwise."""
    gathered = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    for other in gathered[1:]:
        assert torch.equal(other, gathered[0])


def _build_allreduce_ddp():
    """A plain all-reduce DDP over the encoder groups, so the full summed
    gradient is observable on every rank (the ZeRO-1 path reduce-scatters and
    leaves each rank only its shard; its semantics are covered by the
    optimizer-step test)."""
    from megatron.core.distributed import DistributedDataParallel

    model_config = _tiny_config()
    return DistributedDataParallel(
        config=model_config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False, overlap_grad_reduce=False, overlap_param_gather=False
        ),
        module=_TinyEncoder(model_config).cuda(),
        pg_collection=_build_encoder_pgs(),
    )


@requires_distributed
@pytest.mark.parametrize(
    "num_tokens, divisor",
    [
        pytest.param(40.0, 40.0, id="scaled_by_token_count"),
        # clamp(T_global, min=1): a zero token count must leave the sum alone
        # rather than divide by zero.
        pytest.param(0.0, 1.0, id="zero_token_count_means_no_scaling"),
    ],
)
def test_world_sum_reduction_with_prescale_one_and_token_scaling(num_tokens, divisor):
    ddp = _build_allreduce_ddp()
    ddp.zero_grad_buffer()

    # Rank-distinct work: gradients must be summed over WORLD (prescale 1),
    # then scaled by 1/clamp(T_global, 1) exactly once.
    rank = torch.distributed.get_rank()
    ddp(torch.full((4, 8), float(rank + 1), device="cuda")).sum().backward()

    param = next(ddp.module.parameters())
    expected = param.main_grad.clone()
    torch.distributed.all_reduce(expected)  # WORLD sum, no pre-division
    expected /= divisor

    finalize_encoder_grads(ddp, globally_reduced_num_tokens=torch.tensor(num_tokens, device="cuda"))
    assert torch.allclose(param.main_grad, expected, rtol=1e-6, atol=1e-6)
    _assert_identical_on_every_rank(param.main_grad)


@requires_distributed
def test_optimizer_steps_identically_on_all_ranks():
    domain = _build_domain()
    ddp = domain.encoder_ddp
    ddp.zero_grad_buffer()
    ddp(torch.ones(2, 8, device="cuda")).sum().backward()
    finalize_encoder_grads(ddp, globally_reduced_num_tokens=torch.tensor(16.0, device="cuda"))
    success, _, _ = domain.encoder_optimizer.step()
    assert success
    _assert_identical_on_every_rank(next(ddp.module.parameters()).data)


@requires_distributed
def test_disjointness_assertion_catches_leaked_parameter():
    domain = _build_domain()

    leaky_chunk = torch.nn.Sequential(torch.nn.Linear(4, 4), domain.encoder_ddp.module)
    with pytest.raises(MdpConfigurationError, match="contains encoder parameters"):
        assert_parameter_disjointness(domain.encoder_ddp, [leaky_chunk])
    # And a clean chunk passes.
    assert_parameter_disjointness(domain.encoder_ddp, [torch.nn.Linear(4, 4).cuda()])
