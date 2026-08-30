# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    BucketingPolicy,
    _get_parameter_groups,
    _is_expert_parallel_parameter,
)


class _ExpertTestModule(torch.nn.Module):
    """
    Mock module whose params are routed under `.experts.` to trigger
    is_expert_param=True. The outer `layer` attribute puts a dot before
    `experts` in the parameter path (e.g. `layer.experts.linear_fc1`).
    """

    def __init__(self, shapes):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.experts = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.empty(shape)) for name, shape in shapes.items()}
        )


def _get_bucket_signatures(module):
    bucket_groups, _, _ = _get_parameter_groups(
        module, BucketingPolicy(suggested_bucket_size=None), meta_device_init_fp8_params={}
    )
    param_to_name = {param: name for name, param in module.named_parameters()}
    return [
        {
            "chunk_size_factor": group.chunk_size_factor,
            "params": [(param_to_name[param], tuple(param.shape)) for param in group.params],
        }
        for group in bucket_groups
    ]


def test_grouped_expert_weights_split_when_chunk_size_factors_differ():
    """Grouped expert weights with mismatched chunk size factors get routed to separate buckets."""
    num_local_experts = 4
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (num_local_experts, 2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (num_local_experts, hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc1"])[1:].numel(),
            "params": [("layer.experts.linear_fc1", shapes["linear_fc1"])],
        },
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc2"])[1:].numel(),
            "params": [("layer.experts.linear_fc2", shapes["linear_fc2"])],
        },
    ]


def test_per_expert_2d_weights_merge_via_lcm():
    """Per-expert 2D weights merge into a single bucket via LCM chunk size factor."""
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": math.lcm(
                torch.Size(shapes["linear_fc1"])[1:].numel(),
                torch.Size(shapes["linear_fc2"])[1:].numel(),
            ),
            "params": [
                ("layer.experts.linear_fc1", shapes["linear_fc1"]),
                ("layer.experts.linear_fc2", shapes["linear_fc2"]),
            ],
        }
    ]


def test_allreduce_false_parameter_uses_expert_fsdp_bucket_without_moe_name():
    """Engram-style EP parameters use the expert mesh without a `.experts.` path."""
    module = torch.nn.Module()
    module.engram = torch.nn.Module()
    module.engram.embedding = torch.nn.Module()
    module.engram.embedding.weight = torch.nn.Parameter(torch.empty(11, 4))
    module.engram.embedding.weight.allreduce = False

    parameter_groups, _, _ = _get_parameter_groups(
        module, BucketingPolicy(suggested_bucket_size=None), meta_device_init_fp8_params={}
    )

    assert _is_expert_parallel_parameter("engram.embedding.weight", module.engram.embedding.weight)
    assert len(parameter_groups) == 1
    assert parameter_groups[0].is_expert_param


def test_engram_tables_use_independent_expert_fsdp_buckets():
    """Engram payload cannot land entirely beside unrelated MoE weights on one expert-DP rank."""
    module = torch.nn.Module()
    module.layer = torch.nn.Module()
    module.layer.experts = torch.nn.ParameterDict(
        {"weight": torch.nn.Parameter(torch.empty(16, 4))}
    )
    module.layer.experts.weight.allreduce = False
    module.engram = torch.nn.Module()
    module.engram.embedding = torch.nn.Module()
    module.engram.embedding.tables = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.empty(16, 4)) for _ in range(2)]
    )
    for table in module.engram.embedding.tables:
        table.allreduce = False
        table.is_engram_embedding = True

    parameter_groups, param_to_group, _ = _get_parameter_groups(
        module,
        BucketingPolicy(
            suggested_bucket_size=None,
            data_parallel_sharding_strategy="optim_grads_params",
        ),
        meta_device_init_fp8_params={},
    )

    expert_weight_group = param_to_group[module.layer.experts.weight]
    table_groups = [param_to_group[table] for table in module.engram.embedding.tables]
    assert len(set([expert_weight_group, *table_groups])) == 3
    assert all(parameter_groups[group_id].is_expert_param for group_id in table_groups)
    assert all(len(parameter_groups[group_id].params) == 1 for group_id in table_groups)
