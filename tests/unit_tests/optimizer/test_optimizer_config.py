# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch

from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import ParamKey, ParamPredicate


def test_paramkey_matches():
    len_1_predicate = ParamPredicate(name="param_len_1", fn=lambda param: len(param.shape) == 1)
    endswith_bias = ParamKey(name="*.bias")
    has_dotbias = ParamKey(name="*.bias*")
    len_1_param = ParamKey(predicate=len_1_predicate)
    has_bias_or_len1_param = ParamKey(name="*.bias", predicate=len_1_predicate)
    has_attr = ParamKey(attr="is_embedding_or_output_parameter")

    assert endswith_bias.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias")
    assert not endswith_bias.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "something.bias.other"
    )
    assert has_dotbias.matches(torch.nn.Parameter(torch.empty(10)), "random.biasstuff")
    assert not has_dotbias.matches(torch.nn.Parameter(torch.empty(10, 10)), "random_bias_name")
    assert len_1_param.matches(torch.nn.Parameter(torch.empty(10)), "interesting.bias")
    assert not len_1_param.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting_bias")
    assert has_bias_or_len1_param.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias"
    )
    assert has_bias_or_len1_param.matches(torch.nn.Parameter(torch.empty(10)), "interesting_bias")
    assert not has_bias_or_len1_param.matches(
        torch.nn.Parameter(torch.empty(10, 10)), "random_bias_name"
    )
    p_with_attr = torch.nn.Parameter(torch.empty(10, 10))
    setattr(p_with_attr, "is_embedding_or_output_parameter", True)
    assert has_attr.matches(p_with_attr, "interesting.bias")
    assert not has_attr.matches(torch.nn.Parameter(torch.empty(10, 10)), "interesting.bias")

    # We expect that if the return of the attribute is False, it should not match even if
    #  it has the attribute.
    setattr(p_with_attr, "is_embedding_or_output_parameter", False)
    assert not has_attr.matches(p_with_attr, "interesting.bias")


def test_filter_and_reorder_param_groups_preserves_duplicate_identifiers():
    """Groups with distinct schedules may share the legacy checkpoint identifier."""
    common = {"wd_mult": 1.0, "lr_mult": 1.0, "is_expert_parallel": True, "is_decoupled_lr": False}
    current_groups = [
        {**common, "params": [object(), object()], "max_lr": 1.0},
        {**common, "params": [object()], "max_lr": 5.0},
    ]
    state_dict_groups = [
        {**common, "params": [10, 11], "max_lr": 1.0},
        {**common, "params": [12], "max_lr": 5.0},
    ]

    restored_groups = MegatronOptimizer._filter_and_reorder_param_groups(
        current_groups, state_dict_groups
    )

    assert [group["params"] for group in restored_groups] == [[10, 11], [12]]
    assert [group["max_lr"] for group in restored_groups] == [1.0, 5.0]
    assert restored_groups[0] is not restored_groups[1]
