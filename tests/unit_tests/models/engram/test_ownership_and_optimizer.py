# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.distributed_embedding import get_contiguous_row_range
from megatron.core.optimizer import OptimizerConfig, ParamKey, get_engram_config_overrides
from megatron.training.utils import broadcast_tokens_across_pipeline


def test_uneven_row_ownership_exactly_covers_global_table():
    ranges = [get_contiguous_row_range(11, rank, 4) for rank in range(4)]
    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 11)]
    assert [row for start, end in ranges for row in range(start, end)] == list(range(11))


def test_engram_optimizer_override_is_sparse_only():
    config = OptimizerConfig(lr=2.0e-4, min_lr=1.0e-5, optimizer="muon")
    overrides = get_engram_config_overrides(config, lr_multiplier=5.0, weight_decay=0.0)
    assert len(overrides) == 1
    key, override = next(iter(overrides.items()))
    assert isinstance(key, ParamKey)
    table = torch.nn.Parameter(torch.ones(2, 2))
    table.is_engram_embedding = True
    dense = torch.nn.Parameter(torch.ones(2, 2))
    assert key.matches(table, "embedding.weight")
    assert not key.matches(dense, "value_projection.weight")
    assert override == {
        "optimizer": "adam",
        "max_lr": pytest.approx(1.0e-3),
        "min_lr": pytest.approx(5.0e-5),
        "start_wd": 0.0,
        "end_wd": 0.0,
        "wd_mult": 1.0,
    }


def test_pp_token_helper_has_no_cached_state(monkeypatch):
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 1)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)
    tokens = torch.arange(8).view(2, 4)
    output = broadcast_tokens_across_pipeline(tokens, 2, 4, object())
    assert output.data_ptr() == tokens.data_ptr()
    assert not hasattr(broadcast_tokens_across_pipeline, "cache")
    with pytest.raises(ValueError, match="expected token shape"):
        broadcast_tokens_across_pipeline(tokens, 1, 8, object())
