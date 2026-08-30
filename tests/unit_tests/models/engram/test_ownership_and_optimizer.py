# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.distributed_embedding import (
    EPShardedEmbeddingTable,
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from megatron.core.optimizer import (
    OptimizerConfig,
    ParamKey,
    _group_param_groups_by_optimizer,
    get_engram_config_overrides,
)
from megatron.training.utils import get_pipeline_prefetched_tokens, prepare_tokens_for_pipeline

from ._test_utils import make_module_config


def test_uneven_row_ownership_exactly_covers_global_table():
    ranges = [get_contiguous_row_range(11, rank, 4) for rank in range(4)]
    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 11)]
    assert [row for start, end in ranges for row in range(start, end)] == list(range(11))


def test_multi_table_checkpoint_keeps_global_logical_shape():
    embedding = EPShardedMultiTableEmbedding(
        config=make_module_config(dtype=torch.float32),
        table_sizes=(11, 13),
        embedding_dim=3,
        init_method=lambda _: None,
    )
    state = embedding.sharded_state_dict()
    assert state["tables.0.weight"].global_shape == (11, 3)
    assert state["tables.1.weight"].global_shape == (13, 3)


def test_deterministic_embedding_backward_accumulates_repeated_rows():
    table = EPShardedEmbeddingTable(
        config=make_module_config(dtype=torch.float32, deterministic_mode=True),
        global_num_embeddings=7,
        embedding_dim=3,
        init_method=torch.nn.init.zeros_,
    )
    row_ids = torch.tensor([2, 1, 2, 5, 2, 1], dtype=torch.int64)
    output_grad = torch.arange(18, dtype=torch.float32).view(6, 3)

    table(row_ids).backward(output_grad)

    expected = torch.zeros_like(table.weight)
    for row_id, row_grad in zip(row_ids, output_grad):
        expected[row_id] += row_grad
    torch.testing.assert_close(table.weight.grad, expected, rtol=0, atol=0)


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


def test_standard_optimizer_groups_honor_sparse_adam_override():
    dense_group = {"params": [torch.nn.Parameter(torch.zeros(1))]}
    sparse_group = {"params": [torch.nn.Parameter(torch.zeros(1))], "optimizer": "adam"}
    grouped = _group_param_groups_by_optimizer([dense_group, sparse_group], "sgd")
    assert grouped == {"sgd": [dense_group], "adam": [sparse_group]}


def test_pp_token_prefetch_replays_source_batches_in_order(monkeypatch):
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 1)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)
    batches = [
        {"tokens": torch.arange(8).view(2, 4), "id": 0},
        {"tokens": torch.arange(8, 16).view(2, 4), "id": 1},
    ]
    iterator = prepare_tokens_for_pipeline(iter(batches), 2, 2, 4, object(), object())

    assert next(iterator) is batches[0]
    first_tokens = get_pipeline_prefetched_tokens(iterator)
    torch.testing.assert_close(first_tokens, batches[0]["tokens"].to(first_tokens.device))
    assert next(iterator) is batches[1]
    second_tokens = get_pipeline_prefetched_tokens(iterator)
    torch.testing.assert_close(second_tokens, batches[1]["tokens"].to(second_tokens.device))
    with pytest.raises(RuntimeError, match="queue is exhausted"):
        get_pipeline_prefetched_tokens(iterator)

    malformed = iter([{"tokens": torch.arange(8).view(2, 4)}])
    with pytest.raises(ValueError, match="expected token shape"):
        prepare_tokens_for_pipeline(malformed, 1, 1, 8, object(), object())


def test_pp_token_prefetch_accepts_vanilla_multimodal_batch(monkeypatch):
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 1)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)
    batch = [
        {"input_ids": torch.tensor([10, 11, 12, 13]), "sample_id": 0},
        {"input_ids": torch.tensor([20, 21, 22, 23]), "sample_id": 1},
    ]

    iterator = prepare_tokens_for_pipeline(iter([batch]), 1, 2, 4, object(), object())

    assert next(iterator) is batch
    tokens = get_pipeline_prefetched_tokens(iterator)
    torch.testing.assert_close(tokens.cpu(), torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]))


def test_pp_token_prefetch_rejects_variable_length_multimodal_batch(monkeypatch):
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 1)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)
    batch = [
        {"input_ids": torch.tensor([10, 11, 12])},
        {"input_ids": torch.tensor([20, 21])},
    ]

    with pytest.raises(ValueError, match="fixed-length input_ids"):
        prepare_tokens_for_pipeline(iter([batch]), 1, 2, 3, object(), object())
