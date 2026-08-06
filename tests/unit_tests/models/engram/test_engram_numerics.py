# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.engram import Engram
from megatron.core.models.engram.hashing import build_ngram_hashes

from ._test_utils import make_module_config, make_pg_collection, write_tokenizer_map


def _official_reference(module: Engram, hidden_states: torch.Tensor, input_ids: torch.Tensor):
    hashes = build_ngram_hashes(
        input_ids,
        module.tokenizer_remap,
        module.hash_multipliers,
        module.table_sizes,
        module.engram_config.max_ngram_order,
        module.engram_config.num_hash_heads,
        module.engram_config.compressed_pad_token_id,
    )
    embeddings = torch.cat(
        [
            F.embedding(hashes[..., table_id], table.weight)
            for table_id, table in enumerate(module.embedding.tables)
        ],
        dim=-1,
    )
    streams = hidden_states.transpose(0, 1).view(
        input_ids.shape[0], input_ids.shape[1], module.num_streams, module.hidden_size
    )
    gates = []
    for stream_index in range(module.num_streams):
        key = module.key_norms[stream_index](module.key_projections[stream_index](embeddings))
        query = module.query_norms[stream_index](streams[:, :, stream_index])
        score = (key * query).sum(dim=-1) / math.sqrt(module.hidden_size)
        score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
        gates.append(score.sigmoid())
    gate = torch.stack(gates, dim=2).unsqueeze(-1)
    value = gate * module.value_projection(embeddings).unsqueeze(2)
    normed = torch.cat(
        [module.conv_norms[index](value[:, :, index]) for index in range(module.num_streams)],
        dim=-1,
    )
    convolved = F.conv1d(
        normed.transpose(1, 2),
        module.short_conv.weight,
        padding=module.short_conv.padding,
        dilation=module.short_conv.dilation,
        groups=module.short_conv.groups,
    )[..., : input_ids.shape[1]]
    output = value + F.silu(convolved).transpose(1, 2).view_as(value)
    return output.view(input_ids.shape[0], input_ids.shape[1], -1).transpose(0, 1)


@pytest.mark.parametrize("num_streams", [1, 4])
def test_official_forward_and_backward_math(tmp_path, num_streams):
    torch.manual_seed(123)
    artifact = write_tokenizer_map(tmp_path / "map.json", vocab_size=32, layer_ids=(1,))
    config = EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        pad_token_id=0,
        tokenizer_map_path=str(artifact),
    )
    module = Engram(
        config=make_module_config(num_streams=num_streams),
        engram_config=config,
        layer_number=1,
        pg_collection=make_pg_collection(),
    )
    assert torch.count_nonzero(module.short_conv.weight) == 0
    with torch.no_grad():
        module.short_conv.weight.normal_(mean=0.0, std=0.03)

    input_ids = torch.tensor([[1, 4, 2, 7], [3, 5, 9, 2]], dtype=torch.int64)
    hidden = torch.randn(4, 2, 8 * num_streams, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    actual = module(hidden, input_ids)
    expected = _official_reference(module, reference_hidden, input_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    parameters = [hidden, *module.parameters()]
    reference_parameters = [reference_hidden, *module.parameters()]
    projection = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * projection).sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(
        (expected * projection).sum(), reference_parameters, retain_graph=True
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-9)
    assert module.embedding.tables[0].weight.is_engram_embedding
    assert not module.embedding.tables[0].weight.allreduce
