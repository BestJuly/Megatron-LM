# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram under context parallelism on the multimodal per-sample padded THD layout.

multimodal_dev packs every sample padded individually: ``cu_seqlens_q`` keeps the real
lengths, ``cu_seqlens_q_padded`` is the layout (each padded length a multiple of
``2 * cp``), the padding slots carry the pad token, and ``padding_mask`` (True at padding)
is CP-local after the model's split. The first ``cp_size`` ranks form one CP group; each
rank forwards its per-document zigzag shard of the pack. The outputs must equal the
full-pack (CP=1) reference at the shard's rows -- n-gram windows reset at document
starts, the causal convolution crosses ranks, padding stays out of the lookup -- and the
CP-summed parameter gradients must equal the reference gradients.
"""

import os

import pytest
import torch

from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.cp_layout import thd_partition_index
from megatron.core.models.engram.engram import Engram
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.test_utilities import Utils

from ._test_utils import make_module_config, write_tokenizer_map

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8),
    reason="requires torchrun with two, four, or eight ranks",
)

_HIDDEN_SIZE = 8
_PAD = 0
# Real sample lengths; each is padded to a multiple of 2 * cp (8 covers cp 2 and 4).
_REAL_LENGTHS = (13, 30, 5, 21)
_ALIGN = 8


def _build_module(tmp_path, cp_group):
    from megatron.core.process_groups_config import ProcessGroupCollection

    artifact = tmp_path / "map.json"
    if not artifact.exists():
        write_tokenizer_map(artifact, vocab_size=32, layer_ids=(1,), pad_token_id=_PAD)
    engram_config = EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=_PAD,
        tokenizer_map_path=str(artifact),
    )
    torch.manual_seed(2026)
    module = Engram(
        config=make_module_config(),
        engram_config=engram_config,
        layer_number=1,
        pg_collection=ProcessGroupCollection(cp=cp_group),
    )
    with torch.no_grad():
        # A nonzero convolution branch so the cross-rank convolution and its gradients matter.
        module.short_conv.weight.normal_(mean=0.0, std=0.05)
    return module.to(torch.cuda.current_device())


def _layout(device):
    padded = [(n + _ALIGN - 1) // _ALIGN * _ALIGN for n in _REAL_LENGTHS]
    real_cu = torch.tensor([0, *torch.tensor(_REAL_LENGTHS).cumsum(0).tolist()])
    padded_cu = torch.tensor([0, *torch.tensor(padded).cumsum(0).tolist()])
    total = int(padded_cu[-1])
    padding = torch.ones(total, dtype=torch.bool)
    for start, n in zip(padded_cu[:-1].tolist(), _REAL_LENGTHS):
        padding[start : start + n] = False
    return (
        real_cu.to(device=device, dtype=torch.int32),
        padded_cu.to(device=device, dtype=torch.int32),
        padding.to(device),
        total,
    )


def _params(real_cu, padded_cu, total, padding_mask):
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real_cu,
        cu_seqlens_kv=real_cu,
        cu_seqlens_q_padded=padded_cu,
        cu_seqlens_kv_padded=padded_cu,
        max_seqlen_q=total,
        max_seqlen_kv=total,
        padding_mask=padding_mask.view(1, -1),
    )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_shards_match_the_full_pack(tmp_path_factory, cp_size):
    Utils.initialize_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size < cp_size:
        pytest.skip(f"requires at least {cp_size} ranks")
    # Collective on the default group: every rank must participate in group creation.
    cp_group = torch.distributed.new_group(ranks=list(range(cp_size)))
    single = [torch.distributed.new_group(ranks=[r]) for r in range(world_size)]
    if rank >= cp_size:
        return
    device = torch.device("cuda", torch.cuda.current_device())
    tmp_path = tmp_path_factory.getbasetemp()

    cp_module = _build_module(tmp_path, cp_group)
    reference = _build_module(tmp_path, single[rank])
    for a, b in zip(cp_module.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b)

    real_cu, padded_cu, padding, total = _layout(device)
    generator = torch.Generator(device="cpu").manual_seed(11)
    tokens = torch.randint(1, 32, (1, total), generator=generator, dtype=torch.int64)
    tokens[0, padding.cpu()] = _PAD
    tokens = tokens.to(device)
    hidden = torch.randn(total, 1, _HIDDEN_SIZE, generator=generator, dtype=torch.float64)
    projection = torch.randn(total, 1, _HIDDEN_SIZE, generator=generator, dtype=torch.float64)
    hidden, projection = hidden.to(device), projection.to(device)

    ref_out = reference(hidden, tokens, packed_seq_params=_params(real_cu, padded_cu, total, padding))
    (ref_out * projection).sum().backward()

    rows = thd_partition_index(padded_cu, cp_size, rank)
    local_params = _params(real_cu, padded_cu, total, padding.index_select(0, rows))
    out = cp_module(
        hidden.index_select(0, rows), tokens.index_select(1, rows), packed_seq_params=local_params
    )
    torch.testing.assert_close(out, ref_out.detach().index_select(0, rows))
    (out * projection.index_select(0, rows)).sum().backward()

    for (name, param), ref_param in zip(cp_module.named_parameters(), reference.parameters()):
        assert param.grad is not None, name
        grad = param.grad.clone()
        # The data-parallel reduction runs over dp x cp: CP shards sum to the full-pack grad.
        torch.distributed.all_reduce(grad, group=cp_group)
        torch.testing.assert_close(grad, ref_param.grad, msg=name)

