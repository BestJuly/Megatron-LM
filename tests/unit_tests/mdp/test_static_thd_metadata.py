# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for static THD dummy segmentation, independent of packing policy."""

import argparse

import pytest
import torch

from examples.multimodal_dev.arguments import add_multimodal_args
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import build_static_thd_metadata
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import add_megatron_arguments


def _build(valid=(0, 5, 12), physical=(0, 6, 14), **kwargs):
    options = dict(target_len=32, max_num_seqs=8)
    options.update(kwargs)
    return build_static_thd_metadata(
        torch.tensor(valid, dtype=torch.int32), torch.tensor(physical, dtype=torch.int32), **options
    )


def test_disabled_preserves_one_dummy_and_real_flops():
    valid, physical, real = _build()
    assert valid.tolist() == [0, 5, 12, 30, 30, 30, 30, 30, 30]
    assert physical.tolist() == [0, 6, 14, 32, 32, 32, 32, 32, 32]
    assert real.tolist() == [0, 5, 12]


def test_balanced_segments_preserve_distinct_coordinate_spaces():
    valid, physical, real = _build(dummy_seq_length=8)
    assert valid.tolist() == [0, 5, 12, 18, 24, 30, 30, 30, 30]
    assert physical.tolist() == [0, 6, 14, 20, 26, 32, 32, 32, 32]
    assert real.tolist() == [0, 5, 12]


@pytest.mark.parametrize("slots,tail_lengths", [(3, [18]), (4, [9, 9]), (5, [6, 6, 6])])
def test_slot_pressure_changes_only_dummy_lengths(slots, tail_lengths):
    valid, physical, real = _build(max_num_seqs=slots, dummy_seq_length=8)
    assert valid[:3].tolist() == [0, 5, 12]
    assert physical[:3].tolist() == [0, 6, 14]
    assert torch.diff(physical)[2:].tolist() == tail_lengths
    assert real.tolist() == [0, 5, 12]
    assert valid.numel() == physical.numel() == slots + 1


@pytest.mark.parametrize("length", [None, 1, 8, 100])
def test_exact_fit_needs_no_dummy_or_flops_override(length):
    valid, physical, real = _build(
        valid=(0, 16, 32), physical=(0, 16, 32), max_num_seqs=2, dummy_seq_length=length
    )
    assert valid.tolist() == physical.tolist() == [0, 16, 32]
    assert real is None


@pytest.mark.parametrize(
    "cp,mode,quantum", [(1, "zigzag", 1), (2, "zigzag", 4), (4, "contiguous", 4)]
)
def test_cp_alignment_and_rounding(cp, mode, quantum):
    valid, physical, real = _build(
        valid=(0, 5, 12),
        physical=(0, 8, 16),
        cp_size=cp,
        cp_partition_mode=mode,
        dummy_seq_length=9,
    )
    assert valid[:3].tolist() == [0, 5, 12]
    assert physical[:3].tolist() == [0, 8, 16]
    tails = torch.diff(physical)[2:]
    assert tails.sum().item() == 16
    assert (tails % quantum == 0).all()
    assert tails.max().item() <= 9
    assert real.tolist() == [0, 5, 12]


@pytest.mark.parametrize("length", [0, -1])
def test_nonpositive_target_is_rejected_even_without_padding(length):
    with pytest.raises(ValueError, match="positive integer"):
        _build(valid=(0, 32), physical=(0, 32), dummy_seq_length=length)


def test_target_below_cp_quantum_is_rejected():
    with pytest.raises(ValueError, match="CP partition alignment"):
        _build(cp_size=4, dummy_seq_length=7)


def test_unaligned_contiguous_tail_is_rejected():
    with pytest.raises(AssertionError, match="CP partition alignment"):
        _build(cp_size=4, cp_partition_mode="contiguous", dummy_seq_length=8)


def test_unknown_cp_partition_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown CP partition mode"):
        _build(cp_size=2, cp_partition_mode="invalid", dummy_seq_length=8)


@pytest.mark.parametrize("length", [None, 8])
def test_no_dummy_slot_is_an_error(length):
    with pytest.raises(AssertionError, match="thd_max_packed_sequences"):
        _build(max_num_seqs=2, dummy_seq_length=length)


def test_input_metadata_is_not_mutated():
    cu = torch.tensor([0, 5, 12], dtype=torch.int32)
    physical = torch.tensor([0, 6, 14], dtype=torch.int32)
    _, _, real = build_static_thd_metadata(
        cu, physical, target_len=32, max_num_seqs=8, dummy_seq_length=8
    )
    assert real is cu
    assert cu.tolist() == [0, 5, 12]
    assert physical.tolist() == [0, 6, 14]


def test_many_layouts_conserve_rows_bound_slots_and_reduce_attention_work():
    generator = torch.Generator().manual_seed(1234)
    for _ in range(100):
        count = int(torch.randint(1, 16, (), generator=generator))
        lengths = torch.randint(1, 30, (count,), generator=generator, dtype=torch.int32)
        cu = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
        tail = int(torch.randint(1, 100, (), generator=generator))
        slots = count + int(torch.randint(1, 12, (), generator=generator))
        valid, physical, real = build_static_thd_metadata(
            cu, cu.clone(), target_len=int(cu[-1]) + tail, max_num_seqs=slots, dummy_seq_length=8
        )
        assert torch.equal(real, cu)
        assert torch.equal(valid, physical)
        assert torch.equal(valid[: count + 1], cu)
        assert valid.numel() == slots + 1
        dummy = torch.diff(valid)[count:]
        assert (dummy >= 0).all()
        assert dummy.sum().item() == tail
        assert dummy.square().sum().item() <= tail**2
        nonempty = dummy[dummy > 0]
        assert nonempty.max() - nonempty.min() <= 1
        if slots - count >= (tail + 7) // 8:
            assert nonempty.max().item() <= 8


@pytest.mark.parametrize("length", [0, -1])
def test_config_rejects_nonpositive_length(length):
    with pytest.raises(ValueError, match="thd_dummy_seq_length must"):
        ModelParallelConfig(thd_dummy_seq_length=length)


def test_config_requires_static_thd():
    with pytest.raises(ValueError, match="requires thd_static_packing"):
        ModelParallelConfig(thd_dummy_seq_length=8)


@pytest.mark.parametrize("policy", [None, "greedy", "ffd"])
def test_cli_accepts_segmentation_independently_of_grouping(policy):
    parser = add_multimodal_args(
        add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    )
    flags = ["--thd-static-packing", "--thd-dummy-seq-length", "8192"]
    if policy:
        flags.append(f"--mdp-{policy}-packing")
    args = parser.parse_args(flags)
    config = ModelParallelConfig(
        thd_static_packing=args.thd_static_packing,
        thd_dummy_seq_length=args.thd_dummy_seq_length,
        max_seqlen_per_dp_cp_rank=131072,
        pad_packed_seq_alignment="max",
    )
    assert config.thd_dummy_seq_length == 8192


@pytest.mark.parametrize(
    "routing",
    [
        {"moe_expert_capacity_factor": 1.0},
        {"moe_expert_rank_capacity_factor": 1.0},
        {"moe_router_load_balancing_type": "sinkhorn"},
    ],
)
def test_segmentation_rejects_moe_routing_that_couples_padding_to_real_tokens(routing):
    with pytest.raises(ValueError, match="dropless, non-Sinkhorn"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="ncclep",
            thd_static_packing=True,
            thd_dummy_seq_length=8,
            max_seqlen_per_dp_cp_rank=1024,
            thd_max_packed_sequences=8,
            pad_packed_seq_alignment="max",
            **routing,
        )
