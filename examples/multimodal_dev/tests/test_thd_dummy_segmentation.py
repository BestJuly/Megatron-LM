# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU numerical parity for real rows when only static dummy boundaries change.

Run with torchrun --nproc-per-node 1 -m pytest --experimental -s <this file>.
"""

import json

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams, build_static_thd_metadata
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    _roll_tensor_packed_seq,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def distributed():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    yield
    MTPLossLoggingHelper.tracker = {}
    Utils.destroy_model_parallel()


def _metadata(segment_length, *, target=1024, slots=16):
    cu = torch.tensor([0, 63, 192], dtype=torch.int32, device="cuda")
    q, padded, real = build_static_thd_metadata(
        cu, cu.clone(), target_len=target, max_num_seqs=slots, dummy_seq_length=segment_length
    )
    assert torch.equal(real, cu)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=q,
        cu_seqlens_kv=q,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        max_seqlen_q=target,
        max_seqlen_kv=target,
        total_tokens=target,
        pad_between_seqs=False,
    )


def _config(**overrides):
    options = dict(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        ffn_hidden_size=512,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        normalization="RMSNorm",
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
    )
    options.update(overrides)
    return TransformerConfig(**options)


def _check(before, after, label):
    torch.testing.assert_close(after, before, atol=5e-4, rtol=1e-2, msg=label)
    a, b = after.float(), before.float()
    print(
        "DUMMY_PARITY "
        + json.dumps(
            {
                "tensor": label,
                "max_abs": (a - b).abs().max().item(),
                "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
            }
        ),
        flush=True,
    )


def _gradients(model):
    return {
        name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None
    }


@pytest.mark.parametrize("slots", [4, 16])
def test_gdn_real_output_and_all_gradients(slots):
    config = _config(
        num_layers=1,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        activation_func=F.silu,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )
    groups = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    model = (
        GatedDeltaNet(
            config,
            submodules=get_gated_delta_net_module_spec(config).submodules,
            layer_number=1,
            pg_collection=groups,
        )
        .cuda()
        .bfloat16()
    )
    hidden = torch.randn(1024, 1, 256, device="cuda", dtype=torch.bfloat16)
    probe = torch.randn(192, 1, 256, device="cuda") / 192
    results = []
    for segment in (None, 128):
        model.zero_grad(set_to_none=True)
        x = hidden.detach().clone().requires_grad_(True)
        output, _ = model(x, None, packed_seq_params=_metadata(segment, slots=slots))
        (output[:192].float() * probe).sum().backward()
        results.append((output[:192].detach(), x.grad.detach(), _gradients(model)))
    _check(results[0][0], results[1][0], "gdn.real_output")
    _check(results[0][1], results[1][1], "gdn.input_gradient")
    assert results[0][2].keys() == results[1][2].keys()
    assert results[0][2]
    for name in results[0][2]:
        _check(results[0][2][name], results[1][2][name], f"gdn.{name}.gradient")


@pytest.mark.parametrize("experts", [None, 4])
def test_gpt_attention_and_mtp_loss_and_parameter_gradients(experts):
    config = _config(
        mtp_num_layers=1,
        calculate_per_token_loss=True,
        apply_rope_fusion=True,
        num_moe_experts=experts,
        moe_router_topk=2,
        moe_aux_loss_coeff=1e-3 if experts else 0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
    )
    layer = get_gpt_layer_with_transformer_engine_spec(num_experts=experts)
    model = (
        GPTModel(
            config=config,
            transformer_layer_spec=layer,
            mtp_block_spec=get_gpt_mtp_block_spec(config, layer, use_transformer_engine=True),
            vocab_size=1024,
            max_sequence_length=1024,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
        )
        .cuda()
        .bfloat16()
    )
    tokens = torch.randint(0, 1024, (1, 1024), device="cuda")
    labels = torch.randint(0, 1024, (1, 1024), device="cuda")
    positions = torch.arange(1024, device="cuda").unsqueeze(0)
    mask = torch.zeros(1, 1024, device="cuda")
    mask[:, :192] = 1
    results = []
    for segment in (None, 128):
        model.zero_grad(set_to_none=True)
        MTPLossLoggingHelper.tracker = {}
        loss = model(
            tokens,
            positions,
            None,
            labels=labels,
            loss_mask=mask,
            padding_mask=~mask.bool(),
            packed_seq_params=_metadata(segment),
        )
        (loss.float() * mask).sum().div(mask.sum()).backward()
        results.append((loss[:, :192].detach(), _gradients(model)))
    _check(results[0][0], results[1][0], "gpt.real_primary_loss")
    assert results[0][1].keys() == results[1][1].keys()
    assert any("mtp" in name for name in results[0][1])
    for name in results[0][1]:
        _check(results[0][1][name], results[1][1][name], f"gpt.{name}.gradient")


def test_mtp_shifts_preserve_labels_and_loss_masks():
    labels = torch.full((1, 1024), -100, device="cuda", dtype=torch.long)
    labels[:, :192] = torch.arange(192, device="cuda")
    mask = (labels != -100).float()
    baseline = [labels.clone(), mask.clone()]
    segmented = [labels.clone(), mask.clone()]
    for _ in range(3):
        for index, fill in enumerate((-100, 0)):
            baseline[index], _ = _roll_tensor_packed_seq(
                baseline[index], -1, -1, _metadata(None), fill_value=fill
            )
            segmented[index], _ = _roll_tensor_packed_seq(
                segmented[index], -1, -1, _metadata(128), fill_value=fill
            )
            assert torch.equal(baseline[index], segmented[index])
