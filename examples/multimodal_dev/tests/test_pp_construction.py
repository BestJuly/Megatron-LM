# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""PP (pipeline parallelism) construction + forward unit test for Qwen3.5-VL.

Validates the PP plumbing added in `lit/qwen35_dev_pp_support`:

  1. ``factory.build_model(pre_process, post_process)`` builds the right
     parts on each PP rank (vision encoder only on first stage).
  2. ``MultimodalModel.pre_process`` / ``post_process`` flags and the
     underlying ``GPTModel`` flags match the PP rank.
  3. A single forward pass through the model on each PP rank produces
     finite tensors of the expected shape.

Two-rank PP test, launched via torchrun on 2 GPUs::

    torchrun --nproc_per_node=2 \\
        examples/multimodal_dev/tests/test_pp_construction.py
"""

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../.."),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    get_qwen35_vl_language_config,
    get_qwen35_vl_vision_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import build_model, post_language_config
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils

# ===================================================================
# Tiny config — single transformer layer per PP stage, no MoE.
# ===================================================================

def _tiny_args(pp_size: int):
    """Build an argparse Namespace mimicking the bits of Megatron args
    that ``factory.build_model`` consults.
    """
    return argparse.Namespace(
        padded_vocab_size=512,
        max_position_embeddings=128,
        image_token_id=248056,
        mtp_num_layers=0,
        transformer_impl="transformer_engine",
        untie_embeddings_and_output_weights=True,
    )


def _tiny_language_config(pp_size: int):
    """Tiny dense language config with NUM_LAYERS == pp_size so each PP
    rank lands exactly one decoder layer.  No MoE, no GatedDeltaNet (pure
    self-attention) so we don't depend on EP.
    """
    cfg = get_qwen35_vl_language_config(
        variant="proxy",
        # Architecture — small enough to fit comfortably in 80GB H100.
        num_layers=pp_size,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        # Drop hybrid attention and MoE — we just want a vanilla decoder.
        experimental_attention_variant=None,
        linear_attention_freq=0,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        # No MoE
        num_moe_experts=None,
        moe_router_topk=None,
        moe_ffn_hidden_size=None,
        moe_shared_expert_intermediate_size=None,
        # Parallelism (set by ``Utils.initialize_model_parallel``).
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        pipeline_model_parallel_size=pp_size,
        # Misc
        bf16=True,
        pipeline_dtype=torch.bfloat16,
    )
    return cfg


def _tiny_vision_config():
    cfg = get_qwen35_vl_vision_config(num_layers_override=2, variant=None)
    cfg.bf16 = True
    return cfg


# ===================================================================
# Test 1: Construction structure (PP=2)
# ===================================================================


def _check_construction(expected_pp_size: int):
    """Verify PP model construction places the vision encoder on rank 0
    only, and that ``pre_process`` / ``post_process`` flags match.

    Returns the constructed model and language config so the caller can
    feed them into the forward-pass check.
    """
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    assert pp_size == expected_pp_size

    args = _tiny_args(pp_size)
    lcfg = _tiny_language_config(pp_size)
    post_language_config(lcfg, args)
    vcfg = _tiny_vision_config()

    pre_process = pp_rank == 0
    post_process = pp_rank == pp_size - 1
    model = build_model(
        args=args,
        language_config=lcfg,
        vision_config=vcfg,
        pre_process=pre_process,
        post_process=post_process,
    )
    # In real PP training the framework wraps with ``Float16Module``;
    # here we cast directly so TE attention sees bf16 params matching
    # the bf16 activations that pass between stages.
    model = model.cuda().bfloat16()

    # Vision encoder on rank 0 only
    if pp_rank == 0:
        assert model.vision_model is not None, (
            "vision_model must be built on PP rank 0"
        )
    else:
        assert model.vision_model is None, (
            f"vision_model must be None on PP rank {pp_rank}"
        )

    # MultimodalModel-side flags
    assert model.pre_process == pre_process
    assert model.post_process == post_process

    # GPTModel-side flags
    assert model.language_model.pre_process == pre_process
    assert model.language_model.post_process == post_process

    return model, lcfg


# ===================================================================
# Test 2: Single forward pass on each rank
# ===================================================================


def _check_forward(model, lcfg):
    """Run a forward pass on each PP rank.  We bypass the PP scheduler
    by manually calling ``set_input_tensor`` on non-first ranks with a
    fake activation, since this test focuses on the per-rank forward
    logic, not the inter-stage send/recv.
    """
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1

    B, S = 2, 16  # small fixed batch
    device = "cuda"
    vocab_size = 512

    torch.manual_seed(123)
    input_ids = torch.randint(0, vocab_size, (B, S), device=device)
    labels = torch.randint(0, vocab_size, (B, S), device=device)
    loss_mask = torch.ones(B, S, device=device)
    image_grid_thw = None  # text-only batch
    pixel_values = None

    if is_first and is_last:
        # PP=1 case — single-stage model handles input → loss internally.
        out = model(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            packed_seq_params=None,
        )
        assert out is not None
        assert torch.isfinite(out).all(), "PP=1 loss contains NaN/Inf"
        torch.cuda.synchronize()
        return out.detach()

    if is_first:
        # First stage: model builds decoder_input from input_ids.
        # Output is the hidden state to send to the next stage.
        out = model(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            labels=None,  # never compute loss on non-last stages
            loss_mask=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            packed_seq_params=None,
        )
        assert out is not None, "first stage should produce hidden states"
        assert out.dim() == 3, (
            f"expected [S, B, H] hidden states, got {tuple(out.shape)}"
        )
        assert out.shape[0] == S, (
            f"first dim should be S={S}, got {out.shape[0]}"
        )
        assert out.shape[1] == B, (
            f"second dim should be B={B}, got {out.shape[1]}"
        )
        assert torch.isfinite(out).all(), (
            "first-stage hidden states contain NaN/Inf"
        )
        torch.cuda.synchronize()
        return out.detach()

    # Last (or middle) stage: feed a synthetic activation.
    H = lcfg.hidden_size
    fake_act = torch.randn(S, B, H, device=device, dtype=torch.bfloat16)
    model.set_input_tensor(fake_act)
    out = model(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=None,
        labels=labels if is_last else None,
        loss_mask=loss_mask if is_last else None,
        pixel_values=None,
        image_grid_thw=None,
        packed_seq_params=None,
    )
    assert out is not None, "non-first stage should produce a tensor"
    assert torch.isfinite(out).all(), "non-first-stage output contains NaN/Inf"
    torch.cuda.synchronize()
    return out.detach()


# ===================================================================
# Main — orchestrate setup and run tests.
# ===================================================================


def _print_banner(title):
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="PP construction + forward test")
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pp_size = args.pp
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
    )
    model_parallel_cuda_manual_seed(args.seed)

    rank = parallel_state.get_pipeline_model_parallel_rank()

    _print_banner(f"Test 1 — PP={pp_size} construction")
    model, lcfg = _check_construction(expected_pp_size=pp_size)
    print(f"  [pp_rank={rank}] construction PASS")

    _print_banner(f"Test 2 — PP={pp_size} forward pass")
    out = _check_forward(model, lcfg)
    print(f"  [pp_rank={rank}] forward output shape={tuple(out.shape)} PASS")

    Utils.destroy_model_parallel()
    if rank == 0:
        print("\n  ALL TESTS PASSED\n")


if __name__ == "__main__":
    main()
