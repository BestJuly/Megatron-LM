#!/usr/bin/env python3
"""Forward-only parity test for Kimi K2.5 VL (Bridge side).

Constructs the SAME deterministic batch as parity_forward_mimo.py,
loads the Bridge checkpoint via Bridge's model construction, and
runs a single forward pass. Prints loss and logits for comparison.

Usage (from Megatron-Bridge repo root):
    export PYTHONPATH=src:3rdparty/Megatron-LM
    torchrun --nproc_per_node=8 /path/to/parity_forward_bridge.py \
        --ckpt /path/to/kimi-test \
        --hf-model-path moonshotai/Kimi-K2.5
"""

import argparse
import os
import sys

import torch


def make_deterministic_batch(
    batch_size: int = 1,
    seq_len: int = 2048,
    image_token_id: int = 163605,
    vocab_size: int = 163840,
    device: str = "cuda",
):
    """Create a deterministic batch — IDENTICAL to parity_forward_mimo.py."""
    MERGED_PATCHES = 64
    TOTAL_RAW_PATCHES = 256
    PATCH_SIZE = 14
    IN_CHANNELS = 3

    image_part = torch.full((MERGED_PATCHES,), image_token_id, dtype=torch.long)
    text_part = torch.arange(1, seq_len - MERGED_PATCHES + 1, dtype=torch.long) % (image_token_id - 1) + 1
    single_ids = torch.cat([image_part, text_part])
    input_ids = single_ids.unsqueeze(0).expand(batch_size, -1).contiguous().to(device)

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = 0
    labels[input_ids == image_token_id] = -100

    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)
    loss_mask[input_ids == image_token_id] = 0.0

    per_patch_dim = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 588
    total_patches = batch_size * TOTAL_RAW_PATCHES
    # Flat for MIMO; Bridge model reshapes internally
    pixel_values_flat = (
        torch.arange(total_patches * per_patch_dim, dtype=torch.float32, device=device)
        .reshape(total_patches, per_patch_dim)
        * 1e-4
    )
    # Bridge MoonViT3d expects (N, C, H, W)
    pixel_values = pixel_values_flat.reshape(total_patches, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)

    grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long, device=device).expand(batch_size, -1).contiguous()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Bridge checkpoint path")
    parser.add_argument("--hf-model-path", default="moonshotai/Kimi-K2.5")
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    # --- Distributed init ---
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
    )
    model_parallel_cuda_manual_seed(1234)

    if rank == 0:
        print("=" * 70)
        print("  Bridge Parity Forward Test")
        print("=" * 70)

    # --- Build model via Bridge AutoBridge ---
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    # Override to proxy-size to match checkpoint
    provider.hidden_size = 7168
    provider.ffn_hidden_size = 1024
    provider.num_moe_experts = 16
    provider.moe_ffn_hidden_size = 64
    provider.num_layers = 4
    provider.seq_length = args.seq_len
    # moe_layer_freq must match num_layers (first dense, rest MoE)
    provider.moe_layer_freq = [0, 1, 1, 1]

    # Parallelism
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 8
    provider.sequence_parallel = False
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16

    provider.finalize()

    model = provider.provide(pre_process=True, post_process=True)
    model = model.cuda().bfloat16()

    # --- Load checkpoint ---
    from megatron.core import dist_checkpointing

    ckpt_dir = args.ckpt
    latest_file = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(latest_file):
        with open(latest_file) as f:
            iteration = f.read().strip()
        ckpt_dir = os.path.join(ckpt_dir, f"iter_{int(iteration):07d}")

    sharded_sd = model.sharded_state_dict(prefix="")
    for k in list(sharded_sd.keys()):
        if "extra_state" in k:
            del sharded_sd[k]

    loaded = dist_checkpointing.load({"state_dict": sharded_sd}, checkpoint_dir=ckpt_dir)
    model.load_state_dict(loaded["state_dict"], strict=False)
    if rank == 0:
        print(f"Checkpoint loaded from {ckpt_dir}")

    # --- Deterministic batch ---
    batch = make_deterministic_batch(
        batch_size=1,
        seq_len=args.seq_len,
        device="cuda",
    )

    # --- Forward pass (no grad) ---
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
        )

    # --- Report ---
    if rank == 0:
        if isinstance(output, tuple):
            output = output[0]
        print(f"\nOutput shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")

        if output.dim() == 0:
            print(f"Loss (scalar): {output.item():.6f}")
        else:
            loss_flat = output.float().view(-1)
            lm = batch["loss_mask"].float().view(-1)
            total_loss = (loss_flat * lm).sum()
            total_tokens = lm.sum()
            avg_loss = total_loss / total_tokens if total_tokens > 0 else total_loss
            print(f"Avg lm loss: {avg_loss.item():.6f}")
            print(f"Total tokens: {total_tokens.item():.0f}")
            print(f"First 10 per-token losses: {loss_flat[:10].tolist()}")
            print(f"Last 10 per-token losses:  {loss_flat[-10:].tolist()}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
