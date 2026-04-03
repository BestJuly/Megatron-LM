#!/usr/bin/env python3
"""Forward-only parity test for Kimi K2.5 VL (MIMO side).

Constructs a deterministic batch, loads the Bridge checkpoint, and
runs a single forward pass through the standalone KimiK25VLModel.
Prints loss and first few logit values for comparison with Bridge.

Usage:
    torchrun --nproc_per_node=8 examples/mimo/scripts/parity_forward_mimo.py \
        --ckpt /path/to/kimi-test
"""

import argparse
import os
import sys

import torch

# Add repo root and examples/mimo to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", "..", ".."))
sys.path.insert(0, os.path.join(_script_dir, ".."))


def make_deterministic_batch(
    batch_size: int = 1,
    seq_len: int = 2048,
    image_token_id: int = 163605,
    vocab_size: int = 163840,
    device: str = "cuda",
):
    """Create a deterministic batch identical across Bridge and MIMO.

    Image geometry: MoonViT3d Conv2d patch_size=14, 224px image
      grid_thw = [1, 16, 16], total_patches = 256, merged = 64
      pixel_values: (B*256, 3, 14, 14) -> flat (B*256, 588)
    """
    MERGED_PATCHES = 64
    TOTAL_RAW_PATCHES = 256
    PATCH_SIZE = 14
    IN_CHANNELS = 3

    # Deterministic input_ids: image placeholders + sequential text tokens
    image_part = torch.full((MERGED_PATCHES,), image_token_id, dtype=torch.long)
    text_part = torch.arange(1, seq_len - MERGED_PATCHES + 1, dtype=torch.long) % (image_token_id - 1) + 1
    single_ids = torch.cat([image_part, text_part])
    input_ids = single_ids.unsqueeze(0).expand(batch_size, -1).contiguous().to(device)

    # Labels: shifted by 1
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = 0
    labels[input_ids == image_token_id] = -100

    # Loss mask
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)
    loss_mask[input_ids == image_token_id] = 0.0

    # Deterministic pixel values: use arange pattern, flat for TP broadcast
    per_patch_dim = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 588
    total_patches = batch_size * TOTAL_RAW_PATCHES
    pixel_values = (
        torch.arange(total_patches * per_patch_dim, dtype=torch.float32, device=device)
        .reshape(total_patches, per_patch_dim)
        * 1e-4  # small scale to avoid numerical issues
    )

    # Grid THW
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
    parser.add_argument("--variant", default="proxy")
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
        print("  MIMO Parity Forward Test")
        print("=" * 70)

    # --- Build model ---
    from model_providers.kimi_k25 import get_kimi_k25_language_config, KIMI_K25_VOCAB_SIZE, KIMI_K25_IMAGE_TOKEN_ID
    from model_providers.kimi_k25.model import KimiK25VLModel
    from model_providers.kimi_k25.specs import get_kimi_k25_language_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    language_config = get_kimi_k25_language_config(variant=args.variant)
    language_config.tensor_model_parallel_size = 1
    language_config.pipeline_model_parallel_size = 1
    language_config.expert_model_parallel_size = 8
    language_config.sequence_parallel = False
    language_config.bf16 = True

    layer_spec = get_kimi_k25_language_spec(language_config)

    language_model = GPTModel(
        config=language_config,
        transformer_layer_spec=layer_spec,
        vocab_size=KIMI_K25_VOCAB_SIZE,
        max_sequence_length=args.seq_len,
        pre_process=True,
        post_process=True,
        position_embedding_type="rope",
    )

    model = KimiK25VLModel(
        language_model=language_model,
        hf_model_path=args.hf_model_path,
        media_placeholder_token_id=KIMI_K25_IMAGE_TOKEN_ID,
        freeze_vision_model=True,
        freeze_vision_projection=True,
        pre_process=True,
    ).cuda().bfloat16()

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
        image_token_id=KIMI_K25_IMAGE_TOKEN_ID,
        device="cuda",
    )

    # --- Forward pass (no grad) ---
    model.eval()
    with torch.no_grad():
        output, loss_mask_out = model(**batch)

    # --- Report ---
    if rank == 0:
        print(f"\nOutput shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")

        if output.dim() == 0:
            # Loss scalar (when labels are provided)
            print(f"Loss (scalar): {output.item():.6f}")
        else:
            # Per-token losses
            loss_flat = output.float().view(-1)
            lm = loss_mask_out.float().view(-1) if loss_mask_out is not None else torch.ones_like(loss_flat)
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
