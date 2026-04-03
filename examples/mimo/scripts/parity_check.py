#!/usr/bin/env python3
"""Bitwise parity check: Bridge vs MIMO Kimi K2.5 VL forward pass.

Runs both models with identical deterministic input and the same
checkpoint, then compares every element of the output tensor.

Saves output tensors to /tmp/ so you can inspect them afterwards.

Usage — Step 1 (Bridge side, from Megatron-Bridge repo root):
    export PYTHONPATH=src:3rdparty/Megatron-LM
    torchrun --nproc_per_node=8 /path/to/parity_check.py \
        --side bridge \
        --ckpt /path/to/kimi-test

  Step 2 (MIMO side, from Megatron-LM/examples/mimo):
    export PYTHONPATH=/path/to/Megatron-LM
    torchrun --nproc_per_node=8 /path/to/parity_check.py \
        --side mimo \
        --ckpt /path/to/kimi-test

  Step 3 (compare, any single process):
    python /path/to/parity_check.py --side compare
"""

import argparse
import os
import sys

import torch

SAVE_DIR = "/tmp/kimi_k25_parity"
BRIDGE_FILE = os.path.join(SAVE_DIR, "bridge_output.pt")
MIMO_FILE = os.path.join(SAVE_DIR, "mimo_output.pt")


# ======================================================================
# Shared: deterministic batch construction
# ======================================================================

def make_deterministic_batch(
    batch_size: int = 1,
    seq_len: int = 2048,
    image_token_id: int = 163605,
    device: str = "cuda",
):
    """Identical deterministic batch for both Bridge and MIMO."""
    MERGED_PATCHES = 64   # MoonViT3d: (16/2)*(16/2) after sd2_tpool
    TOTAL_RAW_PATCHES = 256  # 1*16*16
    PATCH_SIZE = 14
    IN_CHANNELS = 3

    # input_ids: 64 image placeholders + sequential text tokens
    image_part = torch.full((MERGED_PATCHES,), image_token_id, dtype=torch.long)
    text_part = (
        torch.arange(1, seq_len - MERGED_PATCHES + 1, dtype=torch.long)
        % (image_token_id - 1) + 1
    )
    input_ids = torch.cat([image_part, text_part]).unsqueeze(0).expand(batch_size, -1).contiguous().to(device)

    # labels: shifted by 1, image positions = -100
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = 0
    labels[input_ids == image_token_id] = -100

    # loss_mask: 0 for image positions
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)
    loss_mask[input_ids == image_token_id] = 0.0

    # pixel_values: deterministic arange pattern
    per_patch_dim = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 588
    total_patches = batch_size * TOTAL_RAW_PATCHES
    pixel_values_flat = (
        torch.arange(total_patches * per_patch_dim, dtype=torch.float32, device=device)
        .reshape(total_patches, per_patch_dim)
        * 1e-4
    )

    grid_thw = torch.tensor(
        [[1, 16, 16]], dtype=torch.long, device=device
    ).expand(batch_size, -1).contiguous()

    return input_ids, labels, loss_mask, pixel_values_flat, grid_thw


# ======================================================================
# Bridge forward
# ======================================================================

def run_bridge(args):
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=8)
    model_parallel_cuda_manual_seed(1234)

    if rank == 0:
        print("=" * 70)
        print("  [1/2]  Bridge forward")
        print("=" * 70)

    # Build model
    from megatron.bridge import AutoBridge
    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.hidden_size = 7168
    provider.ffn_hidden_size = 1024
    provider.num_moe_experts = 16
    provider.moe_ffn_hidden_size = 64
    provider.num_layers = 4
    provider.seq_length = args.seq_len
    provider.moe_layer_freq = [0, 1, 1, 1]
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 8
    provider.sequence_parallel = False
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()

    model = provider.provide(pre_process=True, post_process=True).cuda().bfloat16()

    # Load checkpoint
    from megatron.core import dist_checkpointing
    ckpt_dir = _resolve_ckpt(args.ckpt)
    sd = model.sharded_state_dict(prefix="")
    for k in list(sd):
        if "extra_state" in k: del sd[k]
    loaded = dist_checkpointing.load({"state_dict": sd}, checkpoint_dir=ckpt_dir)
    model.load_state_dict(loaded["state_dict"], strict=False)
    if rank == 0:
        print(f"  Checkpoint: {ckpt_dir}")

    # Deterministic batch — Bridge expects 4D pixels
    input_ids, labels, loss_mask, pv_flat, grid_thw = make_deterministic_batch(device="cuda", seq_len=args.seq_len)
    pixel_values = pv_flat.reshape(-1, 3, 14, 14)

    # Forward
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            labels=labels,
            loss_mask=loss_mask,
        )

    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save({"output": output.cpu(), "loss_mask": loss_mask.cpu()}, BRIDGE_FILE)
        print(f"  Saved → {BRIDGE_FILE}")
        _print_summary("Bridge", output, loss_mask)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


# ======================================================================
# MIMO forward
# ======================================================================

def run_mimo(args):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=8)
    model_parallel_cuda_manual_seed(1234)

    if rank == 0:
        print("=" * 70)
        print("  [1/2]  MIMO forward")
        print("=" * 70)

    # Build model
    from model_providers.kimi_k25 import get_kimi_k25_language_config, KIMI_K25_VOCAB_SIZE, KIMI_K25_IMAGE_TOKEN_ID
    from model_providers.kimi_k25.model import KimiK25VLModel
    from model_providers.kimi_k25.specs import get_kimi_k25_language_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    cfg = get_kimi_k25_language_config(variant=args.variant)
    cfg.tensor_model_parallel_size = 1
    cfg.pipeline_model_parallel_size = 1
    cfg.expert_model_parallel_size = 8
    cfg.sequence_parallel = False
    cfg.bf16 = True

    lm = GPTModel(
        config=cfg,
        transformer_layer_spec=get_kimi_k25_language_spec(cfg),
        vocab_size=KIMI_K25_VOCAB_SIZE,
        max_sequence_length=args.seq_len,
        pre_process=True, post_process=True,
        position_embedding_type="rope",
    )
    model = KimiK25VLModel(
        language_model=lm,
        hf_model_path=args.hf_model_path,
        media_placeholder_token_id=KIMI_K25_IMAGE_TOKEN_ID,
        freeze_vision_model=True, freeze_vision_projection=True,
        pre_process=True,
    ).cuda().bfloat16()

    # Load checkpoint
    from megatron.core import dist_checkpointing
    ckpt_dir = _resolve_ckpt(args.ckpt)
    sd = model.sharded_state_dict(prefix="")
    for k in list(sd):
        if "extra_state" in k: del sd[k]
    loaded = dist_checkpointing.load({"state_dict": sd}, checkpoint_dir=ckpt_dir)
    model.load_state_dict(loaded["state_dict"], strict=False)
    if rank == 0:
        print(f"  Checkpoint: {ckpt_dir}")

    # Deterministic batch — MIMO gets flat pixels (reshape in model)
    input_ids, labels, loss_mask, pv_flat, grid_thw = make_deterministic_batch(device="cuda", seq_len=args.seq_len)

    # Forward
    model.eval()
    with torch.no_grad():
        output, _ = model(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            pixel_values=pv_flat,
            image_grid_thw=grid_thw,
        )

    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save({"output": output.cpu(), "loss_mask": loss_mask.cpu()}, MIMO_FILE)
        print(f"  Saved → {MIMO_FILE}")
        _print_summary("MIMO", output, loss_mask)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


# ======================================================================
# Compare (no GPU needed)
# ======================================================================

def run_compare():
    print("=" * 70)
    print("  [2/2]  Bitwise Comparison")
    print("=" * 70)

    if not os.path.exists(BRIDGE_FILE):
        print(f"  ERROR: {BRIDGE_FILE} not found. Run --side bridge first.")
        return
    if not os.path.exists(MIMO_FILE):
        print(f"  ERROR: {MIMO_FILE} not found. Run --side mimo first.")
        return

    b = torch.load(BRIDGE_FILE, map_location="cpu")
    m = torch.load(MIMO_FILE, map_location="cpu")

    b_out = b["output"].float()
    m_out = m["output"].float()
    b_lm = b["loss_mask"].float()
    m_lm = m["loss_mask"].float()

    print(f"\n  Bridge output shape: {b_out.shape},  dtype: {b['output'].dtype}")
    print(f"  MIMO   output shape: {m_out.shape},  dtype: {m['output'].dtype}")

    # Shape check
    if b_out.shape != m_out.shape:
        print(f"\n  ❌ SHAPE MISMATCH: Bridge {b_out.shape} vs MIMO {m_out.shape}")
        return

    # Bitwise check
    exact_match = torch.equal(b["output"], m["output"])
    print(f"\n  Bitwise identical (torch.equal): {exact_match}")

    # Element-wise analysis
    diff = (b_out - m_out).abs()
    num_total = diff.numel()
    num_zero = (diff == 0).sum().item()
    num_nonzero = num_total - num_zero
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    print(f"\n  Total elements:       {num_total}")
    print(f"  Exactly matching:     {num_zero}  ({100*num_zero/num_total:.2f}%)")
    print(f"  Non-zero diff:        {num_nonzero}  ({100*num_nonzero/num_total:.2f}%)")
    print(f"  Max  abs diff:        {max_abs_diff:.2e}")
    print(f"  Mean abs diff:        {mean_abs_diff:.2e}")

    # Relative diff (only on non-zero elements)
    nonzero_mask = b_out.abs() > 1e-12
    if nonzero_mask.any():
        rel_diff = diff[nonzero_mask] / b_out.abs()[nonzero_mask]
        print(f"  Max  rel diff:        {rel_diff.max().item():.2e}")
        print(f"  Mean rel diff:        {rel_diff.mean().item():.2e}")

    # Loss comparison
    b_loss = (b_out.view(-1) * b_lm.view(-1)).sum() / b_lm.sum()
    m_loss = (m_out.view(-1) * m_lm.view(-1)).sum() / m_lm.sum()
    print(f"\n  Bridge avg lm loss:   {b_loss.item():.10f}")
    print(f"  MIMO   avg lm loss:   {m_loss.item():.10f}")
    print(f"  Loss diff:            {abs(b_loss.item() - m_loss.item()):.2e}")

    # Per-position spot check
    b_flat = b_out.view(-1)
    m_flat = m_out.view(-1)
    print(f"\n  Per-token loss spot check (position: bridge, mimo, diff):")
    positions = [0, 1, 63, 64, 65, 100, 500, 1000, 1500, 2000, 2047]
    for p in positions:
        if p < b_flat.numel():
            d = abs(b_flat[p].item() - m_flat[p].item())
            tag = "✓" if d == 0 else f"Δ={d:.2e}"
            print(f"    [{p:5d}]  {b_flat[p].item():14.6f}  {m_flat[p].item():14.6f}  {tag}")

    if exact_match:
        print(f"\n  ✅ BITWISE IDENTICAL — all {num_total} elements match exactly.")
    elif max_abs_diff < 1e-6:
        print(f"\n  ⚠️  NOT bitwise identical, but very close (max diff {max_abs_diff:.2e}).")
    else:
        print(f"\n  ❌ SIGNIFICANT DIFFERENCE (max diff {max_abs_diff:.2e}).")

    print(f"\n  Files:")
    print(f"    {BRIDGE_FILE}")
    print(f"    {MIMO_FILE}")
    print()


# ======================================================================
# Helpers
# ======================================================================

def _resolve_ckpt(path):
    latest = os.path.join(path, "latest_checkpointed_iteration.txt")
    if os.path.exists(latest):
        with open(latest) as f:
            it = f.read().strip()
        return os.path.join(path, f"iter_{int(it):07d}")
    return path


def _print_summary(name, output, loss_mask):
    if isinstance(output, tuple):
        output = output[0]
    loss_flat = output.float().view(-1)
    lm = loss_mask.float().view(-1)
    avg = (loss_flat * lm).sum() / lm.sum()
    print(f"  {name} avg lm loss: {avg.item():.10f}")
    print(f"  {name} first 5: {loss_flat[:5].tolist()}")
    print(f"  {name} last  5: {loss_flat[-5:].tolist()}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Kimi K2.5 VL bitwise parity check")
    parser.add_argument("--side", required=True, choices=["bridge", "mimo", "compare"],
                        help="Which side to run: bridge, mimo, or compare")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path (for bridge/mimo)")
    parser.add_argument("--variant", default="proxy", help="MIMO model variant")
    parser.add_argument("--hf-model-path", default="moonshotai/Kimi-K2.5")
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    if args.side == "compare":
        run_compare()
    elif args.side == "bridge":
        assert args.ckpt, "--ckpt required for bridge"
        run_bridge(args)
    elif args.side == "mimo":
        assert args.ckpt, "--ckpt required for mimo"
        run_mimo(args)


if __name__ == "__main__":
    main()
