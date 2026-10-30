# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A tiny Kimi-K3 VL built twice: HF remote code and MCore, with matching shapes.

Used by the full-model cross-check and the parameter-name dump. The HF config
is the real ``config.json`` (pinned revision) with every width scaled down;
the MCore model goes through the real entry-point path
(``pretrain_multimodal.model_provider`` -> ``kimi_k3_vl`` factory ->
``HybridModelBuilder``) with the matching CLI arguments.

Decoder: 4 Kimi layers = KDA+dense, KDA+MoE, KDA+MoE, MLA+MoE
(MCore pattern ``K-KEKE+E``), attention residuals on (one HF layer per block,
``attn_res_block_layers=2`` HybridModel entries), latent MoE with its norm,
2 shared experts, no MTP. Vision: MoonViT widths, 2 layers.
"""

import copy
import json
import os
import sys

import torch

from examples.multimodal_dev.tests._kimi_hf import hf_reference_dir, import_hf_module

HIDDEN = 512
FFN = 1024
HEADS = 8
HEAD_DIM = 64  # KDA head dim (== f/g low-rank width in HF)
Q_LORA, KV_LORA, QK_NOPE, QK_ROPE, V_HEAD = 128, 64, 64, 32, 64
EXPERTS, TOPK, MOE_FFN, LATENT, SHARED = 8, 2, 128, 256, 2
VIT_LAYERS = 2
PATTERN = "K-KEKE+E"


def hf_config():
    """Tiny ``KimiK3Config`` derived from the pinned ``config.json``."""
    cfg_mod = import_hf_module("configuration_kimi_k3")
    with open(os.path.join(hf_reference_dir(), "config.json")) as f:
        raw = json.load(f)
    text = copy.deepcopy(raw["text_config"])
    text.update(
        hidden_size=HIDDEN,
        intermediate_size=FFN,
        num_hidden_layers=4,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD,
        num_experts=EXPERTS,
        num_experts_per_token=TOPK,
        moe_intermediate_size=MOE_FFN,
        num_shared_experts=SHARED,
        routed_expert_hidden_size=LATENT,
        first_k_dense_replace=1,
        attn_res_block_size=1,
        num_nextn_predict_layers=0,
    )
    text["linear_attn_config"] = dict(
        text["linear_attn_config"],
        kda_layers=[1, 2, 3],
        full_attn_layers=[4],
        head_dim=HEAD_DIM,
        num_heads=HEADS,
    )
    vision = dict(raw["vision_config"], vt_num_hidden_layers=VIT_LAYERS, text_hidden_size=HIDDEN)
    vision["_attn_implementation"] = "eager"
    top = {k: v for k, v in raw.items() if k not in ("text_config", "vision_config", "auto_map")}
    return cfg_mod.KimiK3Config(text_config=text, vision_config=vision, **top)


def build_hf_model(device="cuda", seed=0):
    """Random-init HF ``KimiK3ForConditionalGeneration`` in fp32, eval mode."""
    mdl = import_hf_module("modeling_kimi_k3")
    torch.manual_seed(seed)
    model = mdl.KimiK3ForConditionalGeneration(hf_config())
    # HF's _init_weights covers Linear/Conv2d/Embedding only; tensors created
    # with torch.empty (KDA dt_bias, the router's e_score_correction_bias) would
    # otherwise hold whatever memory they got. Give every tensor a defined,
    # seeded value: norms around 1, biases and gates small, the router bias zero
    # (routing defined, and equal to an MCore router without an expert bias).
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("e_score_correction_bias"):
                p.zero_()
            elif name.endswith("dt_bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
            elif not torch.isfinite(p).all():
                raise AssertionError(f"HF parameter {name} is not initialized")
    for block in model.vision_tower.encoder.blocks:
        block.attn_implementation = "eager"
    # The text MLA defaults to flash_attention_2, which has no fp32 kernel.
    model.language_model.config._attn_implementation = "sdpa"
    return model.float().to(device).eval()


MCORE_ARGS = [
    "--model-arch",
    "kimi_k3_vl",
    "--model-variant",
    "proxy",
    "--vision-num-layers",
    str(VIT_LAYERS),
    "--use-packed-sequence",
    "--use-vanilla-collate-fn",
    "--tokenizer-type",
    "NullTokenizer",
    "--vocab-size",
    "163840",
    "--make-vocab-size-divisible-by",
    "1",
    "--spec",
    "megatron.core.models.hybrid.hybrid_layer_specs",
    "hybrid_stack_spec",
    "--enable-experimental",
    "--num-layers",
    str(len(PATTERN)),
    "--hybrid-layer-pattern",
    PATTERN,
    "--experimental-attention-variant",
    "kda",
    "--hidden-size",
    str(HIDDEN),
    "--ffn-hidden-size",
    str(FFN),
    "--num-attention-heads",
    str(HEADS),
    "--kv-channels",
    str(HEAD_DIM),
    "--max-position-embeddings",
    "4096",
    "--seq-length",
    "1024",
    "--linear-conv-kernel-dim",
    "4",
    "--linear-key-head-dim",
    str(HEAD_DIM),
    "--linear-value-head-dim",
    str(HEAD_DIM),
    "--linear-num-key-heads",
    str(HEADS),
    "--linear-num-value-heads",
    str(HEADS),
    "--kda-f-lora-rank",
    str(HEAD_DIM),
    "--kda-safe-gate",
    "--kda-lower-bound",
    "-5.0",
    "--multi-latent-attention",
    "--q-lora-rank",
    str(Q_LORA),
    "--kv-lora-rank",
    str(KV_LORA),
    "--qk-head-dim",
    str(QK_NOPE),
    "--qk-pos-emb-head-dim",
    str(QK_ROPE),
    "--v-head-dim",
    str(V_HEAD),
    "--qk-layernorm",
    "--attention-latent-norm-epsilon",
    "1e-6",
    "--attention-output-gate",
    "--gated-attention-proj-granularity",
    "elementwise",
    "--no-rope-freq",
    "1",
    "--position-embedding-type",
    "none",
    "--enable-attention-residuals",
    "--attn-res-block-layers",
    "2",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-5",
    "--situ-glu",
    "--situ-glu-beta1",
    "4.0",
    "--situ-glu-beta2",
    "25.0",
    "--disable-bias-linear",
    "--untie-embeddings-and-output-weights",
    "--num-experts",
    str(EXPERTS),
    "--moe-ffn-hidden-size",
    str(MOE_FFN),
    "--moe-latent-size",
    str(LATENT),
    "--moe-latent-up-projection-rmsnorm",
    "--moe-shared-expert-intermediate-size",
    str(SHARED * MOE_FFN),
    "--moe-router-topk",
    str(TOPK),
    "--moe-router-score-function",
    "sigmoid",
    "--moe-router-dtype",
    "fp32",
    "--moe-router-topk-scaling-factor",
    "1.0",
    "--moe-router-load-balancing-type",
    "none",
    "--moe-aux-loss-coeff",
    "0.0",
    "--moe-token-dispatcher-type",
    "alltoall",
    "--moe-grouped-gemm",
    "--attention-dropout",
    "0.0",
    "--hidden-dropout",
    "0.0",
    "--micro-batch-size",
    "1",
    "--global-batch-size",
    "1",
    "--train-iters",
    "1",
    "--lr",
    "1e-4",
    "--transformer-impl",
    "transformer_engine",
    "--no-gradient-accumulation-fusion",
]


def build_mcore_model(extra_args=(), device="cuda"):
    """The tiny Kimi-K3 VL through ``pretrain_multimodal.model_provider`` (fp32)."""
    from examples.multimodal_dev.arguments import add_multimodal_args
    from megatron.training.argument_utils import (
        pretrain_cfg_container_from_args,
        resolve_tokenizer_vocab_size,
    )
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.global_vars import get_args, initialize_runtime_services, set_run_config
    from megatron.training.initialize import initialize_megatron

    # Same sequence as pretrain_multimodal.py's __main__ + pretrain().
    argv = sys.argv
    sys.argv = [argv[0]] + MCORE_ARGS + list(extra_args)
    try:
        args = parse_and_validate_args(extra_args_provider=add_multimodal_args, args_defaults={})
    finally:
        sys.argv = argv
    full_config = pretrain_cfg_container_from_args(args)
    set_run_config(full_config)
    initialize_runtime_services(args, training=False)
    resolve_tokenizer_vocab_size(full_config, args.padded_vocab_size)
    initialize_megatron()
    from examples.multimodal_dev.pretrain_multimodal import model_provider

    get_args().model_arch = "kimi_k3_vl"
    model = model_provider(pre_process=True, post_process=True)
    return model.to(device)
