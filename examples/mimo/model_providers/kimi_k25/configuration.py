# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Configuration helpers for Kimi K2.5 VL vision-language model.

Provides MLATransformerConfig builders for the language decoder.
The vision encoder is dynamically loaded from HuggingFace (MoonViT3d),
so no vision TransformerConfig is needed here.

The language backbone uses MoE with Multi-Latent Attention (MLA),
sharing architecture with DeepSeek V2/V3 and Kimi K2.

Supported language variants:
    ``proxy``    4 layers, 16 experts — single-node testing
    ``full``     61 layers, 256 experts — production Kimi K2.5 VL
"""

import torch

from megatron.core.transformer.transformer_config import MLATransformerConfig

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Token IDs from Kimi K2.5 HF config defaults
KIMI_K25_IMAGE_TOKEN_ID: int = 163605  # media_placeholder_token_id
KIMI_K25_BOS_TOKEN_ID: int = 163584
KIMI_K25_EOS_TOKEN_ID: int = 163585
KIMI_K25_PAD_TOKEN_ID: int = 163839

# Vocabulary size (must be divisible by make_vocab_size_divisible_by=1280)
KIMI_K25_VOCAB_SIZE: int = 164480  # 163839 rounded up to nearest multiple of 1280

# ---------------------------------------------------------------------------
# Language config variants
# ---------------------------------------------------------------------------

# Kimi K2 / K2.5 architecture values.
# The "full" variant matches the Kimi K2.5 VL production model:
#   61 decoder layers (first 2 dense, rest MoE), 256 routed experts,
#   MLA with q_lora_rank=1536, kv_lora_rank=512.
# The "proxy" variant shrinks to 4 layers / 16 experts for quick testing.
_VARIANT_CONFIGS = {
    "proxy": {
        "num_layers": 4,
        "hidden_size": 4096,
        "ffn_hidden_size": 12288,
        "num_attention_heads": 32,
        "num_query_groups": 32,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_head_dim": 128,
        "qk_pos_emb_head_dim": 64,
        "v_head_dim": 128,
        "num_moe_experts": 16,
        "moe_router_topk": 4,
        "moe_ffn_hidden_size": 1536,
        "moe_shared_expert_intermediate_size": 1536,
        "first_k_dense_replace": 1,
    },
    "full": {
        "num_layers": 61,
        "hidden_size": 4096,
        "ffn_hidden_size": 12288,
        "num_attention_heads": 32,
        "num_query_groups": 32,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_head_dim": 128,
        "qk_pos_emb_head_dim": 64,
        "v_head_dim": 128,
        "num_moe_experts": 256,
        "moe_router_topk": 8,
        "moe_ffn_hidden_size": 1536,
        "moe_shared_expert_intermediate_size": 1536,
        "first_k_dense_replace": 2,
    },
}


def get_kimi_k25_language_config(
    variant: str = "proxy",
    **overrides,
) -> MLATransformerConfig:
    """MLATransformerConfig for the Kimi K2.5 VL language decoder.

    Args:
        variant: One of ``proxy``, ``full``.
        **overrides: Override any MLATransformerConfig field.

    Returns:
        Fully-populated MLATransformerConfig.
    """
    if variant not in _VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from {list(_VARIANT_CONFIGS.keys())}"
        )

    v = _VARIANT_CONFIGS[variant]

    # Build moe_layer_freq: first_k_dense_replace dense layers, rest MoE
    first_k = v["first_k_dense_replace"]
    total = v["num_layers"]
    moe_layer_freq = [0] * first_k + [1] * (total - first_k)

    kwargs = dict(
        # Architecture
        num_layers=v["num_layers"],
        hidden_size=v["hidden_size"],
        ffn_hidden_size=v["ffn_hidden_size"],
        num_attention_heads=v["num_attention_heads"],
        num_query_groups=v["num_query_groups"],
        # MLA-specific
        multi_latent_attention=True,
        q_lora_rank=v["q_lora_rank"],
        kv_lora_rank=v["kv_lora_rank"],
        qk_head_dim=v["qk_head_dim"],
        qk_pos_emb_head_dim=v["qk_pos_emb_head_dim"],
        v_head_dim=v["v_head_dim"],
        # Normalization & activation
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        # Attention
        qk_layernorm=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        # RoPE — Kimi K2 uses YaRN RoPE
        # NOTE: position_embedding_type is set on GPTModel, not TransformerConfig
        rope_type="yarn",
        rotary_base=10000,
        rotary_scaling_factor=40,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=0.0,
        # MoE
        num_moe_experts=v["num_moe_experts"],
        moe_router_topk=v["moe_router_topk"],
        moe_ffn_hidden_size=v["moe_ffn_hidden_size"],
        moe_shared_expert_intermediate_size=v["moe_shared_expert_intermediate_size"],
        moe_layer_freq=moe_layer_freq,
        moe_grouped_gemm=True,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="seq_aux_loss",
        moe_shared_expert_overlap=True,
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=1e-3,
        # Kernel / TE fusions
        apply_rope_fusion=False,
        bias_activation_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        # Misc
        # NOTE: share_embeddings_and_output_weights is set on GPTModel, not TransformerConfig
        attention_softmax_in_fp32=False,
        # Precision
        bf16=True,
    )

    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)
