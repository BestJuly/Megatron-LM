# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Layer spec helper for the Qwen3 text-only MoE LLM.

Returns a MoE TransformerEngine layer spec without monkey-patches or the
fp32-RoPE wrapper used for vision-specific RoPE handling. The implementation
reuses Megatron-core's existing Qwen3-compatible decoder helper.
"""

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec


def get_qwen3_language_spec(config, vp_stage=None, pp_rank=None):
    """Build the Qwen3 language decoder block spec.

    Uses ``get_gpt_decoder_block_spec`` which assembles a per-layer
    TransformerEngine spec respecting MoE settings on the config
    (num_moe_experts, moe_router_topk, moe_ffn_hidden_size, etc.).

    This helper delegates decoder construction to the standard Megatron-core
    GPT block builder.
    """
    return get_gpt_decoder_block_spec(config=config, use_transformer_engine=True, vp_stage=vp_stage)
