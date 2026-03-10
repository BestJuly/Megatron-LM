# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL model definition using pure mcore specs.

Architecture:
    - Vision encoder: 27-layer ViT with Conv3d patches + PatchMerger
    - Language decoder: Qwen3-Next with hybrid GatedDeltaNet/full-attention, MoE
    - Integration: Single-pass vision embeddings scattered into text embeddings
    - Position encoding: MRoPE with sections [11, 11, 10] and partial_rotary_factor=0.25

Supported variants:
    - "9b": Dense 9B model
    - "397b_a17b": MoE 397B-A17B model (512 experts top-2)
    - "proxy": Small proxy model for testing (4 layers, 4 experts)
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from multimodal.models.base import MultimodalModel


# ---------------------------------------------------------------------------
# Vision config
# ---------------------------------------------------------------------------

def get_qwen35_vl_vision_config() -> TransformerConfig:
    """TransformerConfig for the Qwen3.5-VL vision encoder (27-layer ViT)."""
    return TransformerConfig(
        num_layers=27,
        hidden_size=1152,
        num_attention_heads=16,
        kv_channels=72,  # 1152 / 16
        ffn_hidden_size=4304,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="LayerNorm",
        gated_linear_unit=False,
        activation_func=torch.nn.functional.gelu,
        bias_activation_fusion=False,
        apply_query_key_layer_scaling=False,
        bf16=True,
    )


# rotary_base and rotary_percent are GPTModel constructor args, not TransformerConfig fields.
_ROTARY_BASE = 10000000
_ROTARY_PERCENT = 0.25  # partial_rotary_factor

VISION_KWARGS = {
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": 3584,
    "max_num_positions": 2304,
}


# ---------------------------------------------------------------------------
# Language config variants
# ---------------------------------------------------------------------------

_VARIANT_CONFIGS = {
    "9b": {
        "num_layers": 60,
        "hidden_size": 4096,
        "ffn_hidden_size": 12288,
        "num_attention_heads": 16,
        "num_query_groups": 4,
        "kv_channels": 256,
        "num_moe_experts": None,  # dense
        "moe_router_topk": None,
        "moe_ffn_hidden_size": None,
        "moe_shared_expert_intermediate_size": None,
    },
    "397b_a17b": {
        "num_layers": 60,
        "hidden_size": 4096,
        "ffn_hidden_size": 10240,
        "num_attention_heads": 32,
        "num_query_groups": 2,
        "kv_channels": 256,
        "num_moe_experts": 512,
        "moe_router_topk": 10,
        "moe_ffn_hidden_size": 1024,
        "moe_shared_expert_intermediate_size": 1024,
    },
    "proxy": {
        "num_layers": 4,
        "hidden_size": 4096,
        "ffn_hidden_size": 10240,
        "num_attention_heads": 32,
        "num_query_groups": 2,
        "kv_channels": 256,
        "num_moe_experts": 4,
        "moe_router_topk": 2,
        "moe_ffn_hidden_size": 1024,
        "moe_shared_expert_intermediate_size": 1024,
    },
}


def get_qwen35_vl_language_config(
    variant: str = "proxy",
    **overrides,
) -> TransformerConfig:
    """TransformerConfig for Qwen3.5-VL language decoder.

    Args:
        variant: One of "9b", "397b_a17b", "proxy".
        **overrides: Override any config field.

    Returns:
        TransformerConfig for the language decoder.
    """
    if variant not in _VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_VARIANT_CONFIGS.keys())}")

    v = _VARIANT_CONFIGS[variant]

    kwargs = dict(
        # Architecture
        num_layers=v["num_layers"],
        hidden_size=v["hidden_size"],
        ffn_hidden_size=v["ffn_hidden_size"],
        num_attention_heads=v["num_attention_heads"],
        num_query_groups=v["num_query_groups"],
        kv_channels=v["kv_channels"],
        # Normalization & activation
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        layernorm_zero_centered_gamma=True,
        apply_residual_connection_post_layernorm=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        # MRoPE section (valid TransformerConfig field; rotary_base / rotary_percent /
        # position_embedding_type are NOT TransformerConfig fields — they are passed
        # to GPTModel directly via MultimodalModel.__init__)
        mrope_section=[11, 11, 10],
        rotary_interleaved=False,
        # Attention
        qk_layernorm=True,
        attention_output_gate=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        # Hybrid attention (GatedDeltaNet)
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        # Kernel / TE fusions
        bias_activation_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        apply_rope_fusion=True,
        # Precision
        bf16=True,
    )

    # MoE config (only for MoE variants)
    if v["num_moe_experts"] is not None:
        kwargs.update(
            num_moe_experts=v["num_moe_experts"],
            moe_router_topk=v["moe_router_topk"],
            moe_ffn_hidden_size=v["moe_ffn_hidden_size"],
            moe_shared_expert_intermediate_size=v["moe_shared_expert_intermediate_size"],
            moe_shared_expert_gate=True,
            moe_layer_freq=1,
            moe_router_pre_softmax=False,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1e-3,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_dtype="fp32",
        )

    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Language layer spec
# ---------------------------------------------------------------------------

def get_qwen35_vl_language_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Build transformer block spec for Qwen3.5-VL language decoder.

    Uses the experimental attention variant infrastructure to build hybrid
    GatedDeltaNet + full-attention layers with optional MoE interleaving.

    Args:
        config: Language decoder TransformerConfig.
        vp_stage: Virtual pipeline stage.
        pp_rank: Pipeline parallel rank.

    Returns:
        TransformerBlockSubmodules with per-layer specs.
    """
    return get_transformer_block_with_experimental_attention_variant_spec(
        config=config,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )


def get_qwen35_vl_vision_spec() -> ModuleSpec:
    """Build ModuleSpec for vision encoder transformer layers."""
    return get_vit_layer_with_transformer_engine_spec()


# ---------------------------------------------------------------------------
# MRoPE position ID computation
# ---------------------------------------------------------------------------

def compute_mrope_position_ids(
    input_ids: Tensor,
    image_grid_thw: Optional[Tensor],
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> Tensor:
    """Compute 3D MRoPE position IDs for Qwen3.5-VL.

    For text tokens: sequential positions on all 3 dimensions.
    For image tokens: 2D spatial grid positions (temporal, height, width).

    Args:
        input_ids: [B, S] token IDs.
        image_grid_thw: [num_images, 3] with (temporal, height, width) per image.
        image_token_id: Token ID for image placeholders.
        spatial_merge_size: Merge factor (positions are in merged grid coordinates).

    Returns:
        Position IDs [3, B, S] for MRoPE (temporal, height, width).
    """
    B, S = input_ids.shape
    device = input_ids.device

    # Initialize with sequential positions
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    # [3, B, S]: all 3 dims start as sequential
    mrope_ids = position_ids.unsqueeze(0).expand(3, -1, -1).clone()

    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return mrope_ids

    # Process each sample in the batch
    for b in range(B):
        image_mask = (input_ids[b] == image_token_id)
        if not image_mask.any():
            continue

        image_positions = image_mask.nonzero(as_tuple=True)[0]

        # Track which image we're processing
        img_idx = 0
        pos_offset = 0
        text_pos = 0

        for s in range(S):
            if input_ids[b, s] == image_token_id:
                # This is a visual token — assign spatial position
                if pos_offset == 0:
                    # Starting a new image
                    if img_idx >= image_grid_thw.shape[0]:
                        break
                    t = int(image_grid_thw[img_idx, 0].item())
                    h = int(image_grid_thw[img_idx, 1].item()) // spatial_merge_size
                    w = int(image_grid_thw[img_idx, 2].item()) // spatial_merge_size
                    n_tokens = t * h * w

                temporal_idx = pos_offset // (h * w)
                spatial_idx = pos_offset % (h * w)
                h_idx = spatial_idx // w
                w_idx = spatial_idx % w

                mrope_ids[0, b, s] = temporal_idx + text_pos  # temporal
                mrope_ids[1, b, s] = h_idx + text_pos  # height
                mrope_ids[2, b, s] = w_idx + text_pos  # width

                pos_offset += 1
                if pos_offset >= n_tokens:
                    # Finished this image
                    text_pos += max(t, h, w)
                    pos_offset = 0
                    img_idx += 1
            else:
                # Text token — sequential position
                mrope_ids[0, b, s] = text_pos
                mrope_ids[1, b, s] = text_pos
                mrope_ids[2, b, s] = text_pos
                text_pos += 1

    return mrope_ids


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class Qwen35VLModel(MultimodalModel):
    """Qwen3.5-VL multimodal model.

    Composes a mcore ViT vision encoder with a GPTModel language decoder
    using MRoPE and hybrid GatedDeltaNet/full-attention layers.

    Args:
        language_config: TransformerConfig for language decoder.
        language_spec: ModuleSpec for language decoder layers.
        vision_config: TransformerConfig for vision encoder.
        vision_spec: ModuleSpec for vision encoder layers.
        vocab_size: Vocabulary size.
        max_sequence_length: Maximum sequence length.
        image_token_id: Token ID for image placeholders.
        spatial_merge_size: Vision encoder spatial merge factor.
        pre_process: PP first stage flag.
        post_process: PP last stage flag.
        add_encoder: Build vision encoder.
        add_decoder: Build language decoder.
        parallel_output: Keep outputs split across TP.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        language_spec: ModuleSpec,
        vision_config: TransformerConfig,
        vision_spec: ModuleSpec = None,
        vocab_size: int = 248320,
        max_sequence_length: int = 262144,
        image_token_id: int = 248056,
        spatial_merge_size: int = 2,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ):
        if vision_spec is None:
            vision_spec = get_qwen35_vl_vision_spec()

        self.spatial_merge_size = spatial_merge_size

        vision_kwargs = dict(VISION_KWARGS)
        vision_kwargs["spatial_merge_size"] = spatial_merge_size
        # out_hidden_size must match language decoder hidden_size
        vision_kwargs["out_hidden_size"] = language_config.hidden_size

        super().__init__(
            language_config=language_config,
            language_spec=language_spec,
            vision_config=vision_config,
            vision_spec=vision_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            image_token_id=image_token_id,
            position_embedding_type="mrope",
            rotary_percent=_ROTARY_PERCENT,
            rotary_base=_ROTARY_BASE,
            mrope_section=language_config.mrope_section,
            vision_kwargs=vision_kwargs,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

    def compute_position_ids(
        self,
        input_ids: Tensor,
        image_grid_thw: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute 3D MRoPE position IDs for Qwen3.5-VL.

        Returns:
            [3, B, S] position IDs for MRoPE.
        """
        return compute_mrope_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            image_token_id=self.image_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )
