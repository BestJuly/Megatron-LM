# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for Kimi-K3 VL model construction (registry hooks)."""

import functools

import torch

from examples.multimodal_dev.models.kimi_k3_vl.configuration import (
    VISION_KWARGS,
    resolve_image_token_id,
    vision_norm_epsilon,
)


def align_vision_norm_epsilon(vision_config) -> None:
    """Set MoonViT's dtype-dependent RMSNorm epsilon on *vision_config*.

    The entry point sets ``bf16`` / ``fp16`` on the vision config after the
    registry built it, so the epsilon is re-derived here, before any norm is
    constructed. See :func:`configuration.vision_norm_epsilon`.
    """
    if vision_config.bf16:
        dtype = torch.bfloat16
    elif vision_config.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    vision_config.layernorm_epsilon = vision_norm_epsilon(dtype)


def set_vision_flops_metadata(args, language_config, vision_config):
    """MoonViT dimensions for the vision-encoder FLOPs term.

    Per-batch row counts come from the collate spec's ``workload_of``; this only
    sets shapes. The patch embedding is 2-D (``T_p = 1``), the MLP is plain
    GELU, and the projector is ``4096 -> 4096 -> hidden``.
    """
    merge = VISION_KWARGS["spatial_merge_size"]
    args.count_vision_model_flops = True
    args.vision_num_layers = vision_config.num_layers
    args.vision_hidden_size = vision_config.hidden_size
    args.vision_ffn_hidden_size = vision_config.ffn_hidden_size
    args.vision_num_attention_heads = vision_config.num_attention_heads
    args.vision_kv_channels = vision_config.kv_channels
    args.vision_in_channels = VISION_KWARGS["in_channels"]
    args.vision_patch_size = VISION_KWARGS["patch_size"]
    args.vision_temporal_patch_size = 1
    args.vision_out_hidden_size = language_config.hidden_size
    args.vision_gated_mlp = False
    args.vision_merge_dim = vision_config.hidden_size * merge * merge
    args.vision_merger_hidden_size = args.vision_merge_dim


def _check_decoder_args(args) -> None:
    """Reject recipes that would build something other than the Kimi-K3 decoder."""
    if getattr(args, "hybrid_layer_pattern", None) is None:
        raise ValueError(
            "Kimi-K3 VL uses HybridModel and requires --hybrid-layer-pattern "
            "(K=KDA, +=MLA, -=dense MLP, E=latent MoE, /+E per MTP depth)."
        )
    position_embedding_type = getattr(args, "position_embedding_type", None)
    if position_embedding_type != "none":
        raise ValueError(
            "Kimi-K3 VL requires --position-embedding-type none: the decoder is "
            f"NoPE MLA + KDA, got '{position_embedding_type}'."
        )


def build_model(
    args,
    language_config,
    vision_config,
    pre_process: bool = True,
    post_process: bool = True,
    language_model_builder=None,
    **kwargs,
):
    """Build a :class:`KimiK3VLModel` for this pipeline stage.

    The decoder comes from *language_model_builder* (default
    ``models.base.build_language_model``: ``hybrid_config_from_args`` +
    ``HybridModelBuilder``), so every Kimi-K3 decoder feature the recipe enables
    -- KDA, MLA, latent MoE, attention residuals, SiTU-GLU, MTP -- is built
    exactly as ``pretrain_hybrid.py`` builds it. Nothing is pinned here.
    """
    _check_decoder_args(args)
    vp_stage = kwargs.get("vp_stage", None)

    from examples.multimodal_dev.models.kimi_k3_vl.model import KimiK3VLModel

    if language_model_builder is None:
        from examples.multimodal_dev.models.base import build_language_model

        language_model_builder = functools.partial(
            build_language_model,
            args,
            language_config,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    align_vision_norm_epsilon(vision_config)
    return KimiK3VLModel(
        language_config=language_config,
        language_model_builder=language_model_builder,
        vision_config=vision_config,
        image_token_id=resolve_image_token_id(args),
        pre_process=pre_process,
        vp_stage=vp_stage,
        build_vision_encoder=not getattr(args, "mdp_enable", False),
    )
