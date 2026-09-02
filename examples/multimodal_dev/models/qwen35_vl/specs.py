# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer spec helpers for Qwen3.5-VL vision encoder and language decoder.

Provides ModuleSpec builders that define the transformer layer composition.
Both the standalone and MIMO training paths import from here.
"""

import dataclasses
import math
from typing import Optional

import torch.nn.functional as F

from examples.multimodal_dev.models.base import _NO_CP_GROUP
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


def _apply_rope_fp32(
    t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None, max_seqlen=None
):
    """Apply rotary positional embedding in fp32, then cast back to original dtype.

    Mirrors ``Qwen3VLSelfAttention.apply_rotary_pos_emb_absolute`` in Megatron-Bridge
    with ``apply_rotary_pos_emb_in_fp32=True``.
    """
    from megatron.core.models.common.embeddings import rope_utils
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    orig_dtype = t.dtype
    if (
        cu_seqlens is not None
        and getattr(config, "apply_rope_fusion", False)
        and getattr(config, "mrope_section", None) is not None
        and getattr(config, "rotary_interleaved", False) is False
        and getattr(config, "multi_latent_attention", False) is False
        and mscale == 1.0
        and t.dim() == 3
        and freqs.dim() == 4
        and freqs.shape[0] == 3
        and cp_group is not None
        and rope_utils.fused_apply_mrope_thd is not None
        and rope_utils.get_fused_mrope_thd_unavailable_reason is not None
    ):
        unavailable_reason = rope_utils.get_fused_mrope_thd_unavailable_reason(
            t,
            cu_seqlens,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            cp_size=cp_group.size(),
            cp_rank=cp_group.rank(),
        )
        if unavailable_reason is None:
            return rope_utils.fused_apply_mrope_thd(
                t,
                cu_seqlens,
                freqs,
                config.mrope_section,
                interleaved_mrope=config.mrope_interleaved,
                rotary_interleaved=config.rotary_interleaved,
                cp_size=cp_group.size(),
                cp_rank=cp_group.rank(),
                fp32_compute=True,
            )

    t_fp32 = t.float()
    out = apply_rotary_pos_emb(
        t_fp32,
        freqs,
        config=config,
        cu_seqlens=cu_seqlens,
        mscale=mscale,
        cp_group=cp_group,
        mla_rotary_interleaved=getattr(config, 'multi_latent_attention', False),
        max_seqlen=max_seqlen,
    )
    return out.to(orig_dtype)


def _apply_rope_fp32_no_cp(
    t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None, max_seqlen=None
):
    """Same as ``_apply_rope_fp32`` but forces CP-size=1.

    The vision encoder uses THD packed sequences for variable-resolution
    images.  When the language model uses CP>1, the global CP group would
    incorrectly split the vision seqlens.  This wrapper substitutes a
    trivial group so the vision RoPE sees the full packed sequence.
    """
    range_name = "qwen35_vl.vision_encoder.rope_apply"
    nvtx_range_push(range_name)
    try:
        return _apply_rope_fp32(
            t,
            freqs,
            config,
            cu_seqlens,
            mscale,
            cp_group=_NO_CP_GROUP,
            max_seqlen=max_seqlen,
        )
    finally:
        nvtx_range_pop(range_name)


class Qwen35VLVisionSelfAttention(SelfAttention):
    """ViT self-attention with RoPE applied in fp32.

    Matches Bridge's ``Qwen3VLSelfAttention`` behaviour when
    ``apply_rotary_pos_emb_in_fp32=True``:  query and key are cast to float32
    before the rotary multiply and cast back to bf16 afterwards.  The
    monkey-patch approach avoids duplicating the 300-line ``SelfAttention.forward``
    while keeping the change local to this class.
    """

    def forward(self, *args, **kwargs):
        import megatron.core.transformer.attention as _attn_mod

        _orig = _attn_mod.apply_rotary_pos_emb
        _attn_mod.apply_rotary_pos_emb = _apply_rope_fp32_no_cp
        try:
            return super().forward(*args, **kwargs)
        finally:
            _attn_mod.apply_rotary_pos_emb = _orig


def get_qwen35_vl_language_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Transformer block spec for the Qwen3.5-VL language decoder.

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


# head_dims where cuDNN's fused THD attention BACKWARD requests a grossly
# oversized workspace. Measured on TE 2.18.0 / cuDNN 9.25.0.15 / GB300 with the
# vision encoder's real packed batch (206 sequences, 61,524 tokens, max_seqlen
# 1024): head_dim 72 asks for 75,436 MB, while 64/80/96/128 all land in
# 865-1,722 MB. That single transient was 55.7% of the training step's
# allocation peak. FlashAttention is not an escape hatch -- at head_dim 72 it
# fails outright with "ICE IR Verification Failed".
#
# This is an EMPIRICAL table, not an exhaustive one: only 64/72/80/96/128/144
# were measured, and 72 was the sole outlier. Revisit when TE or cuDNN moves.
# See agent_works/mdp-fast-pass-0901/repro/CONCLUSION.md for the reproducer.
_PADDED_ATTENTION_HEAD_DIMS = {72: 80}


class PaddedHeadDimDotProductAttention(TEDotProductAttention):
    """Widen head_dim into a well-supported size for the attention call only.

    q/k/v are zero-padded on the head dimension before the attention and the
    output is sliced back down, so the surrounding module is unchanged:
    ``linear_qkv`` still emits ``num_heads * real_kv`` and ``linear_proj`` still
    consumes it, which keeps checkpoints compatible. Zeros contribute nothing to
    ``Q.K^T`` and only produce zero columns in the output, so the result is
    mathematically identical -- provided ``softmax_scale`` stays at
    ``1/sqrt(real_kv)``. TE would otherwise derive ``1/sqrt(padded)`` from
    ``config.kv_channels`` and silently change the numerics; that is the one
    thing this class must not get wrong.

    Both the pad and the slice have static shapes (they depend on tensor.shape,
    never on tensor values), so they remain CUDA-graph capturable.
    """

    def __init__(self, config, *args, **kwargs):
        real_kv = config.kv_channels
        pad_kv = _PADDED_ATTENTION_HEAD_DIMS.get(real_kv)
        if pad_kv:
            # Only the copy handed to TE is widened; the caller's config, and
            # therefore every projection around this module, keeps real_kv.
            config = dataclasses.replace(config, kv_channels=pad_kv)
            kwargs.setdefault("softmax_scale", 1.0 / math.sqrt(real_kv))
        super().__init__(config, *args, **kwargs)
        self._real_kv = real_kv
        self._pad_kv = pad_kv
        self._pad_width = pad_kv - real_kv if pad_kv else 0

    def forward(self, query, key, value, *args, **kwargs):
        """Pad, attend, slice back."""
        if not self._pad_width:
            return super().forward(query, key, value, *args, **kwargs)
        query, key, value = (
            F.pad(t, (0, self._pad_width)) for t in (query, key, value)
        )
        out = super().forward(query, key, value, *args, **kwargs)
        return (
            out.view(*out.shape[:-1], -1, self._pad_kv)[..., : self._real_kv]
            .reshape(*out.shape[:-1], -1)
        )


def get_qwen35_vl_vision_spec() -> ModuleSpec:
    """ModuleSpec for vision encoder transformer layers.

    Uses ``TEDotProductAttention`` which supports packed-sequence (THD)
    attention via ``PackedSeqParams`` for variable-length images.

    ``Qwen35VLVisionSelfAttention`` replaces the default ``SelfAttention`` so
    that RoPE is applied in fp32, matching Bridge's
    ``apply_rotary_pos_emb_in_fp32=True`` behaviour.
    """
    spec = get_vit_layer_with_transformer_engine_spec()
    spec.submodules.self_attention.module = Qwen35VLVisionSelfAttention
    # Works around an oversized cuDNN THD backward workspace at head_dim 72;
    # inert for every other head_dim. See _PADDED_ATTENTION_HEAD_DIMS.
    spec.submodules.self_attention.submodules.core_attention = (
        PaddedHeadDimDotProductAttention
    )
    return spec
