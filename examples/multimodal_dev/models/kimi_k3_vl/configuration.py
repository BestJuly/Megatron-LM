# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configuration helpers for the Kimi-K3 vision-language model.

The reference is the HuggingFace Hub remote code of ``moonshotai/Kimi-K3`` at
revision ``f831ab66814297da540d832a5235f8e904f29d06``
(``modeling_kimi_k3.py``: ``MoonViT3dPretrainedModel`` + ``PatchMergerMLPV2``).
Names below follow that file's ``vision_config`` keys.

The vision tower is MoonViT 3D:

* patch embed: ``Conv2d(3, 1024, 14, 14, bias=False)`` over pre-patchified
  ``3 x 14 x 14`` rows, i.e. one GEMM per 588-wide row;
* a learnable ``64 x 64`` absolute position table, bilinear-interpolated to the
  item's ``(h, w)``, plus a fixed 1-D sin-cos time embedding for ``t > 1``;
* 27 pre-norm layers (RMSNorm, 12 heads x 128 = 1536-wide QKV over a
  1024-wide residual, non-gated tanh-GELU MLP of 4096, no biases) with an
  interleaved 2-D RoPE and bidirectional attention over each whole item
  (all frames of a video item attend to each other);
* a final RMSNorm, then ``sd2_tpool``: a 2x2 spatial merge after averaging
  over all frames of the item.

The projector (``patchmergerv2``) maps the 4096-wide merged rows to the
decoder width with ``Linear -> GELU(erf) -> Linear -> RMSNorm(eps=1e-5)``.
"""

from typing import Optional

import torch

from megatron.core.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Token ids (tokenizer_config.json at the pinned revision)
# ---------------------------------------------------------------------------

KIMI_K3_BOS_TOKEN_ID: int = 163584
KIMI_K3_EOS_TOKEN_ID: int = 163586
KIMI_K3_PAD_TOKEN_ID: int = 163839
KIMI_K3_MEDIA_BEGIN_TOKEN_ID: int = 163602
KIMI_K3_MEDIA_CONTENT_TOKEN_ID: int = 163603
KIMI_K3_MEDIA_END_TOKEN_ID: int = 163604
#: ``<|media_pad|>`` -- ``media_placeholder_token_id`` in config.json. The HF
#: processor emits one per item and the model expands it; multimodal_dev expands
#: it in the data pipeline instead, one slot per merged vision row.
KIMI_K3_IMAGE_TOKEN_ID: int = 163605
KIMI_K3_VOCAB_SIZE: int = 163840

#: HF Hub revision every reference in this package was checked against.
KIMI_K3_HF_REVISION: str = "f831ab66814297da540d832a5235f8e904f29d06"


def resolve_image_token_id(args) -> int:
    """``--image-token-id`` when given, else ``<|media_pad|>``."""
    value = getattr(args, "image_token_id", None)
    return KIMI_K3_IMAGE_TOKEN_ID if value is None else int(value)


# ---------------------------------------------------------------------------
# Vision config
# ---------------------------------------------------------------------------

VISION_KWARGS = {
    "in_channels": 3,
    "patch_size": 14,
    # merge_kernel_size [2, 2]; MoonViT only supports a square kernel here.
    "spatial_merge_size": 2,
    # Learnable absolute position table (init_pos_emb_height/width) and the
    # number of fixed time-embedding rows (init_pos_emb_time). A video item may
    # carry at most this many frames.
    "pos_emb_height": 64,
    "pos_emb_width": 64,
    "pos_emb_time": 4,
    # Rope2DPosEmbRepeated(head_dim, 512, 512): the largest grid side.
    "rope_max_side": 512,
    "rope_theta": 10000.0,
    # PatchMergerMLPV2.post_norm eps (projector_ln_eps).
    "projector_norm_eps": 1e-5,
    "out_hidden_size": 7168,
}

_VISION_FULL = {
    "num_layers": 27,
    "hidden_size": 1024,
    "num_attention_heads": 12,
    "kv_channels": 128,
    "ffn_hidden_size": 4096,
}

#: ``--model-variant`` -> vision tower depth. Both keep MoonViT's widths; the
#: proxy only drops layers.
VISION_VARIANT_NUM_LAYERS = {"full": 27, "proxy": 4}


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``gelu_pytorch_tanh``, MoonViT's MLP activation."""
    return torch.nn.functional.gelu(x, approximate="tanh")


def vision_norm_epsilon(params_dtype: torch.dtype) -> float:
    """MoonViT's RMSNorm epsilon for parameters of *params_dtype*.

    MoonViT builds every ViT norm as ``nn.RMSNorm(hidden)`` without ``eps``, so
    PyTorch uses ``torch.finfo(x.dtype).eps`` of the input: 7.8e-3 in bf16 and
    1.2e-7 in fp32. The released model runs in bf16, so bf16 training matches
    HF bf16 inference with the bf16 value; fp32 parity tests use the fp32 one.
    """
    return float(torch.finfo(params_dtype).eps)


def get_kimi_k3_vl_vision_config(
    num_layers_override: Optional[int] = None, variant: Optional[str] = None, **overrides
) -> TransformerConfig:
    """TransformerConfig for the MoonViT vision tower.

    Args:
        num_layers_override: ``--vision-num-layers``; wins over the variant.
        variant: ``--model-variant`` (``full`` or ``proxy``); ``None`` means
            ``full``.
        **overrides: Any TransformerConfig field.

    The norm epsilon defaults to the bf16 value (see
    :func:`vision_norm_epsilon`); the factory re-derives it from the parameter
    dtype once the entry point has set ``bf16`` / ``fp16`` on the config.
    """
    variant = variant or "full"
    if variant not in VISION_VARIANT_NUM_LAYERS:
        raise ValueError(
            f"Unknown Kimi-K3 VL variant '{variant}'. "
            f"Choose from {list(VISION_VARIANT_NUM_LAYERS)}"
        )
    num_layers = (
        num_layers_override
        if num_layers_override is not None
        else VISION_VARIANT_NUM_LAYERS[variant]
    )
    kwargs = dict(
        num_layers=num_layers,
        hidden_size=_VISION_FULL["hidden_size"],
        num_attention_heads=_VISION_FULL["num_attention_heads"],
        num_query_groups=_VISION_FULL["num_attention_heads"],
        kv_channels=_VISION_FULL["kv_channels"],
        ffn_hidden_size=_VISION_FULL["ffn_hidden_size"],
        normalization="RMSNorm",
        layernorm_epsilon=vision_norm_epsilon(torch.bfloat16),
        add_bias_linear=False,
        add_qkv_bias=False,
        gated_linear_unit=False,
        activation_func=gelu_tanh,
        bias_activation_fusion=False,
        # MoonViT rotates adjacent (even, odd) channel pairs.
        rotary_interleaved=True,
        apply_rope_fusion=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)
