# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for Qwen3.8-Flash-Next VL.

The decoder is built by the registry's default hook (``hybrid_config_from_args`` +
``HybridModelBuilder``), which already selects the gated-residual stack spec, composes QSA
and attaches Engram from the CLI arguments; nothing here constructs decoder modules. The
factory pins the RoPE constants, checks the arguments that would otherwise build a decoder
that silently differs from ``Qwen4ExpForConditionalGeneration``, and builds the Qwen3.5 ViT
before the decoder (initialization RNG order).
"""

import functools

from examples.multimodal_dev.models.qwen35_vl.factory import (
    set_vision_flops_metadata as _qwen35_vl_vision_flops,
)
from examples.multimodal_dev.models.qwen38_vl.configuration import (
    MROPE_SECTION,
    QWEN38_VL_EOS_TOKEN_ID,
    ROTARY_BASE,
    ROTARY_PERCENT,
    resolve_image_token_id,
)


def post_language_config(language_config, args):
    """Pin ``mrope_interleaved``: Qwen3.8 interleaves T/H/W like Qwen3.5 (no CLI default)."""
    language_config.mrope_interleaved = True


def set_vision_flops_metadata(args, language_config, vision_config):
    """Same ViT as Qwen3.5-VL, so the same FLOPs knobs."""
    _qwen35_vl_vision_flops(args, language_config, vision_config)


def check_qwen38_vl_args(args, language_config) -> None:
    """Reject arguments that build a decoder other than HF ``qwen4_exp``'s.

    Each check names a mismatch that would still train to a plausible loss.
    """
    pattern = getattr(args, "hybrid_layer_pattern", None)
    if pattern is None:
        raise ValueError(
            "Qwen3.8-VL uses HybridModel and requires --hybrid-layer-pattern "
            "(e.g. GEGEGEQE... with one attention symbol G/Q and one MoE symbol E per layer)."
        )
    if getattr(args, "position_embedding_type", None) != "mrope":
        raise ValueError(
            "Qwen3.8-VL requires --position-embedding-type mrope: the decoder applies 3-D "
            "mRoPE, and QSA switches its indexer to per-token frequencies only under mRoPE."
        )
    if list(getattr(language_config, "mrope_section", None) or []) != list(MROPE_SECTION):
        raise ValueError(
            f"Qwen3.8-VL requires --mrope-section {' '.join(map(str, MROPE_SECTION))}, got "
            f"{getattr(language_config, 'mrope_section', None)}."
        )
    if not (
        getattr(args, "enable_hyper_connections", False)
        and getattr(args, "hyper_connection_variant", None) == "gated_residual"
    ):
        raise ValueError(
            "Qwen3.8-VL requires --enable-hyper-connections --hyper-connection-variant "
            "gated_residual (HF hc_count streams with the gated-residual mixer)."
        )
    if "Q" in pattern.split("/")[0] and getattr(args, "use_packed_sequence", False):
        if not getattr(args, "qsa_use_sparse_attention", False):
            raise ValueError(
                "QSA with --use-packed-sequence (THD, and therefore MDP) needs "
                "--qsa-use-sparse-attention: the dense-mask bridge is BSHD only."
            )
    if getattr(args, "engram_enabled", False):
        if getattr(args, "engram_variant", None) != "qwen":
            raise ValueError("Qwen3.8-VL's PLE is the Engram qwen variant (--engram-variant qwen).")
        if getattr(args, "engram_eos_token_id", None) != QWEN38_VL_EOS_TOKEN_ID:
            raise ValueError(
                f"--engram-eos-token-id must be the Qwen3.8 EOS {QWEN38_VL_EOS_TOKEN_ID}: it is "
                "the n-gram window boundary and the token collate padding is hashed as."
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
    """Build a :class:`Qwen38VLModel` for this pipeline stage.

    Args as ``qwen35_vl.factory.build_model``; ``language_model_builder`` is
    ``(**config_overrides) -> decoder`` (the registry hook bound by the entry point).
    """
    vp_stage = kwargs.get("vp_stage", None)
    check_qwen38_vl_args(args, language_config)

    from examples.multimodal_dev.models.qwen38_vl.model import Qwen38VLModel

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

    def _build_decoder():
        # Architectural: they set the rotary width the mRoPE section split is checked against.
        return language_model_builder(rotary_percent=ROTARY_PERCENT, rotary_base=ROTARY_BASE)

    return Qwen38VLModel(
        language_config=language_config,
        language_model_builder=_build_decoder,
        vision_config=vision_config,
        build_vision_encoder=not getattr(args, "mdp_enable", False),
        image_token_id=resolve_image_token_id(args),
        pre_process=pre_process,
        vp_stage=vp_stage,
    )
