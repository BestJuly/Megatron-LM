# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for Qwen3.5-VL model construction.

Encapsulates all Qwen3.5-VL-specific logic needed by ``pretrain_multimodal.py``
so that the training entry point remains model-agnostic.
"""

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    MROPE_SECTION,
    ROTARY_BASE,
    ROTARY_PERCENT,
    VISION_KWARGS,
    resolve_image_token_id,
)


def post_language_config(language_config, args):
    """Apply Qwen3.5-VL-specific settings to the language TransformerConfig.

    Called after ``core_transformer_config_from_args`` to inject model-specific
    fields that cannot be expressed via CLI args alone.

    ``mrope_section`` is deliberately not set here: it is a ``TransformerConfig``
    field fed by ``--mrope-section``, so overriding it would let the recipe and
    the constructed model disagree. :func:`build_model` checks the value the
    recipe supplied against the architecture constant instead.

    ``mrope_interleaved`` goes the other way. ``--mrope-interleaved`` also exists
    as a generated flag, but it is ``store_true`` with a ``False`` default, so a
    recipe that forgot it would silently build a non-interleaved decoder.
    Qwen3.5 always interleaves the T/H/W sections, so it is pinned here as an
    architectural constant, alongside ``ROTARY_PERCENT`` / ``ROTARY_BASE`` in
    ``model.py``. ``mrope_section`` cannot get the same treatment because
    ``validate_args`` requires it on the CLI before the model is built.
    """
    language_config.mrope_interleaved = True


def set_vision_flops_metadata(args, language_config, vision_config):
    """Expose Qwen3.5-VL vision-model dimensions for FLOPs estimation.

    Only dimensions and shape flags live here. The per-batch row counts the
    formula needs (patch rows, packed-attention ``L^2``, merged rows) are
    measured at collate time through ``VisionCollateSpec.workload_of``; nothing
    downstream reads these attributes to derive geometry.
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
    args.vision_temporal_patch_size = VISION_KWARGS["temporal_patch_size"]
    args.vision_out_hidden_size = language_config.hidden_size
    # Plain (non-gated) GELU MLP, and a two-layer merger over the concatenated
    # m x m patch window.
    args.vision_gated_mlp = False
    args.vision_merge_dim = vision_config.hidden_size * merge * merge
    args.vision_merger_hidden_size = args.vision_merge_dim


def build_model(
    args,
    language_config,
    vision_config,
    pre_process: bool = True,
    post_process: bool = True,
    language_model_builder=None,
    **kwargs,
):
    """Build a complete Qwen3.5-VL model instance.

    The decoder comes from *language_model_builder* (the registry hook bound by
    the entry point; default ``models.base.build_language_model``, i.e.
    ``hybrid_config_from_args`` + ``HybridModelBuilder``), so ``--spec`` and the
    builder's full keyword set apply. Qwen3.5-VL pins only what is
    architectural: the RoPE fraction and base.

    Args:
        args: Megatron parsed arguments.
        language_config: ``TransformerConfig`` for the language decoder
            (already post-processed by :func:`post_language_config`).
        vision_config: ``TransformerConfig`` for the vision encoder.
        pre_process: First PP stage flag — vision encoder + embedding live here.
        post_process: Last PP stage flag — output projection + loss live here.
        language_model_builder: ``(**config_overrides) -> decoder``. When
            ``None`` (direct callers, tests) the default hook is bound here.
        **kwargs: Extra keyword arguments (e.g. ``vp_stage``).

    Returns:
        A :class:`Qwen35VLModel` instance.
    """
    vp_stage = kwargs.get("vp_stage", None)

    hybrid_layer_pattern = getattr(args, "hybrid_layer_pattern", None)
    if hybrid_layer_pattern is None:
        raise ValueError(
            "Qwen3.5-VL uses HybridModel and requires --hybrid-layer-pattern. "
            "Use GEGEGE*E per four MoE blocks (or G-G-G-*- for dense blocks), "
            "and append /*E or /*- for each MTP depth."
        )

    position_embedding_type = getattr(args, "position_embedding_type", None)
    if position_embedding_type != "mrope":
        raise ValueError(
            "Qwen3.5-VL requires --position-embedding-type mrope, but got "
            f"'{position_embedding_type}'. The decoder applies 3D MRoPE, so any "
            "other setting would make the parsed args and checkpoint metadata "
            "describe a different model than the one being trained."
        )

    # The T/H/W split is architectural, not a tuning knob: it is tied to
    # kv_channels and ROTARY_PERCENT. A section with the right total width but
    # the wrong partition builds an incompatible rotary layout that trains to a
    # plausible loss, so check the recipe value rather than documenting it.
    mrope_section = getattr(language_config, "mrope_section", None)
    if list(mrope_section or []) != list(MROPE_SECTION):
        raise ValueError(
            f"Qwen3.5-VL requires --mrope-section {' '.join(str(s) for s in MROPE_SECTION)}, "
            f"but got {mrope_section}. The section split is tied to kv_channels and "
            f"ROTARY_PERCENT; a mismatch silently misplaces the T/H/W channel boundaries."
        )

    from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel

    if language_model_builder is None:
        import functools

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
        # RoPE fraction and base are fixed by the architecture (they set the
        # rotary width the mRoPE section split above is checked against), so
        # they are pinned here rather than read from the recipe.
        return language_model_builder(rotary_percent=ROTARY_PERCENT, rotary_base=ROTARY_BASE)

    return Qwen35VLModel(
        language_config=language_config,
        language_model_builder=_build_decoder,
        vision_config=vision_config,
        # Under MDP the replicated encoder domain owns the vision weights.
        build_vision_encoder=not getattr(args, "mdp_enable", False),
        image_token_id=resolve_image_token_id(args),
        pre_process=pre_process,
        vp_stage=vp_stage,
    )
