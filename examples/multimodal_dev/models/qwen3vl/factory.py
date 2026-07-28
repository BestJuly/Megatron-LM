# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for Qwen3-VL hybrid model construction.

Hybrid architecture:
    * Vision encoder, MRoPE, multimodal vocab, compute_position_ids <-
      ``qwen35_vl`` (production-tested vision tower; 27 layers, hidden=1152).
    * LLM decoder spec <- ``qwen3`` via ``get_gpt_decoder_block_spec``.
    * LLM decoder hyperparams (from launcher): 48 layers, hidden=2048,
      ffn=6144, num_attention_heads=32, num_query_groups=4, kv_channels=128,
      128 experts x top-8, no shared expert.
    * MRoPE adapted for qwen3's head_dim=128: ``rotary_percent=0.5`` and
      ``mrope_section=[11, 11, 10]`` (32 pairs = 64 elements; preserves
      the qwen35-VL [t,h,w] partition ratio).
    * Vocab: qwen35-VL's 248320 (preserves ``image_token_id=248056`` and
      the multimodal token block).
"""

from examples.multimodal_dev.models.qwen3.specs import get_qwen3_language_spec
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    MROPE_SECTION,
    QWEN35_VL_IMAGE_TOKEN_ID,
    get_qwen35_vl_vision_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import (
    set_vision_flops_metadata as _qwen35_vl_set_vision_flops_metadata,
)
from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel
from examples.multimodal_dev.mdp_pipeline_sidecar import (
    pp_cp_replicated_vision_requested,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

# qwen35-VL uses rotary_percent=0.25 with
# kv_channels=256 (= 64 RoPE elements = 32 pairs); qwen3 has head_dim=128.
# rotary_percent=0.5 keeps the same 32-pair RoPE budget for MRoPE
# section [11, 11, 10] (matches qwen35-VL's vision-side [t,h,w] partition).
QWEN3VL_ROTARY_PERCENT: float = 0.5


def get_qwen3vl_vision_config(num_layers_override=None, variant=None):
    """Vision config - same 27-layer ViT as qwen35-VL.

    Identity import; the hybrid does not modify the vision tower.
    """
    return get_qwen35_vl_vision_config(num_layers_override=num_layers_override, variant=variant)


def post_language_config(language_config, args):
    """Apply qwen3-VL hybrid settings to the language TransformerConfig.

    Sets MRoPE carried over from qwen35-VL and applies the Qwen3 language
    settings used by this workload.
    """
    language_config.mrope_section = list(MROPE_SECTION)
    language_config.mrope_interleaved = True
    if hasattr(language_config, "linear_attention_freq"):
        language_config.linear_attention_freq = None


def set_vision_flops_metadata(args, language_config, vision_config):
    """Vision FLOPs metadata - identical to qwen35-VL (real vision tower)."""
    _qwen35_vl_set_vision_flops_metadata(args, language_config, vision_config)


def build_model(args, language_config, vision_config, **kwargs):
    """Build the Qwen3-VL hybrid model.

    Args:
        args: Megatron parsed arguments.
        language_config: ``TransformerConfig`` for the qwen3-spec decoder
            (already post-processed by :func:`post_language_config`).
        vision_config: ``TransformerConfig`` for the qwen35-VL ViT.
        **kwargs: Optional model-construction arguments.

    Returns:
        A :class:`Qwen35VLModel` instance with qwen3 language spec and
        ``rotary_percent=0.5``. Class is reused from the qwen35_vl path
        (its ``compute_position_ids`` MRoPE logic is what we want); only
        the LLM layer spec and rotary_percent differ.
    """
    language_spec = get_qwen3_language_spec(
        config=language_config, vp_stage=kwargs.get("vp_stage", None), pp_rank=None
    )

    mtp_block_spec = None
    if getattr(args, "mtp_num_layers", None):
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=language_config,
            spec=language_spec,
            use_transformer_engine=(args.transformer_impl == "transformer_engine"),
            vp_stage=kwargs.get("vp_stage", None),
            pp_rank=None,
        )

    pre_process = kwargs.get("pre_process", True)
    post_process = kwargs.get("post_process", True)
    vp_stage = kwargs.get("vp_stage", None)
    replicate_vision = pp_cp_replicated_vision_requested(args)
    # VPP is now supported: non-sidecar chunks are filtered by pre_process in
    # mdp_model_setup.configure_mdp_model before this factory is called.

    return Qwen35VLModel(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        mtp_block_spec=mtp_block_spec,
        parallel_output=kwargs.get("parallel_output", True),
        rotary_percent=getattr(args, "rotary_percent", QWEN3VL_ROTARY_PERCENT),
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        build_vision_model=pre_process or replicate_vision,
    )
