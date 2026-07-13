# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Factory functions for Qwen3 text-only MoE model construction.

Encapsulates Qwen3-specific logic so ``pretrain_multimodal.py`` remains
model-agnostic via the ``MODEL_REGISTRY`` indirection.
"""

from examples.multimodal_dev.models.qwen3.specs import get_qwen3_language_spec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

QWEN3_ROTARY_BASE = 1_000_000


class Qwen3TextOnlyGPTModel(GPTModel):
    """GPTModel + a thin shim that swallows the multimodal kwargs the shared
    ``forward_step.forward_step`` passes unconditionally
    (``pixel_values``, ``image_grid_thw``).

    The forward step is shared with the qwen35_vl path, where those kwargs
    are consumed by the ``MultimodalModel`` wrapper. A bare ``GPTModel``
    raises ``TypeError`` on the unknown kwargs; this subclass intercepts
    them at the call boundary and forwards only GPT-relevant kwargs to the
    base class. ``build_schedule_plan`` (used only when
    ``--overlap-moe-expert-parallel-comm`` is set) is patched the same way.
    """

    def forward(self, *args, pixel_values=None, image_grid_thw=None, **kwargs):
        return super().forward(*args, **kwargs)

    def build_schedule_plan(self, *args, pixel_values=None, image_grid_thw=None, **kwargs):
        return super().build_schedule_plan(*args, **kwargs)


def get_qwen3_vision_config(num_layers_override=None, variant=None):
    """Stub vision config for the text-only Qwen3 arch.

    The MODEL_REGISTRY interface declares ``vision_config_fn`` as required —
    this returns a tiny throwaway TransformerConfig that ``build_model``
    ignores. Constructed minimally so its attributes don't crash any
    framework-side metadata reads.
    """
    del num_layers_override, variant
    return TransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=1, ffn_hidden_size=64, kv_channels=64
    )


def post_language_config(language_config, args):
    """Apply Qwen3-specific overrides to the language TransformerConfig.

    Called after ``core_transformer_config_from_args`` to select the Qwen3
    language settings used by this model path.
    """
    if hasattr(language_config, "linear_attention_freq"):
        language_config.linear_attention_freq = None
    # Qwen3 uses standard 1D RoPE — disable any MRoPE config.
    if hasattr(language_config, "mrope_section"):
        language_config.mrope_section = None
    if hasattr(language_config, "mrope_interleaved"):
        language_config.mrope_interleaved = False


def set_vision_flops_metadata(args, language_config, vision_config):
    """No-op for Qwen3 (text-only). Sets all vision FLOPs metadata to 0."""
    args.count_vision_model_flops = False
    args.vision_flops_variant = "none"
    args.vision_num_layers = 0
    args.vision_hidden_size = 0
    args.vision_ffn_hidden_size = 0
    args.vision_num_attention_heads = 0
    args.vision_kv_channels = 0
    args.vision_in_channels = 0
    args.vision_patch_size = 0
    args.vision_temporal_patch_size = 0
    args.vision_spatial_merge_size = 0
    args.vision_out_hidden_size = 0


def build_model(args, language_config, vision_config, **kwargs):
    """Build a Qwen3 text-only MoE GPTModel.

    Args:
        args: Megatron parsed arguments (used for vocab_size,
            max_position_embeddings, mtp_num_layers, transformer_impl).
        language_config: TransformerConfig (already post-processed by
            :func:`post_language_config`).
        vision_config: ignored (text-only).
        **kwargs: Recognised: ``vp_stage``, ``mp_padding_needed``.

    Returns:
        GPTModel instance with Qwen3 layer spec.
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

    return Qwen3TextOnlyGPTModel(
        config=language_config,
        transformer_layer_spec=language_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        position_embedding_type="rope",
        rotary_percent=getattr(args, "rotary_percent", 1.0),
        rotary_base=int(getattr(args, "rotary_base", QWEN3_ROTARY_BASE)),
        mtp_block_spec=mtp_block_spec,
        pre_process=kwargs.get("pre_process", True),
        post_process=kwargs.get("post_process", True),
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        vp_stage=kwargs.get("vp_stage", None),
    )
