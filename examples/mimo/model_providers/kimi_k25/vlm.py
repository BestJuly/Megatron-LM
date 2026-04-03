# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model provider for Kimi K2.5 VL.

Builds a standalone ``KimiK25VLModel`` whose submodule names
(``language_model``, ``vision_tower``, ``mm_projector``) match the
Megatron-Bridge checkpoint layout exactly, enabling direct checkpoint
loading with no key remapping.

Supported variants (via ``--model-variant`` CLI arg):
  ``proxy``   4 layers, 16 experts — single-node bring-up
  ``full``    61 layers, 256 experts — production (default)
"""

import torch

from megatron.core.models.gpt.gpt_model import GPTModel

from . import KIMI_K25_IMAGE_TOKEN_ID, KIMI_K25_VOCAB_SIZE, get_kimi_k25_language_config
from .model import KimiK25VLModel
from .specs import get_kimi_k25_language_spec


# Default HF model path
_DEFAULT_HF_MODEL_PATH = "moonshotai/Kimi-K2.5"


def _apply_args_to_language_config(cfg):
    """Apply Megatron CLI args to the language MLATransformerConfig."""
    try:
        from megatron.training import get_args
        args = get_args()
    except (ModuleNotFoundError, AssertionError):
        return cfg

    if getattr(args, "bf16", False):
        cfg.bf16 = True
    if getattr(args, "fp16", False):
        cfg.fp16 = True

    cfg.tensor_model_parallel_size = getattr(args, "tensor_model_parallel_size", 1)
    cfg.pipeline_model_parallel_size = getattr(args, "pipeline_model_parallel_size", 1)
    cfg.expert_model_parallel_size = getattr(args, "expert_model_parallel_size", 1)
    cfg.sequence_parallel = getattr(args, "sequence_parallel", False)
    cfg.context_parallel_size = getattr(args, "context_parallel_size", 1)

    if getattr(args, "use_distributed_optimizer", False):
        cfg.use_distributed_optimizer = True
    if getattr(args, "init_model_with_meta_device", False):
        cfg.init_model_with_meta_device = True
    if getattr(args, "moe_token_dispatcher_type", None) is not None:
        cfg.moe_token_dispatcher_type = args.moe_token_dispatcher_type
    if getattr(args, "moe_grouped_gemm", None) is not None:
        cfg.moe_grouped_gemm = args.moe_grouped_gemm

    return cfg


def _get_variant() -> str:
    try:
        from megatron.training import get_args
        return getattr(get_args(), "model_variant", "full") or "full"
    except (ModuleNotFoundError, AssertionError):
        return "full"


def _get_hf_model_path() -> str:
    import os
    env_path = os.environ.get("KIMI_K25_HF_MODEL_PATH")
    if env_path:
        return env_path
    try:
        from megatron.training import get_args
        path = getattr(get_args(), "hf_model_path", None)
        if path:
            return path
    except (ModuleNotFoundError, AssertionError):
        pass
    return _DEFAULT_HF_MODEL_PATH


def model_provider_kimi_k25_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = KIMI_K25_IMAGE_TOKEN_ID,
):
    """Build a Kimi K2.5 VL model with Bridge-compatible checkpoint layout.

    Submodule names match Bridge exactly:
        language_model.*  /  vision_tower.*  /  mm_projector.*
    """
    variant = _get_variant()
    hf_model_path = _get_hf_model_path()

    # --- Language config & spec ---
    language_config = get_kimi_k25_language_config(variant=variant)
    _apply_args_to_language_config(language_config)
    layer_spec = get_kimi_k25_language_spec(language_config)

    seq_length = 4096
    try:
        from megatron.training import get_args
        seq_length = getattr(get_args(), "seq_length", 4096)
    except (ModuleNotFoundError, AssertionError):
        pass

    # --- Build language model (GPTModel) ---
    language_model = GPTModel(
        config=language_config,
        transformer_layer_spec=layer_spec,
        vocab_size=KIMI_K25_VOCAB_SIZE,
        max_sequence_length=seq_length,
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
    )

    # --- Build VL model ---
    model = KimiK25VLModel(
        language_model=language_model,
        hf_model_path=hf_model_path,
        media_placeholder_token_id=image_special_token_id,
        freeze_vision_model=True,
        freeze_vision_projection=True,
        pre_process=pre_process,
    )

    print("*" * 80)
    print(f"KimiK25VLModel [variant={variant}]")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total / 1e9:.3f}B")
    print(f"  Trainable params: {trainable / 1e9:.3f}B")
    print(f"  Vision frozen:    True")
    print(f"  HF model path:    {hf_model_path}")
    print("*" * 80)

    # --- Load Bridge checkpoint if provided ---
    try:
        from megatron.training import get_args
        _args = get_args()
        ckpt_path = getattr(_args, "language_model_checkpoint", None)
        if ckpt_path is not None:
            _load_bridge_checkpoint(model, ckpt_path)
    except (ModuleNotFoundError, AssertionError):
        pass

    return model


def _load_bridge_checkpoint(model: KimiK25VLModel, ckpt_path: str):
    """Load a Bridge checkpoint directly — keys match exactly."""
    import os
    from megatron.core import dist_checkpointing

    # Resolve iteration directory
    ckpt_dir = ckpt_path
    latest_file = os.path.join(ckpt_path, "latest_checkpointed_iteration.txt")
    if os.path.exists(latest_file):
        with open(latest_file) as f:
            iteration = f.read().strip()
        ckpt_dir = os.path.join(ckpt_path, f"iter_{int(iteration):07d}")

    # Build sharded state dict — keys are language_model.*, vision_tower.*, mm_projector.*
    sharded_sd = model.sharded_state_dict(prefix="")

    # Remove extra_state keys (fp8 states may not exist in checkpoint)
    for k in list(sharded_sd.keys()):
        if "extra_state" in k:
            del sharded_sd[k]

    wrapper = {"state_dict": sharded_sd}
    loaded = dist_checkpointing.load(sharded_state_dict=wrapper, checkpoint_dir=ckpt_dir)

    incompatible = model.load_state_dict(loaded["state_dict"], strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "extra_state" not in k]
    if unexpected:
        print(f"[load_bridge_checkpoint] Unexpected: {unexpected[:5]}...")
    if missing:
        print(f"[load_bridge_checkpoint] Missing: {missing[:5]}...")

    print(f"[load_bridge_checkpoint] Loaded from {ckpt_dir}")
