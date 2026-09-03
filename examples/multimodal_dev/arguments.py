# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Extra CLI arguments for multimodal_dev standalone training."""


def validate_encoder_recompute_args(args) -> None:
    """Validate the shared native/MDP encoder recompute argument matrix."""
    granularity = getattr(args, "encoder_recompute_granularity", None)
    method = getattr(args, "encoder_recompute_method", None)
    num_layers = getattr(args, "encoder_recompute_num_layers", None)
    modules = getattr(args, "encoder_recompute_modules", None)

    if granularity == "whole" and not getattr(args, "mdp_enable", False):
        raise RuntimeError(
            "--encoder-recompute-granularity whole requires --mdp-enable"
        )

    if granularity in (None, "whole"):
        incompatible = {
            "encoder_recompute_method": method,
            "encoder_recompute_num_layers": num_layers,
            "encoder_recompute_modules": modules,
        }
    elif granularity == "selective":
        incompatible = {
            "encoder_recompute_method": method,
            "encoder_recompute_num_layers": num_layers,
        }
    else:  # full
        incompatible = {"encoder_recompute_modules": modules}

    invalid = [
        f"--{name.replace('_', '-')}"
        for name, value in incompatible.items()
        if value is not None
    ]
    if invalid:
        raise RuntimeError(
            f"{', '.join(invalid)} cannot be used with "
            f"--encoder-recompute-granularity {granularity}"
        )


def encoder_recompute_overrides_from_args(args) -> dict:
    """Return native TransformerConfig overrides for encoder recompute."""
    validate_encoder_recompute_args(args)
    granularity = getattr(args, "encoder_recompute_granularity", None)
    if granularity in (None, "whole"):
        return {}

    modules = getattr(args, "encoder_recompute_modules", None)
    return {
        "recompute_granularity": granularity,
        "recompute_method": getattr(args, "encoder_recompute_method", None),
        "recompute_num_layers": getattr(args, "encoder_recompute_num_layers", None),
        "recompute_modules": list(modules) if modules is not None else None,
    }


def validate_encoder_fp8_args(args) -> None:
    """Validate ``--encoder-fp8``.

    The flag runs the vision encoder under the decoder's ``--fp8-format`` /
    ``--fp8-recipe``; it carries no format or recipe of its own. It rides on
    MDP: the payload-row alignment quantized GEMMs need is supplied by the MDP
    adapter's ``encode`` (see ``mdp_adapter.py``), and the native path has no
    equivalent, so it would abort inside TE on the first unaligned vision
    batch. It also needs decoder FP8 to be on -- there is nothing to inherit
    otherwise, and encoder-only FP8 is not supported (measured as pure launch
    overhead on the encoder, and it would need its own recipe plumbing).
    """
    if not getattr(args, "encoder_fp8", False):
        return
    if not getattr(args, "mdp_enable", False):
        raise RuntimeError(
            "--encoder-fp8 requires --mdp-enable: quantized vision GEMMs need the "
            "encoder payload rows padded to the recipe's alignment, which only the "
            "MDP adapter's encode() supplies"
        )
    if getattr(args, "fp8", None) is None:
        raise RuntimeError(
            "--encoder-fp8 requires --fp8-format: the encoder inherits the decoder's "
            "FP8 format and recipe; enable decoder FP8 first (encoder-only FP8 is "
            "not supported)"
        )


def encoder_fp8_overrides_from_args(args) -> dict:
    """Return native TransformerConfig overrides for encoder FP8 (the decoder's)."""
    validate_encoder_fp8_args(args)
    if not getattr(args, "encoder_fp8", False):
        return {}
    return {"fp8": args.fp8, "fp8_recipe": getattr(args, "fp8_recipe", None)}


def validate_encoder_ffn_args(args) -> None:
    """Validate the vision-FFN width override and its zero-padding flag.

    ``--mdp-zero-pad-vision-ffn`` pads the checkpoint architecture's FFN up to
    ``--encoder-ffn-hidden-size``; with no target there is nothing to pad to.
    The zeroing itself runs in ``build_encoder_domain``, so the flag needs MDP.
    The width override alone is an ordinary architecture change (Approach A,
    checkpoint-incompatible) and is allowed on both paths.
    """
    ffn = getattr(args, "encoder_ffn_hidden_size", None)
    if ffn is not None and ffn <= 0:
        raise RuntimeError(f"--encoder-ffn-hidden-size must be positive, got {ffn}")
    if not getattr(args, "mdp_zero_pad_vision_ffn", False):
        return
    if not getattr(args, "mdp_enable", False):
        raise RuntimeError(
            "--mdp-zero-pad-vision-ffn requires --mdp-enable: the padding channels "
            "are zeroed in build_encoder_domain"
        )
    if ffn is None:
        raise RuntimeError(
            "--mdp-zero-pad-vision-ffn requires --encoder-ffn-hidden-size: it pads "
            "the checkpoint architecture's vision FFN up to that width"
        )


def encoder_ffn_overrides_from_args(args) -> dict:
    """Return the native TransformerConfig override for the vision FFN width."""
    validate_encoder_ffn_args(args)
    ffn = getattr(args, "encoder_ffn_hidden_size", None)
    return {} if ffn is None else {"ffn_hidden_size": ffn}


def add_multimodal_args(parser):
    """Add multimodal-specific arguments to the Megatron argument parser."""
    group = parser.add_argument_group(
        "Multimodal", "Multimodal model arguments",
    )

    group.add_argument(
        "--model-arch",
        type=str,
        default="qwen35_vl",
        help="Model architecture. Available: qwen35_vl",
    )
    group.add_argument(
        "--model-variant",
        type=str,
        default="proxy",
        help="Model variant (size). E.g. proxy, 9b, 397b_a17b",
    )
    group.add_argument(
        "--dataset-provider",
        type=str,
        default="mock",
        help="Dataset provider: mock",
    )
    group.add_argument(
        "--image-token-id",
        type=int,
        default=248056,
        help="Token ID for image placeholder tokens",
    )
    group.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (height and width) for mock data",
    )
    group.add_argument(
        "--total-seq-length",
        type=int,
        default=1024,
        help="Total sequence length for mock data",
    )
    group.add_argument(
        "--image-seq-length",
        type=int,
        default=256,
        help="Number of image tokens in mock data",
    )
    group.add_argument(
        "--vision-num-layers",
        type=int,
        default=None,
        help=(
            "Override for vision backbone depth. "
            "Useful for proxy perf runs."
        ),
    )
    group.add_argument(
        "--hf-processor-path",
        type=str,
        default=None,
        help=(
            "HuggingFace processor path for real VLM datasets "
            "(e.g. Qwen/Qwen2.5-VL-7B-Instruct)"
        ),
    )
    group.add_argument(
        "--use-packed-sequence",
        action="store_true",
        default=False,
        help=(
            "Pack variable-length sequences into THD format to eliminate "
            "padding waste."
        ),
    )
    group.add_argument(
        "--mdp-enable",
        action="store_true",
        default=False,
        help=(
            "Enable MDP (modality decoupled parallelism): balance vision "
            "items across each decoder replica's CP x PP encoder worker "
            "pool. Off by default; when absent, training is identical to "
            "the native path."
        ),
    )
    group.add_argument(
        "--mdp-encoder-cp",
        type=int,
        default=1,
        help="MDP encoder context-parallel width (must currently be 1).",
    )
    group.add_argument(
        "--mdp-encoder-max-payload-rows",
        type=int,
        default=None,
        help=(
            "Patch-row cap for one MDP encoder chunk; splitting happens "
            "only at complete vision-item boundaries."
        ),
    )
    group.add_argument(
        "--encoder-recompute-granularity",
        choices=("selective", "full", "whole"),
        default=None,
        help=(
            "Vision-encoder recompute granularity. 'selective' and 'full' use "
            "native MCore Transformer recompute in both native and MDP paths; "
            "'whole' runs the complete encoder under no_grad in P2 and replays "
            "it in P5, and therefore requires --mdp-enable."
        ),
    )
    group.add_argument(
        "--encoder-recompute-method",
        choices=("uniform", "block"),
        default=None,
        help=(
            "Layer partitioning method for --encoder-recompute-granularity full."
        ),
    )
    group.add_argument(
        "--encoder-recompute-num-layers",
        type=int,
        default=None,
        help=(
            "Number of vision Transformer layers per recompute unit for full "
            "Transformer recompute."
        ),
    )
    group.add_argument(
        "--encoder-recompute-modules",
        nargs="+",
        default=None,
        metavar="MODULE",
        help=(
            "Vision Transformer submodules to checkpoint when "
            "--encoder-recompute-granularity selective is enabled."
        ),
    )
    group.add_argument(
        "--encoder-fp8",
        action="store_true",
        default=False,
        help=(
            "Run the vision encoder's GEMMs in FP8 with the decoder's "
            "--fp8-format and --fp8-recipe (the encoder has no recipe of its "
            "own; FP8 attention is not enabled for it). Requires --mdp-enable "
            "and decoder FP8; the decoder recipe must be one of tensorwise, "
            "blockwise, mxfp8 -- delayed scaling is rejected (see "
            "megatron/core/mdp/config.py ENCODER_COMPATIBLE_FP8_RECIPES)."
        ),
    )
    group.add_argument(
        "--encoder-ffn-hidden-size",
        type=int,
        default=None,
        help=(
            "Build the vision encoder's FFN at this width instead of the "
            "architecture's (e.g. 4320 instead of Qwen3.5-VL's 4304, MXFP8's "
            "32-channel block alignment). On its own this changes the "
            "architecture and official checkpoints no longer load; pair it with "
            "--mdp-zero-pad-vision-ffn to keep them loadable."
        ),
    )
    group.add_argument(
        "--mdp-zero-pad-vision-ffn",
        action="store_true",
        default=False,
        help=(
            "Zero-pad the vision FFN's real (checkpoint) ffn_hidden_size up to "
            "--encoder-ffn-hidden-size instead of changing the architecture "
            "outright. The padding channels are zero-initialized on both "
            "linear_fc1's output rows and linear_fc2's input columns; since the "
            "vision MLP has no normalization between them, GELU(0)=0 and the "
            "chain rule keep those channels at exactly zero forever, so the "
            "padded model stays numerically identical to the unpadded one and "
            "loadable from official (unpadded) checkpoints. The reverse "
            "direction is not implemented: a padded model is saved at its "
            "padded width and cannot be read back by the official architecture. "
            "Requires --mdp-enable and --encoder-ffn-hidden-size."
        ),
    )
    group.add_argument(
        "--mdp-locality-slack-permille",
        type=int,
        default=10,
        help="LPT near-equal-load window in per-mille (default 10 = 1%%).",
    )
    group.add_argument(
        "--mdp-row-alignment",
        type=int,
        default=1,
        help="MDP row-capacity alignment (1 in production; tests may use 16).",
    )
    group.add_argument(
        "--mdp-plan-check-interval",
        type=int,
        default=1,
        help=(
            "Plan-digest consistency check interval in iterations; must be "
            ">= 1 (the check can be sampled but never fully disabled)."
        ),
    )
    group.add_argument(
        "--mdp-overlap-window-capture",
        action="store_true",
        default=False,
        help=(
            "Prefetch the next iteration's data window on a background "
            "thread and a dedicated side CUDA stream while the current "
            "iteration runs, hiding the serial P1 window-capture cost "
            "without inserting H2D copies into the main compute stream. "
            "TP=1 only."
        ),
    )
    group.add_argument(
        "--mdp-pixel-locality",
        action="store_true",
        default=False,
        help=(
            "Prefer assigning a vision item to its pixel owner within the LPT slack "
            "(--mdp-locality-slack-permille), trading load balance for less "
            "pixel traffic."
        ),
    )
    group.add_argument(
        "--mdp-debug-plan-payload-check",
        action="store_true",
        default=False,
        help="Additionally compare canonical plan payloads (debug only).",
    )
    group.add_argument(
        "--use-vanilla-collate-fn",
        action="store_true",
        default=False,
        help=(
            "Use vanilla collate function to collate the data."
        ),
    )

    return parser
