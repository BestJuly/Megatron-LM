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
