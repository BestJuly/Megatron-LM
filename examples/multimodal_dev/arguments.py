# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extra CLI arguments for multimodal_dev standalone training."""


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
        "--recompute-vision",
        action="store_true",
        default=False,
        help=(
            "Enable full activation recomputation for vision encoder layers. "
            "Uses uniform method and recomputes every layer. "
            "Independent of the decoder --recompute-* flags."
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
        "--mdp-vision-config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Vision TransformerConfig override entry (repeatable). Keys "
            "are restricted to the MDP allowlist (recompute_granularity, "
            "recompute_method, recompute_num_layers, recompute_modules)."
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
        "--mdp-greedy-packing",
        action="store_true",
        default=False,
        help=(
            "Fill each decoder microbatch to a token budget "
            "(--max-seqlen-per-dp-cp-rank x CP) by consuming as many samples as "
            "it takes, instead of a fixed --micro-batch-size count. "
            "IMPORTANT: this REINTERPRETS --micro-batch-size and "
            "--global-batch-size. They no longer describe what goes into a "
            "microbatch; they only set the number of bins per iteration "
            "(N = GBS / (MBS x DP)). The sample count per iteration then floats, "
            "so --global-batch-size means 'N x token budget' and loss curves are "
            "not iteration-by-iteration comparable against a fixed-GBS run. "
            "Requires --max-seqlen-per-dp-cp-rank. Independent of "
            "--thd-static-packing."
        ),
    )
    group.add_argument(
        "--mdp-mock-dataset-config-json",
        type=str,
        default=None,
        help=(
            "Sequence-length distribution for the MDP mock dataset, as JSON or "
            "a path to a JSON file. Same schema as "
            "--varlen-mock-dataset-config-json, e.g. "
            '\'{"mode":"distribution","type":"lognormal","min_seq_len":512,'
            '"max_seq_len":4096,"mean_seq_len":2048,"lognormal_sigma":1.1}\'. '
            "A dedicated flag because --varlen-mock-dataset-config-json is only "
            "honored under --use-varlen-dataset, which auto-sets the packing "
            "scheduler MDP must not have. Unset keeps the built-in "
            "[1000, 2000] uniform range."
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
