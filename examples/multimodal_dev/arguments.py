# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

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
        help="Model architecture registered in MODEL_REGISTRY.",
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
        help="Dataset provider registered for the selected model architecture",
    )
    group.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split used by the Energon provider",
    )
    group.add_argument(
        "--energon-path",
        type=str,
        default=None,
        help="Prepared Energon dataset or metadataset path",
    )
    group.add_argument(
        "--energon-packing-buffer-size",
        type=int,
        default=1024,
        help="Number of encoded samples available to Energon packing",
    )
    group.add_argument(
        "--energon-shuffle-buffer-size",
        type=int,
        default=1000,
        help="Energon training shuffle buffer size",
    )
    group.add_argument(
        "--energon-max-samples-per-sequence",
        type=int,
        default=100,
        help="Maximum source documents in one packed sequence",
    )
    group.add_argument(
        "--energon-prefetch-factor",
        type=int,
        default=2,
        help="Energon worker prefetch factor",
    )
    group.add_argument(
        "--dataloader-sequence-packing",
        action="store_true",
        help=(
            "Build deterministic packed THD items in the data provider. "
            "The Energon provider already owns this packing path; this flag "
            "preserves the b436 launcher contract."
        ),
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
        "--image-size-w",
        type=int,
        default=None,
        help=(
            "Image width for mock data. Defaults to --image-size for "
            "square images."
        ),
    )
    group.add_argument(
        "--image-sizes-h",
        type=str,
        default=None,
        help=(
            "Delimited per-image heights for mock data. The number of "
            "values must equal --num-images."
        ),
    )
    group.add_argument(
        "--image-sizes-w",
        type=str,
        default=None,
        help=(
            "Delimited per-image widths for mock data. The number of "
            "values must equal --num-images."
        ),
    )
    group.add_argument(
        "--image-min-pixels",
        type=int,
        default=0,
        help="Minimum image area; zero disables the lower bound",
    )
    group.add_argument(
        "--image-max-pixels",
        type=int,
        default=0,
        help="Maximum image area; zero disables the upper bound",
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
        "--num-images",
        type=int,
        default=1,
        help="Number of fixed-shape mock images in each sample",
    )
    group.add_argument(
        "--mock-layout",
        type=str,
        default="single",
        choices=["single", "interleaved"],
        help=(
            "Mock image/text layout. 'single' groups the image blocks; "
            "'interleaved' separates them with text."
        ),
    )
    group.add_argument(
        "--mock-pack-num-docs",
        type=int,
        default=1,
        help=(
            "Number of fixed mock documents packed into one THD sample. "
            "The value must divide --total-seq-length and requires "
            "--mock-layout single."
        ),
    )
    group.add_argument(
        "--text-only",
        action="store_true",
        default=False,
        help=(
            "Mock dataset: emit plain text positions and empty vision "
            "tensors. Implied by --model-arch qwen3."
        ),
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
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional HuggingFace tokenizer path for the Energon provider",
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
        "--use-vanilla-collate-fn",
        action="store_true",
        default=False,
        help=(
            "Use vanilla collate function to collate the data."
        ),
    )

    return parser
