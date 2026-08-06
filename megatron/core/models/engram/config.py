# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration and deterministic table allocation for Engram."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

TOKENIZER_MAP_FORMAT = "megatron-engram-token-map"
TOKENIZER_MAP_VERSION = 1


def is_prime(value: int) -> bool:
    """Return whether ``value`` is prime using deterministic trial division."""
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    limit = math.isqrt(value)
    divisor = 5
    while divisor <= limit:
        if value % divisor == 0 or value % (divisor + 2) == 0:
            return False
        divisor += 6
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Return the first unused prime strictly greater than ``start``."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def allocate_table_sizes(
    global_vocab_sizes: tuple[int, ...], layer_ids: tuple[int, ...], num_hash_heads: int
) -> dict[int, tuple[int, ...]]:
    """Allocate the official distinct prime table sizes for every layer and head."""
    seen_primes: set[int] = set()
    result: dict[int, tuple[int, ...]] = {}
    for layer_id in layer_ids:
        layer_sizes = []
        for vocab_size in global_vocab_sizes:
            search_start = vocab_size - 1
            for _ in range(num_hash_heads):
                prime = find_next_prime(search_start, seen_primes)
                seen_primes.add(prime)
                layer_sizes.append(prime)
                search_start = prime
        result[layer_id] = tuple(layer_sizes)
    return result


@dataclass
class EngramConfig:
    """Configuration for DeepSeek Engram.

    The tokenizer remap and PCG64-generated hash multipliers are loaded from a
    versioned offline artifact. They are Torch tensors so model forward never
    imports a tokenizer or NumPy.
    """

    global_vocab_sizes: tuple[int, ...]
    layer_ids: tuple[int, ...]
    max_ngram_order: int
    num_hash_heads: int
    memory_dim: int
    kernel_size: int
    hash_seed: int
    pad_token_id: int
    tokenizer_map_path: str
    embedding_lr_multiplier: float = 5.0
    embedding_weight_decay: float = 0.0
    tokenizer_remap: torch.Tensor = field(init=False, repr=False)
    compressed_pad_token_id: int = field(init=False)
    compressed_vocab_size: int = field(init=False)
    layer_multipliers: dict[int, tuple[int, ...]] = field(init=False, repr=False)
    table_sizes_by_layer: dict[int, tuple[int, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.global_vocab_sizes = tuple(self.global_vocab_sizes)
        self.layer_ids = tuple(self.layer_ids)
        self._validate_values()
        self._load_tokenizer_map()
        self.table_sizes_by_layer = allocate_table_sizes(
            self.global_vocab_sizes, self.layer_ids, self.num_hash_heads
        )

    @classmethod
    def from_args(cls, args: Any, transformer_config: Any) -> EngramConfig | None:
        """Build and validate Engram configuration from Megatron CLI/YAML arguments."""
        if getattr(args, "engram_vocab_sizes", None) is None:
            return None
        config = cls(
            global_vocab_sizes=tuple(args.engram_vocab_sizes),
            layer_ids=tuple(args.engram_layer_ids or ()),
            max_ngram_order=args.engram_max_ngram_order,
            num_hash_heads=args.engram_num_hash_heads,
            memory_dim=args.engram_memory_dim,
            kernel_size=args.engram_kernel_size,
            hash_seed=args.engram_hash_seed,
            pad_token_id=args.engram_pad_token_id,
            tokenizer_map_path=args.engram_tokenizer_map,
            embedding_lr_multiplier=args.engram_embedding_lr_multiplier,
            embedding_weight_decay=args.engram_embedding_weight_decay,
        )
        config.validate_startup(
            transformer_config,
            expected_tokenizer_vocab_size=args.padded_vocab_size,
            packed_sequences=bool(
                getattr(args, "sft", False)
                or getattr(args, "use_varlen_dataset", False)
                or getattr(args, "sequence_packing_scheduler", None) is not None
            ),
            use_fsdp=bool(
                getattr(args, "use_torch_fsdp2", False) or getattr(args, "use_megatron_fsdp", False)
            ),
        )
        return config

    @property
    def num_tables(self) -> int:
        """Number of prime-sized tables in one Engram module."""
        return (self.max_ngram_order - 1) * self.num_hash_heads

    @property
    def head_dim(self) -> int:
        """Embedding width of one hash head."""
        return self.memory_dim // self.num_hash_heads

    @property
    def total_memory_dim(self) -> int:
        """Concatenated retrieved-memory width over all n-gram orders."""
        return (self.max_ngram_order - 1) * self.memory_dim

    def table_sizes(self, layer_id: int) -> tuple[int, ...]:
        """Return prime table sizes for a selected global layer."""
        try:
            return self.table_sizes_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"Engram is not configured for global layer {layer_id}.") from exc

    def multipliers(self, layer_id: int) -> tuple[int, ...]:
        """Return official hash multipliers for a selected global layer."""
        try:
            return self.layer_multipliers[layer_id]
        except KeyError as exc:
            raise ValueError(f"Tokenizer map has no multipliers for layer {layer_id}.") from exc

    def _validate_values(self) -> None:
        if self.max_ngram_order < 2:
            raise ValueError("engram_max_ngram_order must be at least 2.")
        expected_vocab_sizes = self.max_ngram_order - 1
        if len(self.global_vocab_sizes) != expected_vocab_sizes:
            raise ValueError(
                "engram_vocab_sizes must contain exactly one value for each n-gram order "
                f"2..{self.max_ngram_order}; expected {expected_vocab_sizes}, "
                f"got {len(self.global_vocab_sizes)}."
            )
        if any(size <= 0 for size in self.global_vocab_sizes):
            raise ValueError("Every engram_vocab_sizes value must be positive.")
        if not self.layer_ids:
            raise ValueError("engram_layer_ids must contain at least one 1-based layer ID.")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("engram_layer_ids must be unique.")
        if any(layer_id < 1 for layer_id in self.layer_ids):
            raise ValueError("engram_layer_ids are 1-based and must be positive.")
        if self.num_hash_heads <= 0:
            raise ValueError("engram_num_hash_heads must be positive.")
        if self.memory_dim <= 0:
            raise ValueError("engram_memory_dim must be positive.")
        if self.memory_dim % self.num_hash_heads != 0:
            raise ValueError(
                "engram_memory_dim must be divisible by engram_num_hash_heads; "
                f"got {self.memory_dim} and {self.num_hash_heads}."
            )
        if self.kernel_size <= 0:
            raise ValueError("engram_kernel_size must be positive.")
        if self.pad_token_id < 0:
            raise ValueError("engram_pad_token_id must be nonnegative.")
        if self.embedding_lr_multiplier <= 0:
            raise ValueError("engram_embedding_lr_multiplier must be positive.")
        if self.embedding_weight_decay < 0:
            raise ValueError("engram_embedding_weight_decay must be nonnegative.")
        if not self.tokenizer_map_path:
            raise ValueError("engram_tokenizer_map is required when Engram is enabled.")

    def _load_tokenizer_map(self) -> None:
        path = Path(self.tokenizer_map_path)
        if not path.is_file():
            raise ValueError(f"Engram tokenizer-map artifact does not exist: {path}")
        with path.open("r", encoding="utf-8") as artifact_file:
            artifact = json.load(artifact_file)

        if artifact.get("format") != TOKENIZER_MAP_FORMAT:
            raise ValueError(
                "Invalid Engram tokenizer-map format; expected "
                f"{TOKENIZER_MAP_FORMAT!r}, got {artifact.get('format')!r}."
            )
        if artifact.get("version") != TOKENIZER_MAP_VERSION:
            raise ValueError(
                "Unsupported Engram tokenizer-map version; expected "
                f"{TOKENIZER_MAP_VERSION}, got {artifact.get('version')!r}."
            )
        for field_name, expected in (
            ("max_ngram_order", self.max_ngram_order),
            ("hash_seed", self.hash_seed),
            ("pad_token_id", self.pad_token_id),
        ):
            if artifact.get(field_name) != expected:
                raise ValueError(
                    f"Engram tokenizer-map {field_name} mismatch: expected {expected}, "
                    f"got {artifact.get(field_name)!r}."
                )

        artifact_layer_ids = tuple(artifact.get("layer_ids", ()))
        if artifact_layer_ids != self.layer_ids:
            raise ValueError(
                "Engram tokenizer-map layer_ids mismatch: "
                f"expected {self.layer_ids}, got {artifact_layer_ids}."
            )

        remap = artifact.get("remap")
        source_vocab_size = artifact.get("source_vocab_size")
        if not isinstance(remap, list) or len(remap) != source_vocab_size:
            raise ValueError(
                "Engram tokenizer-map remap length must equal source_vocab_size; "
                f"got {len(remap) if isinstance(remap, list) else type(remap).__name__} "
                f"and {source_vocab_size!r}."
            )
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in remap):
            raise ValueError("Engram tokenizer-map remap values must be nonnegative integers.")

        compressed_vocab_size = artifact.get("compressed_vocab_size")
        if not isinstance(compressed_vocab_size, int) or compressed_vocab_size <= 0:
            raise ValueError("Engram tokenizer-map compressed_vocab_size must be positive.")
        if max(remap, default=-1) >= compressed_vocab_size:
            raise ValueError("Engram tokenizer-map remap contains an out-of-range compressed ID.")

        compressed_pad_token_id = artifact.get("compressed_pad_token_id")
        if compressed_pad_token_id != remap[self.pad_token_id]:
            raise ValueError(
                "Engram tokenizer-map compressed_pad_token_id does not match remap[pad_token_id]."
            )

        raw_multipliers = artifact.get("layer_multipliers", {})
        layer_multipliers: dict[int, tuple[int, ...]] = {}
        for layer_id in self.layer_ids:
            values = raw_multipliers.get(str(layer_id))
            if not isinstance(values, list) or len(values) != self.max_ngram_order:
                raise ValueError(
                    "Engram tokenizer-map layer_multipliers must contain "
                    f"{self.max_ngram_order} values for layer {layer_id}."
                )
            multipliers = tuple(int(value) for value in values)
            if any(value <= 0 or value % 2 == 0 for value in multipliers):
                raise ValueError("Engram hash multipliers must be positive odd integers.")
            layer_multipliers[layer_id] = multipliers

        self.tokenizer_remap = torch.tensor(remap, dtype=torch.int64)
        self.compressed_vocab_size = compressed_vocab_size
        self.compressed_pad_token_id = compressed_pad_token_id
        self.layer_multipliers = layer_multipliers

    def validate_startup(
        self,
        transformer_config: Any,
        expected_tokenizer_vocab_size: int,
        packed_sequences: bool = False,
        use_fsdp: bool = False,
    ) -> None:
        """Validate layer placement, tokenizer compatibility, and supported parallel features."""
        if any(layer_id > transformer_config.num_layers for layer_id in self.layer_ids):
            raise ValueError(
                "engram_layer_ids must fall within the model's 1-based layer range "
                f"[1, {transformer_config.num_layers}]; got {self.layer_ids}."
            )
        if self.tokenizer_remap.numel() != expected_tokenizer_vocab_size:
            raise ValueError(
                "Engram tokenizer-map vocabulary mismatch: artifact has "
                f"{self.tokenizer_remap.numel()} raw tokens but the model expects "
                f"{expected_tokenizer_vocab_size}."
            )
        if transformer_config.context_parallel_size != 1:
            raise ValueError("Engram currently requires context_parallel_size == 1.")
        if not transformer_config.bf16:
            raise ValueError("Engram currently supports ordinary BF16 training only.")
        if transformer_config.fp8 is not None or transformer_config.fp4 is not None:
            raise ValueError("Engram currently supports ordinary BF16 training, not FP8 or FP4.")
        if transformer_config.transformer_impl == "inference_optimized":
            raise ValueError("Engram does not yet support inference-optimized model execution.")
        if transformer_config.mtp_num_layers:
            raise ValueError("Engram does not yet support multi-token prediction layers.")
        if transformer_config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("Engram does not yet support virtual pipeline parallelism.")
        if packed_sequences:
            raise ValueError("Engram does not yet support packed or variable-length sequences.")
        if transformer_config.recompute_granularity is not None:
            raise ValueError("Engram does not yet support activation recomputation.")
        if transformer_config.cuda_graph_impl != "none":
            raise ValueError("Engram does not yet support CUDA graphs.")
        if use_fsdp:
            raise ValueError("Engram does not yet support FSDP.")
