# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""On-device tokenizer compression and official Engram n-gram hashing."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.utils import get_pg_rank, get_pg_size


def compress_token_ids(input_ids: Tensor, tokenizer_remap: Tensor) -> Tensor:
    """Map nonnegative raw token IDs to canonical IDs entirely on device."""
    safe_ids = input_ids.clamp_min(0)
    compressed = tokenizer_remap[safe_ids]
    return torch.where(input_ids >= 0, compressed, input_ids)


def build_ngram_hashes(
    input_ids: Tensor,
    tokenizer_remap: Tensor,
    multipliers: Tensor,
    table_sizes: Tensor,
    max_ngram_order: int,
    num_hash_heads: int,
    compressed_pad_token_id: int,
) -> Tensor:
    """Compute official multiplicative-XOR multi-head hashes.

    Args:
        input_ids: Raw token IDs with shape ``[batch, sequence]``.
        tokenizer_remap: Raw-to-compressed token map with shape ``[vocab]``.
        multipliers: Odd int64 multiplier per suffix position.
        table_sizes: Prime modulus for each order/head in order-major layout.
        max_ngram_order: Largest suffix order to hash.
        num_hash_heads: Number of distinct prime tables per order.
        compressed_pad_token_id: Left-padding value in compressed vocabulary space.

    Returns:
        Hash IDs with shape ``[batch, sequence, (max_ngram_order - 1) * heads]``.
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"Engram input_ids must have shape [batch, sequence], got {input_ids.shape}."
        )
    compressed = compress_token_ids(input_ids.to(torch.int64), tokenizer_remap)
    sequence_length = compressed.shape[1]
    suffixes = [compressed]
    for shift in range(1, max_ngram_order):
        suffixes.append(
            F.pad(compressed, (shift, 0), value=compressed_pad_token_id)[:, :sequence_length]
        )

    hashes = []
    table_index = 0
    for order in range(2, max_ngram_order + 1):
        mixed = suffixes[0] * multipliers[0]
        for suffix_index in range(1, order):
            mixed = torch.bitwise_xor(mixed, suffixes[suffix_index] * multipliers[suffix_index])
        for _ in range(num_hash_heads):
            hashes.append(torch.remainder(mixed, table_sizes[table_index]))
            table_index += 1
    return torch.stack(hashes, dim=-1)


def slice_hashes_for_sequence_parallel(
    hash_ids: Tensor, local_sequence_length: int, tp_group
) -> Tensor:
    """Select the contiguous SP interval after hashes were computed globally."""
    full_sequence_length = hash_ids.shape[1]
    tp_size = get_pg_size(tp_group)
    if full_sequence_length == local_sequence_length:
        return hash_ids
    expected_sequence_length = local_sequence_length * tp_size
    if full_sequence_length != expected_sequence_length:
        raise ValueError(
            "Engram full hash sequence is incompatible with the local SP hidden state: "
            f"full={full_sequence_length}, local={local_sequence_length}, TP={tp_size}."
        )
    start = get_pg_rank(tp_group) * local_sequence_length
    return hash_ids[:, start : start + local_sequence_length]
