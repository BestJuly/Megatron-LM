# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Faithful standard-residual and native-mHC Engram module."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule

from .config import EngramConfig
from .distributed_embedding import EPShardedMultiTableEmbedding
from .hashing import build_ngram_hashes, slice_hashes_for_sequence_parallel

logger = logging.getLogger(__name__)


class Engram(MegatronModule):
    """DeepSeek Engram injection for one selected global transformer layer."""

    def __init__(
        self,
        config,
        engram_config: EngramConfig,
        layer_number: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        self.engram_config = engram_config
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.num_streams = config.num_residual_streams if config.enable_hyper_connections else 1
        self.hidden_size = config.hidden_size
        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        )

        self.register_buffer(
            "tokenizer_remap", engram_config.tokenizer_remap.to(device=device), persistent=False
        )
        self.register_buffer(
            "hash_multipliers",
            torch.tensor(engram_config.multipliers(layer_number), dtype=torch.int64, device=device),
            persistent=False,
        )
        table_sizes = engram_config.table_sizes(layer_number)
        self.register_buffer(
            "table_sizes",
            torch.tensor(table_sizes, dtype=torch.int64, device=device),
            persistent=False,
        )

        self.embedding = EPShardedMultiTableEmbedding(
            config=config,
            table_sizes=table_sizes,
            embedding_dim=engram_config.head_dim,
            init_method=config.init_method,
            ep_group=pg_collection.ep,
            tp_group=pg_collection.tp,
            expt_dp_group=pg_collection.expt_dp,
        )
        factory_kwargs = {"device": device, "dtype": config.params_dtype}
        self.value_projection = nn.Linear(
            engram_config.total_memory_dim, config.hidden_size, **factory_kwargs
        )
        self.key_projections = nn.ModuleList(
            [
                nn.Linear(engram_config.total_memory_dim, config.hidden_size, **factory_kwargs)
                for _ in range(self.num_streams)
            ]
        )
        self.key_norms = nn.ModuleList(
            [
                nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, **factory_kwargs)
                for _ in range(self.num_streams)
            ]
        )
        self.query_norms = nn.ModuleList(
            [
                nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, **factory_kwargs)
                for _ in range(self.num_streams)
            ]
        )
        self.conv_norms = nn.ModuleList(
            [
                nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, **factory_kwargs)
                for _ in range(self.num_streams)
            ]
        )
        channels = self.num_streams * config.hidden_size
        dilation = engram_config.max_ngram_order
        self.short_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=engram_config.kernel_size,
            groups=channels,
            bias=False,
            padding=(engram_config.kernel_size - 1) * dilation,
            dilation=dilation,
            **factory_kwargs,
        )
        self._initialize_dense_parameters()
        self._log_allocation()

    @property
    def local_sparse_parameter_count(self) -> int:
        """Rank-local sparse-table parameter count."""
        return self.embedding.local_parameter_count

    @property
    def global_sparse_parameter_count(self) -> int:
        """Global logical sparse-table parameter count."""
        return self.embedding.global_parameter_count

    def _initialize_dense_parameters(self) -> None:
        if self.config.perform_initialization:
            self.config.init_method(self.value_projection.weight)
            for projection in self.key_projections:
                self.config.init_method(projection.weight)
        with torch.no_grad():
            if self.value_projection.bias is not None:
                self.value_projection.bias.zero_()
            for projection in self.key_projections:
                if projection.bias is not None:
                    projection.bias.zero_()
            self.short_conv.weight.zero_()

        if self.config.sequence_parallel:
            for parameter in self.parameters():
                parameter.sequence_parallel = True

    def _log_allocation(self) -> None:
        """Emit one machine-readable ownership record per instantiated module and rank."""
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        ep_rank = (
            torch.distributed.get_rank(self.pg_collection.ep)
            if torch.distributed.is_initialized() and self.pg_collection.ep is not None
            else 0
        )
        local_rows = [table.local_num_embeddings for table in self.embedding.tables]
        row_ranges = [(table.row_start, table.row_end) for table in self.embedding.tables]
        logger.info(
            "[Engram allocation] rank=%s layer=%s ep_rank=%s local_rows=%s "
            "row_ranges=%s global_rows=%s local_sparse_parameters=%s "
            "global_sparse_parameters=%s",
            global_rank,
            self.layer_number,
            ep_rank,
            local_rows,
            row_ranges,
            list(self.embedding.table_sizes),
            self.local_sparse_parameter_count,
            self.global_sparse_parameter_count,
        )

    def _short_convolution(self, value: Tensor) -> Tensor:
        sequence_length, batch_size, num_streams, hidden_size = value.shape
        normed = torch.stack(
            [self.conv_norms[index](value[:, :, index]) for index in range(num_streams)], dim=2
        )
        channels_first = normed.permute(1, 2, 3, 0).reshape(
            batch_size, num_streams * hidden_size, sequence_length
        )
        convolved = self.short_conv(channels_first)[..., :sequence_length]
        return (
            F.silu(convolved)
            .view(batch_size, num_streams, hidden_size, sequence_length)
            .permute(3, 0, 1, 2)
            .contiguous()
        )

    def forward(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        """Return an Engram residual with the same standard or mHC layout as input."""
        if hidden_states.ndim != 3:
            raise ValueError(f"Engram hidden_states must be [S,B,H], got {hidden_states.shape}.")
        expected_hidden = self.num_streams * self.hidden_size
        if hidden_states.shape[-1] != expected_hidden:
            raise ValueError(
                f"Engram expected hidden width {expected_hidden}, got {hidden_states.shape[-1]}."
            )

        hash_ids = build_ngram_hashes(
            input_ids=input_ids,
            tokenizer_remap=self.tokenizer_remap,
            multipliers=self.hash_multipliers,
            table_sizes=self.table_sizes,
            max_ngram_order=self.engram_config.max_ngram_order,
            num_hash_heads=self.engram_config.num_hash_heads,
            compressed_pad_token_id=self.engram_config.compressed_pad_token_id,
        )
        hash_ids = slice_hashes_for_sequence_parallel(
            hash_ids, hidden_states.shape[0], self.tp_group
        )
        memory = self.embedding(hash_ids).flatten(start_dim=-2).transpose(0, 1).contiguous()
        streams = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_streams, self.hidden_size
        )

        shared_value = self.value_projection(memory)
        gates = []
        for stream_index in range(self.num_streams):
            key = self.key_norms[stream_index](self.key_projections[stream_index](memory))
            query = self.query_norms[stream_index](streams[:, :, stream_index])
            score = (key * query).sum(dim=-1) / math.sqrt(self.hidden_size)
            score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
            gates.append(score.sigmoid())
        gate = torch.stack(gates, dim=2).unsqueeze(-1)
        value = gate * shared_value.unsqueeze(2)
        output = value + self._short_convolution(value)
        return output.reshape(hidden_states.shape)
