# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native differentiable EP-sharded embedding lookup for Engram."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, get_pg_size


def get_contiguous_row_range(global_rows: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return the balanced, unpadded contiguous row interval owned by ``rank``."""
    if global_rows < 0:
        raise ValueError(f"global_rows must be nonnegative, got {global_rows}.")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")
    base, remainder = divmod(global_rows, world_size)
    start = rank * base + min(rank, remainder)
    local_rows = base + int(rank < remainder)
    return start, start + local_rows


class EPShardedEmbeddingTable(MegatronModule):
    """One prime-sized embedding table sharded over an explicit EP group."""

    def __init__(
        self,
        config,
        global_num_embeddings: int,
        embedding_dim: int,
        init_method: Callable[[Tensor], None],
        ep_group=None,
        tp_group=None,
        expt_dp_group=None,
    ) -> None:
        super().__init__(config=config)
        self.global_num_embeddings = global_num_embeddings
        self.embedding_dim = embedding_dim
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.expt_dp_group = expt_dp_group
        self.ep_rank = get_pg_rank(ep_group)
        self.ep_size = get_pg_size(ep_group)
        self.row_start, self.row_end = get_contiguous_row_range(
            global_num_embeddings, self.ep_rank, self.ep_size
        )

        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        )
        self.weight = nn.Parameter(
            torch.empty(
                (self.row_end - self.row_start, embedding_dim),
                device=device,
                dtype=config.params_dtype,
            )
        )
        if config.perform_initialization:
            init_method(self.weight)

        # The row shard is distinct across EP and synchronized only over expert-DP.
        self.weight.allreduce = False
        self.weight.is_engram_embedding = True
        self.weight.is_embedding_or_output_parameter = True
        if config.sequence_parallel:
            self.weight.sequence_parallel = True

    @property
    def local_num_embeddings(self) -> int:
        """Number of rows owned by this rank."""
        return self.row_end - self.row_start

    def forward(self, local_row_ids: Tensor) -> Tensor:
        """Look up owner-local row IDs."""
        return F.embedding(local_row_ids, self.weight)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Represent this uneven EP row shard with its exact global logical shape."""
        del metadata
        prepend_axis_num = len(sharded_offsets)
        global_shape = [1] * prepend_axis_num + [self.global_num_embeddings, self.embedding_dim]
        global_offset = [0] * prepend_axis_num + [self.row_start, 0]
        for axis, rank_offset, axis_fragmentation in sharded_offsets:
            global_shape[axis] = axis_fragmentation
            global_offset[axis] = rank_offset

        replica_id = (0, get_pg_rank(self.tp_group), get_pg_rank(self.expt_dp_group))
        key = f"{prefix}weight"
        return {
            key: ShardedTensor(
                key=key,
                data=self.weight,
                dtype=self.weight.dtype,
                local_shape=tuple(self.weight.shape),
                global_shape=tuple(global_shape),
                global_offset=tuple(global_offset),
                axis_fragmentations=None,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            )
        }


class EPShardedMultiTableEmbedding(MegatronModule):
    """Batch variable-size requests for multiple independently EP-sharded tables."""

    def __init__(
        self,
        config,
        table_sizes: tuple[int, ...],
        embedding_dim: int,
        init_method: Callable[[Tensor], None],
        ep_group=None,
        tp_group=None,
        expt_dp_group=None,
    ) -> None:
        super().__init__(config=config)
        self.table_sizes = tuple(table_sizes)
        self.embedding_dim = embedding_dim
        self.ep_group = ep_group
        self.ep_rank = get_pg_rank(ep_group)
        self.ep_size = get_pg_size(ep_group)
        self.tables = nn.ModuleList(
            [
                EPShardedEmbeddingTable(
                    config=config,
                    global_num_embeddings=table_size,
                    embedding_dim=embedding_dim,
                    init_method=init_method,
                    ep_group=ep_group,
                    tp_group=tp_group,
                    expt_dp_group=expt_dp_group,
                )
                for table_size in self.table_sizes
            ]
        )

        starts = []
        ends = []
        for table_size in self.table_sizes:
            table_starts = []
            table_ends = []
            for rank in range(self.ep_size):
                row_start, row_end = get_contiguous_row_range(table_size, rank, self.ep_size)
                table_starts.append(row_start)
                table_ends.append(row_end)
            starts.append(table_starts)
            ends.append(table_ends)
        self.register_buffer(
            "row_starts",
            torch.tensor(starts, dtype=torch.int64, device=self.tables[0].weight.device),
            persistent=False,
        )
        self.register_buffer(
            "row_ends",
            torch.tensor(ends, dtype=torch.int64, device=self.tables[0].weight.device),
            persistent=False,
        )

    @property
    def local_parameter_count(self) -> int:
        """Local sparse parameter count."""
        return sum(table.weight.numel() for table in self.tables)

    @property
    def global_parameter_count(self) -> int:
        """Global logical sparse parameter count."""
        return sum(self.table_sizes) * self.embedding_dim

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Preserve each table's irregular EP row metadata through the ModuleList container."""
        sharded_state_dict = {}
        for table_id, table in enumerate(self.tables):
            sharded_state_dict.update(
                table.sharded_state_dict(
                    prefix=f"{prefix}tables.{table_id}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        return sharded_state_dict

    def _lookup_received_requests(self, requests: Tensor) -> Tensor:
        output = self.tables[0].weight.new_empty((requests.shape[0], self.embedding_dim))
        received_table_ids = requests[:, 0]
        received_local_rows = requests[:, 1]
        for table_id, table in enumerate(self.tables):
            positions = torch.nonzero(received_table_ids == table_id, as_tuple=False).flatten()
            values = table(received_local_rows.index_select(0, positions))
            output.index_copy_(0, positions, values)
        return output

    def forward(self, hash_ids: Tensor) -> Tensor:
        """Route global table rows to owners and restore token/head ordering.

        Args:
            hash_ids: Per-table row IDs with shape ``[..., num_tables]``.

        Returns:
            Retrieved embeddings with shape ``[..., num_tables, embedding_dim]``.
        """
        if hash_ids.shape[-1] != len(self.tables):
            raise ValueError(
                f"Expected {len(self.tables)} Engram hash heads, got {hash_ids.shape[-1]}."
            )
        original_shape = hash_ids.shape
        rows = hash_ids.reshape(-1).to(torch.int64)
        table_ids = (
            torch.arange(len(self.tables), device=hash_ids.device, dtype=torch.int64)
            .view(*([1] * (hash_ids.ndim - 1)), -1)
            .expand(original_shape)
            .reshape(-1)
        )

        if self.ep_size == 1:
            requests = torch.stack((table_ids, rows), dim=-1)
            output = self._lookup_received_requests(requests)
            return output.view(*original_shape, self.embedding_dim)

        owner_ends = self.row_ends.index_select(0, table_ids)
        owners = torch.sum(rows.unsqueeze(-1) >= owner_ends, dim=-1).to(torch.int64)
        owner_starts = self.row_starts[table_ids, owners]
        local_rows = rows - owner_starts

        owner_order = torch.argsort(owners, stable=True)
        sorted_owners = owners.index_select(0, owner_order)
        requests = torch.stack(
            (table_ids.index_select(0, owner_order), local_rows.index_select(0, owner_order)),
            dim=-1,
        )
        send_counts = torch.bincount(sorted_owners, minlength=self.ep_size).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        received_requests = requests.new_empty((sum(recv_splits), 2))
        torch.distributed.all_to_all_single(
            received_requests,
            requests.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.ep_group,
        )
        owner_embeddings = self._lookup_received_requests(received_requests)
        sorted_embeddings = all_to_all(
            self.ep_group,
            owner_embeddings,
            output_split_sizes_=send_splits,
            input_split_sizes=recv_splits,
        )
        output = torch.empty_like(sorted_embeddings)
        output.index_copy_(0, owner_order, sorted_embeddings)
        return output.view(*original_shape, self.embedding_dim)
