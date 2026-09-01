# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP modality bridge: one ledger and one transport for pixels, embeddings,
and gradients.

Pixel, embedding, and gradient routes use the same ledger builder, packing,
exchange, and unpacking implementation; three separate transports are forbidden.
Data for the same ``(src, dst)`` pair is coalesced across the iteration, local
edges copy, empty edges are omitted, and every planning-group member enters each
bridge phase exactly once — including members with an empty ledger.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.errors import MdpBridgeError
from megatron.core.mdp.observability import nvtx_phase
from megatron.core.mdp.plan import MdpBatchPlan
from megatron.core.mdp.rank_mapping import MdpRankMap


class BridgePhase(Enum):
    """The three payload classes carried over the one transport."""

    PIXEL = "pixel"
    EMBEDDING = "embedding"
    GRADIENT = "gradient"


@dataclass(frozen=True)
class BridgeBufferKey:
    """Identifies one transported buffer.

    ``slice_id`` distinguishes the per-endpoint runs a vision item's decoder
    rows break into under decoder context parallelism. It is 0 for every buffer
    at CP=1, and always 0 in the PIXEL phase, whose payload is per item.
    """

    global_item_id: int
    slice_id: int = 0


@dataclass(frozen=True)
class BridgeLedgerEntry:
    """One directed transfer. ``plan_offset`` is the element offset of this
    entry inside its coalesced ``(src, dst)`` message."""

    phase: BridgePhase
    src_global_rank: int
    dst_global_rank: int
    dtype: torch.dtype
    element_count: int
    plan_offset: int
    key: BridgeBufferKey


@dataclass(frozen=True)
class BridgeLedger:
    """All transfers of one phase for one planning group, in canonical order."""

    phase: BridgePhase
    entries: tuple
    total_bytes: int
    remote_bytes: int


@dataclass(frozen=True)
class BridgeTensorSpec:
    """Sizing for one transported buffer.

    ``capacity_rows`` always comes from ``plan.capacity_policy.capacity_of(valid_rows)``;
    callers must not compute it themselves. Only ``valid_rows`` rows are
    transmitted and unpacked; ``capacity_rows`` only sizes the allocator request.
    """

    valid_rows: int
    capacity_rows: int
    width: int
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class BridgePhaseStats:
    """Completed-phase communication metrics (not asynchronous launch latency)."""

    elapsed_ms: float
    total_bytes: int
    remote_bytes: int
    edges: int
    small_message_count: int


def _entry_sort_key(entry: BridgeLedgerEntry):
    return (
        entry.src_global_rank,
        entry.dst_global_rank,
        entry.key.global_item_id,
        entry.key.slice_id,
        entry.plan_offset,
    )


class ModalityBridge:
    """The single transport implementation shared by all three bridge phases."""

    def __init__(self, allocator: MdpBufferAllocator) -> None:
        self._allocator = allocator
        self._last_stats: dict = {}
        self._in_flight = False

    def build_ledger(
        self,
        phase: BridgePhase,
        plan: MdpBatchPlan,
        rank_map: MdpRankMap,
        tensor_specs: Mapping[BridgeBufferKey, BridgeTensorSpec],
    ) -> BridgeLedger:
        """Deterministically build the full-group ledger for one phase.

        The plan's ``producer_worker_id`` is a logical worker; ``worker_ranks()``
        is the only resolution point to physical ranks. Item rows come from the
        caller's tensor specs (which the caller derives via ``segment_for_item``,
        never a linear scan).
        """
        entries = []
        # Pixels belong to the item, not to a decoder-CP slice of it: an item
        # split across endpoints still has exactly one pixel payload and one
        # owner. Routing per slice here would post the same payload cp times.
        routes = (
            tuple(route for route in plan.routes if route.slice_id == 0)
            if phase is BridgePhase.PIXEL
            else plan.routes
        )
        for route in routes:
            producer_ranks = rank_map.worker_ranks(plan.outer_dp_rank, route.producer_worker_id)
            if len(producer_ranks) != 1:
                raise MdpBridgeError(
                    f"MDP: producer_worker_id={route.producer_worker_id} resolves to "
                    f"{len(producer_ranks)} ranks; the encoder-CP physical expansion is "
                    "not implemented in this version."
                )
            producer_rank = producer_ranks[0]
            if phase is BridgePhase.EMBEDDING:
                src, dst = producer_rank, route.endpoint_rank
            elif phase is BridgePhase.PIXEL:
                owner_ranks = rank_map.worker_ranks(
                    plan.outer_dp_rank, route.owner_worker_id
                )
                if len(owner_ranks) != 1:
                    raise MdpBridgeError(
                        f"MDP: owner_worker_id={route.owner_worker_id} resolves to "
                        f"{len(owner_ranks)} ranks; the encoder-CP physical "
                        "expansion is not implemented in this version."
                    )
                owner_rank = owner_ranks[0]
                src, dst = owner_rank, producer_rank
            else:  # GRADIENT flows owner endpoint -> producer
                src, dst = route.endpoint_rank, producer_rank
            key = BridgeBufferKey(
                global_item_id=route.global_item_id,
                slice_id=0 if phase is BridgePhase.PIXEL else route.slice_id,
            )
            spec = tensor_specs.get(key)
            if spec is None:
                raise MdpBridgeError(
                    f"MDP: key {key} violates: every routed item has a tensor spec."
                )
            element_count = spec.valid_rows * max(1, spec.width)
            if element_count == 0:
                continue  # empty edges are omitted
            entries.append(
                BridgeLedgerEntry(
                    phase=phase,
                    src_global_rank=src,
                    dst_global_rank=dst,
                    dtype=spec.dtype,
                    element_count=element_count,
                    plan_offset=0,  # assigned below in canonical order
                    key=key,
                )
            )

        entries.sort(key=_entry_sort_key)
        # Assign each entry its element offset inside the coalesced (src, dst)
        # message, in the same canonical order used to post requests.
        with_offsets = []
        offsets: dict = {}
        total_bytes = 0
        remote_bytes = 0
        for entry in entries:
            edge = (entry.src_global_rank, entry.dst_global_rank)
            offset = offsets.get(edge, 0)
            offsets[edge] = offset + entry.element_count
            entry = BridgeLedgerEntry(
                phase=entry.phase,
                src_global_rank=entry.src_global_rank,
                dst_global_rank=entry.dst_global_rank,
                dtype=entry.dtype,
                element_count=entry.element_count,
                plan_offset=offset,
                key=entry.key,
            )
            with_offsets.append(entry)
            nbytes = entry.element_count * entry.dtype.itemsize
            total_bytes += nbytes
            if entry.src_global_rank != entry.dst_global_rank:
                remote_bytes += nbytes
        return BridgeLedger(
            phase=phase,
            entries=tuple(with_offsets),
            total_bytes=total_bytes,
            remote_bytes=remote_bytes,
        )

    def exchange_all_to_all(
        self,
        ledger: BridgeLedger,
        local_tensors: Mapping[BridgeBufferKey, Tensor],
        *,
        tensor_specs: Mapping[BridgeBufferKey, BridgeTensorSpec],
        group,
        group_ranks,
        global_rank: int,
        dtype: torch.dtype,
        device: torch.device,
        dest_views: Optional[Mapping[BridgeBufferKey, Tensor]] = None,
    ) -> Mapping[BridgeBufferKey, Tensor]:
        """Execute this rank's part of the ledger with one ``all_to_all_single``.

        Used by PIXEL, EMBEDDING, and GRADIENT. Every planning-group member
        must call this in each phase — a rank with
        nothing to move participates with zero splits (the collective cannot be
        skipped). The payload carries raw rows only, no headers: both sides
        hold the identical ledger and derive the buffer layout (per-destination
        blocks in group-rank order, ``plan_offset`` inside each block). Local
        edges bypass the collective and copy directly.

        No host synchronization: ``all_to_all_single`` with ``async_op=False``
        stream-orders subsequent reads of the receive buffer, and unpacking
        stays on the same stream.
        """
        if self._in_flight:
            raise MdpBridgeError("MDP: bridge violates: one exchange at a time per phase.")
        self._in_flight = True
        start = time.monotonic()
        try:
            received = self._exchange_all_to_all_impl(
                ledger, local_tensors, tensor_specs, group, group_ranks, global_rank,
                dtype, device, dest_views,
            )
        finally:
            self._in_flight = False
        elapsed_ms = (time.monotonic() - start) * 1000.0
        edges = len(
            {
                (e.src_global_rank, e.dst_global_rank)
                for e in ledger.entries
                if global_rank in (e.src_global_rank, e.dst_global_rank)
            }
        )
        self._last_stats[ledger.phase] = BridgePhaseStats(
            elapsed_ms=elapsed_ms,
            total_bytes=ledger.total_bytes,
            remote_bytes=ledger.remote_bytes,
            edges=edges,
            small_message_count=0,
        )
        return received

    def _exchange_all_to_all_impl(
        self,
        ledger: BridgeLedger,
        local_tensors: Mapping[BridgeBufferKey, Tensor],
        tensor_specs: Mapping[BridgeBufferKey, BridgeTensorSpec],
        group,
        group_ranks,
        global_rank: int,
        dtype: torch.dtype,
        device: torch.device,
        dest_views: Optional[Mapping[BridgeBufferKey, Tensor]] = None,
    ) -> Mapping[BridgeBufferKey, Tensor]:
        group_ranks = tuple(group_ranks)
        send_by_dst: dict = {}
        recv_by_src: dict = {}
        local_entries = []
        for entry in ledger.entries:  # already in canonical order
            if entry.dtype != dtype:
                raise MdpBridgeError(
                    f"MDP: all_to_all exchange violates: one dtype per phase "
                    f"(ledger has {entry.dtype}, expected {dtype})."
                )
            src, dst = entry.src_global_rank, entry.dst_global_rank
            if src == dst:
                if src == global_rank:
                    local_entries.append(entry)
            elif src == global_rank:
                send_by_dst.setdefault(dst, []).append(entry)
            elif dst == global_rank:
                recv_by_src.setdefault(src, []).append(entry)

        input_splits = [
            sum(e.element_count for e in send_by_dst.get(rank, ())) for rank in group_ranks
        ]
        output_splits = [
            sum(e.element_count for e in recv_by_src.get(rank, ())) for rank in group_ranks
        ]
        send_buffer = self._allocator.acquire(
            rows=sum(input_splits), width=0, dtype=dtype, device=device, tag="bridge_a2a_send"
        )
        recv_buffer = self._allocator.acquire(
            rows=sum(output_splits), width=0, dtype=dtype, device=device, tag="bridge_a2a_recv"
        )
        base = 0
        pack_dst = []
        pack_src = []
        for rank, split in zip(group_ranks, input_splits):
            for entry in send_by_dst.get(rank, ()):
                offset = base + entry.plan_offset
                pack_dst.append(send_buffer[offset : offset + entry.element_count])
                pack_src.append(self._entry_payload(local_tensors, entry))
            base += split
        if pack_dst:
            # One multi-tensor launch instead of one copy kernel per entry;
            # all slices share the phase dtype, so the fast path applies.
            torch._foreach_copy_(pack_dst, pack_src)

        received: dict = {}

        def _unpack(entry: BridgeLedgerEntry, flat: Tensor):
            self._unpack_entry(
                entry, flat, tensor_specs, dest_views, received, ledger.phase.value
            )

        # Issue the collective asynchronously and unpack the local edges while
        # it is in flight: the local copies run on the current stream, the
        # collective on NCCL's stream, and neither touches the other's
        # buffers. work.wait() then stream-orders everything that reads the
        # receive buffer, exactly like the synchronous form did.
        with nvtx_phase("bridge_alltoall_launch"):
            work = dist.all_to_all_single(
                recv_buffer,
                send_buffer,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=group,
                async_op=True,
            )
        for entry in local_entries:
            _unpack(entry, self._entry_payload(local_tensors, entry))
        with nvtx_phase("bridge_alltoall_wait"):
            if work is not None:
                work.wait()
        # The send buffer stays alive until the wait ordered the collective
        # ahead of any current-stream reuse of its block.
        self._allocator.release(send_buffer)
        base = 0
        for rank, split in zip(group_ranks, output_splits):
            for entry in recv_by_src.get(rank, ()):
                offset = base + entry.plan_offset
                _unpack(entry, recv_buffer[offset : offset + entry.element_count])
            base += split
        self._allocator.release(recv_buffer)
        return received

    def _unpack_entry(
        self,
        entry: BridgeLedgerEntry,
        flat: Tensor,
        tensor_specs: Mapping[BridgeBufferKey, BridgeTensorSpec],
        dest_views: Optional[Mapping[BridgeBufferKey, Tensor]],
        received: dict,
        phase_value: str,
    ) -> None:
        """Copy one received (or local) entry into its destination.

        With a caller-provided destination view the wire data lands directly
        in the consumer buffer (``copy_`` casts if the consumer dtype differs,
        e.g. fp32 gradient-regroup buffers fed by a bf16 wire); otherwise an
        intermediate capacity-sized buffer is allocated as before.
        """
        if entry.key in received:
            raise MdpBridgeError(
                f"MDP: key {entry.key} violates: one received buffer per key."
            )
        dest = dest_views.get(entry.key) if dest_views is not None else None
        if dest is None and dest_views is not None:
            # Partial coverage is always a bug: the runtime derives its
            # destination views from the same plan the ledger came from, so a
            # missing key means the two disagree. Falling through to the
            # allocator below would land the payload in a buffer nobody reads —
            # a silent data loss, which is exactly the shape a decoder-CP
            # slicing mistake takes.
            raise MdpBridgeError(
                f"MDP: key {entry.key} violates: when destination views are "
                "supplied they cover every entry this rank receives."
            )
        if dest is not None:
            if dest.numel() != entry.element_count:
                raise MdpBridgeError(
                    f"MDP: destination view for key {entry.key} holds {dest.numel()} "
                    f"elements; the ledger entry carries {entry.element_count}."
                )
            dest.copy_(flat.view(dest.shape))
            received[entry.key] = dest
            return
        spec = tensor_specs[entry.key]
        width = max(1, spec.width)
        rows = entry.element_count // width
        out = self._allocator.acquire(
            rows=spec.capacity_rows,
            width=spec.width,
            dtype=spec.dtype,
            device=spec.device,
            tag=f"bridge_{phase_value}_out",
        )
        out_valid = out[:rows] if spec.width == 0 else out[:rows, :]
        out_valid.copy_(flat.view(out_valid.shape))
        received[entry.key] = out_valid

    @staticmethod
    def _entry_payload(
        local_tensors: Mapping[BridgeBufferKey, Tensor], entry: BridgeLedgerEntry
    ) -> Tensor:
        tensor = local_tensors.get(entry.key)
        if tensor is None:
            raise MdpBridgeError(
                f"MDP: key {entry.key} violates: the sending rank holds a local tensor "
                "for every entry it sources."
            )
        flat = tensor.reshape(-1)
        if flat.numel() < entry.element_count:
            raise MdpBridgeError(
                f"MDP: key {entry.key} violates: local tensor holds at least "
                f"element_count={entry.element_count} elements (got {flat.numel()})."
            )
        return flat[: entry.element_count]

    def last_stats(self) -> Mapping[str, BridgePhaseStats]:
        """Stats of the most recent exchange per phase, keyed by phase value."""
        return {phase.value: stats for phase, stats in self._last_stats.items()}

    def assert_idle(self) -> None:
        """Lifecycle invariant: no exchange in flight at an iteration boundary."""
        if self._in_flight:
            raise MdpBridgeError("MDP: bridge violates: idle at iteration boundary.")
