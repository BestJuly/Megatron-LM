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

# Remote messages below this size trigger a structured warning: the bridge is
# meant to carry coalesced item payloads, not chatter. Warning-only, not a
# semantic option, hence a module constant rather than an MdpConfig field.
MIN_REMOTE_MSG_BYTES = 4096


class BridgePhase(Enum):
    """The three payload classes carried over the one transport."""

    PIXEL = "pixel"
    EMBEDDING = "embedding"
    GRADIENT = "gradient"


@dataclass(frozen=True)
class BridgeBufferKey:
    """Identifies one transported buffer. ``slice_id`` is always 0 in v1 and
    distinguishes multiple slices of one item once decoder CP lands."""

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
        for route in plan.routes:
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
                # Pixels flow from the worker that owns them. Routes without
                # an owner (pre-owner-sharding plans) fall back to the
                # endpoint, today's star dispatch.
                if route.owner_worker_id is None:
                    owner_rank = route.endpoint_rank
                else:
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
            key = BridgeBufferKey(global_item_id=route.global_item_id)
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

    def exchange(
        self,
        ledger: BridgeLedger,
        local_tensors: Mapping[BridgeBufferKey, Tensor],
        *,
        tensor_specs: Mapping[BridgeBufferKey, BridgeTensorSpec],
        global_rank: int,
        dest_views: Optional[Mapping[BridgeBufferKey, Tensor]] = None,
    ) -> Mapping[BridgeBufferKey, Tensor]:
        """Execute this rank's part of the ledger and return received buffers.

        Receives are posted before sends, both in canonical entry order. The call
        returns only after every request completes; unfinished handles never reach
        the schedule. Each returned tensor exposes exactly ``valid_rows`` rows of a
        capacity-sized allocation. A rank with no edges performs a no-op call.

        ``dest_views`` optionally maps keys to caller-owned ``[valid_rows, width]``
        views (e.g. rows of the encoder payload or the endpoint leaf); a received
        item with a view is unpacked straight into it — the returned mapping then
        aliases the caller's buffer — skipping the intermediate per-item buffer.
        """
        if self._in_flight:
            raise MdpBridgeError("MDP: bridge violates: one exchange at a time per phase.")
        self._in_flight = True
        start = time.monotonic()
        try:
            received = self._exchange_impl(
                ledger, local_tensors, tensor_specs, global_rank, dest_views
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
        small = 0
        edge_bytes: dict = {}
        for entry in ledger.entries:
            if entry.src_global_rank == entry.dst_global_rank:
                continue
            edge = (entry.src_global_rank, entry.dst_global_rank)
            edge_bytes[edge] = (
                edge_bytes.get(edge, 0) + entry.element_count * entry.dtype.itemsize
            )
        for edge, nbytes in edge_bytes.items():
            if nbytes < MIN_REMOTE_MSG_BYTES:
                small += 1
                if global_rank == edge[0]:
                    import logging

                    logging.getLogger(__name__).warning(
                        "MDP bridge: remote message %s -> %s in phase %s is only %d bytes "
                        "(< %d); the plan is producing chatter-sized edges.",
                        edge[0],
                        edge[1],
                        ledger.phase.value,
                        nbytes,
                        MIN_REMOTE_MSG_BYTES,
                    )
        self._last_stats[ledger.phase] = BridgePhaseStats(
            elapsed_ms=elapsed_ms,
            total_bytes=ledger.total_bytes,
            remote_bytes=ledger.remote_bytes,
            edges=edges,
            small_message_count=small,
        )
        return received

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

        Used by the PIXEL phase under owner-sharded pixel reading. Every
        planning-group member must call this every iteration — a rank with
        nothing to move participates with zero splits (the collective cannot be
        skipped). The payload carries raw rows only, no headers: both sides
        hold the identical ledger and derive the buffer layout (per-destination
        blocks in group-rank order, ``plan_offset`` inside each block). Local
        edges bypass the collective and copy directly, like the P2P path.

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
        for rank, split in zip(group_ranks, input_splits):
            for entry in send_by_dst.get(rank, ()):
                offset = base + entry.plan_offset
                send_buffer[offset : offset + entry.element_count].copy_(
                    self._entry_payload(local_tensors, entry)
                )
            base += split

        with nvtx_phase("bridge_pixel_alltoall"):
            dist.all_to_all_single(
                recv_buffer,
                send_buffer,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=group,
            )
        self._allocator.release(send_buffer)

        received: dict = {}

        def _unpack(entry: BridgeLedgerEntry, flat: Tensor):
            self._unpack_entry(
                entry, flat, tensor_specs, dest_views, received, ledger.phase.value
            )

        for entry in local_entries:
            _unpack(entry, self._entry_payload(local_tensors, entry))
        base = 0
        for rank, split in zip(group_ranks, output_splits):
            for entry in recv_by_src.get(rank, ()):
                offset = base + entry.plan_offset
                _unpack(entry, recv_buffer[offset : offset + entry.element_count])
            base += split
        self._allocator.release(recv_buffer)
        return received

    def _exchange_impl(
        self,
        ledger: BridgeLedger,
        local_tensors: Mapping[BridgeBufferKey, Tensor],
        tensor_specs: Mapping[BridgeBufferKey, BridgeTensorSpec],
        global_rank: int,
        dest_views: Optional[Mapping[BridgeBufferKey, Tensor]] = None,
    ) -> Mapping[BridgeBufferKey, Tensor]:
        recv_entries: dict = {}  # (src) -> [entries] for remote receives
        send_entries: dict = {}  # (dst) -> [entries] for remote sends
        local_entries = []
        for entry in ledger.entries:  # already in canonical order
            src, dst = entry.src_global_rank, entry.dst_global_rank
            if src == dst:
                if src == global_rank:
                    local_entries.append(entry)
            elif dst == global_rank:
                recv_entries.setdefault(src, []).append(entry)
            elif src == global_rank:
                send_entries.setdefault(dst, []).append(entry)

        def _device_and_dtype(entries):
            dtypes = {e.dtype for e in entries}
            if len(dtypes) != 1:
                raise MdpBridgeError(
                    f"MDP: coalesced message violates: one dtype per (src, dst) edge "
                    f"(got {dtypes})."
                )
            key = entries[0].key
            return tensor_specs[key].device, entries[0].dtype

        # Post all receives in canonical order, then all sends in canonical order.
        p2p_ops = []
        recv_staging = {}
        for src in sorted(recv_entries):
            entries = recv_entries[src]
            device, dtype = _device_and_dtype(entries)
            total = sum(e.element_count for e in entries)
            staging = self._allocator.acquire(
                rows=total, width=0, dtype=dtype, device=device, tag="bridge_recv"
            )
            recv_staging[src] = (staging, entries)
            p2p_ops.append(dist.P2POp(dist.irecv, staging, peer=src))
        send_staging = []
        for dst in sorted(send_entries):
            entries = send_entries[dst]
            device, dtype = _device_and_dtype(entries)
            total = sum(e.element_count for e in entries)
            staging = self._allocator.acquire(
                rows=total, width=0, dtype=dtype, device=device, tag="bridge_send"
            )
            offset = 0
            for entry in entries:
                payload = self._entry_payload(local_tensors, entry)
                staging[offset : offset + entry.element_count].copy_(payload)
                offset += entry.element_count
            send_staging.append(staging)
            p2p_ops.append(dist.P2POp(dist.isend, staging, peer=dst))

        if p2p_ops:
            with nvtx_phase("bridge_p2p_wait"):
                requests = dist.batch_isend_irecv(p2p_ops)
                for request in requests:
                    request.wait()
                # Batched P2P can leave the copies on a side stream on some
                # NCCL/PyTorch combinations; sync before anything reads the
                # receive buffers. The exchange is phase-synchronous anyway.
                torch.cuda.synchronize()
        # Send buffers stayed alive until the waits above completed.
        for staging in send_staging:
            self._allocator.release(staging)

        # Unpack only after all receives completed.
        received: dict = {}

        def _unpack(entry: BridgeLedgerEntry, flat: Tensor):
            self._unpack_entry(
                entry, flat, tensor_specs, dest_views, received, ledger.phase.value
            )

        for entry in local_entries:
            _unpack(entry, self._entry_payload(local_tensors, entry))
        for src in sorted(recv_staging):
            staging, entries = recv_staging[src]
            offset = 0
            for entry in entries:
                _unpack(entry, staging[offset : offset + entry.element_count])
                offset += entry.element_count
            self._allocator.release(staging)
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
