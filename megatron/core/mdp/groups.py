# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP process groups and descriptor broadcast.

Kept out of ``rank_mapping.py`` so the rank map stays pure compute; this module
is the distributed counterpart that installs groups from the map and moves
fixed-width descriptor records inside each planning group.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankMap

# int64 slots per descriptor record (see API design 5.6; slot 11 carries
# owner_worker_id for owner-sharded pixel reading).
DESCRIPTOR_SLOTS = 12


@dataclass(frozen=True)
class MdpProcessGroups:
    """The process groups one rank participates in."""

    planning_group: dist.ProcessGroup
    encoder_reduction_group: dist.ProcessGroup
    world_group: dist.ProcessGroup


class MdpGroupRegistry:
    """Deduplicating registry for MDP-created process groups.

    Reinstalling the same specification returns the existing handle;
    ``dist.new_group`` is never called twice for one key. The registry exposes
    leaks during teardown and tests.
    """

    def __init__(self) -> None:
        self._groups: dict = {}

    def get_or_create(self, key: tuple, ranks: Sequence[int]) -> dist.ProcessGroup:
        """Create (or return) the group for ``key``; every rank must call this in
        the same order with the same arguments."""
        if key in self._groups:
            existing_ranks, group = self._groups[key]
            if existing_ranks != tuple(ranks):
                raise MdpConfigurationError(
                    f"MDP: group key {key} violates: one rank list per key "
                    f"({existing_ranks} != {tuple(ranks)})."
                )
            return group
        group = dist.new_group(ranks=list(ranks))
        self._groups[key] = (tuple(ranks), group)
        return group

    def created_keys(self) -> Sequence[tuple]:
        """Keys of every group created through this registry."""
        return tuple(self._groups.keys())

    def register_alias(self, key: tuple, ranks: Sequence[int], group) -> None:
        """Record an existing group (e.g. WORLD) under a key without creating one."""
        if key not in self._groups:
            self._groups[key] = (tuple(ranks), group)

    def assert_no_leak(self) -> None:
        """Every created key must be a planning group or a registered alias."""
        for key in self._groups:
            if key[0] not in ("planning", "world_alias"):
                raise MdpConfigurationError(
                    f"MDP: group registry violates: no unexpected groups (found {key})."
                )


def install_mdp_process_groups(
    rank_map: MdpRankMap, *, group_registry: MdpGroupRegistry
) -> MdpProcessGroups:
    """Install MDP process groups; every rank creates groups in the same order.

    One planning group per outer-DP group, in ascending ``outer_dp_rank`` order.
    With ``encoder_cp=1`` the encoder reduction group aliases WORLD; no duplicate
    same-sized group is created.
    """
    if rank_map.spec.encoder_cp != 1:
        raise MdpConfigurationError(
            f"MDP: encoder_cp={rank_map.spec.encoder_cp} violates: encoder_cp == 1. "
            "Encoder-CP group construction requires revalidating DDP/ZeRO semantics."
        )
    world = dist.group.WORLD
    my_rank = dist.get_rank()
    my_planning_group = None
    for outer_dp_rank, ranks in enumerate(rank_map.planning_groups()):
        group = group_registry.get_or_create(("planning", outer_dp_rank), ranks)
        if my_rank in ranks:
            my_planning_group = group
    group_registry.register_alias(
        ("world_alias",), tuple(range(rank_map.spec.world_size)), world
    )
    if my_planning_group is None:
        raise MdpConfigurationError(
            f"MDP: rank {my_rank} violates: every rank belongs to exactly one planning "
            "group."
        )
    return MdpProcessGroups(
        planning_group=my_planning_group,
        encoder_reduction_group=world,
        world_group=world,
    )


def descriptors_to_records(descriptors: Sequence[VisionDescriptor]) -> list:
    """Fixed-width int64 record rows in the wire slot order."""
    records = []
    for d in descriptors:
        records.append(
            [
                d.global_item_id,
                d.sample_id,
                d.image_ordinal,
                d.owner_dp_lane,
                d.microbatch_id,
                d.estimated_cost_units,
                d.payload_rows,
                d.output_rows,
                d.grid_thw[0],
                d.grid_thw[1],
                d.grid_thw[2],
                d.owner_worker_id,
            ]
        )
    return records


def records_to_descriptors(records) -> tuple:
    """Inverse of :func:`descriptors_to_records`."""
    descriptors = []
    for row in records:
        (
            global_item_id,
            sample_id,
            image_ordinal,
            owner_dp_lane,
            microbatch_id,
            estimated_cost_units,
            payload_rows,
            output_rows,
            grid_t,
            grid_h,
            grid_w,
            owner_worker_id,
        ) = (int(v) for v in row)
        descriptors.append(
            VisionDescriptor(
                global_item_id=global_item_id,
                sample_id=sample_id,
                image_ordinal=image_ordinal,
                owner_dp_lane=owner_dp_lane,
                microbatch_id=microbatch_id,
                estimated_cost_units=estimated_cost_units,
                payload_rows=payload_rows,
                output_rows=output_rows,
                grid_thw=(grid_t, grid_h, grid_w),
                owner_worker_id=owner_worker_id,
            )
        )
    return tuple(descriptors)


def broadcast_descriptors(
    local_descriptors: Sequence[VisionDescriptor],
    *,
    planning_group: dist.ProcessGroup,
    endpoint_rank: int,
    num_microbatches: int,
    text_only_flags: Sequence[bool] = (),
    device=None,
) -> tuple:
    """Broadcast the endpoint's descriptors to its planning group.

    Two collectives: a small header (descriptor count and per-microbatch
    ``text_only`` flags), then the fixed-width ``int64[count, 11]`` payload.
    Pickle and object collectives are forbidden. The endpoint emits in
    ``(microbatch_id, sample_id, image_ordinal)`` order, so every member's input
    is byte-identical by construction.

    Returns ``(descriptors, text_only_flags)``.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    my_rank = dist.get_rank()
    is_endpoint = my_rank == endpoint_rank

    header = torch.zeros(1 + num_microbatches, dtype=torch.int64, device=device)
    if is_endpoint:
        if len(text_only_flags) != num_microbatches:
            raise MdpBridgeError(
                f"MDP: text_only_flags length {len(text_only_flags)} violates: one flag "
                f"per microbatch ({num_microbatches})."
            )
        previous = None
        for d in local_descriptors:
            key = (d.microbatch_id, d.sample_id, d.image_ordinal)
            if previous is not None and key <= previous:
                raise MdpBridgeError(
                    f"MDP: descriptor order violates: strictly ascending "
                    f"(microbatch_id, sample_id, image_ordinal) at {key}."
                )
            previous = key
        header[0] = len(local_descriptors)
        for index, flag in enumerate(text_only_flags):
            header[1 + index] = 1 if flag else 0
    dist.broadcast(header, src=endpoint_rank, group=planning_group)
    count = int(header[0].item())
    flags = tuple(bool(v) for v in header[1:].tolist())

    payload = torch.zeros(count, DESCRIPTOR_SLOTS, dtype=torch.int64, device=device)
    if is_endpoint and count:
        payload.copy_(
            torch.tensor(
                descriptors_to_records(local_descriptors), dtype=torch.int64
            )
        )
    if count:
        dist.broadcast(payload, src=endpoint_rank, group=planning_group)
    return records_to_descriptors(payload.tolist() if count else []), flags
