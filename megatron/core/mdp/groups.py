# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP process groups and descriptor broadcast.

Kept out of ``rank_mapping.py`` so the rank map stays pure compute; this module
is the distributed counterpart that installs groups from the map and moves
fixed-width descriptor records inside each planning group.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist

from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankMap

# int64 slots per descriptor record (see API design 5.6; slot 11 carries
# owner_worker_id for owner-sharded pixel reading, and slots 12-14 carry the
# item's decoder span so every member can derive the decoder-CP row split).
DESCRIPTOR_SLOTS = 15


@dataclass(frozen=True)
class MdpProcessGroups:
    """The process groups one rank participates in."""

    planning_group: dist.ProcessGroup
    encoder_reduction_group: dist.ProcessGroup
    world_group: dist.ProcessGroup
    # This rank's logical worker: the encoder_cp ranks that collaboratively
    # encode one vision chunk. A singleton at encoder_cp=1.
    encoder_cp_group: Optional[dist.ProcessGroup] = None


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
            if key[0] not in ("planning", "world_alias", "encoder_cp"):
                raise MdpConfigurationError(
                    f"MDP: group registry violates: no unexpected groups (found {key})."
                )


def install_mdp_process_groups(
    rank_map: MdpRankMap, *, group_registry: MdpGroupRegistry
) -> MdpProcessGroups:
    """Install MDP process groups; every rank creates groups in the same order.

    One planning group per outer-DP group, in ascending ``outer_dp_rank`` order,
    then one encoder-CP group per logical worker in ascending
    ``(outer_dp_rank, worker_id)`` order. ``dist.new_group`` is a WORLD
    collective, so the order must be identical on every rank and every rank must
    enter every creation, including the groups it does not join — a divergence
    here hangs with no message rather than raising.

    With ``encoder_cp=1`` the encoder reduction group aliases WORLD; no duplicate
    same-sized group is created.
    """
    world = dist.group.WORLD
    my_rank = dist.get_rank()
    my_planning_group = None
    for outer_dp_rank, ranks in enumerate(rank_map.planning_groups()):
        group = group_registry.get_or_create(("planning", outer_dp_rank), ranks)
        if my_rank in ranks:
            my_planning_group = group
    my_encoder_cp_group = None
    if rank_map.spec.encoder_cp > 1:
        for outer_dp_rank in range(len(rank_map.planning_groups())):
            for worker_id in range(rank_map.num_workers_per_group):
                ranks = rank_map.worker_ranks(outer_dp_rank, worker_id)
                group = group_registry.get_or_create(
                    ("encoder_cp", outer_dp_rank, worker_id), ranks
                )
                if my_rank in ranks:
                    my_encoder_cp_group = group
    group_registry.register_alias(
        ("world_alias",), tuple(range(rank_map.spec.world_size)), world
    )
    if my_planning_group is None:
        raise MdpConfigurationError(
            f"MDP: rank {my_rank} violates: every rank belongs to exactly one planning "
            "group."
        )
    _assert_rank_map_matches_mpu(rank_map, my_rank)
    _assert_encoder_cp_group_matches_map(rank_map, my_rank, my_encoder_cp_group)
    return MdpProcessGroups(
        planning_group=my_planning_group,
        encoder_reduction_group=world,
        world_group=world,
        encoder_cp_group=my_encoder_cp_group,
    )


def _assert_encoder_cp_group_matches_map(
    rank_map: MdpRankMap, my_rank: int, group
) -> None:
    """Cross-check the encoder-CP group against the rank map's own coordinate.

    Unlike the decoder's CP group there is no MPU counterpart to compare
    against, so the map is the only available truth. It is still worth
    asserting: an off-by-one between ``my_encoder_cp_rank`` and the group's rank
    ordering produces silently wrong vision shards — every rank encodes a
    plausible slice of the chunk, the shapes agree, and the loss is merely
    wrong. Runs once per job.
    """
    encoder_cp = rank_map.spec.encoder_cp
    if encoder_cp == 1:
        if group is not None:
            raise MdpConfigurationError(
                "MDP: encoder_cp=1 violates: no encoder-CP group is created."
            )
        return
    if group is None:
        raise MdpConfigurationError(
            f"MDP: rank {my_rank} violates: every rank belongs to exactly one "
            f"logical worker at encoder_cp={encoder_cp}."
        )
    view = rank_map.view(my_rank)
    size = dist.get_world_size(group=group)
    rank_in_group = dist.get_rank(group=group)
    if size != encoder_cp:
        raise MdpConfigurationError(
            f"MDP: encoder-CP group of rank {my_rank} has size {size} violates: "
            f"size == encoder_cp ({encoder_cp})."
        )
    if rank_in_group != view.my_encoder_cp_rank:
        raise MdpConfigurationError(
            f"MDP: rank {my_rank} is index {rank_in_group} in its encoder-CP group "
            f"violates: it equals my_encoder_cp_rank ({view.my_encoder_cp_rank}). "
            "The vision chunk would be sharded against the wrong coordinate."
        )


def _assert_rank_map_matches_mpu(rank_map: MdpRankMap, my_rank: int) -> None:
    """Cross-check the independently derived rank map against live MPU state.

    MDP rebuilds its own ``RankGenerator`` from arguments and never reads the
    MPU's groups, so the two can disagree — and with decoder CP the
    ``index_in_group -> (cp_rank, pp_rank)`` mapping decides which physical rank
    is a decoder endpoint. A divergence there yields a rank map that is
    internally consistent but names the wrong ranks, which surfaces as a bridge
    hang rather than an error. This runs once per job.
    """
    from megatron.core import parallel_state

    view = rank_map.view(my_rank)
    cp = rank_map.spec.cp
    expected_cp_rank = parallel_state.get_context_parallel_rank()
    expected_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if (view.my_cp_rank, view.my_pp_rank) != (expected_cp_rank, expected_pp_rank):
        raise MdpConfigurationError(
            f"MDP: rank {my_rank} maps to (cp_rank={view.my_cp_rank}, "
            f"pp_rank={view.my_pp_rank}) violates: the MPU reports "
            f"(cp_rank={expected_cp_rank}, pp_rank={expected_pp_rank}). The MDP "
            "rank map and the initialized process groups disagree; every route "
            "and endpoint derived from it would name the wrong rank."
        )
    if view.is_decoder_endpoint != parallel_state.is_pipeline_first_stage(
        ignore_virtual=True
    ):
        raise MdpConfigurationError(
            f"MDP: rank {my_rank} decoder-endpoint status "
            f"{view.is_decoder_endpoint} violates: it equals "
            "is_pipeline_first_stage(). Vision embeddings are consumed exactly "
            "where pre_process runs."
        )
    if len(view.decoder_endpoint_ranks) != cp:
        raise MdpConfigurationError(
            f"MDP: rank {my_rank} sees {len(view.decoder_endpoint_ranks)} decoder "
            f"endpoints violates: one per cp_rank (cp={cp})."
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
                d.sample_padded_start,
                d.sample_padded_len,
                d.decoder_offset_in_sample,
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
            sample_padded_start,
            sample_padded_len,
            decoder_offset_in_sample,
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
                sample_padded_start=sample_padded_start,
                sample_padded_len=sample_padded_len,
                decoder_offset_in_sample=decoder_offset_in_sample,
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
    ``text_only`` flags), then the fixed-width ``int64[count, DESCRIPTOR_SLOTS]``
    payload.
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
    if is_endpoint:
        # The endpoint knows its own header; reading it back from the device
        # would sync against the broadcast it just posted.
        count = len(local_descriptors)
        flags = tuple(bool(flag) for flag in text_only_flags)
    else:
        count = int(header[0].item())
        flags = tuple(bool(v) for v in header[1:].tolist())

    payload = torch.zeros(count, DESCRIPTOR_SLOTS, dtype=torch.int64, device=device)
    if is_endpoint and count:
        records = torch.tensor(
            descriptors_to_records(local_descriptors), dtype=torch.int64
        )
        # Pinned staging: a pageable H2D here blocks until the compute
        # stream drains; pinned + non_blocking stays stream-ordered ahead
        # of the broadcast below without a host wait.
        payload.copy_(records.pin_memory(), non_blocking=True)
    if count:
        dist.broadcast(payload, src=endpoint_rank, group=planning_group)
    if is_endpoint:
        # Wire roundtrip is lossless (unit-tested); skip the D2H readback.
        return tuple(local_descriptors), flags
    return records_to_descriptors(payload.tolist() if count else []), flags
