# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group installation and descriptor-broadcast tests.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_groups.py

The pure record round-trip tests also pass single-process.
"""

import os

import pytest
import torch

from megatron.core.mdp.errors import MdpBridgeError
from megatron.core.mdp.groups import (
    DESCRIPTOR_SLOTS,
    MdpGroupRegistry,
    broadcast_descriptors,
    descriptors_to_records,
    install_mdp_process_groups,
    records_to_descriptors,
)
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map


def _descriptor(
    item_id,
    mb=0,
    sample=0,
    ordinal=0,
    lane=0,
    cost=7,
    grid=(1, 4, 4),
    sample_padded_start=0,
    sample_padded_len=0,
    decoder_offset_in_sample=0,
):
    t, h, w = grid
    return VisionDescriptor(
        global_item_id=item_id,
        sample_id=sample,
        image_ordinal=ordinal,
        owner_dp_lane=lane,
        microbatch_id=mb,
        estimated_cost_units=cost,
        payload_rows=t * h * w,
        output_rows=t * (h // 2) * (w // 2),
        grid_thw=grid,
        owner_worker_id=0,
        sample_padded_start=sample_padded_start,
        sample_padded_len=sample_padded_len,
        decoder_offset_in_sample=decoder_offset_in_sample,
    )


def test_record_round_trip_is_lossless():
    descriptors = (
        _descriptor(0, grid=(2, 6, 8)),
        _descriptor(1, mb=1, sample=3, ordinal=2, cost=123, grid=(1, 4, 4)),
        # The decoder-CP span columns must survive the wire: every planning
        # group member derives its slice table from them, and a member that
        # deserializes them wrong builds a different plan.
        _descriptor(
            2,
            mb=2,
            sample=1,
            sample_padded_start=4096,
            sample_padded_len=2048,
            decoder_offset_in_sample=137,
        ),
    )
    assert records_to_descriptors(descriptors_to_records(descriptors)) == descriptors


def test_record_width_matches_the_declared_slot_count():
    # The broadcast allocates int64[count, DESCRIPTOR_SLOTS]; if the serializer
    # and the constant drift apart the payload is silently truncated or padded
    # on every non-source rank.
    records = descriptors_to_records((_descriptor(0),))
    assert len(records[0]) == DESCRIPTOR_SLOTS


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_install_process_groups_and_registry_dedup():
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    # encoder reduction aliases WORLD; no duplicate same-sized group.
    assert groups.encoder_reduction_group is torch.distributed.group.WORLD
    assert groups.world_group is torch.distributed.group.WORLD
    my_rank = torch.distributed.get_rank()
    view = rank_map.view(my_rank)
    assert (
        torch.distributed.get_world_size(group=groups.planning_group)
        == len(view.planning_group_ranks)
    )
    # Reinstalling returns existing handles: no second new_group per key.
    first_keys = registry.created_keys()
    groups_again = install_mdp_process_groups(rank_map, group_registry=registry)
    assert registry.created_keys() == first_keys
    assert groups_again.planning_group is groups.planning_group
    registry.assert_no_leak()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_broadcast_descriptors_from_endpoint():
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    my_rank = torch.distributed.get_rank()
    view = rank_map.view(my_rank)

    # Endpoints of different groups emit *different* descriptor sets, so the
    # test also proves group isolation.
    lane = view.outer_dp_rank
    endpoint_descriptors = (
        _descriptor(0, mb=0, sample=0, cost=10 + lane, lane=lane, grid=(1, 4, 4)),
        _descriptor(1, mb=1, sample=0, cost=20 + lane, lane=lane, grid=(2, 4, 8)),
    )
    local = endpoint_descriptors if view.lane_id is not None else ()
    flags = (False, False) if view.lane_id is not None else ()
    received, text_only = broadcast_descriptors(
        local,
        planning_group=groups.planning_group,
        endpoint_rank=view.endpoint_rank,
        num_microbatches=2,
        text_only_flags=flags,
    )
    assert received == endpoint_descriptors
    assert text_only == (False, False)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_broadcast_rejects_misordered_descriptors():
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    view = rank_map.view(torch.distributed.get_rank())
    lane = view.outer_dp_rank
    # (microbatch_id, sample_id, image_ordinal) descending: must be rejected
    # on the endpoint before any collective payload is formed.
    bad = (
        _descriptor(0, mb=1, lane=lane),
        _descriptor(1, mb=0, lane=lane),
    )
    if view.lane_id is not None:
        with pytest.raises(MdpBridgeError, match="ascending"):
            broadcast_descriptors(
                bad,
                planning_group=groups.planning_group,
                endpoint_rank=view.endpoint_rank,
                num_microbatches=2,
                text_only_flags=(False, False),
            )
    # Non-endpoint ranks skip; a real run would abort collectively before
    # reaching the broadcast.
