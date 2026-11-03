# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-compute tests for the MDP rank map. No distributed state, no CUDA.

Includes the registered extension-hook test at encoder_cp=2 (design doc 12.1):
worker partitioning, group disjointness, and endpoint ownership must hold
before any encoder-CP runtime exists.
"""

import pytest

from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.parallel_state import RankGenerator


def _spec(**kwargs):
    base = dict(world_size=8, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    base.update(kwargs)
    return MdpRankSpec(**base)


def test_design_doc_example_w8_pp2():
    # Design doc section 6: W=8, PP=2, DP=4 -> groups (0,4),(1,5),(2,6),(3,7),
    # endpoints 0-3, global_rank = dp_rank + 4 * pp_rank.
    rank_map = build_rank_map(_spec())
    assert rank_map.planning_groups() == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert [rank_map.endpoint_rank(d) for d in range(4)] == [0, 1, 2, 3]
    view = rank_map.view(5)
    assert view.outer_dp_rank == 1
    assert view.lane_id is None
    assert view.my_worker_id == 1
    assert view.endpoint_rank == 1
    assert view.planning_group_ranks == (1, 5)
    assert view.worker_ids == (0, 1)
    endpoint_view = rank_map.view(1)
    assert endpoint_view.lane_id == 1
    assert endpoint_view.my_worker_id == 0


def test_groups_form_disjoint_world_partition():
    for pp, cp, world in ((1, 1, 4), (2, 1, 8), (4, 1, 8), (2, 2, 16)):
        rank_map = build_rank_map(_spec(world_size=world, pp=pp, cp=cp))
        seen = set()
        for group in rank_map.planning_groups():
            assert len(group) == pp * cp
            assert not (seen & set(group))
            seen |= set(group)
        assert seen == set(range(world))
        for rank in range(world):
            view = rank_map.view(rank)
            assert rank in view.planning_group_ranks
            assert view.my_worker_id in view.worker_ids


def test_worker_ranks_is_the_single_resolution_point():
    rank_map = build_rank_map(_spec())
    for outer_dp_rank in range(4):
        for worker_id in rank_map.view(rank_map.endpoint_rank(outer_dp_rank)).worker_ids:
            ranks = rank_map.worker_ranks(outer_dp_rank, worker_id)
            assert len(ranks) == 1  # encoder_cp=1: one rank per logical worker
            assert rank_map.view(ranks[0]).my_worker_id == worker_id
    with pytest.raises(MdpConfigurationError, match="worker_id"):
        rank_map.worker_ranks(0, 2)


def test_extension_hook_encoder_cp2():
    # encoder_cp=2 over CP=2, PP=2: 4 workers' ranks collapse to 2 logical
    # workers of 2 ranks each; assignment-visible worker ids are unchanged
    # by the physical expansion.
    rank_map = build_rank_map(_spec(world_size=16, pp=2, cp=2, encoder_cp=2))
    assert rank_map.num_workers_per_group == 2
    seen = set()
    for outer_dp_rank, group in enumerate(rank_map.planning_groups()):
        assert len(group) == 4
        expansion = [rank_map.worker_ranks(outer_dp_rank, w) for w in (0, 1)]
        assert all(len(ranks) == 2 for ranks in expansion)
        # The expansion partitions the group with no overlap.
        flat = [rank for ranks in expansion for rank in ranks]
        assert sorted(flat) == sorted(group)
        assert not (seen & set(flat))
        seen |= set(flat)
        # The endpoint lives in worker 0.
        assert rank_map.endpoint_rank(outer_dp_rank) in expansion[0]
        for rank in group:
            view = rank_map.view(rank)
            assert view.worker_ids == (0, 1)
            assert rank in rank_map.worker_ranks(outer_dp_rank, view.my_worker_id)
    assert seen == set(range(16))


def test_tp2_pp1_every_rank_is_a_worker_and_peers_hold_leaves():
    # W=8, TP=2, PP=1 -> DP=4; rank = tp + 2 * dp, so each planning group is one
    # decoder TP group and the endpoint is its tp=0 rank.
    rank_map = build_rank_map(_spec(world_size=8, tp=2, pp=1))
    assert rank_map.planning_groups() == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert rank_map.num_workers_per_group == 2
    endpoint = rank_map.view(2)
    peer = rank_map.view(3)
    assert endpoint.lane_id == 1 and endpoint.is_endpoint and not endpoint.is_leaf_peer
    assert peer.lane_id is None and peer.is_leaf_peer and not peer.is_endpoint
    assert endpoint.leaf_tp_ranks == peer.leaf_tp_ranks == (2, 3)
    assert endpoint.shard_owners == peer.shard_owners == (2,)
    assert peer.leaf_tp_source == endpoint.leaf_tp_source == 2
    assert (endpoint.my_worker_id, peer.my_worker_id) == (0, 1)
    # Only the TP source loads data, so only its worker owns pixels.
    assert endpoint.pixel_owners == peer.pixel_owners == (0,)
    assert endpoint.worker_ids == (0, 1)


def test_tp2_pp2_groups_workers_and_pixel_owners():
    # W=8, TP=2, PP=2 -> DP=2; rank = tp + 2 * dp + 4 * pp.
    rank_map = build_rank_map(_spec(world_size=8, tp=2, pp=2))
    assert rank_map.planning_groups() == ((0, 1, 4, 5), (2, 3, 6, 7))
    assert rank_map.num_workers_per_group == 4
    for outer_dp_rank, group in enumerate(rank_map.planning_groups()):
        resolved = [rank_map.worker_ranks(outer_dp_rank, w) for w in range(4)]
        assert [ranks[0] for ranks in resolved] == list(group)
        for rank in group:
            view = rank_map.view(rank)
            # Only PP0 ranks read leaves; PP1 has no leaf TP group.
            assert view.leaf_tp_ranks == (group[:2] if rank in group[:2] else ())
            # The TP sources are the PP0 and PP1 tp=0 ranks: indices 0 and 2.
            assert view.pixel_owners == (0, 2)
            assert view.is_leaf_peer == (rank == group[1])
            assert rank_map.worker_ranks(outer_dp_rank, view.my_worker_id) == (rank,)
    # PP1's TP pair is not part of the leaf broadcast: PP1 never reads leaves.
    assert not rank_map.view(5).is_leaf_peer and not rank_map.view(4).is_endpoint


def test_tp1_view_keeps_every_worker_as_pixel_owner():
    rank_map = build_rank_map(_spec())
    view = rank_map.view(4)
    assert view.pixel_owners == view.worker_ids == (0, 1)
    assert view.leaf_tp_ranks == ()  # rank 4 is PP1
    assert view.shard_owners == (0,) and not view.is_shard_owner
    assert not view.is_leaf_peer


def test_local_view_has_no_global_lists():
    # O(W^2) guard: a view carries only its own group, not all groups.
    rank_map = build_rank_map(_spec(world_size=8, pp=2))
    view = rank_map.view(3)
    assert len(view.planning_group_ranks) == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(tp=0), "tp"),
        (dict(world_size=6, pp=4), "world_size"),
        (dict(encoder_cp=3, world_size=16, cp=2), "encoder_cp"),
        (dict(rank_order="tp-ep-dp-pp-cp"), "rank_order"),
        (dict(pp=0), "pp"),
    ],
)
def test_invalid_specs_rejected(kwargs, match):
    with pytest.raises(MdpConfigurationError, match=match):
        build_rank_map(_spec(**kwargs))


@pytest.mark.parametrize(
    "tp,cp,pp",
    [(1, 2, 1), (2, 2, 1), (1, 2, 2), (2, 2, 2), (2, 4, 1), (4, 2, 1), (1, 4, 2)],
)
def test_cp_shard_owners_and_leaf_tp_groups_match_rank_generator(tp, cp, pp):
    # Cross-check the view's index arithmetic against the decoder's own groups:
    # shard owners are the PP0 tp=0 ranks ordered by cp_rank (= the CP group of
    # the endpoint), and every PP0 rank's leaf TP group is its decoder TP group.
    world = 2 * tp * cp * pp
    rank_map = build_rank_map(_spec(world_size=world, tp=tp, pp=pp, cp=cp))
    generator = RankGenerator(tp=tp, ep=1, dp=world // (tp * cp * pp), pp=pp, cp=cp,
                              order="tp-cp-ep-dp-pp")
    tp_groups = {r: tuple(g) for g in generator.get_ranks("tp") for r in g}
    cp_groups = {r: tuple(g) for g in generator.get_ranks("cp") for r in g}
    pp_groups = {r: tuple(g) for g in generator.get_ranks("pp") for r in g}
    for group in rank_map.planning_groups():
        endpoint = group[0]
        pp0 = {r for r in group if pp_groups[r][0] == r}
        for rank in group:
            view = rank_map.view(rank)
            assert view.shard_owners == cp_groups[endpoint]
            assert len(view.shard_owners) == cp
            if rank in pp0:
                assert view.leaf_tp_ranks == tp_groups[rank]
                assert view.leaf_tp_source == tp_groups[rank][0]
                assert view.is_shard_owner == (rank == tp_groups[rank][0])
                assert view.is_leaf_peer == (rank != tp_groups[rank][0])
            else:
                assert view.leaf_tp_ranks == () and not view.is_leaf_peer
                assert not view.is_shard_owner
            assert view.is_endpoint == (rank == endpoint)
            # Every rank is an encoder worker; every TP source owns pixels.
            assert rank_map.num_workers_per_group == tp * cp * pp
            assert view.pixel_owners == tuple(range(0, tp * cp * pp, tp))
