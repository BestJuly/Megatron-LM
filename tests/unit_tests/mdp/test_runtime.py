# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Runtime phase-machine tests with a stub adapter and tiny encoder.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_runtime.py
"""

import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import build_encoder_domain, build_encoder_pg_collection
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1
pytestmark = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


WIDTH = 8  # payload width == hidden size, identity encoder
MERGE = 2
GRIDS = ((1, 4, 4), (1, 8, 8), (2, 4, 4))  # 16/64/32 payload; 4/16/8 output rows


def _sentinel(lane, item_index):
    return float(10 * (lane + 1) + item_index)


class _TinyEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(WIDTH))

    def forward(self, x):
        return self.proj(x)


class _StubAdapter:
    """Deterministic capture: two microbatches, three items in mb0, mb1 text-only."""

    payload_width = WIDTH
    spatial_merge_size = MERGE

    def __init__(self, lane):
        self._lane = lane if lane is not None else 0

    def get_batch(self, iterator):
        mb = next(iterator)
        if mb != 0:
            return CapturedMicrobatch(
                decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
                vision_items=(),
                flat_pixel_payload=None,
                model_payload=MappingProxyType({"microbatch": mb}),
            )
        items = []
        payload_chunks = []
        payload_start = 0
        for index, grid in enumerate(GRIDS):
            t, h, w = grid
            rows = t * h * w
            output_rows = t * (h // MERGE) * (w // MERGE)
            items.append(
                CapturedVisionItem(
                    sample_id=index,
                    image_ordinal=0,
                    grid_thw=grid,
                    payload_row_start=payload_start,
                    payload_rows=rows,
                    decoder_positions=tuple(range(output_rows)),
                )
            )
            payload_chunks.append(
                torch.full((rows, WIDTH), _sentinel(self._lane, index), device="cuda")
            )
            payload_start += rows
        return CapturedMicrobatch(
            decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
            vision_items=tuple(items),
            flat_pixel_payload=torch.cat(payload_chunks),
            model_payload=MappingProxyType({"microbatch": mb}),
        )

    def estimate_cost(self, item):
        return item.payload_rows

    def build_encoder(self, model_config, *, pg_collection):
        return _TinyEncoder(model_config)

    def encode(self, encoder, payload, layout):
        pieces = []
        for segment in layout.segments:
            pieces.append(
                encoder(
                    payload[
                        segment.payload_row_start : segment.payload_row_start
                        + segment.output_rows
                    ]
                )
            )
        return torch.cat(pieces) if pieces else payload[:0]


def _build_runtime():
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    view = rank_map.view(rank)
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
    adapter = _StubAdapter(view.outer_dp_rank)
    model_config = TransformerConfig(
        num_layers=1,
        hidden_size=WIDTH,
        num_attention_heads=1,
        calculate_per_token_loss=True,
        use_cpu_initialization=True,
    )
    domain = build_encoder_domain(
        adapter=adapter,
        model_config=model_config,
        mdp_config=MdpConfig(enable=True),
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1e-3, use_distributed_optimizer=True, clip_grad=1.0
        ),
        encoder_pgs=encoder_pgs,
        wrap_mixed_precision=False,
    )
    allocator = DirectBufferAllocator()
    config = MdpConfig(enable=True)
    runtime = MdpRuntime(
        config=config,
        rank_map=rank_map,
        rank_view=view,
        process_groups=groups,
        adapter=adapter,
        encoder_domain=domain,
        planner=MdpPlanner(
            view,
            locality_slack_permille=config.locality_slack_permille,
            capacity_policy=RowCapacityPolicy(config.row_alignment),
        ),
        bridge=ModalityBridge(allocator),
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=WIDTH,
        params_dtype=torch.float32,
        num_vpp_chunks=1,
    )
    return runtime, view


def _drive_decoder(runtime, view, replay_iters, *, backward):
    """Consume the replay iterator like the native schedule would."""
    records = [next(replay_iters[0]) for _ in range(2)]
    assert [r.model_payload["microbatch"] for r in records] == [0, 1]
    if view.lane_id is not None:
        leaf = runtime.storage.get_leaf(0)
        assert leaf is not None
        assert runtime.storage.get_leaf(1) is None  # text-only
        # Forward routing correctness: every leaf row carries its item's
        # sentinel (identity encoder, sentinel pixels).
        offset = 0
        for index, grid in enumerate(GRIDS):
            t, h, w = grid
            rows = t * (h // MERGE) * (w // MERGE)
            block = leaf[offset : offset + rows]
            assert (block == _sentinel(view.outer_dp_rank, index)).all(), index
            offset += rows
        if backward:
            (leaf * 2.0).sum().backward()
    return records


def test_full_training_iteration_and_state_machine():
    runtime, view = _build_runtime()
    assert runtime.state is MdpRuntimeState.EMPTY

    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    assert runtime.state is MdpRuntimeState.DECODER_READY
    _drive_decoder(runtime, view, replay, backward=True)

    tokens = torch.tensor(20.0, device="cuda")
    runtime.capture_global_num_tokens(tokens)
    assert runtime.consumed_num_tokens() is tokens  # same object, never cloned
    assert runtime.consumed_num_tokens().data_ptr() == tokens.data_ptr()
    runtime.mark_decoder_complete()
    assert runtime.state is MdpRuntimeState.DECODER_DONE
    runtime.end_iteration()
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime.iteration == 1

    # Encoder gradients exist (ZeRO-1 leaves each rank its reduced shard);
    # after one optimizer step the all-gathered parameters must be identical
    # on every rank, proving the WORLD reduction and shared update.
    param = next(runtime.encoder_domain.encoder_ddp.module.parameters())
    assert param.main_grad.abs().sum() > 0
    success, _, _ = runtime.encoder_domain.encoder_optimizer.step()
    assert success
    world = torch.distributed.get_world_size()
    gathered = [torch.empty_like(param.data) for _ in range(world)]
    torch.distributed.all_gather(gathered, param.data)
    for other in gathered[1:]:
        assert torch.equal(other, gathered[0])
    # The step moved the identity weights: gradients were really applied.
    assert not torch.equal(param.data, torch.eye(WIDTH, device=param.device))


def test_forward_only_iteration_captures_nothing_and_cleans_up():
    runtime, view = _build_runtime()
    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=True
    )
    _drive_decoder(runtime, view, replay, backward=False)
    runtime.mark_decoder_complete()  # eval requires no token capture
    runtime.end_iteration()
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()


def test_invalid_transitions_raise():
    runtime, view = _build_runtime()
    with pytest.raises(MdpStateError, match="mark_decoder_complete"):
        runtime.mark_decoder_complete()
    with pytest.raises(MdpStateError, match="end_iteration"):
        runtime.end_iteration()

    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    with pytest.raises(MdpStateError, match="begin_iteration"):
        runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)

    # Training decoder completion without a token capture is an error.
    _drive_decoder(runtime, view, replay, backward=True)
    with pytest.raises(MdpStateError, match="exactly one global"):
        runtime.mark_decoder_complete()
    # Recover: capture once, then a second capture must fail.
    tokens = torch.tensor(8.0, device="cuda")
    runtime.capture_global_num_tokens(tokens)
    with pytest.raises(MdpStateError, match="more than once"):
        runtime.capture_global_num_tokens(tokens)
    runtime.mark_decoder_complete()
    runtime.end_iteration()


def test_iteration_metrics_are_populated():
    runtime, view = _build_runtime()
    assert runtime.last_iteration_metrics() is None
    replay = runtime.begin_iteration(
        iter(range(10)), num_microbatches=2, forward_only=False
    )
    _drive_decoder(runtime, view, replay, backward=True)
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    metrics = runtime.last_iteration_metrics()
    assert metrics.iteration == 0
    assert metrics.outer_dp_rank == view.outer_dp_rank
    assert metrics.plan_build_ms >= 0.0
    assert metrics.decoder_schedule_ms >= 0.0
    assert len(metrics.worker_loads) == len(view.worker_ids)
    assert sum(metrics.worker_loads) == 16 + 64 + 32  # all payload rows
    assert set(metrics.bridge_stats) == {"pixel", "embedding", "gradient"}
    assert metrics.bridge_stats["pixel"].total_bytes > 0
    assert all(count == 0 for count in metrics.allocator_reuse.values())
