# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Four-rank PP=2 x CP=2 replicated-vision checks.

Run with:

```
torchrun --standalone --nproc_per_node=4 -m pytest -q \
  tests/unit_tests/test_mdp_pp_vision_replica.py
```
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev.mdp_parallel_groups import (
    build_pp_cp_inner_dp_group,
    compute_pp_cp_inner_dp_layout,
)
from examples.multimodal_dev.mdp_pipeline_sidecar import broadcast_vision_state
from examples.multimodal_dev.modality_bridge import gather_to_inner_dp_zero
from examples.multimodal_dev.models.base import MultimodalModel
from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, save
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_mdp_pp_cp_vision_grads,
    finalize_model_grads,
)
from megatron.core.pipeline_parallel import schedules
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class _VisionContainer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self._mdp_pp_cp_inner = True
        self.vision_model = torch.nn.Linear(2, 2, bias=False, device=device)


class _PartitionedVisionReplica(torch.nn.Module):
    def __init__(self, group, *, skew_distopt_layout=False):
        super().__init__()
        self._mdp_pp_cp_inner = True
        self.pre_process = parallel_state.is_pipeline_first_stage()
        self.post_process = parallel_state.is_pipeline_last_stage()
        self.vision_model = torch.nn.Linear(2, 2, bias=False)
        if skew_distopt_layout and parallel_state.get_pipeline_model_parallel_rank() == 1:
            self.stage_only = torch.nn.Embedding(1, 129)
        self.group = group
        self.share_embeddings_and_output_weights = False

    def forward(self, pixels, rank_assignment, row_counts, language_gradient):
        local_embeddings = self.vision_model(pixels)
        gathered = gather_to_inner_dp_zero(
            local_embeddings=local_embeddings,
            rank_assignment=rank_assignment,
            encoder_dp_group=self.group,
            global_per_image_row_counts=row_counts,
        )
        output = (gathered * language_gradient).sum()
        if hasattr(self, "stage_only"):
            output = output + self.stage_only.weight.sum() * 0.0
        return output


def _owned_param_indices(model, parameter):
    for buffer in model.buffers:
        if parameter not in buffer.param_index_map:
            continue
        start, end, bucket_id = buffer.param_index_map[parameter]
        bucket = buffer.buckets[bucket_id]
        shard = bucket.grad_data.numel() // dist.get_world_size(group=model.dp_cp_group)
        dp_rank = dist.get_rank(group=model.dp_cp_group)
        owned_start = bucket.offset + dp_rank * shard
        owned_end = owned_start + shard
        overlap_start = max(start, owned_start)
        overlap_end = min(end, owned_end)
        return list(range(overlap_start - start, overlap_end - start))
    raise AssertionError("vision parameter was not assigned to a DDP buffer")


def _checkpoint_model(device):
    model = MultimodalModel.__new__(MultimodalModel)
    torch.nn.Module.__init__(model)
    model.vision_model = torch.nn.Linear(2, 2, bias=False, device=device)
    model.language_model = torch.nn.Module()
    model._mdp_pp_cp_inner = True
    return model


class TestPpCpVisionReplica:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        if Utils.world_size != 4:
            pytest.skip("requires four ranks for PP=2 x CP=2")
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2, context_parallel_size=2)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.pp_group = parallel_state.get_pipeline_model_parallel_group()
        groups = compute_pp_cp_inner_dp_layout(
            world_size=dist.get_world_size(), tp_size=1, cp_size=2, pp_size=2
        )
        self.inner_group, self.inner_ranks, _ = build_pp_cp_inner_dp_group(
            pp_cp_groups=groups, this_rank=dist.get_rank()
        )

    def teardown_method(self):
        dist.destroy_process_group(self.inner_group)
        Utils.destroy_model_parallel()

    def test_exact_pp_sum_including_missing_local_grad(self):
        model = _VisionContainer(self.device)
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        global_rank = torch.distributed.get_rank()
        if pp_rank == 0:
            model.vision_model.weight.grad = torch.full_like(
                model.vision_model.weight, float(global_rank + 1)
            )
        else:
            model.vision_model.weight.grad = None

        _allreduce_mdp_pp_cp_vision_grads([model], pp_group=self.pp_group)

        source_rank = torch.distributed.get_process_group_ranks(self.pp_group)[0]
        expected = torch.full_like(model.vision_model.weight, float(source_rank + 1))
        torch.testing.assert_close(model.vision_model.weight.grad, expected, rtol=0, atol=0)

    def test_initial_state_comes_from_pp_zero(self):
        model = SimpleNamespace(vision_model=torch.nn.Linear(2, 2, bias=False, device=self.device))
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        with torch.no_grad():
            model.vision_model.weight.fill_(float(pp_rank + 1))

        broadcast_vision_state(model, self.pp_group)

        torch.testing.assert_close(
            model.vision_model.weight, torch.ones_like(model.vision_model.weight), rtol=0, atol=0
        )

    def test_distopt_skewed_buffer_sums_before_reduce_scatter(self):
        config = TransformerConfig(
            num_layers=1,
            num_attention_heads=1,
            context_parallel_size=2,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        module = _PartitionedVisionReplica(self.inner_group, skew_distopt_layout=True).to(
            self.device
        )
        with torch.no_grad():
            module.vision_model.weight.copy_(torch.eye(2, device=self.device))
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if pp_rank != 0:
            for parameter in module.vision_model.parameters():
                parameter.shared = True

        model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                use_distributed_optimizer=True,
            ),
            module,
        )
        model.zero_grad_buffer()

        owner = dist.get_rank(group=self.inner_group)
        pixels = torch.tensor([[owner + 1.0, 1.0]], device=self.device)
        assignment = {rank: [(0, rank)] for rank in range(4)}
        row_counts = [1, 1, 1, 1]
        language_gradient = torch.zeros(4, 2, device=self.device)
        if pp_rank == 0:
            if parallel_state.get_context_parallel_rank() == 0:
                language_gradient[0] = torch.tensor([1.0, 2.0], device=self.device)
                language_gradient[2] = torch.tensor([3.0, 4.0], device=self.device)
            else:
                language_gradient[1] = torch.tensor([5.0, 6.0], device=self.device)
                language_gradient[3] = torch.tensor([7.0, 8.0], device=self.device)

        model(pixels, assignment, row_counts, language_gradient).backward()
        finalize_model_grads([model])

        parameter = model.module.vision_model.weight
        owned = _owned_param_indices(model, parameter)
        layouts = [None] * dist.get_world_size()
        dist.all_gather_object(
            layouts,
            {"pp": pp_rank, "cp": parallel_state.get_context_parallel_rank(), "owned": owned},
        )
        ownership = {(item["pp"], item["cp"]): item["owned"] for item in layouts}
        assert any(ownership[(0, cp_rank)] != ownership[(1, cp_rank)] for cp_rank in (0, 1))

        expected = torch.tensor([[24.0, 8.0], [29.0, 10.0]], device=self.device).flatten()
        if owned:
            torch.testing.assert_close(
                parameter.main_grad.flatten()[owned], expected[owned], rtol=0, atol=0
            )

    def test_pipeline_replica_checkpoint_round_trip(self, tmp_path_dist_ckpt):
        model = _checkpoint_model(self.device)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        with torch.no_grad():
            model.vision_model.weight.copy_(expected)
        broadcast_vision_state(model, self.pp_group)

        sharded_state = model.sharded_state_dict()
        replica_id = sharded_state["vision_model.weight"].replica_id
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if isinstance(replica_id, int):
            assert replica_id == pp_rank
        else:
            assert replica_id[0] == pp_rank

        with TempNamedDir(
            tmp_path_dist_ckpt / "test_mdp_pp_vision_replica_checkpoint", sync=True
        ) as checkpoint_dir:
            save_request = save(
                sharded_state, checkpoint_dir, async_sharded_save=True, async_strategy="mcore"
            )
            save_request.execute_sync()
            with torch.no_grad():
                model.vision_model.weight.fill_(dist.get_rank() + 10.0)
            state_dict = load(model.sharded_state_dict(), checkpoint_dir)
            model.load_state_dict(state_dict)

            torch.testing.assert_close(model.vision_model.weight, expected, rtol=0, atol=0)

    def test_schedule_hooks_are_active_on_all_pp_cp_ranks(self):
        calls = []

        def pre_forward(**kwargs):
            calls.append(kwargs["current_microbatch"])

        model = SimpleNamespace(
            _pipeline_sidecar_enabled=True,
            pipeline_sidecar_pre_forward=pre_forward,
            pipeline_sidecar_post_backward=lambda: None,
        )
        pre_hook, post_hook = schedules._get_pipeline_sidecar_hooks(model)
        assert post_hook is not None
        schedules._prefetch_pipeline_sidecar(
            pre_hook, data_iterator=object(), num_microbatches=3, forward_only=False
        )
        assert calls == [0, 1, 2]

        total_calls = torch.tensor([len(calls)], device=self.device)
        dist.all_reduce(total_calls)
        assert int(total_calls.item()) == 12
