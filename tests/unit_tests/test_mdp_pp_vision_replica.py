# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Four-rank PP=2 x CP=2 replicated-vision checks."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev import forward_step as multimodal_forward_step
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
from megatron.core.enums import ModelType
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


class _TinyVision(torch.nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()
        self.projection = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)

    def forward(self, pixel_values, image_grid_thw):
        del image_grid_thw
        return self.projection(pixel_values)


class _FusedPipelineSidecarModel(torch.nn.Module):
    """Tiny pipeline stage using the production fused sidecar implementation."""

    pipeline_sidecar_pre_forward = MultimodalModel.pipeline_sidecar_pre_forward
    mdp_pp_cp_sidecar_pop_cache = MultimodalModel.mdp_pp_cp_sidecar_pop_cache
    mdp_pp_cp_sidecar_activate_cache = MultimodalModel.mdp_pp_cp_sidecar_activate_cache
    mdp_pp_cp_sidecar_compute_vision = MultimodalModel.mdp_pp_cp_sidecar_compute_vision
    _run_mdp_vision_bridge = MultimodalModel._run_mdp_vision_bridge

    def __init__(self, device, config, inner_group):
        super().__init__()
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = parallel_state.is_pipeline_first_stage()
        self.post_process = parallel_state.is_pipeline_last_stage()
        self.vision_model = _TinyVision(config.hidden_size, device)
        self.language_model = torch.nn.Linear(
            config.hidden_size, config.hidden_size, bias=False, device=device
        )
        self._input_tensor = None
        self._pipeline_sidecar_enabled = True
        self._pp_cp_batch_sidecar = False
        self._mdp_enabled = True
        self._mdp_pp_cp_inner = True
        self._mdp_inner_dp_group = inner_group
        self.vp_stage = None
        self.forward_calls = 0
        self.post_backward_calls = 0

    def set_input_tensor(self, input_tensors):
        self._input_tensor = input_tensors[0]

    def pipeline_sidecar_post_backward(self):
        MultimodalModel.pipeline_sidecar_post_backward(self)
        self.post_backward_calls += 1

    def forward(self, first_stage_input):
        cache = self.mdp_pp_cp_sidecar_pop_cache()
        assert cache is not None
        assert cache.get("fused_backward_entries") is not None
        self.mdp_pp_cp_sidecar_activate_cache(cache)
        input_tensor = first_stage_input if self.pre_process else self._input_tensor
        assert input_tensor is not None
        output = self.language_model(input_tensor)
        if self.pre_process:
            output = output + self._mdp_pp_cp_active_vision_embeddings.sum()
        self.forward_calls += 1
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


def _fused_sidecar_batches(device, inner_group, *, count, hidden_size):
    world_size = dist.get_world_size(group=inner_group)
    group_rank = dist.get_rank(group=inner_group)
    assignment = torch.tensor(
        [[rank, 0, rank] for rank in range(world_size)], dtype=torch.int32, device=device
    )
    global_grid = torch.ones((world_size, 3), dtype=torch.int64, device=device)
    local_grid = torch.ones((1, 3), dtype=torch.int64, device=device)
    batches = []
    for microbatch in range(count):
        batches.append(
            {
                "pixel_values": torch.full(
                    (1, hidden_size), float(microbatch * world_size + group_rank + 1), device=device
                ),
                "image_grid_thw": global_grid.clone(),
                "_mdp_prepartitioned_assignment": assignment.clone(),
                "_mdp_prepartitioned_row_counts": torch.ones(
                    world_size, dtype=torch.int64, device=device
                ),
                "_mdp_prepartitioned_image_grid_thw": local_grid.clone(),
                "_mdp_prepartitioned_local_raw_counts": torch.ones(
                    1, dtype=torch.int64, device=device
                ),
            }
        )
    return batches


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

    @pytest.mark.parametrize("num_microbatches", [1, 16])
    @pytest.mark.parametrize("backward_mode", ["retain", "recompute"])
    def test_fused_sidecar_collectives_do_not_deadlock_pipeline_schedule(
        self, monkeypatch, backward_mode, num_microbatches
    ):
        sequence_length = 8
        hidden_size = 4
        args = SimpleNamespace(
            mdp_fused_vision_window=True,
            mdp_vision_encoder_max_sequence_length=1024,
            mdp_fused_vision_backward=backward_mode,
        )
        monkeypatch.setattr("examples.multimodal_dev.models.base.get_args", lambda: args)
        monkeypatch.setattr(multimodal_forward_step, "get_args", lambda: args)
        monkeypatch.setattr(
            multimodal_forward_step, "get_batch", lambda data_iterator: next(data_iterator, None)
        )

        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            context_parallel_size=2,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            deallocate_pipeline_outputs=False,
        )
        model = _FusedPipelineSidecarModel(self.device, config, self.inner_group)
        batches = _fused_sidecar_batches(
            self.device, self.inner_group, count=num_microbatches, hidden_size=hidden_size
        )

        def forward_step_func(data_iterator, stage_model):
            del data_iterator
            microbatch = stage_model.forward_calls
            first_stage_input = None
            if stage_model.pre_process:
                first_stage_input = torch.full(
                    (sequence_length // 2, 1, hidden_size),
                    float(microbatch + 1),
                    device=self.device,
                )
            output = stage_model(first_stage_input)

            def loss_func(output_tensor):
                return output_tensor.sum(), {"loss": output_tensor.detach().sum()}

            return output, loss_func

        losses = schedules.forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step_func,
            data_iterator=iter(batches),
            model=model,
            num_microbatches=num_microbatches,
            seq_length=sequence_length,
            micro_batch_size=1,
            forward_only=False,
        )

        assert model.forward_calls == num_microbatches
        assert model.post_backward_calls == num_microbatches
        assert not model._mdp_pp_cp_sidecar_cache
        assert not model._mdp_pp_cp_sidecar_backward_cache
        assert model.vision_model.projection.weight.grad is not None
        assert torch.isfinite(model.vision_model.projection.weight.grad).all().item()
        if model.pre_process:
            assert model.vision_model.projection.weight.grad.abs().sum().item() > 0
        assert len(losses) == (num_microbatches if model.post_process else 0)
