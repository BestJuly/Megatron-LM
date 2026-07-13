# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PP x CP replicated-vision setup for the MDP pipeline sidecar."""

import torch

from megatron.core import parallel_state as ps


def pp_cp_replicated_vision_requested(args) -> bool:
    """Return whether the b436 PP x CP vision-replica path is selected."""
    return (
        bool(getattr(args, "mdp_encoder_mode", False))
        and int(getattr(args, "pipeline_model_parallel_size", 1)) > 1
        and getattr(args, "mdp_inner_dp_scope", "cp") == "pp_cp"
        and not bool(getattr(args, "text_only", False))
    )


def broadcast_vision_state(model, group, group_name: str = "PP") -> None:
    """Initialize every replicated vision tower from the first group rank."""
    if group is None or torch.distributed.get_world_size(group=group) <= 1:
        return
    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        raise RuntimeError(
            "PP x CP replicated vision requires vision_model on every pipeline stage"
        )
    source = int(torch.distributed.get_process_group_ranks(group)[0])

    def broadcast_tensor(tensor):
        if tensor.is_cuda:
            torch.distributed.broadcast(tensor, src=source, group=group)
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"replicated vision {group_name} broadcast requires CUDA for CPU staging"
            )
        staged = tensor.to(
            device=torch.device("cuda", torch.cuda.current_device()),
            non_blocking=False,
        )
        torch.distributed.broadcast(staged, src=source, group=group)
        tensor.copy_(staged.cpu())

    with torch.no_grad():
        for tensor in (*vision_model.parameters(), *vision_model.buffers()):
            broadcast_tensor(tensor.data)


def mark_downstream_pp_vision_params_shared(model, pp_rank: int) -> None:
    """Exclude non-PP0 replicas from global parameter-norm accounting."""
    if int(pp_rank) == 0:
        return
    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        return
    for parameter in vision_model.parameters():
        parameter.shared = True


def _build_pp_cp_groups(args):
    from examples.multimodal_dev.mdp_parallel_groups import (
        build_local_process_group,
        build_pp_cp_inner_dp_group,
        get_parallel_order,
    )

    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    tp_size = int(getattr(args, "tensor_model_parallel_size", 1))
    cp_size = int(getattr(args, "context_parallel_size", 1))
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1))
    model_parallel_size = tp_size * cp_size * pp_size
    if model_parallel_size <= 0 or world_size % model_parallel_size:
        raise RuntimeError(
            "PP x CP replicated vision requires world_size divisible by "
            f"TP*CP*PP; got world={world_size}, TP={tp_size}, "
            f"CP={cp_size}, PP={pp_size}"
        )

    rank_generator = ps.RankGenerator(
        tp=tp_size,
        ep=1,
        dp=world_size // model_parallel_size,
        pp=pp_size,
        cp=cp_size,
        order=get_parallel_order(args),
    )
    inner_group, _, _ = build_pp_cp_inner_dp_group(
        pp_cp_groups=rank_generator.get_ranks("pp-cp"),
        this_rank=rank,
    )
    if tp_size > 1:
        tp_source_group, _, _ = build_local_process_group(
            rank_groups=rank_generator.get_ranks("tp"),
            this_rank=rank,
            backend="gloo",
        )
    else:
        tp_source_group = None
    return inner_group, tp_source_group


def configure_pp_cp_replicated_vision(model, args) -> bool:
    """Attach PP x CP InnerDP, replica, checkpoint, and schedule metadata."""
    if not pp_cp_replicated_vision_requested(args):
        return False
    if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
        raise RuntimeError(
            "PP x CP replicated vision supports only non-interleaved pipeline schedules"
        )
    if not bool(getattr(args, "use_packed_sequence", False)):
        raise RuntimeError(
            "PP x CP replicated vision requires packed THD input"
        )
    if bool(getattr(args, "use_megatron_fsdp", False)) or bool(
        getattr(args, "use_torch_fsdp2", False)
    ):
        raise RuntimeError("PP x CP replicated vision does not support FSDP")
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "PP x CP replicated vision setup requires torch.distributed"
        )
    if getattr(model, "vision_model", None) is None:
        raise RuntimeError(
            "PP x CP replicated vision requires vision_model on every pipeline stage"
        )

    inner_group, tp_source_group = _build_pp_cp_groups(args)
    pp_group = ps.get_pipeline_model_parallel_group()
    broadcast_vision_state(model, pp_group)
    mark_downstream_pp_vision_params_shared(
        model,
        ps.get_pipeline_model_parallel_rank(),
    )

    attrs = {
        "_mdp_enabled": True,
        "_mdp_inner_dp_group": inner_group,
        "_mdp_tp_source_group": tp_source_group,
        "_mdp_tp_source_group_device": (
            "cpu" if tp_source_group is not None else None
        ),
        "_mdp_pp_cp_inner": True,
        "_pp_cp_batch_sidecar": False,
        "_pipeline_sidecar_enabled": True,
        "_mdp_rank_assignment": None,
        "_mdp_rank_assignment_row_counts": None,
    }
    for name, value in attrs.items():
        setattr(model, name, value)
    return True
