# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""PP x CP replicated-vision setup for the MDP pipeline sidecar."""

import torch

from examples.multimodal_dev.sidecar_prefetch import validate_fused_vision_window
from megatron.core import parallel_state as ps


def pp_cp_replicated_vision_requested(args) -> bool:
    """Return whether the b436 PP x CP vision-replica path is selected."""
    return (
        bool(getattr(args, "mdp_encoder_mode", False))
        and int(getattr(args, "pipeline_model_parallel_size", 1)) > 1
        and getattr(args, "mdp_inner_dp_scope", "cp") == "pp_cp"
        and not bool(getattr(args, "text_only", False))
    )


def cp_fused_vision_requested(args) -> bool:
    """Return whether PP=1 should precompute fused CP vision windows."""
    return (
        bool(getattr(args, "mdp_encoder_mode", False))
        and int(getattr(args, "pipeline_model_parallel_size", 1)) == 1
        and int(getattr(args, "context_parallel_size", 1)) > 1
        and getattr(args, "mdp_inner_dp_scope", "cp") in ("cp", "pp_cp")
        and not bool(getattr(args, "text_only", False))
        and bool(getattr(args, "use_packed_sequence", False))
        and validate_fused_vision_window(
            getattr(args, "mdp_fused_vision_window", False),
            int(
                getattr(args, "mdp_vision_encoder_max_sequence_length", 0)
                or 0
            ),
        )
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

    # Warm up NCCL topology discovery with a scalar all-reduce before
    # broadcasting large tensors.  Without this, the first broadcast on a
    # cold GPU (no prior NCCL collective in the group) triggers rendezvous
    # which can take O(minutes) on multi-node setups.
    device = torch.device("cuda", torch.cuda.current_device())
    warmup = torch.zeros(1, device=device)
    torch.distributed.all_reduce(warmup, group=group)

    def broadcast_tensor(tensor):
        if tensor.is_cuda:
            torch.distributed.broadcast(tensor, src=source, group=group)
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"replicated vision {group_name} broadcast requires CUDA for CPU staging"
            )
        staged = tensor.to(device=device, non_blocking=False)
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
    """Build the encoder CP gather group (PP0 only) and PP vision sync group.

    Restricting the gather group to PP0 CP ranks eliminates the need for PP1
    to participate in the all-gather/reduce-scatter collectives.  PP1 vision
    encoder params receive gradients via param.shared=True + finalize_model_grads
    after the full pipeline backward completes.
    """
    from examples.multimodal_dev.mdp_parallel_groups import (
        compute_encoder_cp_groups,
        build_local_process_group,
        get_parallel_order,
    )

    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    tp_size = int(getattr(args, "tensor_model_parallel_size", 1))
    cp_size = int(getattr(args, "context_parallel_size", 1))
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1))
    encoder_cp_size = int(getattr(args, "encoder_context_parallel_size", None) or cp_size)
    model_parallel_size = tp_size * cp_size * pp_size
    if model_parallel_size <= 0 or world_size % model_parallel_size:
        raise RuntimeError(
            "PP x CP replicated vision requires world_size divisible by "
            f"TP*CP*PP; got world={world_size}, TP={tp_size}, "
            f"CP={cp_size}, PP={pp_size}"
        )

    order = get_parallel_order(args)
    enc_gather_groups, pp_sync_groups = compute_encoder_cp_groups(
        world_size=world_size,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        encoder_cp_size=encoder_cp_size,
        order=order,
    )

    # enc_gather_groups contains only PP0 ranks; ALL ranks must call new_group()
    # for collective consistency even if not included in the group.
    # A scalar all_reduce immediately after each new_group() forces eager NCCL
    # communicator initialization.  Without this, lazy NCCL init on cold
    # multi-node GPUs hangs when the first real collective fires later
    # (reproducible with VPP=2 on 16+ nodes).
    device = torch.device("cuda", torch.cuda.current_device())
    enc_gather_group = None
    for grp in enc_gather_groups:
        ranks = [int(r) for r in grp]
        pg = torch.distributed.new_group(ranks=ranks)
        torch.distributed.all_reduce(torch.zeros(1, device=device), group=pg)
        if rank in ranks:
            enc_gather_group = pg

    pp_vision_sync_group, _, _ = build_local_process_group(
        rank_groups=pp_sync_groups, this_rank=rank
    )
    torch.distributed.all_reduce(
        torch.zeros(1, device=device), group=pp_vision_sync_group
    )

    if tp_size > 1:
        rank_generator = ps.RankGenerator(
            tp=tp_size, ep=1, dp=world_size // model_parallel_size,
            pp=pp_size, cp=cp_size, order=order,
        )
        tp_source_group, _, _ = build_local_process_group(
            rank_groups=rank_generator.get_ranks("tp"), this_rank=rank, backend="gloo",
        )
    else:
        tp_source_group = None

    return enc_gather_group, pp_vision_sync_group, tp_source_group


def configure_pp_cp_replicated_vision(model, args) -> bool:
    """Attach encoder-CP gather group and PP vision sync group to ``model``.

    Called for every VP0 chunk (both PP0 and PP1) so that new_group() collectives
    run on all ranks.  PP0 uses enc_gather_group for the embedding all-gather;
    PP1 records _mdp_inner_dp_group=None and returns a zero-dep scalar from
    _run_mdp_vision_bridge instead of calling the collective.

    For VPP, higher virtual-stage chunks (vp_stage>0) are skipped by
    configure_mdp_model before this function is ever called.
    """
    if not pp_cp_replicated_vision_requested(args):
        return False

    enc_gather_group, pp_vision_sync_group, tp_source_group = _build_pp_cp_groups(args)
    pp_group = ps.get_pipeline_model_parallel_group()
    broadcast_vision_state(model, pp_group)
    pp_rank = ps.get_pipeline_model_parallel_rank()
    mark_downstream_pp_vision_params_shared(model, pp_rank)

    attrs = {
        "_mdp_enabled": True,
        # PP0 owns the encoder CP gather; PP1 records None and skips the collective.
        "_mdp_inner_dp_group": enc_gather_group if pp_rank == 0 else None,
        "_mdp_pp_vision_sync_group": pp_vision_sync_group,
        "_mdp_is_pp0_gather_rank": pp_rank == 0,
        "_mdp_tp_source_group": tp_source_group,
        "_mdp_tp_source_group_device": (
            "cpu" if tp_source_group is not None else None
        ),
        "_mdp_pp_cp_inner": True,
        "_mdp_cp_fused_sidecar": False,
        "_pp_cp_batch_sidecar": False,
        "_pipeline_sidecar_enabled": True,
        "_mdp_rank_assignment": None,
        "_mdp_rank_assignment_row_counts": None,
    }
    for name, value in attrs.items():
        setattr(model, name, value)
    return True
