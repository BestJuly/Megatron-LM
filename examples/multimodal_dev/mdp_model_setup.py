# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Build-time CP-local and PP x CP MDP wiring for multimodal training."""

import torch

from megatron.core import parallel_state as ps
from megatron.training import print_rank_0


def _set_model_attrs(model, **attrs) -> None:
    for name, value in attrs.items():
        setattr(model, name, value)


def _mark_vision_params_no_overlap(args, vision_model) -> None:
    """Tag trainable vision parameters to use synchronous DDP buckets.

    CP rank image ownership varies across microbatches, so async grad-reduce
    hooks on vision parameters are not safe.  Marking them with
    ``disable_ddp_overlap=True`` places them in a separate synchronous bucket
    while the decoder keeps full DDP overlap.
    """
    if vision_model is None or bool(getattr(args, "text_only", False)):
        return
    overlap_requested = bool(getattr(args, "overlap_grad_reduce", False)) or bool(
        getattr(args, "overlap_param_gather", False)
    )
    if not overlap_requested:
        return
    trainable = [p for p in vision_model.parameters() if p.requires_grad]
    if not trainable:
        return
    for param in trainable:
        param.disable_ddp_overlap = True
    print_rank_0(
        "> Multimodal MDP: vision parameters placed in synchronous DDP buckets; "
        "decoder grad/param overlap remains enabled."
    )


def _validate_mdp_off_pp_batch_sidecar(args, *, mdp_enabled: bool) -> None:
    """Validate the fixed-shape packed contract used by the no-MDP PP sidecar."""
    if (
        mdp_enabled
        or bool(getattr(args, "text_only", False))
        or int(getattr(args, "pipeline_model_parallel_size", 1)) <= 1
    ):
        return
    if not bool(getattr(args, "use_packed_sequence", False)):
        raise RuntimeError(
            "MDP-off multimodal pipeline training requires --use-packed-sequence "
            "so every pipeline stage uses fixed THD communication shapes"
        )
    if int(getattr(args, "micro_batch_size", 1)) != 1:
        raise RuntimeError(
            "MDP-off multimodal pipeline packed THD currently requires "
            "micro_batch_size=1"
        )


def configure_mdp_model(model, args):
    """Attach the MDP process-group contract to ``model``.

    For VPP (virtual pipeline parallelism), only the chunk that owns
    ``pre_process=True`` (vp_stage 0 on PP0) acts as the sidecar.  All
    other VPP chunks receive a no-sidecar configuration so the encoder
    fires exactly once per optimization step.
    """
    if getattr(args, "model_arch", None) == "qwen3":
        args.text_only = True
    mdp_enabled = bool(getattr(args, "mdp_encoder_mode", True))
    cp_size = int(getattr(args, "context_parallel_size", 1))
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1))
    inner_dp_scope = getattr(args, "mdp_inner_dp_scope", "cp")
    dataset_provider = getattr(args, "dataset_provider", "energon")
    vision_model = getattr(model, "vision_model", None)

    # VPP sidecar ownership: only the first virtual stage (vp_stage=0) runs the
    # vision encoder sidecar.  Higher stages are pure-decoder and must not
    # participate in the group-creation or encoder-gather collectives.
    #
    # torch.distributed.new_group() is a global collective — every rank must
    # call it for the same groups in the same order.  For vp_stage=0, all ranks
    # (PP0 and PP1) must call configure_pp_cp_replicated_vision so that the
    # enc_gather_group and pp_vision_sync_group are built consistently.
    #
    # vp_stage is stored on model.language_model.vp_stage at construction time
    # (training.py sets model.vp_stage only AFTER model_provider returns).
    virtual_size = getattr(args, "virtual_pipeline_model_parallel_size", None)
    model_pre_process = getattr(model, "pre_process", True)
    vp_stage_from_model = getattr(
        getattr(model, "language_model", None), "vp_stage", None
    )
    is_vpp = virtual_size is not None and vp_stage_from_model is not None
    is_higher_vpp_chunk = is_vpp and int(vp_stage_from_model) > 0

    if is_higher_vpp_chunk:
        # vp_stage>0: no group creation needed, just disable the sidecar.
        _set_model_attrs(
            model,
            _mdp_enabled=False,
            _mdp_inner_dp_group=None,
            _mdp_tp_source_group=None,
            _mdp_tp_source_group_device=None,
            _mdp_pp_cp_inner=False,
            _mdp_cp_fused_sidecar=False,
            _pp_cp_batch_sidecar=False,
            _pipeline_sidecar_enabled=False,
            _mdp_rank_assignment=None,
            _mdp_rank_assignment_row_counts=None,
        )
        return model
    # For vp_stage=0 all PP ranks call configure_pp_cp_replicated_vision so the
    # group-creation collectives fire on every rank.  PP1 is not in enc_gather_group
    # (_mdp_inner_dp_group=None for PP1), so _run_mdp_vision_bridge returns a
    # zero-dep scalar without calling the all-gather.  PP1 vision params receive
    # gradients via param.shared=True + finalize_model_grads after backward.

    if bool(getattr(args, "text_only", False)):
        # Text-only models such as qwen3 share this training entrypoint but
        # have no image ownership or loader-prepartition contract.
        args.mdp_encoder_mode = False
        _set_model_attrs(
            model,
            _mdp_enabled=False,
            _mdp_inner_dp_group=None,
            _mdp_tp_source_group=None,
            _mdp_tp_source_group_device=None,
            _mdp_pp_cp_inner=False,
            _mdp_cp_fused_sidecar=False,
            _pp_cp_batch_sidecar=False,
            _pipeline_sidecar_enabled=False,
            _mdp_rank_assignment=None,
            _mdp_rank_assignment_row_counts=None,
        )
        return model

    if mdp_enabled and inner_dp_scope not in ("cp", "pp_cp"):
        raise RuntimeError(
            "--mdp-inner-dp-scope must be either cp or pp_cp; "
            f"got {inner_dp_scope!r}"
        )

    if mdp_enabled and inner_dp_scope == "cp" and cp_size <= 1:
        # A one-rank CP scope has no distributed owner set. Keep the loader
        # and model on their ordinary full-vision path together.
        args.mdp_encoder_mode = False
        mdp_enabled = False

    direct_packing_requested = (
        bool(getattr(args, "dataloader_sequence_packing", False))
        or int(getattr(args, "pack_samples_per_item", 1) or 1) > 1
        or int(getattr(args, "mock_pack_num_docs", 1) or 1) > 1
    )
    if dataset_provider == "energon" or (
        dataset_provider in {"blend", "mock", "mock_mdp"}
        and (mdp_enabled or direct_packing_requested)
    ):
        # Descriptor-backed providers emit fixed THD batches. Establish the
        # contract before model/DDP construction; providers are built later
        # and cannot safely toggle overlap or sidecar state retroactively.
        args.use_packed_sequence = True

    if vision_model is None and mdp_enabled:
        # Preserve the ordinary non-vision fallback for model variants that do
        # not construct a vision tower. A genuinely disabled MDP pipeline is
        # handled below so downstream PP stages can still receive batch data.
        args.mdp_encoder_mode = False
        _set_model_attrs(
            model,
            _mdp_enabled=False,
            _mdp_inner_dp_group=None,
            _mdp_tp_source_group=None,
            _mdp_tp_source_group_device=None,
            _mdp_pp_cp_inner=False,
            _mdp_cp_fused_sidecar=False,
            _pp_cp_batch_sidecar=False,
            _pipeline_sidecar_enabled=False,
            _mdp_rank_assignment=None,
            _mdp_rank_assignment_row_counts=None,
        )
        return model

    if mdp_enabled and inner_dp_scope == "cp" and pp_size != 1:
        raise RuntimeError(
            "CP-local MDP requires pipeline_model_parallel_size=1; "
            "use --mdp-inner-dp-scope pp_cp for replicated pipeline vision"
        )
    from examples.multimodal_dev.mdp_pipeline_sidecar import cp_fused_vision_requested

    pp1_cp_fused = (
        pp_size == 1
        and cp_fused_vision_requested(args)
    )
    if mdp_enabled and inner_dp_scope == "pp_cp" and pp_size <= 1 and not pp1_cp_fused:
        raise RuntimeError(
            "PP=1 pp_cp MDP requires CP>1 fused vision prefetch"
        )
    if mdp_enabled and int(getattr(args, "micro_batch_size", 1)) != 1:
        raise RuntimeError("MDP packed multimodal data requires micro_batch_size=1")
    mdp_dataset_providers = {"blend", "energon", "mock", "mock_mdp"}
    if mdp_enabled and dataset_provider not in mdp_dataset_providers:
        raise RuntimeError(
            "MDP mode requires a descriptor-backed --dataset-provider in "
            f"{sorted(mdp_dataset_providers)}; got {dataset_provider!r}"
        )
    if mdp_enabled and bool(getattr(args, "dynamic_context_parallel", False)):
        raise RuntimeError("MDP mode requires static context parallelism")
    if mdp_enabled and (
        bool(getattr(args, "use_megatron_fsdp", False))
        or bool(getattr(args, "use_torch_fsdp2", False))
    ):
        raise RuntimeError("MDP mode does not support FSDP")

    _validate_mdp_off_pp_batch_sidecar(args, mdp_enabled=mdp_enabled)

    _mark_vision_params_no_overlap(args, vision_model)

    if not mdp_enabled:
        pp_batch_sidecar = (
            pp_size > 1 and bool(getattr(args, "use_packed_sequence", False))
        )
        _set_model_attrs(
            model,
            _mdp_enabled=False,
            _mdp_inner_dp_group=None,
            _mdp_tp_source_group=None,
            _mdp_tp_source_group_device=None,
            _mdp_pp_cp_inner=False,
            _mdp_cp_fused_sidecar=False,
            _pp_cp_batch_sidecar=pp_batch_sidecar,
            _pipeline_sidecar_enabled=pp_batch_sidecar,
            _mdp_rank_assignment=None,
            _mdp_rank_assignment_row_counts=None,
        )
        return model

    if inner_dp_scope == "pp_cp" and pp_size > 1:
        from examples.multimodal_dev.mdp_pipeline_sidecar import (
            configure_pp_cp_replicated_vision,
        )

        if not configure_pp_cp_replicated_vision(model, args):
            raise RuntimeError(
                "PP x CP replicated vision was selected but sidecar setup was not activated"
            )
        # Do NOT disable _pipeline_sidecar_enabled for PP1+VP0 ranks.
        # All VP0 ranks (PP0 and PP1) need to call the sidecar's gather to
        # participate in the NCCL all-gather inside _run_mdp_vision_bridge.
        # The existing return_zero_dependency_only path in _run_mdp_vision_bridge
        # handles non-pre_process ranks: they join the collective but discard the
        # gathered embeddings.
        print_rank_0(
            "> MDP PPxCP multimodal path enabled: "
            f"PP={pp_size}, CP={cp_size}, "
            "packed THD input, vision encoder replicated on every pipeline stage."
            )
        return model

    if not torch.distributed.is_initialized():
        raise RuntimeError("MDP CP-local model setup requires torch.distributed to be initialized")
    rank = int(torch.distributed.get_rank())
    cp_group = ps.get_context_parallel_group()
    if cp_group is None:
        raise RuntimeError(
            "MDP CP-local mode requires Megatron context-parallel process "
            "group handles when CP > 1."
        )
    cp_ranks = [
        int(group_rank) for group_rank in torch.distributed.get_process_group_ranks(cp_group)
    ]
    if rank not in cp_ranks:
        raise RuntimeError(
            f"MDP CP-local mode: rank {rank} is not in its Megatron CP " f"group {cp_ranks}."
        )

    _set_model_attrs(
        model,
        _mdp_enabled=True,
        _mdp_inner_dp_group=cp_group,
        _mdp_tp_source_group=None,
        _mdp_tp_source_group_device=None,
        _mdp_pp_cp_inner=False,
        _mdp_cp_fused_sidecar=pp1_cp_fused,
        _pp_cp_batch_sidecar=False,
        _pipeline_sidecar_enabled=pp1_cp_fused,
        _mdp_rank_assignment=None,
        _mdp_rank_assignment_row_counts=None,
    )
    return model
