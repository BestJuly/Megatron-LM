# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Build-time CP-local MDP wiring for multimodal training."""

import torch

from megatron.core import parallel_state as ps
from megatron.training import print_rank_0


def _set_model_attrs(model, **attrs) -> None:
    for name, value in attrs.items():
        setattr(model, name, value)


def _disable_unsafe_packed_vision_overlap(args, vision_model, *, mdp_enabled: bool) -> None:
    """Disable DDP overlap before the model is wrapped by DDP.

    Loader-side image ownership can leave a CP rank with no local vision
    work. Async DDP hooks cannot safely assume identical vision-module use
    across those ranks, so the canonical b436 path disables both overlap
    modes while ``model_provider`` still owns the mutable argument object.
    """
    trainable_vision_params = (
        any(param.requires_grad for param in vision_model.parameters())
        if vision_model is not None
        else not bool(getattr(args, "freeze_ViT", False))
    )
    if not (
        trainable_vision_params
        and not bool(getattr(args, "text_only", False))
        and (mdp_enabled or bool(getattr(args, "use_packed_sequence", False)))
    ):
        return

    disabled = []
    for name, flag in (
        ("overlap_grad_reduce", "--overlap-grad-reduce"),
        ("overlap_param_gather", "--overlap-param-gather"),
        ("overlap_param_gather_with_optimizer_step", "--overlap-param-gather-with-optimizer-step"),
    ):
        if bool(getattr(args, name, False)):
            setattr(args, name, False)
            disabled.append(flag)
    if not disabled:
        return

    print_rank_0(
        "> Multimodal MDP: disabled "
        + ", ".join(disabled)
        + " before DDP setup because CP owner ranks can execute different "
        "vision-module graphs."
    )


def configure_mdp_model(model, args):
    """Attach the b436 CP-local MDP process-group contract to ``model``."""
    if getattr(args, "model_arch", None) == "qwen3":
        args.text_only = True
    mdp_enabled = bool(getattr(args, "mdp_encoder_mode", False))
    cp_size = int(getattr(args, "context_parallel_size", 1))
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1))
    inner_dp_scope = getattr(args, "mdp_inner_dp_scope", "cp")
    vision_model = getattr(model, "vision_model", None)
    if getattr(args, "dataset_provider", None) == "energon":
        # Energon emits THD-packed batches. Establish that contract before
        # model/DDP construction; the dataset provider is created later and
        # cannot safely toggle overlap flags retroactively.
        args.use_packed_sequence = True

    if vision_model is None or bool(getattr(args, "text_only", False)):
        # Text-only models such as qwen3 share this training entrypoint but
        # have no image ownership or loader-prepartition contract.
        args.mdp_encoder_mode = False
        _set_model_attrs(
            model,
            _mdp_enabled=False,
            _mdp_inner_dp_group=None,
            _mdp_rank_assignment=None,
            _mdp_rank_assignment_row_counts=None,
        )
        return model

    if mdp_enabled and inner_dp_scope != "cp":
        raise RuntimeError(
            "CP-local MDP requires --mdp-inner-dp-scope cp."
        )

    if mdp_enabled and cp_size <= 1:
        # A one-rank CP scope has no distributed owner set. Keep the loader
        # and model on their ordinary full-vision path together.
        args.mdp_encoder_mode = False
        mdp_enabled = False

    if mdp_enabled and pp_size != 1:
        raise RuntimeError(
            "CP-local MDP requires pipeline_model_parallel_size=1."
        )
    if mdp_enabled and int(getattr(args, "micro_batch_size", 1)) != 1:
        raise RuntimeError("MDP CP-local Energon packing requires micro_batch_size=1")
    if mdp_enabled and getattr(args, "dataset_provider", "energon") != "energon":
        raise RuntimeError("MDP CP-local mode requires --dataset-provider energon")
    if mdp_enabled and bool(getattr(args, "dynamic_context_parallel", False)):
        raise RuntimeError("MDP CP-local mode requires static context parallelism")
    if mdp_enabled and (
        bool(getattr(args, "use_megatron_fsdp", False))
        or bool(getattr(args, "use_torch_fsdp2", False))
    ):
        raise RuntimeError("MDP CP-local mode does not support FSDP")

    _disable_unsafe_packed_vision_overlap(args, vision_model, mdp_enabled=mdp_enabled)

    if not mdp_enabled:
        _set_model_attrs(
            model,
            _mdp_enabled=False,
            _mdp_inner_dp_group=None,
            _mdp_rank_assignment=None,
            _mdp_rank_assignment_row_counts=None,
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
        _mdp_rank_assignment=None,
        _mdp_rank_assignment_row_counts=None,
    )
    return model
