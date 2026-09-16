# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed-sequence metadata helpers for CP partition-mode tracking."""

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


def _validate_internal_gdr_64_aligned_packed_seq_params(
    packed_seq_params: "PackedSeqParams",
) -> None:
    """Reject packed layouts unsupported by the temporary internal GDR kernel."""
    # [REMOVE BEFORE MERGE] The current internal GDR CuTe kernels used by the
    # ptyche experiments only support positive 64-token-aligned packed
    # sequences. Keep this workaround centralized here so GDN/backend code can
    # trust finalized packed-sequence metadata without duplicating value checks.
    #
    # Check only the array the GDR kernel is actually handed. _GDNBase._resolve_cu_seqlens
    # takes the padded variant whenever it exists and the unpadded one otherwise, and
    # gdn.py passes that single resolved tensor as `cu_seqlens` to every backend. Under
    # --pad-packed-seq-alignment the raw cu_seqlens_q still carries the unaligned real
    # token counts, so validating it rejects packings the kernel would never see.
    for padded_name, raw_name in (
        ("cu_seqlens_q_padded", "cu_seqlens_q"),
        ("cu_seqlens_kv_padded", "cu_seqlens_kv"),
    ):
        cu_seqlens = getattr(packed_seq_params, padded_name, None)
        name = padded_name
        if cu_seqlens is None:
            cu_seqlens = getattr(packed_seq_params, raw_name, None)
            name = raw_name
        if cu_seqlens is None:
            continue
        offsets = cu_seqlens.detach().cpu() if cu_seqlens.device.type != "cpu" else cu_seqlens
        if offsets.numel() < 2:
            raise ValueError(f"{name} must contain at least two offsets.")

        final_offset = offsets[-1].item()
        logical_entries = offsets.numel()
        while logical_entries > 1 and offsets[logical_entries - 2].item() == final_offset:
            logical_entries -= 1
        offsets = offsets[:logical_entries]
        if offsets.numel() < 2 or offsets[0].item() != 0:
            raise ValueError(f"{name} must start at 0 and contain at least one real sequence.")

        lengths = offsets[1:] - offsets[:-1]
        if bool((lengths <= 0).any().item()) or bool((lengths % 64 != 0).any().item()):
            raise ValueError(
                f"[REMOVE BEFORE MERGE] Current internal GDR experiments require every "
                f"packed sequence length in {name} to be a positive multiple of 64; "
                f"got lengths: {lengths.tolist()}."
            )


def get_packed_seq_params_cp_partition_cu_seqlens(
    packed_seq_params: Optional["PackedSeqParams"],
) -> Optional[torch.Tensor]:
    """Return THD cumulative sequence lengths used for CP layout conversion.

    ``packed_seq_params=None`` represents the ordinary SBHD path. Only THD
    metadata carries global packed-token boundaries.
    """
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None
    return (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )


def finalize_packed_seq_params(
    packed_seq_params: Optional["PackedSeqParams"],
    gdn_gdr_backend: Optional[str] = None,
) -> Optional["PackedSeqParams"]:
    """Resolve CP metadata and prebuild the THD layout route for a microbatch.

    Args:
        packed_seq_params: THD metadata for this microbatch, or None for SBHD.
        gdn_gdr_backend: value of ``TransformerConfig.gdn_gdr_backend``. Only
            ``"internal"`` triggers the temporary 64-alignment check below;
            callers that omit it get no backend-specific validation.
    """
    if packed_seq_params is None:
        return None

    # Keep these imports local: routes depends on this module for metadata access.
    from megatron.core.context_parallel_layout.routes import prebuild_thd_cp_partition_routes
    from megatron.core.packed_seq_params import resolve_cp_group
    from megatron.core.parallel_state import get_context_parallel_group

    # Gate on the backend that actually has the restriction. This check is a
    # [REMOVE BEFORE MERGE] workaround for the internal GDR CuTe kernels; the
    # cuDNN and fla backends have no 64-alignment requirement, and running it
    # for them rejects packings they handle correctly. It was unconditional
    # until 2026-09-16, which broke every THD run whose packed lengths are not
    # multiples of 64 regardless of backend -- e.g. the 128-GPU THD-4K
    # benchmark on gdn_gdr_backend='cudnn' died in get_batch on lengths
    # [4095, 1] (job 760373, zero iterations).
    if gdn_gdr_backend == "internal":
        _validate_internal_gdr_64_aligned_packed_seq_params(packed_seq_params)

    cp_group = resolve_cp_group(get_context_parallel_group(), packed_seq_params)
    packed_seq_params.cp_group = cp_group
    prebuild_thd_cp_partition_routes(packed_seq_params, cp_group)
    return packed_seq_params
