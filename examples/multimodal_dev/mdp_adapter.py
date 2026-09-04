# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL model adapter for MDP.

The adapter is everything model-specific MDP core needs: native batch
collation into a :class:`CapturedMicrobatch`, an integer LPT cost, the shared
vision-encoder factory, and a chunk-oblivious ``encode``. It lives in
``examples/multimodal_dev`` because ``megatron/core/mdp`` must not import
model packages.
"""

from types import MappingProxyType
from typing import Iterator, Optional

import torch

from examples.multimodal_dev.models.qwen35_vl.configuration import VISION_KWARGS
from examples.multimodal_dev.models.qwen35_vl.specs import get_qwen35_vl_vision_spec
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem


class Qwen35VLMdpAdapter:
    """MdpModelAdapter implementation for Qwen3.5-VL.

    Args:
        out_hidden_size: Language decoder hidden size (patch-merger output).
        vision_kwargs: Optional override of the Qwen3.5-VL vision kwargs.
    """

    def __init__(self, out_hidden_size: int, vision_kwargs: Optional[dict] = None):
        self._vision_kwargs = dict(vision_kwargs or VISION_KWARGS)
        self._vision_kwargs["out_hidden_size"] = out_hidden_size
        self.spatial_merge_size = self._vision_kwargs["spatial_merge_size"]
        self.payload_width = (
            self._vision_kwargs["in_channels"]
            * self._vision_kwargs["temporal_patch_size"]
            * self._vision_kwargs["patch_size"] ** 2
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def get_batch(self, data_iterator: Iterator) -> Optional[CapturedMicrobatch]:
        """One microbatch through the native THD collation path.

        Requires the vision sidecar (``--mdp-enable`` makes the collator emit
        it), from which the per-item records and decoder positions are cut.
        The pixel payload and the sidecar keys are removed from the replayed
        model payload: pixels never enter the decoder.
        """
        from examples.multimodal_dev.forward_step import get_batch

        batch = get_batch(data_iterator)
        if batch is None:
            return None
        if "vision_item_meta" not in batch:
            raise RuntimeError(
                "MDP adapter needs the vision sidecar; the collator must run with "
                "with_vision_sidecar=True (set by --mdp-enable)."
            )
        meta = batch.pop("vision_item_meta")
        positions = batch.pop("vision_decoder_positions")
        pixels = batch.pop("pixel_values", None)
        merge = self.spatial_merge_size

        items = []
        position_cursor = 0
        # One D2H transfer for the whole positions tensor; slicing the CPU
        # copy per item avoids a device sync per vision item.
        positions_cpu = positions.cpu().tolist()
        for row in meta.cpu().tolist():
            sample_id, ordinal, t, h, w, payload_row_start = (int(v) for v in row)
            output_rows = t * (h // merge) * (w // merge)
            decoder_positions = tuple(
                positions_cpu[position_cursor : position_cursor + output_rows]
            )
            position_cursor += output_rows
            items.append(
                CapturedVisionItem(
                    sample_id=sample_id,
                    image_ordinal=ordinal,
                    grid_thw=(t, h, w),
                    payload_row_start=payload_row_start,
                    payload_rows=t * h * w,
                    decoder_positions=decoder_positions,
                )
            )
        if position_cursor != positions.numel():
            raise RuntimeError(
                f"vision sidecar mismatch: consumed {position_cursor} decoder positions "
                f"of {positions.numel()}"
            )

        packed_seq_params = batch.pop("packed_seq_params", None)
        if pixels is not None and pixels.shape[0] == 0:
            pixels = None
        if pixels is not None and pixels.dtype == torch.float32:
            pixels = pixels.bfloat16()
        return CapturedMicrobatch(
            decoder_packed_seq_params=packed_seq_params,
            vision_items=tuple(items),
            flat_pixel_payload=pixels,
            model_payload=MappingProxyType(batch),
        )

    # ------------------------------------------------------------------
    # Planning cost
    # ------------------------------------------------------------------

    def estimate_cost(self, item: CapturedVisionItem) -> int:
        """Patch rows as the LPT ordering cost; never sizes any buffer."""
        return item.payload_rows

    # ------------------------------------------------------------------
    # Encoder factory and forward
    # ------------------------------------------------------------------

    def build_encoder(self, model_config, *, pg_collection) -> torch.nn.Module:
        """Same factory as the non-MDP path (models/qwen35_vl/model.py)."""
        del pg_collection  # the encoder has no model parallelism in v1
        kwargs = self._vision_kwargs
        return Qwen35VLVisionEncoder(
            config=model_config,
            transformer_layer_spec=get_qwen35_vl_vision_spec(),
            in_channels=kwargs["in_channels"],
            patch_size=kwargs["patch_size"],
            temporal_patch_size=kwargs["temporal_patch_size"],
            spatial_merge_size=kwargs["spatial_merge_size"],
            out_hidden_size=kwargs["out_hidden_size"],
            max_num_positions=kwargs["max_num_positions"],
        )

    def encode(self, encoder: torch.nn.Module, payload: torch.Tensor, layout) -> torch.Tensor:
        """Encoder forward for one (already rebased) chunk sub-layout.

        The encoder builds its vision-only THD ``PackedSeqParams`` internally
        from ``grid_thw`` (one sub-sequence per temporal frame); the decoder
        ``PackedSeqParams`` is never read here.
        """
        # A CPU tensor, deliberately: with the grid cache enabled the encoder
        # consumes grid_thw exclusively as Python lists (tolist), so a device
        # tensor here cost a blocking pageable H2D on the busy compute stream
        # (~2.4 ms/iter measured) followed by D2H readbacks inside the
        # encoder. The encoder moves it to the device itself on the uncached
        # (QWEN35_VL_GRID_CACHE=0) fallback paths that do tensor math on it.
        grids = [segment.grid_thw for segment in layout.segments]

        # TE's forward-pass assert_dim_for_fp8_exec requires the packed leading
        # dim (payload.shape[0]) divisible by 8, but the backward wgrad GEMM
        # (cublaslt_gemm.cu CanonicalizeGemmInput) is stricter: it requires the
        # leading dim divisible by 16 ("Caller must pad", observed empirically
        # via a training-time crash, not just the forward-only assert). MXFP8 is
        # stricter still: TE's quantizer.cpp create_tensor asserts
        # flat_first_dim % MXFP8_BLOCK_SIZE == 0 (MXFP8_BLOCK_SIZE=32, the
        # native OCP microscaling block granularity) — observed empirically as
        # a training-time crash with 16-aligned padding ("MXFP8 requires
        # tensor dims that are divisible by 32"). get_fp8_align_size encodes
        # exactly that split (32 for mxfp8, 16 for every other recipe) and is
        # what the decoder side pads to as well (forward_step.py's
        # quantized_row_alignment). Its 16 fallback would be too small for a
        # block-scaled 'custom' quantizer, but ENCODER_COMPATIBLE_FP8_RECIPES
        # rejects 'custom' before any encoder is built.
        #
        # Every real grid entry's row count t*h*w is a multiple of
        # spatial_merge_size**2 (h and w are always even, being patch-grid
        # units of an even-merge image), so payload.shape[0] itself is already
        # a multiple of merge**2 (4 for this model). Only one extra synthetic
        # all-zero grid item is ever needed to reach the next multiple of
        # ALIGN: pad by at most (ALIGN - merge**2) rows, never more. This
        # padding is entirely local to this call — MDP core's
        # `chunk.total_payload_rows` and everything in bridge.py/storage.py
        # are unaware of it: we strip the padding item's output rows before
        # returning, so the returned tensor has exactly the same shape core
        # already expects (sum of each real segment's merged output rows).
        #
        # The strip is exact in shape and in gradient — the synthetic item is
        # its own THD sub-sequence so attention never mixes it with real
        # tokens, the merger is per-token, and the stripped rows never reach
        # the loss, so they contribute no gradient. It is not numerically
        # identical to an unpadded run, though: the synthetic rows are all-zero
        # only in the input payload (the encoder's bias terms make them
        # non-zero downstream) and they are quantized in the same tensors as
        # the real rows, so they enter the per-tensor amax under 'tensorwise'
        # and share the E8M0 scale of the final 32-row block under columnwise
        # MXFP8.
        encoder_config = self._unwrap_config(encoder)
        fp8_enabled = getattr(encoder_config, "fp8", None) is not None
        fp8_recipe = getattr(encoder_config, "fp8_recipe", None)
        pad_grid = None
        ALIGN = get_fp8_align_size(fp8_recipe)
        if fp8_enabled and payload.shape[0] % ALIGN != 0:
            merge = self.spatial_merge_size
            pad_rows = (ALIGN - payload.shape[0] % ALIGN) % ALIGN
            unit = merge * merge
            if pad_rows % unit != 0:
                raise RuntimeError(
                    "MDP encoder FP8 padding: payload.shape[0]="
                    f"{payload.shape[0]} needs {pad_rows} more rows to reach a "
                    f"multiple of {ALIGN}, but {pad_rows} is not a multiple of "
                    f"spatial_merge_size**2 ({unit}); the 'every real grid "
                    "row-count is a multiple of merge**2' assumption this padding "
                    "relies on does not hold for this vision config — pad_grid "
                    "needs a general solution, not just a single (k, merge, merge) "
                    "item."
                )
            pad_grid = (pad_rows // unit, merge, merge)
            payload = torch.cat(
                [payload, payload.new_zeros(pad_rows, payload.shape[1])], dim=0
            )
            grids = grids + [pad_grid]

        grid_thw = torch.tensor(grids, dtype=torch.long)
        output = encoder(payload, grid_thw)
        if pad_grid is not None:
            merge = self.spatial_merge_size
            pad_output_rows = pad_grid[0] * (pad_grid[1] // merge) * (pad_grid[2] // merge)
            if pad_output_rows:
                output = output[:-pad_output_rows]
        return output

    @staticmethod
    def _unwrap_config(encoder: torch.nn.Module):
        """The effective vision config from the outermost wrapper that has one.

        ``build_encoder_domain`` hands ``DistributedDataParallel`` the effective
        vision config -- the one ``apply_encoder_fp8_config`` wrote ``fp8`` /
        ``fp8_recipe`` onto -- so DDP's own ``.config`` is exactly the object
        wanted here, and the walk stops there. The fallback through ``.module``
        only matters for a bare encoder handed in without a wrapper (tests).
        """
        module = encoder
        while not hasattr(module, "config") and hasattr(module, "module"):
            module = module.module
        return getattr(module, "config", None)


def build_mdp_adapter(args, language_config) -> Qwen35VLMdpAdapter:
    """Adapter factory used by the pretrain entry point."""
    return Qwen35VLMdpAdapter(out_hidden_size=language_config.hidden_size)
