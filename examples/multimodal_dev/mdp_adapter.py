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
            (
                sample_id,
                ordinal,
                t,
                h,
                w,
                payload_row_start,
                sample_padded_start,
                sample_padded_len,
            ) = (int(v) for v in row)
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
                    sample_padded_start=sample_padded_start,
                    sample_padded_len=sample_padded_len,
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
        """Same factory as the non-MDP path (models/qwen35_vl/model.py).

        ``pg_collection`` MUST be threaded through. It used to be discarded,
        which made TransformerBlock fall back to ``use_mpu_process_groups()`` --
        i.e. the vision attention would ring over the DECODER's context-parallel
        group. That is invisible at ``encoder_cp == cp``, where the logical
        worker's ranks and the decoder CP group are the same set.
        """
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
            pg_collection=pg_collection,
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
        grid_thw = torch.tensor(
            [segment.grid_thw for segment in layout.segments], dtype=torch.long
        )
        # Under encoder CP the runtime delivers this rank's zigzag shard of the
        # chunk (1/e of the rows), not the whole chunk; the encoder must not
        # shard it a second time. At encoder_cp=1 the shard IS the chunk and the
        # flag is inert.
        return encoder(payload, grid_thw, pixels_are_sharded=True)


def build_mdp_adapter(args, language_config) -> Qwen35VLMdpAdapter:
    """Adapter factory used by the pretrain entry point."""
    return Qwen35VLMdpAdapter(out_hidden_size=language_config.hidden_size)
