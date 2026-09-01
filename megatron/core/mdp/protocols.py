# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP model-adapter protocol and the carrier types between adapter and core.

``megatron/core/mdp`` must not import examples or model-specific packages; model
behavior is injected through :class:`MdpModelAdapter`. Pixel slicing and
descriptor assembly are pure data transformations and belong in core (the
window), not in the adapter.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Protocol

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.nn import Module

    from megatron.core.mdp.plan import EncoderThdLayout
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass(frozen=True)
class CapturedVisionItem:
    """One vision item as captured from the model's native collation.

    ``payload_row_start/payload_rows`` index rows of the microbatch's
    ``flat_pixel_payload``. ``decoder_positions`` are absolute token offsets in the
    current decoder microbatch THD ``[1, T_dec]`` (following the physical layout in
    ``cu_seqlens_q_padded`` when alignment padding exists); the per-row tuple stays
    endpoint-local and never enters descriptors, routes, or the plan — only the
    derived span below does.

    ``sample_padded_start/sample_padded_len`` are the enclosing sample's span in
    ``cu_seqlens_q_padded``. Unlike ``decoder_positions`` these DO enter the
    descriptor, because the decoder-CP owner of a row is a function of the row's
    offset inside its sample and that sample's padded length, and every planning
    group member must derive the same split from the broadcast records alone
    (see :mod:`megatron.core.mdp.cp_partition`). One item's slots are contiguous,
    so its start offset plus ``output_rows`` is the whole span — the per-row
    ``decoder_positions`` tuple stays out of the wire format.
    """

    sample_id: int
    image_ordinal: int
    grid_thw: tuple
    payload_row_start: int
    payload_rows: int
    decoder_positions: tuple
    sample_padded_start: int = 0
    sample_padded_len: int = 0


@dataclass(frozen=True)
class CapturedMicrobatch:
    """The single carrier type between the adapter and the iteration window.

    - ``vision_items`` is ordered by ``(sample_id, image_ordinal)``.
    - ``model_payload`` is an opaque replay payload owned by the adapter. Core
      never interprets its contents; it must be an immutable mapping exclusively
      referenced by the window and must not be mutated after capture.
    - ``decoder_packed_seq_params.qkv_format`` must be ``"thd"``; it is used only
      for decoder replay and never passed to the vision encoder.
    """

    decoder_packed_seq_params: "PackedSeqParams"
    vision_items: tuple
    flat_pixel_payload: Optional["Tensor"]
    model_payload: Mapping[str, Any]


@dataclass(frozen=True)
class VisionDescriptor:
    """The planner's only input type; assembled by the window, broadcast as
    fixed-width int64 records.

    Invariants: ``global_item_id`` is stable and unique within its outer-DP
    planning group; ``estimated_cost_units`` is a non-negative integer used only
    for ordering and never sizes a buffer; for spatial merge size ``m``,
    ``payload_rows == t*h*w`` and ``output_rows == t*(h/m)*(w/m)``.

    ``owner_worker_id`` is the logical worker holding this item's pixels at
    dispatch time: ``microbatch_id % num_workers``.

    ``sample_padded_start``, ``sample_padded_len`` and ``decoder_offset_in_sample``
    locate the item's contiguous run of decoder rows inside its packed sample.
    They are the only position data on the wire, and they are here because the
    decoder-CP row owner is a function of exactly these three integers plus
    ``output_rows`` (:mod:`megatron.core.mdp.cp_partition`); every planning-group
    member must reach the same split from the broadcast records alone. At CP=1
    they are inert: the split is the identity.
    """

    global_item_id: int
    sample_id: int
    image_ordinal: int
    owner_dp_lane: int
    microbatch_id: int
    estimated_cost_units: int
    payload_rows: int
    output_rows: int
    grid_thw: tuple
    owner_worker_id: int
    sample_padded_start: int = 0
    sample_padded_len: int = 0
    decoder_offset_in_sample: int = 0


class MdpModelAdapter(Protocol):
    """Everything model-specific MDP core needs, and nothing more."""

    payload_width: int
    spatial_merge_size: int

    def get_batch(self, data_iterator: Iterator) -> Optional[CapturedMicrobatch]:
        """Reuse native model collation for one microbatch."""
        ...

    def estimate_cost(self, item: CapturedVisionItem) -> int:
        """Integer ordering cost for LPT; must never size any buffer."""
        ...

    def build_encoder(
        self, model_config: "TransformerConfig", *, pg_collection: "ProcessGroupCollection"
    ) -> "Module":
        """Build the vision encoder through the same factory as the non-MDP path."""
        ...

    def encode(
        self, encoder: "Module", payload: "Tensor", layout: "EncoderThdLayout"
    ) -> "Tensor":
        """Run encoder forward on one already-rebased chunk sub-layout.

        The adapter reads the ordered ``grid_thw`` from ``layout.segments`` and
        constructs a vision-only ``PackedSeqParams(qkv_format="thd")``; it must
        never read or reuse the decoder ``PackedSeqParams``, and it is unaware
        that chunking exists. During training the return value stays
        graph-connected; only the view passed to the EMBEDDING bridge may be
        detached.
        """
        ...
