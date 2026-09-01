# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDP model-adapter protocol and the carrier types between adapter and core.

``megatron/core/mdp`` must not import examples or model-specific packages; model
behavior is injected through :class:`MdpModelAdapter`. Pixel slicing and
descriptor assembly are pure data transformations and belong in core (the
window), not in the adapter.

Core is deliberately blind to vision geometry. It moves rows between ranks and
never interprets what a row means, so every item is described to core by four
integers plus an opaque ``shape_key``:

* ``payload_rows``  -- rows of the flat pixel payload (PIXEL transport size);
* ``output_rows``   -- rows the encoder emits (EMBEDDING/GRADIENT transport size
  and endpoint leaf size), computed by the adapter, never by core;
* ``estimated_cost_units`` -- an integer LPT ordering key;
* ``shape_key``     -- a fixed-width tuple of non-negative integers that the
  adapter alone interprets. Core ships it over the wire and folds it into the
  plan digest so a producer can rebuild encoder metadata for items whose pixels
  were materialized on another rank.

For a patch-grid ViT the shape key is ``(t, h, w)``; nothing in core requires
that reading.
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
    ``flat_pixel_payload``. ``output_rows`` is the adapter's own count of the rows
    its encoder will emit for this item; core uses it verbatim to size transports
    and the endpoint leaf. ``shape_key`` is opaque to core (see the module
    docstring). ``decoder_positions`` are absolute token offsets in the current
    decoder microbatch THD ``[1, T_dec]`` (following the physical layout in
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
    shape_key: tuple
    payload_row_start: int
    payload_rows: int
    output_rows: int
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
    for ordering and never sizes a buffer; ``payload_rows`` and ``output_rows``
    are positive; ``shape_key`` holds exactly ``adapter.shape_key_width``
    non-negative integers.

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
    shape_key: tuple
    owner_worker_id: int
    sample_padded_start: int = 0
    sample_padded_len: int = 0
    decoder_offset_in_sample: int = 0


class MdpModelAdapter(Protocol):
    """Everything model-specific MDP core needs, and nothing more."""

    #: Trailing width of one flat pixel payload row.
    payload_width: int
    #: Number of int64 slots one ``shape_key`` occupies on the descriptor wire.
    shape_key_width: int
    #: Hidden width of one encoder output row; sizes the EMBEDDING/GRADIENT
    #: buffers and the endpoint leaf. Usually the decoder hidden size, because
    #: the modality projector is the last stage of the encoder.
    output_hidden_size: int

    def get_batch(self, data_iterator: Iterator) -> Optional[CapturedMicrobatch]:
        """Reuse native model collation for one microbatch."""
        ...

    def estimate_cost(self, item: CapturedVisionItem) -> int:
        """Integer ordering cost for LPT; must never size any buffer."""
        ...

    def validate_item(self, item: CapturedVisionItem) -> None:
        """Model-specific geometry check for one captured item.

        Called by the window before any distributed transport, once per item, on
        every worker. Core has already checked the model-independent invariants
        (ordering, positive row counts, payload interval inside the flat
        payload, ``len(shape_key) == shape_key_width``); this hook owns whatever
        relates ``shape_key`` to ``payload_rows``/``output_rows`` for this model.
        Raise :class:`~megatron.core.mdp.errors.MdpConfigurationError` on
        violation.
        """
        ...

    def build_encoder(
        self, model_config: "TransformerConfig", *, pg_collection: "ProcessGroupCollection"
    ) -> "Module":
        """Build the vision encoder through the same factory as the non-MDP path."""
        ...

    def encode(self, encoder: "Module", payload: "Tensor", layout: "EncoderThdLayout") -> "Tensor":
        """Run encoder forward on one already-rebased chunk sub-layout.

        The adapter reads the ordered ``shape_key`` from ``layout.segments`` and
        constructs a vision-only ``PackedSeqParams(qkv_format="thd")``; it must
        never read or reuse the decoder ``PackedSeqParams``, and it is unaware
        that chunking exists. The return value must have exactly
        ``layout.total_output_rows`` rows. In the default training mode the
        return value stays graph-connected; in complete-encoder recompute mode
        core first calls it under ``no_grad`` in P2 and then calls it again with
        gradients enabled in P5. Only detached views cross the EMBEDDING bridge.
        """
        ...
