# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDP encoder autograd: retained-graph and complete-replay forward handles.

The default path retains graph-connected P2 outputs and uses native MCore
``TransformerConfig`` checkpointing when requested.  The Design-Doc ``all``
mode instead retains pixels, layouts, output metadata, and RNG recipes; P5
replays the complete model one chunk at a time before backward.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.errors import MdpStateError
from megatron.core.mdp.plan import (
    EncoderThdLayout,
    EncoderThdSegment,
    frame_cu_seqlens,
    frame_lengths,
)
from megatron.core.packed_seq_params import PackedSeqParams


def capture_encoder_rng_state() -> tuple:
    """Capture CPU, CUDA, and model-parallel tracker RNG state for replay."""
    from megatron.core.tensor_parallel.random import _get_all_rng_states

    return _get_all_rng_states()


@dataclass(frozen=True)
class EncoderOutputMetadata:
    """Immutable P2 output contract needed to validate and route P5 gradients."""

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_tensor(cls, output: Tensor) -> "EncoderOutputMetadata":
        """Capture metadata without retaining the output or its autograd graph."""
        return cls(shape=output.shape, dtype=output.dtype, device=output.device)


def build_encoder_packed_seq_params(
    segments: Sequence[EncoderThdSegment], *, allocator: MdpBufferAllocator, device
) -> PackedSeqParams:
    """Vision-only ``PackedSeqParams(qkv_format="thd")`` for one chunk sub-layout.

    Frame boundaries are derived from each segment's ``grid_thw``; this metadata
    is unrelated to — and must never be mixed with — decoder sample ``cu_seqlens``.
    """
    cu = frame_cu_seqlens(segments)
    lengths = frame_lengths(segments)
    max_seqlen = max(lengths) if lengths else 0
    cu_tensor = allocator.acquire(
        rows=len(cu), width=0, dtype=torch.int32, device=device, tag="encoder_cu_seqlens"
    )
    cu_tensor.copy_(torch.tensor(cu, dtype=torch.int32))
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_tensor,
        cu_seqlens_kv=cu_tensor,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


@dataclass
class EncoderForwardHandle:
    """The producer's retained P2 state: graph-connected chunk outputs.

    Chunk outputs stay in a list and are never concatenated: ``torch.cat`` would
    allocate a second full copy while all chunk outputs are alive, cancelling
    exactly the memory ``encoder_max_payload_rows`` was meant to save.
    """

    iteration: int
    producer_worker_id: int
    chunk_outputs: tuple[Tensor, ...]
    chunk_layouts: tuple[EncoderThdLayout, ...]

    def __post_init__(self):
        if len(self.chunk_outputs) != len(self.chunk_layouts):
            raise MdpStateError(
                "MDP: forward handle violates: one layout per chunk output "
                f"({len(self.chunk_outputs)} outputs, {len(self.chunk_layouts)} layouts)."
            )
        for index, (output, layout) in enumerate(
            zip(self.chunk_outputs, self.chunk_layouts)
        ):
            if output.shape[0] != layout.total_output_rows:
                raise MdpStateError(
                    f"MDP: chunk {index} violates: output rows == "
                    f"layout.total_output_rows ({output.shape[0]} != "
                    f"{layout.total_output_rows})."
                )
        self._backward_done = False
        self._released = False

    def detached_outputs(self) -> tuple[Tensor, ...]:
        """Detached views for the EMBEDDING bridge; cross-rank communication
        never joins the autograd graph."""
        return tuple(output.detach() for output in self.chunk_outputs)

    def backward(self, chunk_grads: Sequence[Tensor]) -> None:
        """One multi-tensor backward over all chunk outputs."""
        if self._backward_done:
            raise MdpStateError("MDP: forward handle violates: exactly one backward.")
        if len(chunk_grads) != len(self.chunk_outputs):
            raise MdpStateError(
                f"MDP: backward violates: one gradient per chunk output "
                f"({len(chunk_grads)} grads, {len(self.chunk_outputs)} outputs)."
            )
        for index, (output, grad) in enumerate(zip(self.chunk_outputs, chunk_grads)):
            if grad.shape != output.shape or grad.dtype != output.dtype:
                raise MdpStateError(
                    f"MDP: chunk {index} gradient violates: shape and dtype match the "
                    f"output ({tuple(grad.shape)}/{grad.dtype} vs "
                    f"{tuple(output.shape)}/{output.dtype})."
                )
            if grad.device != output.device:
                raise MdpStateError(
                    f"MDP: chunk {index} gradient violates: device matches the output."
                )
        # MCore checkpoint nodes replay selected/full layers here as configured;
        # retain_graph must not be used.
        torch.autograd.backward(self.chunk_outputs, tuple(chunk_grads))
        self._backward_done = True

    def release(self) -> None:
        """Valid only after backward; drops outputs, graphs, and activations."""
        if not self._backward_done:
            raise MdpStateError(
                "MDP: forward handle violates: release only after backward (training)."
            )
        self._release_now()

    def release_forward_only(self) -> None:
        """Evaluation-path release: no graph exists, no backward will come."""
        self._release_now()

    def _release_now(self) -> None:
        self.chunk_outputs = ()
        self.chunk_layouts = ()
        self._released = True

    @property
    def consumed(self) -> bool:
        """Whether this handle has been fully consumed (lifecycle invariant)."""
        return self._released

    def output_dtype(self, chunk_index: int) -> torch.dtype:
        """Boundary dtype used to allocate the matching routed gradient."""
        return self.chunk_outputs[chunk_index].dtype


@dataclass
class EncoderWholeRecomputeHandle:
    """The producer's P2 recipe for complete encoder replay in P5.

    Unlike :class:`EncoderForwardHandle`, this handle owns no graph-connected
    output.  It retains exactly the valid packed-pixel views, immutable chunk
    layouts, P2 output metadata, and the RNG state observed before each chunk.
    P5 replays and backpropagates one chunk at a time so
    ``encoder_max_payload_rows`` bounds the rebuilt graph as well as forward
    workspace. It does not bound the pixels retained across P4 or the complete
    set of routed output gradients materialized before replay. Processed pixel
    and gradient references are dropped after each chunk backward, reducing
    their remaining lifetime but not the initial P5 peak.
    """

    iteration: int
    producer_worker_id: int
    chunk_payloads: list[Tensor | None]
    chunk_layouts: tuple[EncoderThdLayout, ...]
    output_metadata: tuple[EncoderOutputMetadata, ...]
    chunk_rng_states: tuple[tuple, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.chunk_payloads),
            len(self.chunk_layouts),
            len(self.output_metadata),
            len(self.chunk_rng_states),
        }
        if len(lengths) != 1:
            raise MdpStateError(
                "MDP: whole-recompute handle violates: one payload, layout, output "
                "metadata record, and RNG state per chunk."
            )
        for index, (payload, layout, metadata) in enumerate(
            zip(self.chunk_payloads, self.chunk_layouts, self.output_metadata)
        ):
            if payload.shape[0] != layout.total_payload_rows:
                raise MdpStateError(
                    f"MDP: whole-recompute chunk {index} violates: payload rows == "
                    f"layout.total_payload_rows ({payload.shape[0]} != "
                    f"{layout.total_payload_rows})."
                )
            if metadata.shape[0] != layout.total_output_rows:
                raise MdpStateError(
                    f"MDP: whole-recompute chunk {index} violates: output rows == "
                    f"layout.total_output_rows ({metadata.shape[0]} != "
                    f"{layout.total_output_rows})."
                )
        self._backward_done = False
        self._released = False

    def output_dtype(self, chunk_index: int) -> torch.dtype:
        """Boundary dtype used to allocate the matching routed gradient."""
        return self.output_metadata[chunk_index].dtype

    def backward(
        self,
        chunk_grads: list[Tensor | None],
        *,
        encoder: torch.nn.Module,
        encode: Callable[[torch.nn.Module, Tensor, EncoderThdLayout], Tensor],
    ) -> None:
        """Replay each chunk, backpropagate it, and consume its pixels and gradient."""
        if self._backward_done:
            raise MdpStateError(
                "MDP: whole-recompute handle violates: exactly one backward."
            )
        if len(chunk_grads) != len(self.chunk_layouts):
            raise MdpStateError(
                "MDP: whole-recompute backward violates: one gradient per chunk "
                f"({len(chunk_grads)} grads, {len(self.chunk_layouts)} chunks)."
            )
        for index, (grad, metadata) in enumerate(
            zip(chunk_grads, self.output_metadata)
        ):
            if grad is None:
                raise MdpStateError(
                    f"MDP: whole-recompute chunk {index} gradient was already consumed."
                )
            if grad.shape != metadata.shape or grad.dtype != metadata.dtype:
                raise MdpStateError(
                    f"MDP: whole-recompute chunk {index} gradient violates: shape and "
                    f"dtype match the P2 output ({tuple(grad.shape)}/{grad.dtype} vs "
                    f"{tuple(metadata.shape)}/{metadata.dtype})."
                )
            if grad.device != metadata.device:
                raise MdpStateError(
                    f"MDP: whole-recompute chunk {index} gradient violates: device "
                    "matches the P2 output."
                )

        from megatron.core.tensor_parallel.random import _fork_rng, _set_all_rng_states

        # The outer fork restores the RNG state visible at P5 entry.  Per-chunk
        # states are restored independently because a backward kernel is allowed
        # to consume RNG between consecutive replayed forwards.
        with _fork_rng():
            for index, (layout, rng_state, metadata) in enumerate(
                zip(
                    self.chunk_layouts,
                    self.chunk_rng_states,
                    self.output_metadata,
                )
            ):
                payload = self.chunk_payloads[index]
                grad = chunk_grads[index]
                if payload is None or grad is None:
                    raise MdpStateError(
                        f"MDP: whole-recompute chunk {index} replay inputs were "
                        "already consumed."
                    )
                _set_all_rng_states(*rng_state)
                with torch.enable_grad():
                    output = encode(encoder, payload, layout)
                if (
                    output.shape != metadata.shape
                    or output.dtype != metadata.dtype
                    or output.device != metadata.device
                ):
                    raise MdpStateError(
                        f"MDP: whole-recompute chunk {index} violates: replay output "
                        "metadata matches P2."
                    )
                if output.shape[0] and (
                    not output.requires_grad or output.grad_fn is None
                ):
                    raise MdpStateError(
                        f"MDP: whole-recompute chunk {index} output is not "
                        "graph-connected during P5 replay."
                    )
                torch.autograd.backward(output, grad)
                del output
                self.chunk_payloads[index] = None
                chunk_grads[index] = None
                del payload, grad
        self._backward_done = True

    def release(self) -> None:
        """Drop pixels and replay metadata after the one required backward."""
        if not self._backward_done:
            raise MdpStateError(
                "MDP: whole-recompute handle violates: release only after backward."
            )
        self.chunk_payloads = []
        self.chunk_layouts = ()
        self.output_metadata = ()
        self.chunk_rng_states = ()
        self._released = True

    @property
    def consumed(self) -> bool:
        """Whether this handle has been fully consumed (lifecycle invariant)."""
        return self._released
