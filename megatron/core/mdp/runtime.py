# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP runtime: the P0-P6 phase machine.

Three observable states::

    begin_iteration:       EMPTY -> DECODER_READY      # runs P1-P3
    mark_decoder_complete: DECODER_READY -> DECODER_DONE
    end_iteration:         DECODER_DONE -> EMPTY       # P5 for training, cleanup for eval

Every other transition raises :class:`MdpStateError`. All planning-group
members execute every group-local operation and every required WORLD
collective; text-only and empty-worker ranks contribute empty metadata,
empty ledgers, zero local encoder work, and zero encoder gradients.
"""

import logging
import threading
import time
from enum import Enum, auto
from typing import Iterator, Optional, Sequence, Union

import torch

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.bridge import (
    BridgeBufferKey,
    BridgePhase,
    BridgeTensorSpec,
    ModalityBridge,
)
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import EncoderDomain, finalize_encoder_grads
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError
from megatron.core.mdp.groups import MdpProcessGroups, broadcast_descriptors
from megatron.core.mdp.packing import GreedySampleStream, decoder_sample_length
from megatron.core.mdp.observability import (
    MdpIterationMetrics,
    nvtx_phase,
    worker_loads_from_plan,
)
from megatron.core.mdp.plan import MdpBatchPlan, split_encoder_layout
from megatron.core.mdp.planner import MdpPlanner, assert_consistent_plan
from megatron.core.mdp.protocols import MdpModelAdapter
from megatron.core.mdp.rank_mapping import MdpRankMap, MdpRankView
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import MdpIterationWindow

logger = logging.getLogger(__name__)


class MdpRuntimeState(Enum):
    """Observable runtime states; see module docstring for transitions."""

    EMPTY = auto()
    DECODER_READY = auto()
    DECODER_DONE = auto()


class MdpRuntime:
    """Owns the per-rank MDP iteration state and drives the phase machine."""

    def __init__(
        self,
        *,
        config: MdpConfig,
        rank_map: MdpRankMap,
        rank_view: MdpRankView,
        process_groups: MdpProcessGroups,
        adapter: MdpModelAdapter,
        encoder_domain: EncoderDomain,
        planner: MdpPlanner,
        bridge: ModalityBridge,
        storage: MdpEmbeddingStorage,
        allocator: MdpBufferAllocator,
        hidden_size: int,
        params_dtype: torch.dtype,
        num_vpp_chunks: int = 1,
        device: Optional[torch.device] = None,
        greedy_token_budget: Optional[int] = None,
        greedy_max_num_seqs: Optional[int] = None,
        greedy_row_alignment: int = 1,
    ) -> None:
        self.config = config
        self.rank_map = rank_map
        self.rank_view = rank_view
        self.process_groups = process_groups
        self.adapter = adapter
        self.encoder_domain = encoder_domain
        self.planner = planner
        self.bridge = bridge
        self.storage = storage
        self.allocator = allocator
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.num_vpp_chunks = num_vpp_chunks
        self.device = device or torch.device("cuda", torch.cuda.current_device())

        self._state = MdpRuntimeState.EMPTY
        self._iteration = 0
        self._forward_only = False
        self._window: Optional[MdpIterationWindow] = None
        self._plan: Optional[MdpBatchPlan] = None
        self._iter_specs: dict = {}
        self._iter_ledgers: dict = {}
        self._handle: Optional[EncoderForwardHandle] = None
        self._eval_outputs: Sequence = ()
        self._chunk_layouts: Sequence = ()
        self._chunk_of_item: dict = {}
        self._captured_num_tokens: Optional[torch.Tensor] = None
        self._token_capture_count = 0
        self._token_consumed = False
        self._plan_build_ms = 0.0
        self._encoder_forward_ms = 0.0
        self._decoder_schedule_ms = 0.0
        self._decoder_start = 0.0
        self._last_metrics: Optional[MdpIterationMetrics] = None
        # Window-capture overlap: one in-flight prefetch keyed by the data
        # iterator's identity, so an interleaved eval (different iterator)
        # leaves a pending train prefetch untouched. The prefetch thread runs
        # capture under a dedicated side stream so its H2D traffic never
        # enters (or blocks) the main compute stream; the consumer orders
        # itself via the recorded event.
        self._prefetch_key = None
        self._prefetch_thread = None
        self._prefetch_box: Optional[dict] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        # Greedy token-budget packing (--mdp-greedy-packing). One
        # GreedySampleStream per underlying data iterator, so train and eval keep
        # independent sample buffers: an eval window must never consume (or be
        # consumed by) the training stream's leftovers. Keyed by iterator
        # identity, which the training loop keeps stable for the whole run.
        self._greedy_token_budget = greedy_token_budget
        self._greedy_max_num_seqs = greedy_max_num_seqs
        self._greedy_row_alignment = greedy_row_alignment
        self._greedy_streams: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> MdpRuntimeState:
        """Current phase-machine state."""
        return self._state

    @property
    def iteration(self) -> int:
        """Zero-based iteration counter."""
        return self._iteration

    def begin_iteration(
        self,
        data_iterators: Union[Iterator, Sequence[Iterator]],
        *,
        num_microbatches: int,
        forward_only: bool,
    ) -> Sequence[Iterator]:
        """P0-P3: capture, plan, dispatch pixels, encode, route embeddings.

        Returns the replay iterators the native schedule consumes. The
        ``forward_only`` flag is recorded once here; ``end_iteration`` uses it
        so inconsistent values cannot be passed at two call sites.
        """
        self._require_state(MdpRuntimeState.EMPTY, "begin_iteration")
        self._forward_only = forward_only

        # P0: clear encoder gradients and iteration state.
        if not forward_only:
            with nvtx_phase("p0_zero_grad"):
                self.encoder_domain.encoder_ddp.zero_grad_buffer()
        self._captured_num_tokens = None
        self._token_capture_count = 0
        self._token_consumed = False

        # P1: window capture, descriptor broadcast, plan, pixel dispatch.
        plan_start = time.monotonic()
        window, pending_greedy = self._take_prefetched_window(data_iterators, num_microbatches)
        if window is None:
            with nvtx_phase("p1_window_capture"):
                window, pending_greedy = self._capture_window(data_iterators, num_microbatches)
        # This window is now the iteration's; its samples are consumed, not just
        # drained. Anything the prefetch thread fills for the *next* iteration
        # stays uncommitted until that iteration installs it. Committed before
        # the prefetch starts: commit takes the stream's buffer lock, which the
        # prefetch thread holds for the whole capture it is supposed to overlap.
        if pending_greedy is not None:
            stream, drained = pending_greedy
            stream.commit(drained)
        # The data iterator is fully consumed for this iteration; the next
        # window can be captured concurrently with everything that follows.
        if self.config.overlap_window_capture and not forward_only:
            self._start_window_prefetch(data_iterators, num_microbatches)
        self._window = window
        local_flags = tuple(record.text_only for record in window.records())
        with nvtx_phase("p1_broadcast_descriptors"):
            descriptors, text_only_flags = broadcast_descriptors(
                window.descriptors(),
                planning_group=self.process_groups.planning_group,
                endpoint_rank=self.rank_view.endpoint_rank,
                num_microbatches=num_microbatches,
                text_only_flags=local_flags if self.rank_view.lane_id is not None else (),
                device=self.device,
            )
        if local_flags != text_only_flags:
            raise MdpStateError(
                f"MDP: text-only flags diverge between local records {local_flags} and "
                f"the endpoint broadcast {text_only_flags}; group members are not "
                "consuming identical sampler data."
            )
        with nvtx_phase("p1_build_plan"):
            plan = self.planner.build_plan(
                self._iteration, descriptors, list(range(num_microbatches))
            )
            assert_consistent_plan(
                plan,
                planning_group=self.process_groups.planning_group,
                iteration=self._iteration,
                interval=self.config.plan_check_interval,
                debug_payload_check=self.config.debug_plan_payload_check,
            )
        self._plan = plan
        # Specs and ledgers are pure functions of the plan; derive them once
        # per iteration instead of once per phase (EMBEDDING and GRADIENT
        # share identical specs, and each build_ledger re-sorted the routes).
        pixel_specs = self._tensor_specs(plan, pixels=True)
        io_specs = self._tensor_specs(plan, pixels=False)
        self._iter_specs = {BridgePhase.PIXEL: pixel_specs}
        self._iter_ledgers = {}
        for phase in (BridgePhase.PIXEL, BridgePhase.EMBEDDING, BridgePhase.GRADIENT):
            specs = pixel_specs if phase is BridgePhase.PIXEL else io_specs
            self._iter_specs[phase] = specs
            self._iter_ledgers[phase] = self.bridge.build_ledger(
                phase, plan, self.rank_map, specs
            )
        self._plan_build_ms = (time.monotonic() - plan_start) * 1000.0

        # The producer chunk layouts are known from the plan alone; carve the
        # encoder payload buffers first so the PIXEL exchange can deposit
        # every routed item directly at its final payload offset (no per-item
        # intermediate buffer + repack pass).
        my_layout = plan.encoder_layout_for_producer(self.rank_view.my_worker_id)
        chunk_layouts = split_encoder_layout(
            my_layout, max_payload_rows=self.config.encoder_max_payload_rows
        )
        self._chunk_layouts = chunk_layouts if my_layout.segments else ()
        self._chunk_of_item = {}
        chunk_payloads = []
        pixel_dest = {}
        with nvtx_phase("p2_pack_payload"):
            for chunk_index, chunk in enumerate(self._chunk_layouts):
                payload = self.allocator.acquire(
                    rows=plan.capacity_policy.capacity_of(chunk.total_payload_rows),
                    width=self.adapter.payload_width,
                    dtype=self.params_dtype,
                    device=self.device,
                    tag="packed_pixels",
                )
                chunk_payloads.append(payload)
                for segment in chunk.segments:
                    pixel_dest[BridgeBufferKey(segment.global_item_id)] = payload[
                        segment.payload_row_start : segment.payload_row_start
                        + segment.payload_rows
                    ]
                    self._chunk_of_item[segment.global_item_id] = (chunk_index, segment)

        with nvtx_phase("p1_pixel_dispatch"):
            pixel_specs = self._iter_specs[BridgePhase.PIXEL]
            sidecar = window.payload_sidecar()
            pixel_local = {
                BridgeBufferKey(item_id): tensor.to(self.params_dtype)
                for item_id, tensor in sidecar.items()
            }
            pixel_ledger = self._iter_ledgers[BridgePhase.PIXEL]
            # Owners -> producers in one collective; every group member
            # participates (with zero splits when it has nothing to move).
            self.bridge.exchange_all_to_all(
                pixel_ledger,
                pixel_local,
                tensor_specs=pixel_specs,
                group=self.process_groups.planning_group,
                group_ranks=self.rank_view.planning_group_ranks,
                global_rank=self.rank_view.global_rank,
                dtype=self.params_dtype,
                device=self.device,
                dest_views=pixel_dest,
            )
            window.release_pixels()

        # P2: grad-enabled encoder forward per chunk (no_grad for evaluation).
        chunk_outputs = []
        encoder = self.encoder_domain.encoder_ddp
        forward_start = time.monotonic()
        for chunk_index, chunk in enumerate(self._chunk_layouts):
            payload = chunk_payloads[chunk_index]
            payload_valid = payload[: chunk.total_payload_rows]
            if forward_only:
                with torch.no_grad(), nvtx_phase("p2_encoder_forward"):
                    output = self.adapter.encode(encoder, payload_valid, chunk)
            else:
                with nvtx_phase("p2_encoder_forward"):
                    output = self.adapter.encode(encoder, payload_valid, chunk)
                if output.shape[0] and (not output.requires_grad or output.grad_fn is None):
                    raise MdpStateError(
                        "MDP: encoder chunk output is not graph-connected in training; "
                        "adapter.encode must run with gradients enabled."
                    )
            chunk_outputs.append(output)

        self._encoder_forward_ms = (time.monotonic() - forward_start) * 1000.0

        if self._chunk_layouts and not forward_only:
            self._handle = EncoderForwardHandle(
                iteration=self._iteration,
                producer_worker_id=self.rank_view.my_worker_id,
                chunk_outputs=tuple(chunk_outputs),
                chunk_layouts=tuple(self._chunk_layouts),
            )
            detached = self._handle.detached_outputs()
        else:
            self._handle = None
            self._eval_outputs = tuple(chunk_outputs)
            detached = tuple(output.detach() for output in chunk_outputs)

        # P3: embedding exchange straight into the endpoint leaves.
        emb_dest = {}
        leaves = []  # (microbatch_id, valid leaf view)
        if self.rank_view.is_decoder_endpoint:
            with nvtx_phase("p3_leaf_assembly"):
                # Only this endpoint's layouts: at CP>1 the plan also carries
                # every CP peer's leaf layout, and building those here would
                # allocate cp leaves per microbatch on every endpoint.
                for layout in plan.layouts_for_endpoint(self.rank_view.global_rank):
                    if layout.text_only:
                        continue
                    leaf = self.allocator.acquire(
                        rows=plan.capacity_policy.capacity_of(layout.total_output_rows),
                        width=self.hidden_size,
                        dtype=self.params_dtype,
                        device=self.device,
                        tag="leaf",
                    )
                    for segment in layout.segments:
                        emb_dest[
                            BridgeBufferKey(segment.global_item_id, segment.slice_id)
                        ] = leaf[
                            segment.leaf_row_start : segment.leaf_row_start
                            + segment.output_rows
                        ]
                    leaves.append((layout.microbatch_id, leaf[: layout.total_output_rows]))
        with nvtx_phase("p3_embedding_exchange"):
            emb_specs = self._iter_specs[BridgePhase.EMBEDDING]
            emb_local = {}
            for route in plan.routes_for_producer(self.rank_view.my_worker_id):
                chunk_index, segment = self._chunk_of_item[route.global_item_id]
                start = segment.output_row_start + route.item_row_start
                emb_local[BridgeBufferKey(route.global_item_id, route.slice_id)] = detached[
                    chunk_index
                ][start : start + route.item_rows]
            # One alltoall instead of batched P2P: same ledger and payload,
            # but no per-edge kernel pairs and no torch.cuda.synchronize
            # workaround (all_to_all_single stream-orders its receive buffer).
            self.bridge.exchange_all_to_all(
                self._iter_ledgers[BridgePhase.EMBEDDING],
                emb_local,
                tensor_specs=emb_specs,
                group=self.process_groups.planning_group,
                group_ranks=self.rank_view.planning_group_ranks,
                global_rank=self.rank_view.global_rank,
                dtype=self.params_dtype,
                device=self.device,
                dest_views=emb_dest,
            )
        # requires_grad only after every exchange copy into the leaf is done.
        for microbatch_id, leaf_valid in leaves:
            leaf_valid.requires_grad_(True)
            self.storage.put_leaf(microbatch_id, leaf_valid)
        if forward_only and self._eval_outputs:
            # Evaluation releases producer outputs once the bridge completed.
            self._eval_outputs = ()

        self._state = MdpRuntimeState.DECODER_READY
        self._decoder_start = time.monotonic()
        return window.replay_iterators()

    def expects_leaf(self, microbatch_id: int) -> bool:
        """Whether this rank should hold an embedding leaf for a microbatch.

        ``MdpMicrobatchRecord.text_only`` is a property of the whole microbatch,
        but leaf presence is a property of ``(microbatch, this endpoint)``: under
        decoder CP an endpoint legitimately receives no vision rows from a
        microbatch that has them globally, because the zigzag split put all of
        that microbatch's image tokens on its peers. The plan is the authority.
        """
        if not self.rank_view.is_decoder_endpoint or self._plan is None:
            return False
        try:
            layout = self._plan.layout_for_microbatch(
                microbatch_id, self.rank_view.global_rank
            )
        except MdpPlanError:
            return False
        return not layout.text_only

    def capture_global_num_tokens(self, token_tensor: Optional[torch.Tensor]) -> None:
        """Store a reference to the in-place reduced global token tensor.

        Called from the ``finalize_model_grads_func`` wrapper after the native
        finalizer's collectives. Never clones. Evaluation captures nothing.
        """
        if token_tensor is None:
            raise MdpConfigurationError(
                "MDP: the decoder finalizer received num_tokens=None; "
                "calculate_per_token_loss must be True so the global token count "
                "exists to normalize encoder gradients."
            )
        if self._token_capture_count != 0:
            raise MdpStateError(
                "MDP: the global token tensor was captured more than once this "
                "iteration."
            )
        self._captured_num_tokens = token_tensor
        self._token_capture_count = 1

    def mark_decoder_complete(self) -> None:
        """P4 ended: the native schedule (and its finalizer) returned."""
        self._require_state(MdpRuntimeState.DECODER_READY, "mark_decoder_complete")
        if not self._forward_only and self._token_capture_count != 1:
            raise MdpStateError(
                "MDP: the decoder schedule completed without exactly one global "
                "token capture; is finalize_model_grads_func wrapped?"
            )
        self._decoder_schedule_ms = (time.monotonic() - self._decoder_start) * 1000.0
        self._state = MdpRuntimeState.DECODER_DONE

    def end_iteration(self) -> None:
        """P5 (training) or cleanup (evaluation), then lifecycle asserts."""
        self._require_state(MdpRuntimeState.DECODER_DONE, "end_iteration")
        plan = self._plan
        backward_start = time.monotonic()
        if self._forward_only:
            for layout in plan.layouts:
                self.storage.release(layout.microbatch_id)
        else:
            # P5: gradient exchange, producer multi-tensor backward, WORLD
            # encoder-gradient reduction and 1/T_global normalization.
            # Regroup buffers first: the exchange writes each routed gradient
            # straight to its chunk offset (the wire is params_dtype; the
            # destination copy casts to the chunk output dtype, exactly like
            # the former two-step unpack + regroup did).
            chunk_grads = []
            grad_dest = {}
            if self._handle is not None:
                with nvtx_phase("p5_grad_regroup"):
                    for chunk_index, chunk in enumerate(self._chunk_layouts):
                        # Match the chunk output dtype: a mixed-precision wrapper
                        # (Float16Module) returns fp32 at the module boundary even
                        # when parameters and transport run in bf16.
                        output_dtype = self._handle.chunk_outputs[chunk_index].dtype
                        grad_buffer = self.allocator.acquire(
                            rows=plan.capacity_policy.capacity_of(chunk.total_output_rows),
                            width=self.hidden_size,
                            dtype=output_dtype,
                            device=self.device,
                            tag="grad_regroup",
                        )
                        for segment in chunk.segments:
                            for route in plan.routes_for_item(segment.global_item_id):
                                start = segment.output_row_start + route.item_row_start
                                grad_dest[
                                    BridgeBufferKey(
                                        segment.global_item_id, route.slice_id
                                    )
                                ] = grad_buffer[start : start + route.item_rows]
                        chunk_grads.append(grad_buffer[: chunk.total_output_rows])
            with nvtx_phase("p5_grad_exchange"):
                grad_specs = self._iter_specs[BridgePhase.GRADIENT]
                grad_local = {}
                if self.rank_view.is_decoder_endpoint:
                    for layout in plan.layouts_for_endpoint(self.rank_view.global_rank):
                        if layout.text_only:
                            continue
                        grad = self.storage.pop_grad(layout.microbatch_id)
                        for segment in layout.segments:
                            grad_local[
                                BridgeBufferKey(segment.global_item_id, segment.slice_id)
                            ] = grad[
                                segment.leaf_row_start : segment.leaf_row_start
                                + segment.output_rows
                            ]
                self.bridge.exchange_all_to_all(
                    self._iter_ledgers[BridgePhase.GRADIENT],
                    grad_local,
                    tensor_specs=grad_specs,
                    group=self.process_groups.planning_group,
                    group_ranks=self.rank_view.planning_group_ranks,
                    global_rank=self.rank_view.global_rank,
                    dtype=self.params_dtype,
                    device=self.device,
                    dest_views=grad_dest,
                )
            if self._handle is not None:
                with nvtx_phase("p5_encoder_backward"):
                    self._handle.backward(chunk_grads)
                    self._handle.release()
            with nvtx_phase("p5_finalize_encoder_grads"):
                finalize_encoder_grads(
                    self.encoder_domain.encoder_ddp,
                    globally_reduced_num_tokens=self._captured_num_tokens,
                )
            self._token_consumed = True

        encoder_backward_ms = (
            0.0 if self._forward_only else (time.monotonic() - backward_start) * 1000.0
        )
        worker_loads = worker_loads_from_plan(plan, len(self.rank_view.worker_ids))
        self._last_metrics = MdpIterationMetrics(
            iteration=self._iteration,
            outer_dp_rank=self.rank_view.outer_dp_rank,
            plan_build_ms=self._plan_build_ms,
            encoder_forward_ms=self._encoder_forward_ms,
            decoder_schedule_ms=self._decoder_schedule_ms,
            encoder_backward_ms=encoder_backward_ms,
            worker_loads=worker_loads,
            empty_workers=sum(1 for load in worker_loads if load == 0),
            bridge_stats=self.bridge.last_stats(),
            allocator_reuse=self.allocator.reuse_stats(),
        )
        logger.debug("MDP metrics: %s", self._last_metrics)
        self._assert_iteration_boundary()
        self._window = None
        self._plan = None
        self._iter_specs = {}
        self._iter_ledgers = {}
        self._handle = None
        self._eval_outputs = ()
        self._chunk_layouts = ()
        self._chunk_of_item = {}
        self._captured_num_tokens = None
        self._iteration += 1
        self._state = MdpRuntimeState.EMPTY

    def last_iteration_metrics(self) -> Optional[MdpIterationMetrics]:
        """Metrics of the most recently completed iteration."""
        return self._last_metrics

    def consumed_num_tokens(self) -> Optional[torch.Tensor]:
        """The captured token tensor (test hook for the data_ptr assertion)."""
        return self._captured_num_tokens

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _first_iterator(data_iterators):
        if isinstance(data_iterators, (list, tuple)):
            return data_iterators[0] if data_iterators else None
        return data_iterators

    def _greedy_stream(self, data_iterators):
        """The greedy sample stream for this data iterator, created on first use."""
        iterator = self._first_iterator(data_iterators)
        entry = self._greedy_streams.get(id(iterator))
        if entry is None:
            entry = (
                GreedySampleStream(
                    iterator,
                    token_budget=self._greedy_token_budget,
                    max_num_seqs=self._greedy_max_num_seqs,
                    align=self._greedy_row_alignment,
                    length_of=decoder_sample_length,
                ),
                # Evaluation runs forward_only; recorded so its consumption is
                # kept out of consumed_train_samples.
                self._forward_only,
            )
            self._greedy_streams[id(iterator)] = entry
        return entry[0]

    def consumed_samples(self) -> Optional[int]:
        """Real samples consumed by greedy *training* packing, or ``None`` when off.

        ``training.py`` reads the delta per iteration because the closed form
        ``dp x mbs x num_microbatches`` is wrong under greedy packing. Evaluation
        streams are excluded: they consume their own samples, and folding them in
        would charge an eval pass to the next training iteration.

        Counts committed, not drained, samples. Under
        ``--mdp-overlap-window-capture`` the prefetch thread drains iteration
        i+1's samples during iteration i and the final prefetch is dropped
        unconsumed; commit happens when the window is installed for its
        iteration, so neither shifts the count.
        """
        if not self.config.greedy_packing:
            return None
        return sum(
            stream.consumed_samples
            for stream, forward_only in self._greedy_streams.values()
            if not forward_only
        )

    def _capture_window(self, data_iterators, num_microbatches: int):
        """Capture one window, plus the greedy commit it owes.

        Returns ``(window, pending)`` where ``pending`` is ``(stream, count)``
        for the samples this window drained, or ``None`` without greedy packing.
        The caller commits the count only once the window is installed for an
        iteration; a prefetched window that is never consumed commits nothing.
        """
        if not self.config.greedy_packing:
            return self._capture(data_iterators, num_microbatches), None
        stream = self._greedy_stream(data_iterators)
        drained_before = stream.drained_samples
        try:
            window = self._capture(stream, num_microbatches)
            return window, (stream, stream.drained_samples - drained_before)
        except MdpStateError as error:
            if not stream.exhausted:
                raise
            # Greedy fills a fixed number of bins to a token budget, so an
            # iteration eats roughly token_budget/mean_sample_len samples per
            # bin, not micro_batch_size. Megatron provisions the sampler as
            # train_iters x global_batch_size *samples*, which under-counts
            # whenever the mean sample is shorter than the per-bin share.
            raise MdpStateError(
                f"{error} Under --mdp-greedy-packing the sample stream must be "
                "provisioned by tokens, not by samples: each bin consumes about "
                f"{self._greedy_token_budget} tokens' worth of samples, so raise "
                "--train-samples / the dataset size (roughly by "
                "token_budget / (mean_sample_len x micro_batch_size)), or lower "
                "--max-seqlen-per-dp-cp-rank."
            ) from error

    def _capture(self, data_iterators, num_microbatches: int) -> MdpIterationWindow:
        return MdpIterationWindow.capture(
            data_iterators,
            num_microbatches=num_microbatches,
            adapter=self.adapter,
            num_vpp_chunks=self.num_vpp_chunks,
            lane_id=self.rank_view.lane_id,
            my_worker_id=self.rank_view.my_worker_id,
            num_workers=len(self.rank_view.worker_ids),
        )

    @staticmethod
    def _window_prefetch_key(data_iterators, num_microbatches: int):
        if isinstance(data_iterators, (list, tuple)):
            iterator = data_iterators[0] if data_iterators else None
        else:
            iterator = data_iterators
        return (id(iterator), num_microbatches)

    def _take_prefetched_window(self, data_iterators, num_microbatches: int):
        """Return ``(window, pending_greedy)``, or ``(None, None)`` if unavailable."""
        if self._prefetch_thread is None:
            return None, None
        if self._prefetch_key != self._window_prefetch_key(data_iterators, num_microbatches):
            return None, None  # different iterator (eval); keep the pending prefetch
        with nvtx_phase("p1_window_prefetch_join"):
            self._prefetch_thread.join()
        box = self._prefetch_box
        self._prefetch_key = None
        self._prefetch_thread = None
        self._prefetch_box = None
        if "error" in box:
            raise box["error"]
        window = box["window"]
        # Order every subsequent main-stream op after the side-stream capture
        # work (H2D copies of the window tensors). A stream-level wait, not a
        # host sync: the main stream simply refuses to run ahead of the event.
        current = torch.cuda.current_stream()
        current.wait_event(box["event"])
        # Belt and braces for the caching allocator: the window tensors were
        # allocated on the side stream but are consumed (and eventually freed)
        # from main-stream code; mark them so their blocks are not handed back
        # to the side-stream pool while main-stream work is still pending.
        seen = set()

        def _record(value):
            if torch.is_tensor(value) and value.is_cuda and value.data_ptr() not in seen:
                seen.add(value.data_ptr())
                value.record_stream(current)

        for record in window.records():
            for value in record.model_payload.values():
                _record(value)
            params = record.decoder_packed_seq_params
            for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded",
                         "cu_seqlens_kv_padded"):
                _record(getattr(params, name, None))
        for tensor in window.payload_sidecar().values():
            _record(tensor)
        return window, box.get("greedy")

    def _start_window_prefetch(self, data_iterators, num_microbatches: int) -> None:
        """Capture the next window on a background thread and a side stream.

        The side stream keeps the capture's H2D traffic out of the main
        compute stream, so the copies overlap the decoder schedule instead of
        interleaving with (and delaying) its kernels. Concurrent capture is
        validated for TP=1 only (see --mdp-overlap-window-capture); with TP=1
        the capture path performs no collectives, so the thread never touches
        NCCL. The prefetch after the final training iteration is captured but
        never consumed; a capture failure there stays inside its box and is
        only re-raised if a later iteration actually asks for the window.
        """
        if self._prefetch_thread is not None:
            return  # one in-flight prefetch; an unconsumed one stays cached
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
        box: dict = {}
        stream = self._prefetch_stream

        def _run():
            try:
                # A fresh thread defaults to cuda:0; capture moves tensors to
                # "cuda", which must resolve to this rank's device.
                torch.cuda.set_device(self.device)
                with torch.cuda.stream(stream):
                    with nvtx_phase("p1_window_capture_prefetch"):
                        box["window"], box["greedy"] = self._capture_window(
                            data_iterators, num_microbatches
                        )
                    event = torch.cuda.Event()
                    event.record(stream)
                    box["event"] = event
            except BaseException as exc:  # surfaced on consumption
                box["error"] = exc

        self._prefetch_key = self._window_prefetch_key(data_iterators, num_microbatches)
        self._prefetch_box = box
        self._prefetch_thread = threading.Thread(
            target=_run, name="mdp-window-prefetch", daemon=True
        )
        self._prefetch_thread.start()

    def _require_state(self, expected: MdpRuntimeState, operation: str) -> None:
        if self._state is not expected:
            raise MdpStateError(
                f"MDP: {operation} at iteration {self._iteration} on rank "
                f"{self.rank_view.global_rank} violates: state {expected.name} "
                f"(current: {self._state.name})."
            )

    def _tensor_specs(self, plan: MdpBatchPlan, *, pixels: bool) -> dict:
        """Buffer sizing per transported key.

        PIXEL is keyed by item and sized by ``payload_rows``: one payload per
        item regardless of how many decoder endpoints its rows reach.
        EMBEDDING/GRADIENT are keyed by ``(item, slice_id)`` and sized by that
        slice's rows, which sum to the item's ``output_rows``.
        """
        specs = {}
        for route in plan.routes:
            segment = plan.segment_for_item(route.global_item_id)
            if pixels:
                if route.slice_id != 0:
                    continue
                key = BridgeBufferKey(route.global_item_id)
                valid = segment.payload_rows
                width = self.adapter.payload_width
            else:
                key = BridgeBufferKey(route.global_item_id, route.slice_id)
                valid = route.item_rows
                width = self.hidden_size
            specs[key] = BridgeTensorSpec(
                valid_rows=valid,
                capacity_rows=plan.capacity_policy.capacity_of(valid),
                width=width,
                dtype=self.params_dtype,
                device=self.device,
            )
        return specs

    def _assert_iteration_boundary(self) -> None:
        """Lifecycle invariants at every iteration boundary."""
        if self._handle is not None and not self._handle.consumed:
            raise MdpStateError(
                "MDP: an unconsumed producer forward handle survived the iteration."
            )
        self.storage.assert_empty()
        self.bridge.assert_idle()
        if not self._forward_only and not self._token_consumed:
            raise MdpStateError(
                "MDP: the global token tensor was captured but never consumed."
            )
