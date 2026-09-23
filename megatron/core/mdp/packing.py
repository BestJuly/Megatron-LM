# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Greedy and buffered first-fit decreasing token-budget packing for the MDP decoder data path.

Without this module a microbatch is exactly ``--micro-batch-size`` samples, so
the packed THD length ``T`` is whatever those samples happen to sum to. With it,
a microbatch is a *bin*: samples are appended while they fit a token budget and
a real-sequence cap, so ``T`` is bounded and the padding waste is decoupled from
``--micro-batch-size``.

The grouping rule is the in-order greedy fill of
``DpBalancedScheduler.get_groups_and_subsamples``
(``megatron/core/datasets/data_schedule.py``). MDP cannot use that scheduler --
it asserts on GPT-only sample keys, drops ``pixel_values`` /
``image_grid_thw``, and reroutes samples across DP with an all-to-all that has
no notion of variable-size pixel payloads (see ``mdp/README.md``). So MDP reuses
the *logic*, not the path.

Variant: a fixed number of bins per iteration (``num_microbatches``), with the
number of samples consumed floating. ``num_microbatches`` therefore stays
static, which the PP schedule, the VPP replay cursors, and per-layer CUDA graph
slot sizing all depend on. The cost is that ``--global-batch-size`` /
``--micro-batch-size`` stop describing sample counts and become pure bin-count
knobs; MDP already hard-requires ``calculate_per_token_loss=True``, so loss
normalization is unaffected by a varying sample count per iteration.
"""

import threading
from collections import deque
from typing import Any, Callable, Iterator, List, Optional

from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError


def decoder_sample_length(sample: Any) -> int:
    """Token count of one decoder sample dict.

    ``input_ids`` is the MDP dataset contract (``examples/multimodal_dev/data``);
    image slots are already materialized in it, so it is the packed row count.
    """
    try:
        return int(sample["input_ids"].shape[0])
    except (KeyError, TypeError, AttributeError) as exc:
        raise MdpConfigurationError(
            "MDP: packing violates: every sample carries a 1-D 'input_ids' "
            f"tensor (got {type(sample).__name__})."
        ) from exc


class PackedSampleStream:
    """Shared sample buffering and commit accounting for token-budget streams.

    The underlying data iterator yields whole ``--micro-batch-size`` lists (the
    dataloader uses an identity collate over a ``batch_sampler``), so this holds
    a **sample buffer**: it pulls those lists and drains them sample by sample
    into bins. The batch_sampler itself is untouched -- shrinking it to size 1
    would change sampler bookkeeping and the shuffle order.

    The buffer is training state that carries across iterations: after filling
    an iteration's bins it usually holds a partial list, and dropping those
    samples would silently skip data. It is **not** checkpointed, which is why
    ``validate_mdp_config`` rejects ``--save`` / ``--load`` under greedy packing
    unless ``--mdp-greedy-packing-approximate-resume`` is passed.

    Sample counting is two-stage. ``__next__`` *drains* samples into a bin, but
    a bin is only *committed* when the iteration that owns it actually runs:
    under ``--mdp-overlap-window-capture`` the next iteration's window is filled
    on a prefetch thread during the current one, and the final prefetch is never
    consumed at all. ``consumed_samples`` therefore reports committed samples,
    so ``consumed_train_samples`` counts exactly the samples that were trained
    on.

    Args:
        iterator: The underlying iterator of sample-dict lists.
        token_budget: Maximum aligned token count per bin,
            ``max_seqlen_per_dp_cp_rank * cp_size``.
        max_num_seqs: Maximum real sequences per bin
            (``thd_max_packed_sequences``), or ``None`` for no cap.
        align: Per-sample row alignment applied by the collator; a sample of
            length ``L`` occupies ``ceil(L / align) * align`` rows in the pack,
            and that is what is charged against the budget.
        length_of: Extracts a sample's unaligned token count.
    """

    def __init__(
        self,
        iterator: Iterator,
        *,
        token_budget: int,
        max_num_seqs: Optional[int] = None,
        align: int = 1,
        length_of: Callable[[Any], int],
    ) -> None:
        if token_budget <= 0:
            raise MdpConfigurationError(
                f"MDP: packing violates: token_budget > 0 (got {token_budget}). "
                "Set --max-seqlen-per-dp-cp-rank."
            )
        if align < 1 or token_budget % align != 0:
            raise MdpConfigurationError(
                f"MDP: packing violates: token_budget ({token_budget}) is "
                f"divisible by the collator row alignment ({align}); otherwise a full "
                "bin cannot be split legally across CP/SP ranks."
            )
        if max_num_seqs is not None and max_num_seqs < 1:
            raise MdpConfigurationError("MDP: max_num_seqs must be positive or None.")
        self._iterator = iterator
        self._token_budget = token_budget
        self._max_num_seqs = max_num_seqs
        self._align = align
        self._length_of = length_of
        self._buffer: List[Any] = []
        self._buffer_cursor = 0
        self._exhausted = False
        self._drained_samples = 0
        self._committed_samples = 0
        # --mdp-overlap-window-capture captures the next iteration's window on a
        # background thread. Only one capture per iterator is ever in flight (the
        # consumer joins the prefetch before capturing again), but the buffer is
        # mutable state shared with that thread, so guard it rather than rely on
        # the caller's ordering.
        self._lock = threading.Lock()

    @property
    def drained_samples(self) -> int:
        """Real samples pulled into bins since construction, committed or not."""
        return self._drained_samples

    @property
    def consumed_samples(self) -> int:
        """Real samples whose bins were actually consumed by an iteration."""
        return self._committed_samples

    def commit(self, count: int) -> None:
        """Account ``count`` drained samples as consumed.

        Called by the runtime once the window built from those bins is installed
        for the iteration, never at capture time -- a prefetched window may be
        captured and then dropped.
        """
        if count < 0:
            raise MdpStateError(
                f"MDP: packing violates: committed sample count >= 0 (got {count})."
            )
        with self._lock:
            if self._committed_samples + count > self._drained_samples:
                raise MdpStateError(
                    "MDP: packing violates: committed samples "
                    f"({self._committed_samples} + {count}) <= drained samples "
                    f"({self._drained_samples}); a window was committed twice."
                )
            self._committed_samples += count

    @property
    def exhausted(self) -> bool:
        """True once the underlying iterator has raised ``StopIteration``."""
        return self._exhausted and self._buffer_cursor >= len(self._buffer)

    def __iter__(self) -> "PackedSampleStream":
        return self

    def _next_sample(self) -> Optional[Any]:
        """One sample from the buffer, refilling from the iterator as needed."""
        while self._buffer_cursor >= len(self._buffer):
            if self._exhausted:
                return None
            try:
                self._buffer = list(next(self._iterator))
            except StopIteration:
                self._exhausted = True
                return None
            self._buffer_cursor = 0
        sample = self._buffer[self._buffer_cursor]
        self._buffer_cursor += 1
        return sample

    def _unread(self) -> None:
        """Push the last sample back; it starts the next bin."""
        self._buffer_cursor -= 1

    def _aligned_length(self, sample: Any) -> int:
        length = int(self._length_of(sample))
        if length <= 0:
            raise MdpStateError(f"MDP: sample length must be positive (got {length}).")
        align = self._align
        return ((length + align - 1) // align) * align


class GreedySampleStream(PackedSampleStream):
    """Pack samples in source order, closing each bin when the next cannot fit."""

    def __next__(self) -> List[Any]:
        """One greedy bin, or raise ``StopIteration`` at end of stream.

        A bin is closed when the next sample would exceed the token budget or
        the real-sequence cap. At end of stream a partially filled bin is
        returned as is -- correct, and never an *empty* pack.
        """
        with self._lock:
            bin_samples: List[Any] = []
            total = 0
            while True:
                sample = self._next_sample()
                if sample is None:
                    break
                length = self._aligned_length(sample)
                if not bin_samples and length > self._token_budget:
                    raise MdpStateError(
                        f"MDP: sample violates: aligned length ({length}) <= the greedy "
                        f"token budget ({self._token_budget}). Raise "
                        "--max-seqlen-per-dp-cp-rank or filter overlong samples."
                    )
                if bin_samples and total + length > self._token_budget:
                    self._unread()
                    break
                if self._max_num_seqs is not None and len(bin_samples) >= self._max_num_seqs:
                    self._unread()
                    break
                bin_samples.append(sample)
                total += length
            if not bin_samples:
                raise StopIteration
            self._drained_samples += len(bin_samples)
            return bin_samples


class FfdSampleStream(PackedSampleStream):
    """First-fit decreasing packing over bounded, disjoint sample windows.

    Sort each window by descending aligned length (stable source-order ties),
    then place each complete sample into the first eligible bin in creation
    order. Refill freed buffer slots before emitting each subsequent bin, like
    Energon's reading/prepacking buffers. Form the next window's bins only after
    the current bins drain. The sample-count bound excludes the dataloader's
    current list and worker prefetch; large image payloads can use substantial
    host memory. Read-ahead is not counted as drained or committed consumption.

    Args:
        buffer_size: Maximum number of samples in one sorting window.
        **kwargs: Shared token budget, sequence cap, alignment and length reader.
    """

    def __init__(self, iterator: Iterator, *, buffer_size: int = 128, **kwargs) -> None:
        super().__init__(iterator, **kwargs)
        if buffer_size < 1:
            raise MdpConfigurationError("MDP: FFD buffer_size must be positive.")
        self._buffer_size = buffer_size
        self._bins: deque[List[Any]] = deque()
        self._pending_samples = 0
        self._reading_window: List[tuple[int, Any]] = []

    @property
    def exhausted(self) -> bool:
        """True only after both source samples and prepacked bins are exhausted."""
        return super().exhausted and not self._bins and not self._reading_window

    def __next__(self) -> List[Any]:
        """Emit one FFD bin, without consuming or dropping any other planned bin."""
        with self._lock:
            # Replenish only the slots released by emitted packs. This keeps
            # the same disjoint sorting windows while letting loader workers
            # refill throughout training instead of stalling on a whole window
            # at every boundary. Pending + unread samples never exceed the cap.
            while len(self._reading_window) + self._pending_samples < self._buffer_size:
                sample = self._next_sample()
                if sample is None:
                    break
                length = self._aligned_length(sample)
                if length > self._token_budget:
                    raise MdpStateError(
                        f"MDP: sample aligned length ({length}) exceeds the FFD "
                        f"token budget ({self._token_budget}). Raise "
                        "--max-seqlen-per-dp-cp-rank or filter overlong samples."
                    )
                self._reading_window.append((length, sample))
            if not self._bins:
                window = self._reading_window
                self._reading_window = []
                # Python's stable sort preserves source order for equal lengths.
                window.sort(key=lambda item: -item[0])
                bins: List[List[Any]] = []
                totals: List[int] = []
                for length, sample in window:
                    for index, samples in enumerate(bins):
                        if totals[index] + length <= self._token_budget and (
                            self._max_num_seqs is None or len(samples) < self._max_num_seqs
                        ):
                            samples.append(sample)
                            totals[index] += length
                            break
                    else:
                        bins.append([sample])
                        totals.append(length)
                self._bins.extend(bins)
                self._pending_samples = len(window)
            if not self._bins:
                raise StopIteration
            samples = self._bins.popleft()
            self._pending_samples -= len(samples)
            self._drained_samples += len(samples)
            return samples
