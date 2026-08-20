# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Greedy token-budget packing for the MDP decoder data path.

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
            "MDP: greedy packing violates: every sample carries a 1-D 'input_ids' "
            f"tensor (got {type(sample).__name__})."
        ) from exc


def greedy_bin_sizes(
    lengths: List[int], *, token_budget: int, max_num_seqs: Optional[int], num_bins: int
) -> List[int]:
    """Number of samples taken into each of ``num_bins`` in-order greedy bins.

    Pure function over a length vector, so the grouping rule is testable without
    a dataset. ``lengths`` must already be aligned (see
    :class:`GreedySampleStream`). Raises when ``lengths`` cannot fill
    ``num_bins`` bins.
    """
    sizes = []
    cursor = 0
    for _ in range(num_bins):
        taken = 0
        total = 0
        while cursor + taken < len(lengths):
            length = lengths[cursor + taken]
            if taken and total + length > token_budget:
                break
            if max_num_seqs is not None and taken >= max_num_seqs:
                break
            total += length
            taken += 1
        if taken == 0:
            raise MdpStateError(
                f"MDP: greedy packing violates: {num_bins} non-empty bins per iteration "
                f"(the sample stream ran out after {len(sizes)})."
            )
        sizes.append(taken)
        cursor += taken
    return sizes


class GreedySampleStream:
    """Wrap a microbatch-list iterator so ``next()`` returns one greedy bin.

    The underlying data iterator yields whole ``--micro-batch-size`` lists (the
    dataloader uses an identity collate over a ``batch_sampler``), so this holds
    a **sample buffer**: it pulls those lists and drains them sample by sample
    into bins. The batch_sampler itself is untouched -- shrinking it to size 1
    would change sampler bookkeeping and the shuffle order.

    The buffer is training state that carries across iterations: after filling
    an iteration's bins it usually holds a partial list, and dropping those
    samples would silently skip data. It is **not** checkpointed, so a resume
    under greedy packing restarts on a batch_sampler boundary and is only
    approximately reproducible.

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
                f"MDP: greedy packing violates: token_budget > 0 (got {token_budget}). "
                "Set --max-seqlen-per-dp-cp-rank."
            )
        if align < 1 or token_budget % align != 0:
            raise MdpConfigurationError(
                f"MDP: greedy packing violates: token_budget ({token_budget}) is "
                f"divisible by the collator row alignment ({align}); otherwise a full "
                "bin cannot be split legally across CP/SP ranks."
            )
        self._iterator = iterator
        self._token_budget = token_budget
        self._max_num_seqs = max_num_seqs
        self._align = align
        self._length_of = length_of
        self._buffer: List[Any] = []
        self._buffer_cursor = 0
        self._exhausted = False
        self._consumed_samples = 0
        # --mdp-overlap-window-capture captures the next iteration's window on a
        # background thread. Only one capture per iterator is ever in flight (the
        # consumer joins the prefetch before capturing again), but the buffer is
        # mutable state shared with that thread, so guard it rather than rely on
        # the caller's ordering.
        self._lock = threading.Lock()

    @property
    def consumed_samples(self) -> int:
        """Real samples drained into bins since construction."""
        return self._consumed_samples

    @property
    def exhausted(self) -> bool:
        """True once the underlying iterator has raised ``StopIteration``."""
        return self._exhausted and self._buffer_cursor >= len(self._buffer)

    def __iter__(self) -> "GreedySampleStream":
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
        align = self._align
        return ((length + align - 1) // align) * align

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
            self._consumed_samples += len(bin_samples)
            return bin_samples
