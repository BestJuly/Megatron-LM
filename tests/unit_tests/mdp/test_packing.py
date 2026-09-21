# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Greedy token-budget packing. Pure compute: no distributed state, no CUDA."""

import pytest
import torch

from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.packing import GreedySampleStream, decoder_sample_length
from megatron.core.mdp.runtime import MdpRuntime


def _sample(length, tag=0):
    return {"input_ids": torch.zeros(length, dtype=torch.long), "tag": tag}


def _microbatches(lengths, mbs):
    """Emulate the dataloader: identity collate over an MBS-sized batch_sampler."""
    samples = [_sample(length, tag=i) for i, length in enumerate(lengths)]
    return iter([samples[i : i + mbs] for i in range(0, len(samples), mbs)])


def _bin_lengths(bins):
    return [[int(s["input_ids"].shape[0]) for s in b] for b in bins]


# ---------------------------------------------------------------------------
# GreedySampleStream
# ---------------------------------------------------------------------------


def _stream(lengths, *, mbs, budget, cap=None, align=1):
    return GreedySampleStream(
        _microbatches(lengths, mbs),
        token_budget=budget,
        max_num_seqs=cap,
        align=align,
        length_of=decoder_sample_length,
    )


def test_bins_respect_the_token_budget():
    stream = _stream([400, 400, 400, 100, 900], mbs=5, budget=1000)
    # 400+400 fits; +400 would be 1200 -> close. Then 400+100, +900 -> close.
    assert _bin_lengths([next(stream), next(stream)]) == [[400, 400], [400, 100]]


def test_no_bin_exceeds_the_budget():
    lengths = [137, 998, 5, 640, 640, 1, 512, 512, 511]
    stream = _stream(lengths, mbs=3, budget=1024, cap=8)
    for _ in range(3):
        assert sum(decoder_sample_length(s) for s in next(stream)) <= 1024


def test_exactly_num_bins_are_produced():
    stream = _stream([100] * 50, mbs=10, budget=250, cap=8)
    bins = [next(stream) for _ in range(5)]
    assert [len(b) for b in bins] == [2] * 5


def test_stream_drains_microbatch_lists_sample_by_sample():
    lengths = [300, 300, 300, 300, 300, 300]
    stream = GreedySampleStream(
        _microbatches(lengths, mbs=4), token_budget=900, length_of=decoder_sample_length
    )
    bins = [next(stream), next(stream)]
    # The 4-sample list is split across bins; the leftover carries forward.
    assert _bin_lengths(bins) == [[300, 300, 300], [300, 300, 300]]
    assert stream.drained_samples == 6
    assert stream.consumed_samples == 0  # drained, but no window consumed them yet


def test_leftovers_carry_across_iterations():
    lengths = [500] * 8
    stream = GreedySampleStream(
        _microbatches(lengths, mbs=4), token_budget=1000, length_of=decoder_sample_length
    )
    first_iteration = [next(stream), next(stream)]
    second_iteration = [next(stream), next(stream)]
    tags = [[s["tag"] for s in b] for b in first_iteration + second_iteration]
    assert tags == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_alignment_is_charged_against_the_budget():
    # Aligned to 8, a 5-token sample occupies 8 rows: 3 fit in 24, not 4.
    stream = GreedySampleStream(
        _microbatches([5] * 8, mbs=8),
        token_budget=24,
        align=8,
        length_of=decoder_sample_length,
    )
    assert _bin_lengths([next(stream)]) == [[5, 5, 5]]


def test_sequence_cap_closes_the_bin():
    stream = GreedySampleStream(
        _microbatches([10] * 10, mbs=10),
        token_budget=10_000,
        max_num_seqs=3,
        length_of=decoder_sample_length,
    )
    assert len(next(stream)) == 3


def test_partial_bin_at_end_of_stream_then_stop():
    stream = GreedySampleStream(
        _microbatches([400] * 3, mbs=3), token_budget=1000, length_of=decoder_sample_length
    )
    assert len(next(stream)) == 2
    assert len(next(stream)) == 1  # partial, never empty
    with pytest.raises(StopIteration):
        next(stream)


def test_oversized_sample_names_the_flag():
    stream = GreedySampleStream(
        _microbatches([5000], mbs=1), token_budget=1024, length_of=decoder_sample_length
    )
    with pytest.raises(MdpStateError, match="max-seqlen-per-dp-cp-rank"):
        next(stream)


def test_budget_must_be_divisible_by_the_row_alignment():
    with pytest.raises(MdpConfigurationError, match="row alignment"):
        GreedySampleStream(
            _microbatches([10], mbs=1),
            token_budget=100,
            align=8,
            length_of=decoder_sample_length,
        )


def test_degenerate_distribution_reproduces_fixed_mbs():
    """min=max=mean=L with budget k*L packs exactly k samples per bin.

    This is the exact-equivalence configuration: greedy is then bit-identical
    to today's fixed ``--micro-batch-size k``.
    """
    L, k = 256, 4
    stream = GreedySampleStream(
        _microbatches([L] * 32, mbs=1), token_budget=k * L, length_of=decoder_sample_length
    )
    for _ in range(8):
        assert len(next(stream)) == k


def test_commit_cannot_exceed_what_was_drained():
    stream = _stream([500] * 8, mbs=4, budget=1000)
    next(stream)
    with pytest.raises(MdpStateError, match="committed samples"):
        stream.commit(3)  # only 2 were drained; a double commit would land here


# ---------------------------------------------------------------------------
# Runtime accounting: commit on consumption, not on capture
# ---------------------------------------------------------------------------


class _StubRuntime(MdpRuntime):
    """Only ``MdpRuntime``'s greedy bookkeeping; no distributed state, no CUDA."""

    def __init__(self, *, token_budget, forward_only=False):
        self.config = MdpConfig(enable=True, greedy_packing=True)
        self._greedy_token_budget = token_budget
        self._greedy_max_num_seqs = None
        self._greedy_row_alignment = 1
        self._greedy_streams = {}
        self._forward_only = forward_only

    def _capture(self, data_iterators, num_microbatches):
        return [next(data_iterators) for _ in range(num_microbatches)]


def test_capture_alone_does_not_count_samples_as_consumed():
    # --mdp-overlap-window-capture captures iteration i+1's window during
    # iteration i, and the final prefetch is never consumed at all.
    runtime = _StubRuntime(token_budget=1000)
    iterator = _microbatches([500] * 8, mbs=4)

    _, pending = runtime._capture_window(iterator, 2)
    stream, drained = pending
    assert drained == 4
    assert runtime.consumed_samples() == 0

    stream.commit(drained)  # the window is installed for its iteration
    assert runtime.consumed_samples() == 4

    runtime._capture_window(iterator, 2)  # captured, then dropped unconsumed
    assert runtime.consumed_samples() == 4


def test_evaluation_streams_stay_out_of_the_training_count():
    runtime = _StubRuntime(token_budget=1000, forward_only=True)
    _, (stream, drained) = runtime._capture_window(_microbatches([500] * 8, mbs=4), 2)
    stream.commit(drained)
    assert runtime.consumed_samples() == 0


def test_consumed_samples_is_none_without_greedy_packing():
    runtime = _StubRuntime(token_budget=1000)
    runtime.config = MdpConfig(enable=True, greedy_packing=False)
    assert runtime.consumed_samples() is None
