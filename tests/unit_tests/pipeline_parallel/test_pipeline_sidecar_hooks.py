# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest

from megatron.core.pipeline_parallel import schedules


def test_pipeline_sidecar_hooks_unwrap_model_and_require_pre_forward():
    pre_forward = object()
    post_backward = object()
    model = SimpleNamespace(
        _pipeline_sidecar_enabled=True,
        pipeline_sidecar_pre_forward=pre_forward,
        pipeline_sidecar_post_backward=post_backward,
    )

    assert schedules._get_pipeline_sidecar_hooks(SimpleNamespace(module=model)) == (
        pre_forward,
        post_backward,
    )
    assert schedules._get_pipeline_sidecar_hooks(
        SimpleNamespace(_pipeline_sidecar_enabled=False)
    ) == (None, None)

    with pytest.raises(RuntimeError, match="requires pipeline_sidecar_pre_forward"):
        schedules._get_pipeline_sidecar_hooks(SimpleNamespace(_pipeline_sidecar_enabled=True))


def test_pipeline_sidecar_prefetches_in_microbatch_order():
    events = []

    def pre_forward(**kwargs):
        events.append(
            (kwargs["current_microbatch"], kwargs["num_microbatches"], kwargs["forward_only"])
        )

    schedules._prefetch_pipeline_sidecar(
        pre_forward, data_iterator=object(), num_microbatches=3, forward_only=True
    )

    assert events == [(0, 3, True), (1, 3, True), (2, 3, True)]


def test_supported_schedules_place_sidecar_around_language_work():
    no_pipeline = inspect.getsource(schedules.forward_backward_no_pipelining)
    non_interleaved = inspect.getsource(schedules.forward_backward_pipelining_without_interleaving)

    assert no_pipeline.index("_prefetch_pipeline_sidecar(") < no_pipeline.index(
        "combined_1f1b_schedule_for_no_pipelining("
    )
    assert no_pipeline.count("sidecar_post_backward()") == 2

    assert non_interleaved.index("_prefetch_pipeline_sidecar(") < non_interleaved.index(
        "# Run warmup forward passes."
    )
    assert non_interleaved.count("sidecar_post_backward()") == 2
