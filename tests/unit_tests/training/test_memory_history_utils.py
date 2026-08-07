# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

from megatron.training.utils.utils import (
    ranked_memory_snapshot_path,
    start_memory_history_recording,
)


def test_ranked_memory_snapshot_path_preserves_extension(tmp_path):
    path = tmp_path / "memory.snapshot.pickle"

    assert ranked_memory_snapshot_path(str(path), 7) == str(
        tmp_path / "memory.snapshot_rank-7.pickle"
    )
    assert ranked_memory_snapshot_path(str(tmp_path / "snapshot"), 3) == str(
        tmp_path / "snapshot_rank-3.pickle"
    )


def test_start_memory_history_records_only_selected_ranks(tmp_path):
    profiling = SimpleNamespace(
        record_memory_history=True,
        profile_ranks=[0, 2],
        memory_snapshot_path=str(tmp_path / "snapshot.pickle"),
    )

    with (
        mock.patch("megatron.training.utils.utils.safe_get_rank", return_value=1),
        mock.patch("torch.cuda.memory._record_memory_history") as record_history,
    ):
        start_memory_history_recording(profiling)
    record_history.assert_not_called()

    with (
        mock.patch("megatron.training.utils.utils.safe_get_rank", return_value=2),
        mock.patch("torch.cuda.memory._record_memory_history") as record_history,
        mock.patch("torch._C._cuda_attach_out_of_memory_observer"),
    ):
        start_memory_history_recording(profiling)
    record_history.assert_called_once_with(
        True, trace_alloc_max_entries=100_000, trace_alloc_record_context=True
    )
