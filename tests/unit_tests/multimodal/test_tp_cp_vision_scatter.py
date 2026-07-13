# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from examples.multimodal_dev.models import base


class _ScatterHarness:
    config = SimpleNamespace(sequence_parallel=True)
    image_token_id = 99

    _scatter_vision_embeddings = base.MultimodalModel._scatter_vision_embeddings

    def __init__(self, cp_index):
        self.cp_index = cp_index
        self.indexed_lengths = []

    def _cp_local_thd_index_for_length(self, total_tokens, _packed_seq_params):
        self.indexed_lengths.append(int(total_tokens))
        return self.cp_index


def test_tp_sequence_gather_precedes_cp_partition(monkeypatch):
    monkeypatch.setattr(base, "get_nvtx_range", lambda: lambda _name: nullcontext())
    monkeypatch.setattr(base.parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base.parallel_state, "get_context_parallel_world_size", lambda: 2)

    full_text = torch.arange(8 * 2, dtype=torch.float32).view(8, 1, 2)
    monkeypatch.setattr(
        base.tensor_parallel,
        "gather_from_sequence_parallel_region",
        lambda _tensor, **_kwargs: full_text,
    )
    monkeypatch.setattr(
        base.tensor_parallel, "scatter_to_sequence_parallel_region", lambda tensor: tensor
    )

    harness = _ScatterHarness(torch.tensor([0, 1, 6, 7]))
    input_ids = torch.tensor([[1, 99, 2, 3, 4, 5, 99, 6]])
    vision = torch.tensor([[100.0, 101.0], [200.0, 201.0]])
    packed = SimpleNamespace(qkv_format="thd")

    output = harness._scatter_vision_embeddings(
        input_ids, full_text[:4], vision, packed_seq_params=packed
    )

    assert harness.indexed_lengths == [8]
    assert harness._decoder_input_already_cp_partitioned is True
    torch.testing.assert_close(output[1, 0], vision[0])
    torch.testing.assert_close(output[2, 0], vision[1])
