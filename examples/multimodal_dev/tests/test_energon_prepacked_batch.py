# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from examples.multimodal_dev.forward_step import _broadcast_tensor, _prepare_prepacked_batch
from examples.multimodal_dev.models.base import MultimodalModel
from megatron.core import parallel_state


def test_broadcast_tensor_preserves_scalar_tensors(monkeypatch):
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *_args, **_kwargs: None)

    scalar = torch.tensor(64, dtype=torch.int32)

    actual = _broadcast_tensor(scalar, src=0, group=None, device="cpu")

    assert actual is scalar
    assert actual.shape == torch.Size([])
    assert actual.item() == 64


def test_prepare_prepacked_batch_restores_thd_metadata():
    batch = _prepare_prepacked_batch(
        {
            "input_ids": torch.arange(128),
            "labels": torch.arange(128),
            "loss_mask": torch.ones(128),
            "position_ids": torch.arange(128).repeat(3, 1),
            "pixel_values": torch.zeros(0, 1536),
            "image_grid_thw": torch.zeros(0, 3, dtype=torch.long),
            "cu_seqlens": torch.tensor([0, 64, 96], dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor([0, 64, 128], dtype=torch.int32),
            "max_seqlen": torch.tensor(64, dtype=torch.int32),
        }
    )

    params = batch["packed_seq_params"]
    assert batch["input_ids"].shape == (1, 128)
    assert batch["position_ids"].shape == (3, 1, 128)
    assert params.cu_seqlens_q.tolist() == [0, 64, 96]
    assert params.cu_seqlens_kv.tolist() == [0, 64, 96]
    assert params.cu_seqlens_q_padded.tolist() == [0, 64, 128]
    assert params.cu_seqlens_kv_padded.tolist() == [0, 64, 128]
    assert params.max_seqlen_q == 64
    assert params.total_tokens == 128


def test_prepare_prepacked_batch_rejects_mismatched_container_length():
    batch = {
        "input_ids": torch.arange(128),
        "labels": torch.arange(128),
        "loss_mask": torch.ones(128),
        "cu_seqlens": torch.tensor([0, 64], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 64], dtype=torch.int32),
        "max_seqlen": torch.tensor(64, dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="does not match input length"):
        _prepare_prepacked_batch(batch)


def test_text_only_batch_skips_the_vision_encoder(monkeypatch):
    class LanguageModel(torch.nn.Module):
        def embedding(self, input_ids, position_ids):
            del position_ids
            return torch.zeros(input_ids.shape[1], input_ids.shape[0], 4)

        def forward(self, *, decoder_input, **_kwargs):
            return decoder_input

    class VisionModel(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("vision encoder received an empty image batch")

    model = MultimodalModel.__new__(MultimodalModel)
    torch.nn.Module.__init__(model)
    model.language_model = LanguageModel()
    model.vision_model = VisionModel()
    model.vision_parallel_group = None
    model.pre_process = True
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)

    output = model(
        input_ids=torch.arange(8).unsqueeze(0),
        position_ids=torch.arange(8).unsqueeze(0),
        pixel_values=torch.zeros(0, 1536),
        image_grid_thw=torch.zeros(0, 3, dtype=torch.long),
    )

    assert output.shape == (8, 1, 4)
