from contextlib import nullcontext
from types import SimpleNamespace

import torch

from examples.multimodal_dev.models import base


class _LanguageModel:
    def __init__(self, text_embeddings):
        self._text_embeddings = text_embeddings

    def embedding(self, input_ids, position_ids=None):
        return self._text_embeddings

    def __call__(self, **kwargs):
        return kwargs["decoder_input"]


class _MdpOffHarness:
    forward = base.MultimodalModel.forward
    _scatter_vision_embeddings = base.MultimodalModel._scatter_vision_embeddings
    _cp_local_thd_index_for_length = base.MultimodalModel._cp_local_thd_index_for_length
    _cp_split_for_forward = base.MultimodalModel._cp_split_for_forward
    _thd_mrope_no_cp_override = base.MultimodalModel._thd_mrope_no_cp_override

    def __init__(self, text_embeddings, vision_embeddings):
        self.config = SimpleNamespace(sequence_parallel=False)
        self.pre_process = True
        self.image_token_id = 99
        self.language_model = _LanguageModel(text_embeddings)
        self.vision_model = lambda _pixels, _grid: vision_embeddings
        self._mdp_enabled = False


def test_mdp_off_forward_preserves_legacy_full_vision_scatter(monkeypatch):
    monkeypatch.setattr(base, "get_nvtx_range", lambda: lambda _name: nullcontext())
    monkeypatch.setattr(base.parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(base.parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)

    input_ids = torch.tensor([[1, 99, 2, 99]], dtype=torch.long)
    text = torch.arange(4 * 2, dtype=torch.float32).view(4, 1, 2)
    vision = torch.tensor([[100.0, 101.0], [200.0, 201.0]])
    model = _MdpOffHarness(text, vision)

    output = model.forward(
        input_ids=input_ids,
        position_ids=torch.arange(4).view(1, 4),
        pixel_values=torch.ones(1, 2),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
    )

    expected = text.clone()
    expected[1, 0] = vision[0]
    expected[3, 0] = vision[1]
    torch.testing.assert_close(output, expected)


def test_qwen3_shim_drops_mdp_metadata_for_forward_and_schedule(monkeypatch):
    from examples.multimodal_dev.models.qwen3.factory import Qwen3TextOnlyGPTModel
    from megatron.core.models.gpt.gpt_model import GPTModel

    monkeypatch.setattr(GPTModel, "forward", lambda _self, *args, **kwargs: (args, kwargs))
    monkeypatch.setattr(
        GPTModel, "build_schedule_plan", lambda _self, *args, **kwargs: (args, kwargs)
    )
    model = Qwen3TextOnlyGPTModel.__new__(Qwen3TextOnlyGPTModel)

    _args, forward_kwargs = model.forward(
        input_ids=torch.tensor([[1]]),
        pixel_values=torch.empty(0),
        image_grid_thw=torch.empty(0, 3),
        mdp_cp_local_plan={"unused": True},
    )
    _args, schedule_kwargs = model.build_schedule_plan(
        input_ids=torch.tensor([[1]]),
        pixel_values=torch.empty(0),
        image_grid_thw=torch.empty(0, 3),
        mdp_cp_local_plan={"unused": True},
    )

    assert set(forward_kwargs) == {"input_ids"}
    assert set(schedule_kwargs) == {"input_ids"}
