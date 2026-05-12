# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pure-CPU unit tests for PP support in multimodal_dev.

Covers the PP-specific gating logic added on
`lit/qwen35_dev_pp_support` without requiring a CUDA runtime or
distributed init.  These tests exercise:

  * ``MultimodalModel.__init__``: vision-encoder gating on
    ``pre_process``, plumbing of ``pre_process`` / ``post_process`` /
    ``vp_stage`` into the inner ``GPTModel``.
  * ``MultimodalModel.set_input_tensor``: forwards into the language model.
  * ``MultimodalModel.forward``: vision encoder + embedding skipped on
    non-first PP stages; ``compute_position_ids`` called on every
    stage; CP-split helper invoked unconditionally.
  * ``forward_step``: ``pixel_values`` dropped on non-first stages,
    ``image_grid_thw`` kept on every stage; loss-mask CP-split gated to
    last PP stage.
  * ``factory.build_model`` signature accepts ``pre_process`` /
    ``post_process``.

We mock ``GPTModel`` and the vision encoder so this runs on CPU.
"""

from __future__ import annotations

import inspect
import os
import sys
from unittest import mock

import pytest
import torch

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../.."),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ===================================================================
# Helpers
# ===================================================================


class _FakeGPTModel(torch.nn.Module):
    """Stand-in GPTModel that records its constructor args and forward
    invocations so tests can introspect them.
    """

    instances = []

    def __init__(self, **kw):
        super().__init__()
        self.kw = kw
        self.pre_process = kw.get("pre_process", True)
        self.post_process = kw.get("post_process", True)
        self.vp_stage = kw.get("vp_stage", None)
        self.embedding_calls = 0
        self.input_tensor = None
        self.last_forward_kwargs = None
        # Stash so tests can introspect.
        _FakeGPTModel.instances.append(self)

    # Match GPTModel.set_input_tensor signature (accepts list or tensor).
    def set_input_tensor(self, t):
        self.input_tensor = t

    # Provide an `embedding` only on first stage to mimic GPTModel.
    @property
    def embedding(self):  # noqa: D401
        if not self.pre_process:
            raise AttributeError("embedding only exists on pre_process stage")

        def _embed(input_ids, position_ids):
            self.embedding_calls += 1
            B, S = input_ids.shape
            return torch.zeros(S, B, 4, dtype=torch.float32)

        return _embed

    def forward(self, **kw):
        self.last_forward_kwargs = kw
        if self.post_process:
            # Per-token loss tensor: shape [B, S] — match what the
            # downstream loss closure expects.
            input_ids = kw.get("input_ids")
            if input_ids is not None:
                return torch.zeros_like(input_ids, dtype=torch.float32)
            return torch.zeros(1, 1, dtype=torch.float32)
        # Hidden states [S, B, H]
        di = kw.get("decoder_input")
        if di is not None:
            return di
        return torch.zeros(1, 1, 4, dtype=torch.float32)


class _FakeVisionEncoder(torch.nn.Module):
    """Stand-in for ``Qwen35VLVisionEncoder``."""

    def forward(self, pixel_values, image_grid_thw):
        return torch.zeros(1, 4)


@pytest.fixture
def fake_gpt(monkeypatch):
    """Patch ``GPTModel`` inside ``models.base`` to ``_FakeGPTModel``."""
    _FakeGPTModel.instances.clear()
    import examples.multimodal_dev.models.base as base_mod

    monkeypatch.setattr(base_mod, "GPTModel", _FakeGPTModel)
    return _FakeGPTModel


@pytest.fixture
def tiny_lang_config():
    """Minimal TransformerConfig with the fields MultimodalModel reads."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        ffn_hidden_size=8,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
    )


# ===================================================================
# 1. Construction gating
# ===================================================================


class TestMultimodalModelConstruction:

    def _build(
        self,
        tiny_lang_config,
        *,
        pre_process: bool,
        post_process: bool,
        vp_stage=None,
    ):
        from examples.multimodal_dev.models.base import MultimodalModel

        return MultimodalModel(
            language_config=tiny_lang_config,
            language_spec=mock.MagicMock(),  # spec not introspected by __init__
            vision_encoder=_FakeVisionEncoder(),
            vocab_size=16,
            max_sequence_length=32,
            image_token_id=0,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    def test_first_stage_has_vision_encoder(self, fake_gpt, tiny_lang_config):
        m = self._build(tiny_lang_config, pre_process=True, post_process=False)
        assert m.vision_model is not None
        assert m.pre_process is True
        assert m.post_process is False

    def test_non_first_stage_drops_vision_encoder(
        self, fake_gpt, tiny_lang_config,
    ):
        m = self._build(tiny_lang_config, pre_process=False, post_process=True)
        assert m.vision_model is None
        assert m.pre_process is False
        assert m.post_process is True

    def test_pp1_keeps_both_flags_true(self, fake_gpt, tiny_lang_config):
        m = self._build(tiny_lang_config, pre_process=True, post_process=True)
        assert m.vision_model is not None
        assert m.pre_process is True
        assert m.post_process is True

    def test_gptmodel_receives_pre_post_process(
        self, fake_gpt, tiny_lang_config,
    ):
        self._build(tiny_lang_config, pre_process=False, post_process=True)
        assert len(fake_gpt.instances) == 1
        gpt_kw = fake_gpt.instances[0].kw
        assert gpt_kw["pre_process"] is False
        assert gpt_kw["post_process"] is True

    def test_gptmodel_receives_vp_stage(self, fake_gpt, tiny_lang_config):
        self._build(
            tiny_lang_config,
            pre_process=True, post_process=True, vp_stage=2,
        )
        assert fake_gpt.instances[0].kw["vp_stage"] == 2

    def test_set_input_tensor_routes_to_language_model(
        self, fake_gpt, tiny_lang_config,
    ):
        m = self._build(tiny_lang_config, pre_process=False, post_process=True)
        t = torch.ones(2, 3, 4)
        m.set_input_tensor(t)
        assert torch.equal(m.language_model.input_tensor, t)

    def test_set_input_tensor_accepts_list(self, fake_gpt, tiny_lang_config):
        m = self._build(tiny_lang_config, pre_process=False, post_process=True)
        t = torch.ones(2, 3, 4)
        m.set_input_tensor([t])
        assert torch.equal(m.language_model.input_tensor, t)


# ===================================================================
# 2. forward() gating
# ===================================================================


class TestMultimodalModelForwardGating:

    def _build(
        self,
        tiny_lang_config,
        *,
        pre_process: bool,
        post_process: bool,
    ):
        from examples.multimodal_dev.models.base import MultimodalModel

        # image_token_id=99 keeps the input_ids tensor below (all zeros)
        # free of image-token positions so masked_scatter has zero hits.
        return MultimodalModel(
            language_config=tiny_lang_config,
            language_spec=mock.MagicMock(),
            vision_encoder=_FakeVisionEncoder(),
            vocab_size=16,
            max_sequence_length=32,
            image_token_id=99,
            pre_process=pre_process,
            post_process=post_process,
        )

    def test_forward_first_stage_runs_vision_and_embedding(
        self, fake_gpt, tiny_lang_config, monkeypatch,
    ):
        m = self._build(tiny_lang_config, pre_process=True, post_process=True)

        # Spy on _scatter_vision_embeddings.
        scatter_spy = mock.MagicMock(wraps=m._scatter_vision_embeddings)
        m._scatter_vision_embeddings = scatter_spy

        B, S = 2, 4
        # input_ids has no image tokens (image_token_id=99, ids are 0..S-1),
        # so masked_scatter receives a fully-zero mask.
        input_ids = torch.arange(S).unsqueeze(0).expand(B, S).contiguous()
        m.forward(
            input_ids=input_ids,
            position_ids=None,
            pixel_values=torch.zeros(1, 3, 8, 8),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
        )
        # First-stage embedding was hit.
        assert m.language_model.embedding_calls == 1
        # Scatter ran because pixel_values was non-None.
        assert scatter_spy.called
        # GPTModel was actually called.
        assert m.language_model.last_forward_kwargs is not None
        # decoder_input was built locally (not None).
        assert m.language_model.last_forward_kwargs["decoder_input"] is not None

    def test_forward_non_first_stage_skips_vision_and_embedding(
        self, fake_gpt, tiny_lang_config,
    ):
        m = self._build(tiny_lang_config, pre_process=False, post_process=True)

        B, S = 2, 4
        input_ids = torch.arange(S).unsqueeze(0).expand(B, S).contiguous()
        m.forward(
            input_ids=input_ids,
            position_ids=None,
            # On non-first stages the dataloader still ships
            # image_grid_thw (it's small + needed by compute_position_ids).
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            pixel_values=None,  # forward_step drops this on non-first stages
        )
        # No embedding lookup, no vision encoder execution.
        assert m.language_model.embedding_calls == 0
        # decoder_input must arrive as None so language_model uses
        # whatever was set via set_input_tensor.
        assert m.language_model.last_forward_kwargs["decoder_input"] is None

    def test_forward_computes_position_ids_on_every_stage(
        self, fake_gpt, tiny_lang_config,
    ):
        """MRoPE/RoPE freqs are computed per PP stage from position_ids
        inside GPTModel, so position_ids must be non-None on every stage.
        """
        m = self._build(tiny_lang_config, pre_process=False, post_process=False)
        B, S = 2, 4
        input_ids = torch.arange(S).unsqueeze(0).expand(B, S).contiguous()
        m.forward(
            input_ids=input_ids,
            position_ids=None,
            image_grid_thw=torch.tensor([[1, 4, 4]]),
        )
        kw = m.language_model.last_forward_kwargs
        assert kw["position_ids"] is not None, (
            "position_ids must be computed on non-first stages too"
        )


# ===================================================================
# 3. forward_step gating
# ===================================================================


class TestForwardStepGating:

    def _patch_pipeline_stage(self, monkeypatch, *, is_first: bool, is_last: bool):
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "is_pipeline_first_stage", lambda: is_first)
        monkeypatch.setattr(fs, "is_pipeline_last_stage", lambda: is_last)

    def _patch_get_batch(self, monkeypatch, batch):
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(fs, "get_batch", lambda _it: batch)

    def _patch_cp_size(self, monkeypatch, cp_size: int):
        import examples.multimodal_dev.forward_step as fs

        monkeypatch.setattr(
            fs, "get_context_parallel_world_size", lambda: cp_size,
        )

    def _make_batch(self):
        return {
            "input_ids": torch.zeros(2, 8, dtype=torch.long),
            "position_ids": None,
            "labels": torch.zeros(2, 8, dtype=torch.long),
            "loss_mask": torch.ones(2, 8),
            "pixel_values": torch.zeros(1, 3, 8, 8),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "packed_seq_params": None,
        }

    def test_forward_step_drops_pixel_values_on_non_first_stage(
        self, monkeypatch,
    ):
        from examples.multimodal_dev.forward_step import forward_step

        batch = self._make_batch()
        self._patch_get_batch(monkeypatch, batch)
        self._patch_pipeline_stage(monkeypatch, is_first=False, is_last=True)
        self._patch_cp_size(monkeypatch, cp_size=1)

        captured = {}

        def fake_model(**kw):
            captured.update(kw)
            return torch.zeros(2, 8)

        out, loss_closure = forward_step(iter([]), fake_model)
        assert captured.get("pixel_values") is None, (
            "pixel_values must be None on non-first PP stages"
        )
        # image_grid_thw must still flow through (compute_position_ids).
        assert captured.get("image_grid_thw") is not None

    def test_forward_step_keeps_pixel_values_on_first_stage(
        self, monkeypatch,
    ):
        from examples.multimodal_dev.forward_step import forward_step

        batch = self._make_batch()
        self._patch_get_batch(monkeypatch, batch)
        self._patch_pipeline_stage(monkeypatch, is_first=True, is_last=False)
        self._patch_cp_size(monkeypatch, cp_size=1)

        captured = {}

        def fake_model(**kw):
            captured.update(kw)
            return torch.zeros(2, 8)

        forward_step(iter([]), fake_model)
        assert captured.get("pixel_values") is not None

    def test_loss_mask_cp_split_only_on_last_stage(self, monkeypatch):
        """When CP>1, the loss-mask CP partition must run on the last
        PP stage but not on first/middle stages (loss closure only
        runs there).
        """
        from examples.multimodal_dev.forward_step import forward_step

        batch = self._make_batch()
        self._patch_get_batch(monkeypatch, batch)

        # Stage = first only (not last).
        self._patch_pipeline_stage(monkeypatch, is_first=True, is_last=False)
        self._patch_cp_size(monkeypatch, cp_size=2)

        # Patch the splitter so we can detect calls.
        with mock.patch(
            "examples.multimodal_dev.models.base._cp_split_tensor"
        ) as split_mock:
            split_mock.side_effect = lambda t, **kw: t
            forward_step(iter([]), lambda **kw: torch.zeros(2, 8))
            assert not split_mock.called, (
                "loss_mask CP-split must NOT run on non-last PP stages"
            )

        # Stage = last (the loss closure path).
        self._patch_pipeline_stage(monkeypatch, is_first=True, is_last=True)
        with mock.patch(
            "examples.multimodal_dev.models.base._cp_split_tensor"
        ) as split_mock:
            split_mock.side_effect = lambda t, **kw: t
            forward_step(iter([]), lambda **kw: torch.zeros(2, 8))
            assert split_mock.called, (
                "loss_mask CP-split must run on last PP stage when CP>1"
            )


# ===================================================================
# 4. Factory signature
# ===================================================================


class TestFactorySignatures:
    def test_build_model_accepts_pre_post_process(self):
        from examples.multimodal_dev.models.qwen35_vl.factory import build_model

        sig = inspect.signature(build_model)
        assert "pre_process" in sig.parameters
        assert "post_process" in sig.parameters
        assert sig.parameters["pre_process"].default is True
        assert sig.parameters["post_process"].default is True

    def test_qwen35vl_model_accepts_pre_post_process(self):
        from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel

        sig = inspect.signature(Qwen35VLModel.__init__)
        assert "pre_process" in sig.parameters
        assert "post_process" in sig.parameters
        assert "vp_stage" in sig.parameters

    def test_multimodal_model_accepts_pre_post_process(self):
        from examples.multimodal_dev.models.base import MultimodalModel

        sig = inspect.signature(MultimodalModel.__init__)
        assert "pre_process" in sig.parameters
        assert "post_process" in sig.parameters
        assert "vp_stage" in sig.parameters
