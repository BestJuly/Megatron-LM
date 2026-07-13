# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


def test_null_multimodal_tokenizer_preserves_qwen_image_token_id():
    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

    args = SimpleNamespace(
        tokenizer_type="NullMultimodalTokenizer",
        vocab_size=248320,
        image_token_id=248056,
        padded_vocab_size=None,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        rank=1,
    )

    tokenizer = build_tokenizer(args)

    assert tokenizer.convert_tokens_to_ids("<image>") == 248056
    assert tokenizer.unique_identifiers["image_token_id"] == "248056"


def test_registry_adds_qwen3_paths_without_replacing_existing_models():
    from examples.multimodal_dev.models import MODEL_REGISTRY

    assert set(MODEL_REGISTRY) == {"kimi_k25", "qwen3", "qwen35_vl", "qwen3vl"}
    assert set(MODEL_REGISTRY["qwen35_vl"]["dataset_providers"]) == {"cord_v2", "mock"}
    assert set(MODEL_REGISTRY["kimi_k25"]["dataset_providers"]) == {"mock"}
    assert set(MODEL_REGISTRY["qwen3vl"]["dataset_providers"]) == {"mock"}
    assert MODEL_REGISTRY["qwen3"]["text_only"] is True


def test_model_provider_marks_qwen3_text_only_and_forwards_stage_flags(monkeypatch):
    from examples.multimodal_dev import pretrain_multimodal
    from examples.multimodal_dev.models import MODEL_REGISTRY

    args = SimpleNamespace(
        model_arch="qwen3",
        model_variant=None,
        vision_num_layers=None,
        recompute_vision=False,
    )
    language_config = SimpleNamespace(bf16=True, fp16=False)
    vision_config = SimpleNamespace()
    captured = {}
    expected = SimpleNamespace()
    entry = MODEL_REGISTRY["qwen3"]

    def build_model(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setitem(
        entry,
        "model_factory_fn",
        build_model,
    )
    monkeypatch.setitem(
        entry,
        "vision_config_fn",
        lambda **_kwargs: vision_config,
    )
    monkeypatch.setitem(entry, "post_language_config_fn", None)
    monkeypatch.setitem(entry, "vision_flops_fn", None)
    monkeypatch.setattr(pretrain_multimodal, "get_args", lambda: args)
    monkeypatch.setattr(
        pretrain_multimodal,
        "core_transformer_config_from_args",
        lambda _args: language_config,
    )

    result = pretrain_multimodal.model_provider(
        pre_process=False,
        post_process=True,
    )

    assert result is expected
    assert args.text_only is True
    assert captured["pre_process"] is False
    assert captured["post_process"] is True


def test_qwen3_factory_forwards_pipeline_stage_flags(monkeypatch):
    from examples.multimodal_dev.models.qwen3 import factory

    captured = {}
    monkeypatch.setattr(
        factory,
        "get_qwen3_language_spec",
        lambda **_kwargs: object(),
    )

    def build_model(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        factory,
        "Qwen3TextOnlyGPTModel",
        build_model,
    )
    args = SimpleNamespace(
        max_position_embeddings=40960,
        mtp_num_layers=None,
        padded_vocab_size=151936,
        rotary_percent=1.0,
        transformer_impl="transformer_engine",
    )

    factory.build_model(
        args,
        SimpleNamespace(),
        SimpleNamespace(),
        pre_process=False,
        post_process=True,
    )

    assert captured["pre_process"] is False
    assert captured["post_process"] is True


def test_qwen3_text_only_mock_contract():
    from examples.multimodal_dev.arguments import add_multimodal_args
    from examples.multimodal_dev.data.mock import MockQwen35VLDataset

    parser = add_multimodal_args(argparse.ArgumentParser())
    assert parser.parse_args(["--text-only"]).text_only is True

    dataset = MockQwen35VLDataset(
        num_samples=1,
        seq_length=8,
        vocab_size=32,
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
        text_only=True,
    )
    sample = dataset[0]

    assert sample["position_ids"].tolist() == list(range(8))
    assert sample["pixel_values"].shape[0] == 0
    assert sample["image_grid_thw"].shape == (0, 3)
    assert sample["cu_seqlens"].tolist() == [0, 8]
    assert torch.equal(sample["tokens"], sample["input_ids"])
    assert not {1, 2, 3}.intersection(sample["input_ids"].tolist())


def test_qwen3vl_uses_qwen35_vision_and_qwen3_language_settings():
    from examples.multimodal_dev.models import MODEL_REGISTRY

    entry = MODEL_REGISTRY["qwen3vl"]
    vision_config = entry["vision_config_fn"](num_layers_override=27, variant="proxy")
    language_config = SimpleNamespace(
        linear_attention_freq=[2, 2, 2], mrope_section=None, mrope_interleaved=False
    )

    entry["post_language_config_fn"](language_config, SimpleNamespace())

    assert vision_config.num_layers == 27
    assert vision_config.hidden_size == 1152
    assert vision_config.num_attention_heads == 16
    assert language_config.linear_attention_freq is None
    assert language_config.mrope_section == [11, 11, 10]
    assert language_config.mrope_interleaved is True


def test_qwen3_decoder_uses_gpt_block_spec():
    from examples.multimodal_dev.models.qwen3 import specs

    config = SimpleNamespace(num_layers=48, linear_attention_freq=None)
    sentinel = object()
    with mock.patch.object(
        specs, "get_gpt_decoder_block_spec", return_value=sentinel
    ) as build_standard_spec:
        result = specs.get_qwen3_language_spec(config, vp_stage=None)

    assert result is sentinel
    build_standard_spec.assert_called_once_with(
        config=config, use_transformer_engine=True, vp_stage=None
    )


def test_qwen3vl_factory_preserves_b436_model_shape_and_qwen35_wrapper():
    from examples.multimodal_dev.models.qwen3vl import factory

    language_config = SimpleNamespace(num_layers=48, linear_attention_freq=None)
    vision_config = SimpleNamespace(num_layers=27, hidden_size=1152)
    args = SimpleNamespace(
        image_token_id=248056,
        max_position_embeddings=40960,
        mtp_num_layers=None,
        padded_vocab_size=248320,
        rotary_percent=0.5,
        transformer_impl="transformer_engine",
    )
    language_spec = object()
    captured = {}

    def capture_model(**kwargs):
        captured.update(kwargs)
        return kwargs

    with (
        mock.patch.object(
            factory, "get_qwen3_language_spec", return_value=language_spec
        ) as get_language_spec,
        mock.patch.object(factory, "Qwen35VLModel", side_effect=capture_model),
    ):
        result = factory.build_model(args, language_config, vision_config)

    get_language_spec.assert_called_once_with(config=language_config, vp_stage=None, pp_rank=None)
    assert result["language_spec"] is language_spec
    assert captured["vision_config"] is vision_config
    assert captured["rotary_percent"] == 0.5
    assert captured["vocab_size"] == 248320
    assert captured["max_sequence_length"] == 40960


def test_qwen35_wrapper_keeps_default_rotary_and_accepts_qwen3vl_override():
    from examples.multimodal_dev.models.qwen35_vl.configuration import ROTARY_PERCENT
    from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel

    parameter = inspect.signature(Qwen35VLModel.__init__).parameters["rotary_percent"]
    assert parameter.default == ROTARY_PERCENT


def test_qwen3_launcher_enforces_text_only_mock_contract(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "dev_qwen3_gb200.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--gpus",
            "1",
            "--nnodes",
            "1",
            "--results-dir",
            str(tmp_path),
            "baseline",
            "ep=1",
            "dispatcher_backend=alltoall",
            "calculate_per_token_loss=1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    command = next(
        line for line in result.stdout.splitlines() if line.startswith("CMD: ")
    )
    assert "--model-arch qwen3" in command
    assert "--dataset-provider mock" in command
    assert "--text-only" in command
    assert "--calculate-per-token-loss" in command
    assert "--num-virtual-stages-per-pipeline-rank" not in command


def test_qwen35_patch_projection_fast_path_matches_conv3d():
    from examples.multimodal_dev.models.qwen35_vl import vision_encoder
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
    )
    module = vision_encoder.Qwen35VLPatchEmbed(
        config=config,
        in_channels=1,
        hidden_size=4,
        patch_size=2,
        temporal_patch_size=1,
    )
    pixels = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = module.proj(pixels.reshape(3, 1, 1, 2, 2)).flatten(1)

    with (
        mock.patch.object(vision_encoder, "_using_megatron_fsdp", return_value=False),
        mock.patch.object(vision_encoder, "_has_partial_storage", return_value=False),
    ):
        actual = module(pixels)

    torch.testing.assert_close(actual, expected)


def test_qwen35_vision_rope_exact_token_path_avoids_generic_thd_sync():
    from examples.multimodal_dev.models.qwen35_vl import specs
    from megatron.core.models.common.embeddings import rope_utils

    tensor = torch.arange(24, dtype=torch.bfloat16).reshape(3, 2, 4)
    frequencies = torch.ones(3, 1, 1, 4, dtype=torch.float32)
    config = SimpleNamespace(
        rotary_interleaved=False,
        multi_latent_attention=False,
    )

    with (
        mock.patch.object(
            rope_utils,
            "_apply_rotary_pos_emb_bshd",
            side_effect=lambda value, *_args, **_kwargs: value,
        ) as bshd,
        mock.patch.object(rope_utils, "_apply_rotary_pos_emb_thd") as thd,
    ):
        output = specs._apply_rope_fp32(
            tensor,
            frequencies,
            config,
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )

    assert output.shape == tensor.shape
    assert output.dtype == tensor.dtype
    bshd.assert_called_once()
    thd.assert_not_called()


def test_qwen3vl_launcher_dry_run_is_mock_only_and_matches_b436_shape(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "dev_qwen3vl_gb200.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--gpus",
            "1",
            "--nnodes",
            "1",
            "--results-dir",
            str(tmp_path),
            "baseline",
            "ep=1",
            "dispatcher_backend=alltoall",
            "calculate_per_token_loss=1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    command = next(line for line in result.stdout.splitlines() if line.startswith("CMD: "))
    expected = (
        "--model-arch qwen3vl",
        "--dataset-provider mock",
        "--num-layers 48",
        "--hidden-size 2048",
        "--num-attention-heads 32",
        "--num-query-groups 4",
        "--kv-channels 128",
        "--num-experts 128",
        "--rotary-percent 0.5",
        "--vision-num-layers 27",
        "--moe-router-load-balancing-type aux_loss",
    )
    for option in expected:
        assert option in command
    assert "--calculate-per-token-loss" in command
    assert "gated_delta_net" not in command
    assert "--dataset-root" not in command
    assert "--energon-path" not in command
    assert "WANDB_API_KEY" not in script.read_text()


def test_qwen3_packed_cp_shards_text_fields_but_not_vision_payload():
    from examples.multimodal_dev import forward_step
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=4,
    )
    pixels = torch.arange(8).reshape(2, 4)
    grid = torch.tensor([[1, 1, 2]])
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13]]),
        "labels": torch.tensor([[20, 21, 22, 23]]),
        "loss_mask": torch.ones(1, 4),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
        "pixel_values": pixels,
        "image_grid_thw": grid,
        "packed_seq_params": packed_seq_params,
    }
    local_indices = torch.tensor([0, 3])
    returned_params = object()

    def shard_text(sequence_batch, cu, cu_padded, max_seqlen, **kwargs):
        assert set(sequence_batch) == {"tokens", "labels", "loss_mask"}
        assert sequence_batch["tokens"].shape == (1, 4)
        assert torch.equal(cu, cu_seqlens)
        assert torch.equal(cu_padded, cu_seqlens)
        assert max_seqlen.tolist() == [4]
        assert kwargs == {"cp_size": 2, "cp_rank": 0}
        return (
            {
                key: value.index_select(1, local_indices)
                for key, value in sequence_batch.items()
            },
            returned_params,
        )

    with (
        mock.patch.object(
            forward_step,
            "get_thd_batch_on_this_cp_rank",
            side_effect=shard_text,
        ),
        mock.patch.object(
            forward_step,
            "get_thd_partitioned_indices",
            return_value=local_indices,
        ) as get_indices,
    ):
        result = forward_step._shard_qwen3_packed_batch_for_cp(
            batch, cp_size=2, cp_rank=0,
        )

    assert result["input_ids"].tolist() == [[10, 13]]
    assert result["labels"].tolist() == [[20, 23]]
    assert result["loss_mask"].shape == (1, 2)
    assert result["position_ids"].tolist() == [[0, 3]]
    assert result["pixel_values"] is pixels
    assert result["image_grid_thw"] is grid
    assert result["packed_seq_params"] is returned_params
    assert result["_data_side_cp_sharded"] is True
    get_indices.assert_called_once_with(cu_seqlens, 4, 2, 0)


def test_forward_step_schedule_plan_uses_local_qwen3_loss_mask_once():
    from examples.multimodal_dev import forward_step

    plan = object()
    captured = {}

    class ScheduleModel:
        def __call__(self, **kwargs):
            raise AssertionError("regular forward must not run in schedule-plan mode")

        def build_schedule_plan(self, **kwargs):
            captured.update(kwargs)
            return plan

    batch = {
        "input_ids": torch.tensor([[10, 13]]),
        "labels": torch.tensor([[20, 23]]),
        "loss_mask": torch.ones(1, 2),
        "pixel_values": torch.ones(1, 4, dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
        "packed_seq_params": object(),
        "_data_side_cp_sharded": True,
    }
    with (
        mock.patch.object(forward_step, "get_batch", return_value=batch),
        mock.patch.object(
            forward_step,
            "get_args",
            return_value=SimpleNamespace(
                overlap_moe_expert_parallel_comm=True,
                mdp_encoder_mode=False,
                model_arch="qwen3",
            ),
        ),
        mock.patch.object(
            forward_step,
            "get_context_parallel_world_size",
            return_value=2,
        ),
    ):
        output, calculate_loss = forward_step.forward_step(
            iter(()), ScheduleModel(), return_schedule_plan=True,
        )

    assert output is plan
    assert captured["input_ids"].shape == (1, 2)
    assert captured["labels"].shape == (1, 2)
    assert captured["loss_mask"].shape == (1, 2)
    assert captured["pixel_values"].dtype == torch.bfloat16
    total_loss, total_tokens, _ = calculate_loss(torch.tensor([[2.0, 3.0]]))
    assert total_loss.item() == 5.0
    assert total_tokens.item() == 2


def test_forward_step_rejects_qwen_vl_packed_cp_schedule_plan():
    from examples.multimodal_dev import forward_step

    class ScheduleModel:
        def build_schedule_plan(self, **_kwargs):
            raise AssertionError("unsafe Qwen VL schedule plan must not build")

    batch = {
        "input_ids": torch.tensor([[10, 11]]),
        "labels": torch.tensor([[20, 21]]),
        "loss_mask": torch.ones(1, 2),
        "packed_seq_params": object(),
    }
    with (
        mock.patch.object(forward_step, "get_batch", return_value=batch),
        mock.patch.object(
            forward_step,
            "get_args",
            return_value=SimpleNamespace(
                overlap_moe_expert_parallel_comm=True,
                mdp_encoder_mode=False,
                model_arch="qwen3vl",
            ),
        ),
        mock.patch.object(
            forward_step,
            "get_context_parallel_world_size",
            return_value=2,
        ),
        pytest.raises(
            RuntimeError,
            match="Qwen VL packed THD with context parallelism",
        ),
    ):
        forward_step.forward_step(
            iter(()), ScheduleModel(), return_schedule_plan=True,
        )


def test_forward_step_default_still_calls_regular_model_path():
    from examples.multimodal_dev import forward_step

    expected = torch.tensor([[2.0, 3.0]])
    captured = {}

    class RegularModel:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return expected

        def build_schedule_plan(self, **kwargs):
            raise AssertionError("schedule plan must remain opt-in")

    batch = {
        "input_ids": torch.tensor([[10, 11]]),
        "labels": torch.tensor([[20, 21]]),
        "loss_mask": torch.ones(1, 2),
    }
    with (
        mock.patch.object(forward_step, "get_batch", return_value=batch),
        mock.patch.object(
            forward_step,
            "get_args",
            return_value=SimpleNamespace(mdp_encoder_mode=False),
        ),
        mock.patch.object(
            forward_step,
            "get_context_parallel_world_size",
            return_value=1,
        ),
    ):
        output, _ = forward_step.forward_step(iter(()), RegularModel())

    assert output is expected
    assert captured["input_ids"].shape == (1, 2)
    assert captured["pixel_values"] is None


def test_multimodal_schedule_plan_prepares_vision_then_cp_local_gpt_inputs():
    from examples.multimodal_dev.models import base
    from megatron.core.packed_seq_params import PackedSeqParams

    plan = object()
    captured = {}

    class Vision(torch.nn.Module):
        def forward(self, pixel_values, image_grid_thw):
            assert pixel_values.shape == (1, 4)
            assert image_grid_thw.tolist() == [[1, 1, 1]]
            return torch.tensor([[90.0, 91.0]])

    class Language(torch.nn.Module):
        pre_process = True

        def embedding(self, input_ids, position_ids):
            assert position_ids is None
            values = input_ids.float().unsqueeze(-1)
            return torch.cat((values, values + 0.5), dim=-1).transpose(0, 1)

        def build_schedule_plan(self, **kwargs):
            captured.update(kwargs)
            return plan

    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False)
    model.image_token_id = 99
    model.vision_model = Vision()
    model.language_model = Language()

    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    local_indices = torch.tensor([1, 2])
    input_ids = torch.tensor([[10, 99, 12, 13]])
    labels = torch.tensor([[20, 21, 22, 23]])
    loss_mask = torch.ones(1, 4)

    with (
        mock.patch.object(
            base.parallel_state,
            "get_context_parallel_world_size",
            return_value=2,
        ),
        mock.patch.object(
            base.parallel_state,
            "get_context_parallel_rank",
            return_value=1,
        ),
        mock.patch.object(
            base,
            "_thd_cp_partition_index",
            return_value=local_indices,
        ),
    ):
        result = model.build_schedule_plan(
            input_ids=input_ids,
            position_ids=None,
            labels=labels,
            loss_mask=loss_mask,
            pixel_values=torch.ones(1, 4),
            image_grid_thw=torch.tensor([[1, 1, 1]]),
            packed_seq_params=packed_seq_params,
        )

    assert result is plan
    assert captured["input_ids"].tolist() == [[99, 12]]
    assert captured["labels"].tolist() == [[21, 22]]
    assert captured["loss_mask"].shape == (1, 2)
    assert captured["decoder_input"].shape == (2, 1, 2)
    assert captured["position_ids"].tolist() == [[0, 1, 2, 3]]
    torch.testing.assert_close(
        captured["decoder_input"][:, 0],
        torch.tensor([[90.0, 91.0], [12.0, 12.5]]),
    )


def test_multimodal_schedule_plan_preserves_downstream_pipeline_input():
    from examples.multimodal_dev.models import base

    plan = object()
    captured = {}

    class DownstreamLanguage(torch.nn.Module):
        pre_process = False

        def embedding(self, **kwargs):
            raise AssertionError("downstream pipeline stage must not embed tokens")

        def build_schedule_plan(self, **kwargs):
            captured.update(kwargs)
            return plan

    class DownstreamVision(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("downstream pipeline stage must not run vision")

    model = base.MultimodalModel.__new__(base.MultimodalModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False)
    model.pre_process = False
    model.image_token_id = 99
    model.vision_model = DownstreamVision()
    model.language_model = DownstreamLanguage()

    with mock.patch.object(
        base.parallel_state,
        "get_context_parallel_world_size",
        return_value=1,
    ):
        result = model.build_schedule_plan(
            input_ids=torch.tensor([[10, 11]]),
            position_ids=torch.tensor([[0, 1]]),
            labels=torch.tensor([[20, 21]]),
            loss_mask=torch.ones(1, 2),
            pixel_values=torch.ones(1, 4),
            image_grid_thw=torch.tensor([[1, 1, 1]]),
        )

    assert result is plan
    assert captured["decoder_input"] is None
    assert captured["input_ids"].tolist() == [[10, 11]]
