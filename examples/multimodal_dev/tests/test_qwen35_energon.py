# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import io
import pickle
import shlex
import subprocess
import tarfile
import zipfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from examples.multimodal_dev.mdp_image_materialize import (
    decode_image_descriptors,
    encode_image_descriptors,
    materialize_descriptor,
)


def test_b436_energon_packing_flag_remains_parseable():
    from examples.multimodal_dev.arguments import add_multimodal_args

    parser = add_multimodal_args(ArgumentParser())
    args = parser.parse_args(["--dataloader-sequence-packing"])

    assert args.dataloader_sequence_packing is True


class _FakeTokenizer:
    all_special_ids = []

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10 + (index % 17) for index, _char in enumerate(str(text))]


def _pattern_image_bytes(width=64, height=32):
    y, x = np.indices((height, width))
    array = np.stack(
        ((x * 7 + y * 3) % 256, (x * 5 + y * 11) % 256, (x * 13 + y * 2) % 256), axis=-1
    ).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _b436_raw_bytes_reference(image_bytes, grid_thw, patch_size=16):
    """Reference the b436 llava_energon raw-byte materialization path."""
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw)
    temporal_patch_size = 2
    merge_size = 2
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB").resize(
            (grid_w * patch_size, grid_h * patch_size), Image.Resampling.BICUBIC
        )
    array = (np.asarray(image, dtype=np.float32) / 255.0 - mean) / std
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    frames = image_tensor.unsqueeze(0).expand(grid_t * temporal_patch_size, -1, -1, -1).contiguous()
    patches = frames.reshape(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    return patches.reshape(
        grid_t * grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size
    )


def test_image_descriptor_json_roundtrips_bytes():
    descriptors = [
        {"kind": "image_bytes", "image_bytes": b"\x89PNG\r\n", "nested": {"payload": b"abc"}}
    ]

    assert decode_image_descriptors(encode_image_descriptors(descriptors)) == descriptors


def test_raw_jpgs_matches_b436_materializer_values():
    from examples.multimodal_dev.data.qwen35_energon.raw_jpgs import materialize_image_descriptor

    image_bytes = _pattern_image_bytes()
    grid = [1, 2, 4]
    actual = materialize_image_descriptor(
        {"_raw_image_bytes": image_bytes, "spatial_merge_size": 2},
        grid,
        pixel_dim=1536,
        patch_size=16,
    )
    expected = _b436_raw_bytes_reference(image_bytes, grid)

    assert actual.shape == (8, 1536)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_raw_jpgs_reads_prepared_tar_payload(tmp_path):
    from examples.multimodal_dev.data.qwen35_energon.raw_jpgs import _image_bytes_from_tar

    image_bytes = _pattern_image_bytes()
    payload = pickle.dumps([image_bytes])
    tar_path = tmp_path / "prepared.tar"
    with tarfile.open(tar_path, "w") as archive:
        member = tarfile.TarInfo("sample_000000000.jpgs")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    assert (
        _image_bytes_from_tar(
            {"tar_path": str(tar_path), "key": "sample_000000000", "image_idx": 0}
        )
        == image_bytes
    )


def test_raw_jpgs_rejects_pickled_global_objects():
    from examples.multimodal_dev.data.qwen35_energon.raw_jpgs import _load_image_bytes_payload

    with pytest.raises(pickle.UnpicklingError, match="global objects"):
        _load_image_bytes_payload(pickle.dumps(ValueError("not image bytes")))


def test_m4_zip_descriptor_materializes_from_temporary_archive(tmp_path):
    image_bytes = _pattern_image_bytes()
    zip_path = tmp_path / "m4.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nested/image.png", image_bytes)

    patches = materialize_descriptor(
        {
            "kind": "zip_image",
            "zip_path": str(zip_path),
            "path": "image.png",
            "candidates": ["missing.png", "nested/image.png"],
            "materializer": "examples.multimodal_dev.data.m4_instruct",
        },
        [1, 2, 4],
        pixel_dim=1536,
        patch_size=16,
    )

    assert patches.shape == (8, 1536)


def test_mantis_parquet_list_descriptor_materializes_temporary_row(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_path = tmp_path / "mantis.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"images": [{"bytes": _pattern_image_bytes(), "path": "image.png"}]}]
        ),
        parquet_path,
    )

    patches = materialize_descriptor(
        {
            "kind": "parquet_list_image",
            "parquet_path": str(parquet_path),
            "row_idx": 0,
            "column": "images",
            "image_idx": 0,
            "materializer": "examples.multimodal_dev.data.mantis_instruct",
        },
        [1, 2, 4],
        pixel_dim=1536,
        patch_size=16,
    )

    assert patches.shape == (8, 1536)


def test_pixmo_parquet_column_descriptor_materializes_temporary_row(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_path = tmp_path / "pixmo.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"image": {"bytes": _pattern_image_bytes(), "path": "image.png"}}]),
        parquet_path,
    )

    patches = materialize_descriptor(
        {
            "kind": "parquet_column_image",
            "parquet_path": str(parquet_path),
            "row_idx": 0,
            "column": "image",
            "materializer": "examples.multimodal_dev.data.pixmo_docs",
        },
        [1, 2, 4],
        pixel_dim=1536,
        patch_size=16,
    )

    assert patches.shape == (8, 1536)


def _encoder():
    from examples.multimodal_dev.data.qwen35_energon.task_encoder import Qwen35EnergonTaskEncoder

    return Qwen35EnergonTaskEncoder(
        tokenizer=_FakeTokenizer(),
        seq_length=256,
        patch_size=16,
        spatial_merge_size=2,
        image_max_pixels=0,
        image_min_pixels=0,
    )


def test_task_encoder_packs_json_only_lazy_zip_descriptor(tmp_path):
    image_bytes = _pattern_image_bytes()
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("image.png", image_bytes)
    encoder = _encoder()
    encoded = encoder.preencode_sample(
        {
            "json": {
                "text": "user: <image> question\nassistant: answer",
                "image_descriptors": [
                    {
                        "kind": "zip_image",
                        "zip_path": str(zip_path),
                        "path": "image.png",
                        "width": 64,
                        "height": 32,
                        "materializer": "examples.multimodal_dev.data.m4_instruct",
                    }
                ],
            }
        }
    )

    packed = encoder.pack_selected_samples(encoder.select_samples_to_pack([encoded])[0])

    assert packed["image_grid_thw"].tolist() == [[1, 2, 4]]
    assert packed["pixel_values"].shape == (8, 1536)


def test_none_jpg_falls_back_to_multi_image_jpgs_bytes():
    encoder = _encoder()
    image_bytes = [_pattern_image_bytes(), _pattern_image_bytes(32, 32)]
    encoded = encoder.preencode_sample(
        {
            "json": {
                "text": "user: <image><image> question\nassistant: answer",
                "image_descriptors": [{"width": 64, "height": 32}, {"width": 32, "height": 32}],
            },
            "jpg": None,
            "jpgs": pickle.dumps(image_bytes),
        }
    )

    packed = encoder.pack_selected_samples(encoder.select_samples_to_pack([encoded])[0])

    assert packed["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 2]]
    assert packed["pixel_values"].shape == (12, 1536)


def test_task_encoder_accepts_vqa_sample_shape():
    from examples.multimodal_dev.data.qwen35_energon.task_encoder import Qwen35EnergonTaskEncoder

    class FakeVQASample:
        context = "What is shown?"
        answers = "A square."
        image = Image.new("RGB", (64, 64), color="white")

    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=_FakeTokenizer(),
        seq_length=128,
        patch_size=16,
        spatial_merge_size=2,
        image_max_pixels=0,
        image_min_pixels=0,
    )

    encoded = encoder.preencode_sample(FakeVQASample())
    batch = encoder.pack_selected_samples([encoded])

    assert batch["pixel_values"].shape == (16, encoder._pixel_dim)
    assert batch["image_grid_thw"].tolist() == [[1, 4, 4]]
    descriptors = decode_image_descriptors(batch["_mdp_image_descriptors_json"])
    assert len(descriptors) == 1
    assert descriptors[0]["kind"] == "image_bytes"


@pytest.mark.parametrize(("eval_iters", "val_built"), [(0, False), (2, True)])
def test_energon_provider_builds_val_only_when_eval_iters_positive(
    monkeypatch, eval_iters, val_built
):
    from types import SimpleNamespace

    from examples.multimodal_dev.data.qwen35_energon import provider

    calls = {"val": 0}

    def fake_get_val_datasets(*_args, **_kwargs):
        calls["val"] += 1
        return [(object(), None)]

    args = SimpleNamespace(
        energon_path="unused",
        dataloader_type="external",
        micro_batch_size=1,
        energon_packing_buffer_size=1,
        energon_max_samples_per_sequence=1,
        energon_prefetch_factor=1,
        energon_shuffle_buffer_size=1,
        dataset_split="train",
        num_workers=1,
        mdp_encoder_mode=False,
        eval_iters=eval_iters,
    )
    monkeypatch.setattr(provider, "get_args", lambda: args)
    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(provider.parallel_state, "is_initialized", lambda: False)
    monkeypatch.setattr(provider, "_tokenizer", lambda _args: object())
    monkeypatch.setattr(provider, "_task_encoder", lambda *a, **k: object())
    monkeypatch.setattr(provider, "get_train_dataset", lambda *a, **k: object())
    monkeypatch.setattr(provider, "get_savable_loader", lambda *a, **k: [])
    monkeypatch.setattr(provider, "get_val_datasets", fake_get_val_datasets)
    monkeypatch.setattr(provider, "get_loader", lambda *a, **k: [])

    _train, val_loader, _test = provider.train_valid_test_datasets_provider([1, 0, 0])

    assert calls["val"] == (1 if val_built else 0)
    assert (val_loader is not None) == val_built


def test_task_encoder_accepts_llava_decoded_dict_shape():
    from examples.multimodal_dev.data.qwen35_energon.task_encoder import Qwen35EnergonTaskEncoder

    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=_FakeTokenizer(),
        seq_length=128,
        patch_size=16,
        spatial_merge_size=2,
        image_max_pixels=0,
        image_min_pixels=0,
    )
    sample = {
        "jpg": Image.new("RGB", (64, 64), color="white"),
        "json": [
            {"from": "human", "value": "<image>\nWhat is shown?"},
            {"from": "gpt", "value": "A square."},
        ],
    }

    encoded = encoder.preencode_sample(sample)
    batch = encoder.pack_selected_samples([encoded])

    assert batch["pixel_values"].shape == (16, encoder._pixel_dim)
    assert batch["image_grid_thw"].tolist() == [[1, 4, 4]]
    descriptors = decode_image_descriptors(batch["_mdp_image_descriptors_json"])
    assert len(descriptors) == 1
    assert descriptors[0]["kind"] == "image_bytes"


def test_task_encoder_prepartitions_vqa_sample_shape():
    from examples.multimodal_dev.data.qwen35_energon.task_encoder import Qwen35EnergonTaskEncoder

    class FakeVQASample:
        context = "What is shown?"
        answers = "A square."
        image = Image.new("RGB", (64, 64), color="white")

    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=_FakeTokenizer(),
        seq_length=128,
        patch_size=16,
        spatial_merge_size=2,
        image_max_pixels=0,
        image_min_pixels=0,
        mdp_loader_prepartition=True,
        mdp_loader_prepartition_materialize=False,
    )

    encoded = encoder.preencode_sample(FakeVQASample())
    batch = encoder.pack_selected_samples([encoded])

    assert batch["pixel_values"].shape == (0, encoder._pixel_dim)
    assert batch["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert len(batch["_mdp_image_descriptors"]) == 1
    assert batch["_mdp_image_descriptors"][0]["kind"] == "image_bytes"


def test_task_encoder_prepartitions_llava_decoded_dict_shape():
    from examples.multimodal_dev.data.qwen35_energon.task_encoder import Qwen35EnergonTaskEncoder

    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=_FakeTokenizer(),
        seq_length=128,
        patch_size=16,
        spatial_merge_size=2,
        image_max_pixels=0,
        image_min_pixels=0,
        mdp_loader_prepartition=True,
        mdp_loader_prepartition_materialize=False,
    )
    sample = {
        "jpg": Image.new("RGB", (64, 64), color="white"),
        "json": [
            {"from": "human", "value": "<image>\nWhat is shown?"},
            {"from": "gpt", "value": "A square."},
        ],
    }

    encoded = encoder.preencode_sample(sample)
    batch = encoder.pack_selected_samples([encoded])

    assert batch["pixel_values"].shape == (0, encoder._pixel_dim)
    assert batch["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert len(batch["_mdp_image_descriptors"]) == 1
    assert batch["_mdp_image_descriptors"][0]["kind"] == "image_bytes"


def test_qwen3vl_launcher_matches_job241536_contract(tmp_path):
    from examples.multimodal_dev.arguments import add_multimodal_args
    from examples.multimodal_dev.models import MODEL_REGISTRY

    assert "energon" in MODEL_REGISTRY["qwen3vl"]["dataset_providers"]
    # Keep fused packs within the GB200 blend3 cap validated at world 64.
    parsed = add_multimodal_args(ArgumentParser()).parse_args(
        [
            "--dataset-provider",
            "energon",
            "--energon-path",
            "/workspace/reference/blend3.yaml",
            "--dataloader-sequence-packing",
            "--mdp-encoder-mode",
            "--mdp-inner-dp-scope",
            "pp_cp",
            "--mdp-loader-prepartition-prefetch-windows",
            "1",
            "--mdp-fused-vision-window",
            "--mdp-vision-encoder-max-sequence-length",
            "131072",
            "--mdp-fused-vision-backward",
            "retain",
        ]
    )
    assert parsed.dataset_provider == "energon"
    assert parsed.energon_path == "/workspace/reference/blend3.yaml"
    assert parsed.dataloader_sequence_packing is True
    assert parsed.mdp_encoder_mode is True
    assert parsed.mdp_inner_dp_scope == "pp_cp"
    assert parsed.mdp_loader_prepartition_prefetch_windows == 1
    assert parsed.mdp_fused_vision_window is True
    assert parsed.mdp_vision_encoder_max_sequence_length == 131072
    assert parsed.mdp_fused_vision_backward == "retain"

    script = Path(__file__).parents[1] / "scripts" / "dev_qwen3vl_gb200.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--gpus",
            "4",
            "--nnodes",
            "32",
            "--train-iters",
            "45",
            "--warmup-iters",
            "5",
            "--results-dir",
            str(tmp_path),
            "job241536",
            "tp=1",
            "ep=8",
            "pp=2",
            "cp=2",
            "etp=1",
            "vpp=0",
            "mbs=1",
            "gbs=512",
            "seq_len=8192",
            "image_size=448",
            "dispatcher_backend=hybridep",
            "a2a_overlap=0",
            "recompute=0",
            "recompute_vision=0",
            "mtp=0",
            "use_packed_sequence=1",
            "dataset_provider=energon",
            (
                "extra_args=--dataloader-type external "
                "--energon-path /workspace/reference/blend3.yaml "
                "--image-min-pixels 0 --image-max-pixels 327680 "
                "--energon-packing-buffer-size 128 "
                "--energon-shuffle-buffer-size 128 "
                "--energon-max-samples-per-sequence 16 "
                "--energon-prefetch-factor 1 --num-workers 1 "
                "--dataloader-sequence-packing --mdp-encoder-mode "
                "--mdp-inner-dp-scope pp_cp "
                "--mdp-loader-prepartition-prefetch-windows 1 "
                "--mdp-fused-vision-window "
                "--mdp-vision-encoder-max-sequence-length 131072 "
                "--mdp-fused-vision-backward retain --eval-iters 0"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    command = next(line for line in result.stdout.splitlines() if line.startswith("CMD: "))
    argv = shlex.split(command.removeprefix("CMD: "))

    def last_value(flag):
        positions = [index for index, value in enumerate(argv) if value == flag]
        assert positions, flag
        return argv[positions[-1] + 1]

    expected = {
        "--model-arch": "qwen3vl",
        "--tensor-model-parallel-size": "1",
        "--pipeline-model-parallel-size": "2",
        "--context-parallel-size": "2",
        "--expert-model-parallel-size": "8",
        "--expert-tensor-parallel-size": "1",
        "--micro-batch-size": "1",
        "--global-batch-size": "512",
        "--train-iters": "50",
        "--seq-length": "8192",
        "--dataset-provider": "energon",
        "--energon-path": "/workspace/reference/blend3.yaml",
        "--image-max-pixels": "327680",
        "--energon-packing-buffer-size": "128",
        "--energon-shuffle-buffer-size": "128",
        "--energon-max-samples-per-sequence": "16",
        "--energon-prefetch-factor": "1",
        "--mdp-inner-dp-scope": "pp_cp",
        "--mdp-loader-prepartition-prefetch-windows": "1",
        "--mdp-vision-encoder-max-sequence-length": "131072",
        "--mdp-fused-vision-backward": "retain",
        "--moe-flex-dispatcher-backend": "hybridep",
        "--eval-iters": "0",
    }
    for flag, value in expected.items():
        assert last_value(flag) == value
    for flag in (
        "--calculate-per-token-loss",
        "--dataloader-sequence-packing",
        "--mdp-encoder-mode",
        "--mdp-fused-vision-window",
        "--use-distributed-optimizer",
        "--use-packed-sequence",
    ):
        assert flag in argv
    for flag in (
        "--delay-wgrad-compute",
        "--overlap-moe-expert-parallel-comm",
        "--recompute-vision",
        "--use-megatron-fsdp",
    ):
        assert flag not in argv
