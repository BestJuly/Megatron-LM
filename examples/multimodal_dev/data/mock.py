# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Fixed-shape mock data for multimodal_dev end-to-end testing.

Image-bearing samples support grouped and interleaved multi-image layouts,
rectangular images, per-image sizes, and a fixed number of packed documents.
Text-only samples keep the same batch keys while emitting empty vision
tensors.
"""

import re

import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.mdp_image_materialize import encode_image_descriptors
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index


def _mock_descriptors_from_grid_rows(rows, *, pixel_dim: int, seed_base: int = 0):
    """Build deterministic lazy descriptors for mock image grids."""
    return [
        {
            "kind": "mock_grid",
            "materializer": "examples.multimodal_dev.data.mock",
            "grid_thw": [int(t), int(h), int(w)],
            "pixel_dim": int(pixel_dim),
            "seed": int(seed_base) + idx,
        }
        for idx, (t, h, w) in enumerate(rows)
    ]


def materialize_image_descriptor(
    desc,
    grid_thw,
    *,
    pixel_dim: int,
    patch_size: int,
):
    """Materialize one deterministic mock image descriptor."""
    del patch_size
    if desc.get("kind") != "mock_grid":
        raise ValueError(
            f"mock materializer cannot handle descriptor: {desc!r}"
        )
    t, h_patches, w_patches = [int(value) for value in grid_thw]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(desc.get("seed", 0)))
    return (
        torch.rand(
            (t * h_patches * w_patches, int(pixel_dim)),
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(2.0)
        .sub_(1.0)
    )


class MockQwen35VLDataset(Dataset):
    """Synthetic Qwen3.5-VL training samples with fixed mock shapes."""

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 1024,
        image_seq_length: int = None,
        vocab_size: int = 248320,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        image_size: int = 224,
        image_size_w: int = None,
        image_sizes_h: list = None,
        image_sizes_w: list = None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        num_images: int = 1,
        layout: str = "single",
        pack_num_docs: int = 1,
        text_only: bool = False,
        metadata_only_batch: bool = False,
    ):
        if num_images < 1:
            raise ValueError(f"num_images must be >= 1, got {num_images}")
        if layout not in ("single", "interleaved"):
            raise ValueError(
                f"layout must be 'single' or 'interleaved', got {layout!r}"
            )
        if pack_num_docs < 1:
            raise ValueError(
                f"pack_num_docs must be >= 1, got {pack_num_docs}"
            )
        if pack_num_docs > 1 and seq_length % pack_num_docs != 0:
            raise ValueError(
                f"seq_length ({seq_length}) must be divisible by "
                f"pack_num_docs ({pack_num_docs})"
            )
        if pack_num_docs > 1 and layout != "single":
            raise ValueError(
                "pack_num_docs > 1 requires layout='single', "
                f"got {layout!r}"
            )

        if image_size_w is None:
            image_size_w = image_size
        if image_sizes_h is None:
            image_sizes_h = [image_size] * num_images
        if image_sizes_w is None:
            image_sizes_w = [image_size_w] * num_images
        if len(image_sizes_h) != num_images:
            raise ValueError(
                f"image_sizes_h length ({len(image_sizes_h)}) must equal "
                f"num_images ({num_images})"
            )
        if len(image_sizes_w) != num_images:
            raise ValueError(
                f"image_sizes_w length ({len(image_sizes_w)}) must equal "
                f"num_images ({num_images})"
            )

        for index, (height, width) in enumerate(
            zip(image_sizes_h, image_sizes_w)
        ):
            if height % patch_size != 0:
                raise ValueError(
                    f"image_sizes_h[{index}]={height} must be divisible by "
                    f"patch_size ({patch_size})"
                )
            if width % patch_size != 0:
                raise ValueError(
                    f"image_sizes_w[{index}]={width} must be divisible by "
                    f"patch_size ({patch_size})"
                )
            if (height // patch_size) % spatial_merge_size != 0:
                raise ValueError(
                    f"image_sizes_h[{index}]/patch_size "
                    f"({height // patch_size}) must be divisible by "
                    f"spatial_merge_size ({spatial_merge_size})"
                )
            if (width // patch_size) % spatial_merge_size != 0:
                raise ValueError(
                    f"image_sizes_w[{index}]/patch_size "
                    f"({width // patch_size}) must be divisible by "
                    f"spatial_merge_size ({spatial_merge_size})"
                )

        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.image_size = image_size
        self.image_size_w = image_size_w
        self.image_sizes_h = list(image_sizes_h)
        self.image_sizes_w = list(image_sizes_w)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.num_images = num_images
        self.layout = layout
        self.pack_num_docs = pack_num_docs
        self.doc_length = seq_length // pack_num_docs
        self.text_only = bool(text_only)
        self.metadata_only_batch = bool(metadata_only_batch)

        h_patches = [height // patch_size for height in self.image_sizes_h]
        w_patches = [width // patch_size for width in self.image_sizes_w]
        t_patches = temporal_patch_size
        self.grid_thw = torch.tensor(
            [
                [t_patches, height, width]
                for height, width in zip(h_patches, w_patches)
            ],
            dtype=torch.long,
        )

        self.tokens_per_image_list = [
            t_patches
            * (height // spatial_merge_size)
            * (width // spatial_merge_size)
            for height, width in zip(h_patches, w_patches)
        ]
        self.tokens_per_image = self.tokens_per_image_list[0]
        self.total_image_tokens = sum(self.tokens_per_image_list)
        if (
            image_seq_length is not None
            and len(set(self.tokens_per_image_list)) == 1
            and image_seq_length != self.tokens_per_image
        ):
            raise ValueError(
                f"image_seq_length={image_seq_length} disagrees with the "
                f"derived per-image token count {self.tokens_per_image}"
            )
        # Backward-compatible aliases used by existing callers and tests.
        self.image_seq_length = self.tokens_per_image
        self.num_merged_tokens = self.tokens_per_image

        self.patches_per_image_list = [
            t_patches * height * width
            for height, width in zip(h_patches, w_patches)
        ]
        self.total_patches_per_image = self.patches_per_image_list[0]
        self.total_patches = sum(self.patches_per_image_list)
        self.image_block_cost = sum(
            1 + tokens for tokens in self.tokens_per_image_list
        )
        self._pixel_dim = (
            3 * temporal_patch_size * patch_size * patch_size
        )

        if not self.text_only and self.image_block_cost >= self.doc_length:
            raise ValueError(
                f"image_block_cost ({self.image_block_cost}) must be smaller "
                f"than the per-document budget ({self.doc_length}); "
                f"seq_length={seq_length}, pack_num_docs={pack_num_docs}, "
                f"num_images={num_images}"
            )

    def __len__(self):
        return self.num_samples

    def _build_text(self, text_length: int) -> torch.Tensor:
        special_ids = {
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
        }
        fallback_id = next(
            (
                token_id
                for token_id in range(1, self.vocab_size)
                if token_id not in special_ids
            ),
            None,
        )
        if fallback_id is None:
            raise ValueError("vocab_size has no non-vision token ID")
        text_tokens = torch.randint(
            1,
            self.vocab_size,
            (text_length,),
            dtype=torch.long,
        )
        for special_id in special_ids:
            text_tokens[text_tokens == special_id] = fallback_id
        return text_tokens

    def _build_targets(self, input_ids):
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0
        loss_mask = (input_ids != self.image_token_id).float()
        loss_mask[-1] = 0
        return labels, loss_mask

    def _build_image_block(self, image_index: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.tensor(
                    [self.vision_start_token_id],
                    dtype=torch.long,
                ),
                torch.full(
                    (self.tokens_per_image_list[image_index],),
                    self.image_token_id,
                    dtype=torch.long,
                ),
            ]
        )

    def _pixel_values(self, total_patches: int) -> torch.Tensor:
        if self.metadata_only_batch:
            return torch.zeros(0, self._pixel_dim)
        return torch.randn(total_patches, self._pixel_dim)

    def _attach_descriptors(self, sample, image_grid_thw):
        if not self.metadata_only_batch:
            return sample
        rows = [
            (int(row[0]), int(row[1]), int(row[2]))
            for row in image_grid_thw.tolist()
        ]
        sample["_mdp_image_descriptors_json"] = encode_image_descriptors(
            _mock_descriptors_from_grid_rows(
                rows,
                pixel_dim=self._pixel_dim,
            )
        )
        return sample

    def _base_sample(
        self,
        input_ids,
        position_ids,
        pixel_values,
        image_grid_thw,
        cu_seqlens,
        max_seqlen,
    ):
        labels, loss_mask = self._build_targets(input_ids)
        return {
            "input_ids": input_ids,
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_padded": cu_seqlens.clone(),
            "max_seqlen": torch.tensor(max_seqlen, dtype=torch.int32),
        }

    def __getitem__(self, idx):
        del idx
        if self.text_only:
            input_ids = self._build_text(self.seq_length)
            cu_seqlens = torch.tensor(
                [0, self.seq_length],
                dtype=torch.int32,
            )
            return self._base_sample(
                input_ids=input_ids,
                position_ids=torch.arange(
                    self.seq_length,
                    dtype=torch.long,
                ),
                pixel_values=torch.empty(0, self._pixel_dim),
                image_grid_thw=torch.empty(0, 3, dtype=torch.long),
                cu_seqlens=cu_seqlens,
                max_seqlen=self.seq_length,
            )

        if self.pack_num_docs > 1:
            return self._build_packed_sample()

        text_length = self.seq_length - self.image_block_cost
        text_tokens = self._build_text(text_length)
        if self.layout == "single":
            prefix_length = text_length // 2
            parts = [text_tokens[:prefix_length]]
            parts.extend(
                self._build_image_block(image_index)
                for image_index in range(self.num_images)
            )
            parts.append(text_tokens[prefix_length:])
        else:
            segment_count = self.num_images + 1
            base_length, remainder = divmod(text_length, segment_count)
            segment_lengths = [
                base_length + int(index < remainder)
                for index in range(segment_count)
            ]
            parts = []
            cursor = 0
            for image_index in range(self.num_images):
                segment_length = segment_lengths[image_index]
                parts.append(
                    text_tokens[cursor : cursor + segment_length]
                )
                cursor += segment_length
                parts.append(self._build_image_block(image_index))
            parts.append(text_tokens[cursor:])
        input_ids = torch.cat(parts)
        if input_ids.numel() != self.seq_length:
            raise RuntimeError(
                f"expected seq_length={self.seq_length}, got "
                f"{input_ids.numel()}"
            )

        image_grid_thw = self.grid_thw.clone()
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        cu_seqlens = torch.tensor(
            [0, self.seq_length],
            dtype=torch.int32,
        )
        sample = self._base_sample(
            input_ids=input_ids,
            position_ids=position_ids.squeeze(1),
            pixel_values=self._pixel_values(self.total_patches),
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            max_seqlen=self.seq_length,
        )
        sample["image_cu_seqlens"] = torch.tensor(
            [0, self.num_images],
            dtype=torch.int32,
        )
        sample["pixel_cu_seqlens"] = torch.tensor(
            [0, self.total_patches],
            dtype=torch.int32,
        )
        return self._attach_descriptors(sample, image_grid_thw)

    def _build_packed_sample(self):
        input_documents = []
        position_documents = []
        sequence_ends = [0]
        per_document_text_length = (
            self.doc_length - self.image_block_cost
        )
        repeated_grids = self.grid_thw.repeat(self.pack_num_docs, 1)

        for document_index in range(self.pack_num_docs):
            parts = [self._build_text(per_document_text_length)]
            parts.extend(
                self._build_image_block(image_index)
                for image_index in range(self.num_images)
            )
            document = torch.cat(parts)
            if document.numel() != self.doc_length:
                raise RuntimeError(
                    f"expected packed document length={self.doc_length}, "
                    f"got {document.numel()}"
                )
            input_documents.append(document)
            sequence_ends.append(sequence_ends[-1] + self.doc_length)

            start = document_index * self.num_images
            document_grids = repeated_grids[
                start : start + self.num_images
            ]
            document_positions, _ = get_rope_index(
                spatial_merge_size=self.spatial_merge_size,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                input_ids=document.unsqueeze(0),
                image_grid_thw=document_grids,
            )
            position_documents.append(document_positions.squeeze(1))

        input_ids = torch.cat(input_documents)
        position_ids = torch.cat(position_documents, dim=1)
        cu_seqlens = torch.tensor(sequence_ends, dtype=torch.int32)
        sample = self._base_sample(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=self._pixel_values(
                self.total_patches * self.pack_num_docs
            ),
            image_grid_thw=repeated_grids,
            cu_seqlens=cu_seqlens,
            max_seqlen=self.doc_length,
        )
        sample["image_cu_seqlens"] = (
            torch.arange(self.pack_num_docs + 1, dtype=torch.int32)
            * self.num_images
        )
        sample["pixel_cu_seqlens"] = (
            torch.arange(self.pack_num_docs + 1, dtype=torch.int32)
            * self.total_patches
        )
        return self._attach_descriptors(sample, repeated_grids)


def mock_collate_fn(batch):
    """Collate mock samples while preserving MRoPE and image dimensions."""
    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key == "position_ids" and values[0].dim() == 2:
            result[key] = torch.stack(values, dim=1)
        elif key in ("image_grid_thw", "pixel_values"):
            result[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = values
    return result


def _parse_int_list(value):
    """Parse comma, underscore, or colon delimited integer lists."""
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    return [
        int(item)
        for item in re.split(r"[,_:]", str(value))
        if item.strip()
    ]


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide fixed-shape mock train, validation, and test datasets."""
    from megatron.training import get_args

    args = get_args()
    kwargs = {
        "seq_length": getattr(args, "total_seq_length", 1024),
        # Image-token counts are derived from each patch grid.
        "image_seq_length": None,
        "vocab_size": getattr(args, "padded_vocab_size", 248320),
        "image_token_id": getattr(args, "image_token_id", 248056),
        "image_size": getattr(args, "image_size", 224),
        "image_size_w": getattr(args, "image_size_w", None),
        "image_sizes_h": _parse_int_list(
            getattr(args, "image_sizes_h", None)
        ),
        "image_sizes_w": _parse_int_list(
            getattr(args, "image_sizes_w", None)
        ),
        "num_images": getattr(args, "num_images", 1),
        "layout": getattr(args, "mock_layout", "single"),
        "pack_num_docs": getattr(args, "mock_pack_num_docs", 1),
        "text_only": getattr(args, "text_only", False),
        # PR2 has no MDP model path. Keep ordinary mock training fully
        # materialized; focused tests can opt into descriptor-only samples.
        "metadata_only_batch": False,
    }

    return tuple(
        MockQwen35VLDataset(num_samples=count, **kwargs)
        for count in train_val_test_num_samples
    )
