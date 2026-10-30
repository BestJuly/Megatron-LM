# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kimi-K3 VL dataset providers: ``mock``, ``mdp_mock`` and ``cord_v2``.

All three emit the per-sample dicts the packed-THD collator consumes
(``--use-packed-sequence --use-vanilla-collate-fn``)::

    input_ids, labels, loss_mask   [S]
    pixel_values                   [sum(t*h*w), 588] raster-ordered patch rows
    image_grid_thw                 [num_items, 3]

Each vision item is written into the text as
``<|media_begin|> ... <|media_pad|> x (h/2)*(w/2) ... <|media_end|>``: the
HF processor emits one ``<|media_pad|>`` per item and the HF model expands it;
here the expansion happens in the data so the decoder sees one slot per merged
vision row. Video items fill the same ``(h/2)*(w/2)`` slots as an image of the
same grid, because MoonViT averages over frames.
"""

import logging
import math
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset, _greedy_sample_scale
from examples.multimodal_dev.data.mdp_scenarios import build_scenarios
from examples.multimodal_dev.models.kimi_k3_vl.configuration import (
    KIMI_K3_IMAGE_TOKEN_ID,
    KIMI_K3_MEDIA_BEGIN_TOKEN_ID,
    KIMI_K3_MEDIA_CONTENT_TOKEN_ID,
    KIMI_K3_MEDIA_END_TOKEN_ID,
    VISION_KWARGS,
    resolve_image_token_id,
)
from examples.multimodal_dev.models.kimi_k3_vl.mdp_adapter import item_output_rows

logger = logging.getLogger(__name__)

PIXEL_DIM = VISION_KWARGS["in_channels"] * VISION_KWARGS["patch_size"] ** 2
_SPECIAL_TOKEN_IDS = (
    KIMI_K3_MEDIA_BEGIN_TOKEN_ID,
    KIMI_K3_MEDIA_CONTENT_TOKEN_ID,
    KIMI_K3_MEDIA_END_TOKEN_ID,
)


def kimi_scenarios(scenarios):
    """Map the shared MDP scenario pool onto MoonViT's frame limit.

    The pool draws video items with ``t in {2, 3}``; MoonViT takes up to 4
    frames per item, so ``t = 3`` is promoted to 4 to cover the full range
    (``t in {1, 2, 4}``). Grids and text chunks are otherwise unchanged.
    """
    out = []
    for grids, text_chunks in scenarios:
        grids = tuple((4 if t == 3 else t, h, w) for t, h, w in grids)
        out.append((grids, text_chunks))
    return tuple(out)


class KimiMdpMockDataset(MdpThdMockDataset):
    """``mdp_mock`` with Kimi-K3 geometry and media markers.

    Pixels keep the base class's per-item integer sentinel so MDP's pixel
    routing can be checked element by element; rows are 588 wide and an item
    fills ``(h/2)*(w/2)`` decoder slots regardless of ``t``.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("image_token_id", KIMI_K3_IMAGE_TOKEN_ID)
        kwargs.setdefault("vision_start_token_id", KIMI_K3_MEDIA_BEGIN_TOKEN_ID)
        super().__init__(
            patch_size=VISION_KWARGS["patch_size"],
            temporal_patch_size=1,
            spatial_merge_size=VISION_KWARGS["spatial_merge_size"],
            **kwargs,
        )
        assert self.pixel_dim == PIXEL_DIM

    def item_tokens(self, grid) -> int:
        """Decoder slots of one item: frames are pooled away."""
        return item_output_rows(grid, self.spatial_merge_size)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # The base class draws text from [1, vocab); keep it off Kimi's media
        # markers so only real item boundaries carry them.
        ids = sample["input_ids"]
        text = ~torch.isin(
            ids, torch.tensor([self.image_token_id, self.vision_start_token_id], dtype=ids.dtype)
        )
        for special in _SPECIAL_TOKEN_IDS[1:]:
            ids[text & (ids == special)] = 1
        sample["labels"][:-1] = ids[1:]
        return sample


class KimiFixedMockDataset(Dataset):
    """``mock``: one ``image_size`` x ``image_size`` image per fixed-length sample.

    Args:
        num_samples: Dataset length.
        seq_length: Tokens per sample (``--total-seq-length``).
        image_size: Square image side in pixels; rounded down to a multiple of
            28 (patch 14 x merge 2).
        vocab_size: Text token range.
        image_token_id: ``<|media_pad|>``.
        seed / split: Per-sample generator seed derivation.
    """

    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        image_size: int,
        vocab_size: int,
        image_token_id: int,
        seed: int = 1234,
        split: int = 0,
    ):
        factor = VISION_KWARGS["patch_size"] * VISION_KWARGS["spatial_merge_size"]
        side = max(factor, (image_size // factor) * factor) // VISION_KWARGS["patch_size"]
        self.grid = (1, side, side)
        self.num_tokens = item_output_rows(self.grid)
        if seq_length < self.num_tokens + 3:
            raise ValueError(
                f"--total-seq-length {seq_length} cannot hold a {image_size}px image "
                f"({self.num_tokens} vision tokens + 2 markers)"
            )
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.seed = seed + 7919 * split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(self.seed + int(idx))
        text_len = self.seq_length - self.num_tokens - 2
        text = torch.randint(1, self.vocab_size, (text_len,), generator=gen)
        for special in (self.image_token_id,) + _SPECIAL_TOKEN_IDS:
            text[text == special] = 1
        half = text_len // 2
        input_ids = torch.cat(
            [
                text[:half],
                torch.tensor([KIMI_K3_MEDIA_BEGIN_TOKEN_ID]),
                torch.full((self.num_tokens,), self.image_token_id),
                torch.tensor([KIMI_K3_MEDIA_END_TOKEN_ID]),
                text[half:],
            ]
        ).long()
        labels = torch.zeros_like(input_ids)
        labels[:-1] = input_ids[1:]
        loss_mask = (input_ids != self.image_token_id).float()
        loss_mask[input_ids == KIMI_K3_MEDIA_BEGIN_TOKEN_ID] = 0.0
        loss_mask[-1] = 0.0
        t, h, w = self.grid
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": torch.randn(t * h * w, PIXEL_DIM, generator=gen),
            "image_grid_thw": torch.tensor([self.grid], dtype=torch.long),
        }


def expand_media_pad(input_ids: List[int], num_tokens: List[int], image_token_id: int):
    """Replace the i-th single ``<|media_pad|>`` by ``num_tokens[i]`` copies."""
    out, item = [], 0
    for tok in input_ids:
        if tok == image_token_id:
            if item >= len(num_tokens):
                raise ValueError("more <|media_pad|> tokens than vision items")
            out.extend([image_token_id] * num_tokens[item])
            item += 1
        else:
            out.append(tok)
    if item != len(num_tokens):
        raise ValueError(f"{len(num_tokens)} vision items but {item} <|media_pad|> tokens")
    return out


class KimiCordV2Dataset(Dataset):
    """CORD-V2 receipts, tokenized with the Kimi-K3 chat encoding.

    Preprocessing is the native port in ``preprocessing.py``; only the
    tokenizer is loaded from the Hub (``trust_remote_code``). The user turn is
    ``[image, "Describe this image."]`` and the loss covers the assistant
    answer only. Samples longer than ``seq_length`` are dropped rather than
    truncated, because truncation could cut through an item's vision slots.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        seq_length: int,
        image_token_id: int,
        target_length: Optional[int] = None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.image_token_id = image_token_id
        self._length = target_length or len(examples)

    def __len__(self):
        return self._length

    def _encode(self, example) -> Optional[Dict]:
        from examples.multimodal_dev.models.kimi_k3_vl.preprocessing import preprocess_images

        image = example["image"]
        vision = preprocess_images([image])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": example["question"]},
                ],
            },
            {"role": "assistant", "content": example["answer"]},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            thinking=False,
            image_prompts=vision["prompts"],
            return_dict=True,
        )
        ids = encoded["input_ids"]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = expand_media_pad(ids, vision["num_tokens"], self.image_token_id)
        if len(ids) > self.seq_length:
            return None
        answer = self.tokenizer.encode(example["answer"], add_special_tokens=False)
        loss_mask = torch.zeros(len(ids))
        for start in range(len(ids) - len(answer), -1, -1):
            if ids[start : start + len(answer)] == answer:
                loss_mask[start : start + len(answer)] = 1.0
                break
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.zeros_like(input_ids)
        labels[:-1] = input_ids[1:]
        # loss_mask[i] weights predicting labels[i] = input_ids[i + 1].
        loss_mask = torch.cat([loss_mask[1:], torch.zeros(1)])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": vision["pixel_values"],
            "image_grid_thw": vision["image_grid_thw"],
        }

    def __getitem__(self, idx):
        n = len(self.examples)
        for probe in range(n):
            sample = self._encode(self.examples[(idx + probe) % n])
            if sample is not None:
                return sample
            logger.warning("cord_v2 example %d exceeds seq_length; skipped", (idx + probe) % n)
        raise ValueError(f"no cord_v2 example fits --total-seq-length {self.seq_length}")


# ---------------------------------------------------------------------------
# Providers (registered in models/__init__.py)
# ---------------------------------------------------------------------------


def mdp_mock_provider(train_val_test_num_samples):
    """``--dataset-provider mdp_mock``."""
    from megatron.training import get_args

    args = get_args()
    length_config = getattr(args, "mdp_mock_dataset_config_json", None)
    if length_config is not None:
        from megatron.training.datasets.utils import load_json_arg

        length_config = load_json_arg(length_config)
    scenarios = kimi_scenarios(build_scenarios(length_config=length_config))
    kwargs = dict(
        vocab_size=getattr(args, "padded_vocab_size", 1024),
        image_token_id=resolve_image_token_id(args),
        scenarios=scenarios,
    )
    scale = _greedy_sample_scale(args, scenarios)
    return tuple(
        KimiMdpMockDataset(num_samples=math.ceil(n * scale), seed=1234 + split, **kwargs)
        for split, n in enumerate(train_val_test_num_samples)
    )


def mock_provider(train_val_test_num_samples):
    """``--dataset-provider mock``."""
    from megatron.training import get_args

    args = get_args()
    kwargs = dict(
        seq_length=getattr(args, "total_seq_length", None) or args.seq_length,
        image_size=getattr(args, "image_size", 224),
        vocab_size=getattr(args, "padded_vocab_size", 163840),
        image_token_id=resolve_image_token_id(args),
        seed=getattr(args, "seed", 1234),
    )
    return tuple(
        KimiFixedMockDataset(num_samples=max(n, 1), split=split, **kwargs)
        for split, n in enumerate(train_val_test_num_samples)
    )


def cord_v2_provider(train_val_test_num_samples):
    """``--dataset-provider cord_v2`` with the Kimi-K3 tokenizer.

    The tokenizer comes from ``--hf-processor-path`` (default
    ``moonshotai/Kimi-K3``) at the pinned revision.
    """
    from transformers import AutoTokenizer

    from examples.multimodal_dev.data.cord_v2 import load_cord_v2
    from examples.multimodal_dev.models.kimi_k3_vl.configuration import KIMI_K3_HF_REVISION
    from megatron.training import get_args

    args = get_args()
    path = getattr(args, "hf_processor_path", None) or "moonshotai/Kimi-K3"
    revision = KIMI_K3_HF_REVISION if path == "moonshotai/Kimi-K3" else None
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=True)
    seq_length = getattr(args, "total_seq_length", None) or args.seq_length
    image_token_id = resolve_image_token_id(args)

    def _make(split, n):
        return KimiCordV2Dataset(
            examples=load_cord_v2(split=split),
            tokenizer=tokenizer,
            seq_length=seq_length,
            image_token_id=image_token_id,
            target_length=max(n, 1),
        )

    return (
        _make("train", train_val_test_num_samples[0]),
        _make("validation", train_val_test_num_samples[1]),
        _make("test", train_val_test_num_samples[2]),
    )
