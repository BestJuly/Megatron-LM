# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Map-style access to verified Nemotron WebDataset records for MDP training."""
from __future__ import annotations

import io
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

REVISION = '7656391d4d4cb11ec3722b34f10d499435de0460'
PROCESSOR = 'Qwen/Qwen3.5-35B-A3B'
PROCESSOR_REVISION = '59d61f3ce65a6d9863b86d2e96597125219dc754'
INDEX = 'mdp-training-index.json'


_LOG = logging.getLogger(__name__)


def tokenize(
    processor: Any, messages: list[dict[str, Any]], images: list[Image.Image]
) -> dict[str, torch.Tensor]:
    """Encode a whole conversation with shifted, assistant-only prediction targets."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = processor(
        text=[text], images=images, return_tensors='pt', min_pixels=4096, max_pixels=262144
    )
    tokens = encoded['input_ids'][0]
    ids = tokens.tolist()
    target_mask = torch.zeros_like(tokens, dtype=torch.float32)
    # Qwen3.5's template normalizes thinking blocks, so raw answer text is
    # not necessarily a substring of the rendered conversation. Use ChatML
    # role delimiters from the actual token stream, including reasoning tokens.
    tokenizer = processor.tokenizer
    start_id = int(tokenizer.convert_tokens_to_ids('<|im_start|>'))
    end_id = int(tokenizer.convert_tokens_to_ids('<|im_end|>'))
    header = tokenizer.encode('assistant\n', add_special_tokens=False)
    if not header or start_id == end_id:
        raise ValueError('The processor does not provide the expected ChatML boundaries')
    marked_turns = 0
    for i, token in enumerate(ids):
        if token != start_id or ids[i + 1 : i + 1 + len(header)] != header:
            continue
        content_start = i + 1 + len(header)
        try:
            content_end = ids.index(end_id, content_start)
        except ValueError as exc:
            raise ValueError('An assistant message has no closing ChatML delimiter') from exc
        target_mask[content_start:content_end] = 1
        marked_turns += 1
    expected_turns = sum(turn['role'] == 'assistant' for turn in messages)
    if marked_turns != expected_turns:
        raise ValueError(f'Assistant turn count mismatch: {marked_turns} != {expected_turns}')
    labels = torch.cat([tokens[1:], torch.tensor([-100], dtype=torch.long)])
    loss_mask = torch.cat([target_mask[1:], torch.zeros(1)])
    special = torch.tensor(processor.tokenizer.all_special_ids, dtype=torch.long)
    loss_mask[torch.isin(labels, special)] = 0
    labels[loss_mask == 0] = -100
    if not bool(loss_mask.any()):
        raise ValueError('Sample has no assistant prediction targets')
    return {
        'input_ids': tokens,
        'labels': labels,
        'loss_mask': loss_mask,
        'pixel_values': encoded['pixel_values'].to(torch.bfloat16),
        'image_grid_thw': encoded['image_grid_thw'],
    }


def verified_manifest(root: str | Path) -> dict[str, Any]:
    """Read the pinned source-verification record from a converted directory."""
    manifest = json.loads((Path(root) / 'manifest.json').read_text())
    if manifest.get('revision') != REVISION or not manifest.get('source_integrity_verified'):
        raise ValueError('Training requires the pinned, integrity-verified Nemotron conversion')
    return manifest


def select_records(
    index: dict[str, Any], selection: dict[str, Any], token_budget: int
) -> dict[str, list[dict[str, Any]]]:
    """Resolve a provenance-bound selection; never change records during training."""
    if (
        selection.get('kind') != 'nemotron-mdp-complete-record-selection'
        or selection.get('format_version') != 1
    ):
        raise ValueError('Unsupported Nemotron selection format')
    if selection['manifest'] != index['manifest']:
        raise ValueError('Selection provenance differs from the training index')
    if (
        selection['processor'],
        selection['processor_revision'],
        selection['min_pixels'],
        selection['max_pixels'],
    ) != (PROCESSOR, PROCESSOR_REVISION, 4096, 262144):
        raise ValueError('Selection processor differs from training')
    if selection['token_budget'] != token_budget:
        raise ValueError('Selection token budget differs from training')
    result = {}
    for split in ('train', 'val'):
        keys = selection['splits'][split]
        available = {r['key']: r for r in index['splits'][split]}
        if not keys or len(set(keys)) != len(keys) or any(k not in available for k in keys):
            raise ValueError(
                f'Selection has empty, duplicate, missing or cross-split keys: {split}'
            )
        if any(k.partition('__')[0] != selection['subset'] for k in keys):
            raise ValueError('Selection includes a different subset')
        if len(keys) != selection['counts'][split]['selected_records']:
            raise ValueError('Selection record count mismatch')
        result[split] = [available[k] for k in keys]
    return result


def load_training_selection(
    path: str | Path, token_budget: int
) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    """Require a budget-specific admission manifest for every real-data family."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            'Nemotron training requires a complete-record selection JSON, not a tar directory. '
            'Prepare and validate the subset for this token budget first; '
            'see examples/multimodal_dev/data/nemotron.md.'
        )
    selection = json.loads(path.read_text())
    root = path.parent
    manifest = verified_manifest(root)
    index = json.loads((root / INDEX).read_text())
    if index['manifest'] != manifest:
        raise ValueError('Training index provenance differs from the preparation manifest')
    return root, select_records(index, selection, token_budget)


class NemotronDataset(Dataset):
    """Read complete verified records on demand; let native samplers own ordering."""

    def __init__(
        self,
        root: str | Path,
        records: list[dict[str, Any]],
        *,
        target_length: int,
        max_length: int,
        vocab_size: int,
        image_token_id: int,
    ) -> None:
        self.root = Path(root)
        self.records = records
        self.target_length = max(len(records), int(target_length))
        self.max_length = int(max_length)
        self.vocab_size = int(vocab_size)
        self.image_token_id = int(image_token_id)
        self._processor = None
        self._files = {}

    def __len__(self) -> int:
        return self.target_length

    def _read(self, filename: str, extent: list[int]) -> bytes:
        if filename not in self._files:
            if len(self._files) >= 8:
                self._files.pop(next(iter(self._files))).close()
            self._files[filename] = (self.root / filename).open('rb')
        f = self._files[filename]
        offset, size = extent
        f.seek(offset)
        data = f.read(size)
        if len(data) != size:
            raise ValueError(f'Truncated prepared component in {filename} at {offset}')
        return data

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['_processor'] = None
        state['_files'] = {}
        return state

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[int(idx) % len(self.records)]
        try:
            if self._processor is None:
                from transformers import AutoProcessor

                self._processor = AutoProcessor.from_pretrained(
                    PROCESSOR, revision=PROCESSOR_REVISION, local_files_only=True
                )
                actual_id = int(
                    self._processor.tokenizer.convert_tokens_to_ids(self._processor.image_token)
                )
                if actual_id != self.image_token_id:
                    raise ValueError(
                        f'Processor image token {actual_id} != model token {self.image_token_id}'
                    )
            messages = json.loads(self._read(record['tar'], record['json']))
            # This is the local converter's trusted serialization, not arbitrary remote pickle.
            payload = pickle.loads(self._read(record['tar'], record['jpgs']))
            if not isinstance(payload, list) or not all(isinstance(x, bytes) for x in payload):
                raise ValueError('Prepared image payload must be a list of original byte strings')
            images = [Image.open(io.BytesIO(data)).convert('RGB') for data in payload]
            sample = tokenize(self._processor, messages, images)
            n = sample['input_ids'].numel()
            if not 0 < n <= self.max_length:
                raise ValueError(
                    f'Complete record has {n} tokens, exceeding budget {self.max_length}'
                )
            if (
                sample['input_ids'].min().item() < 0
                or sample['input_ids'].max().item() >= self.vocab_size
            ):
                raise ValueError('Processor tokens are outside the configured vocabulary')
            grid = sample['image_grid_thw']
            merge = self._processor.image_processor.merge_size
            expected_images = int((grid.prod(dim=1) // (merge * merge)).sum().item())
            actual_images = int((sample['input_ids'] == self.image_token_id).sum().item())
            if actual_images != expected_images or grid.shape[0] != len(images):
                raise ValueError(
                    f'Image placeholder/grid mismatch: {actual_images} != {expected_images}'
                )
            return sample
        except Exception as exc:
            raise RuntimeError(f'Nemotron record {record["key"]}: {exc}') from exc


def train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
) -> tuple[NemotronDataset, NemotronDataset, None]:
    """Supply admitted train/validation records to the native MDP packing path."""
    from megatron.training import get_args

    args = get_args()
    paths = args.data_path
    if not paths or len(paths) != 1:
        raise ValueError(
            'Nemotron provider requires one complete-record selection JSON in --data-path'
        )
    path = Path(paths[0])
    root, splits = load_training_selection(path, args.max_seqlen_per_dp_cp_rank)
    packing = args.mdp_greedy_packing or getattr(args, "mdp_ffd_packing", False)
    cap = int(args.thd_max_packed_sequences) - 1 if packing else 1
    result = []
    for split, requested in zip(('train', 'val'), train_val_test_num_samples[:2]):
        # The request is in nominal pack-slot units; the stream can drain cap records per slot.
        target = math.ceil((int(requested) + int(args.global_batch_size)) * cap * 1.1)
        ds = NemotronDataset(
            root,
            splits[split],
            target_length=target,
            max_length=args.max_seqlen_per_dp_cp_rank,
            vocab_size=args.padded_vocab_size,
            image_token_id=args.image_token_id,
        )
        _LOG.info(
            f'Nemotron {split}: {len(ds.records)} distinct records, {len(ds)} virtual samples; '
            f'processor={PROCESSOR}@{PROCESSOR_REVISION}'
        )
        result.append(ds)
    # No separately held-out test partition is supplied by the prepared 95/5 conversion.
    return result[0], result[1], None
