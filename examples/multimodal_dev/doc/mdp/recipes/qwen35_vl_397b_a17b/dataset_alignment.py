"""CPU-only reference for the existing MDP uniform-text numerical rule."""
import hashlib
import json
import os

import torch

IMAGE_ID = 248056
VISION_START_ID = 248053
RULE = "mdp_uniform_text_v1"
_audited = set()


def draw_text(length, vocab_size, generator, excluded=(IMAGE_ID, VISION_START_ID)):
    tokens = torch.randint(1, vocab_size, (length,), dtype=torch.long, generator=generator)
    for special in excluded:
        tokens[tokens == special] = 1
    return tokens


def uniform_text(length, sample_id, vocab_size, seed=1234):
    generator = torch.Generator().manual_seed(seed + int(sample_id))
    return draw_text(length, vocab_size, generator)


def labels_and_mask(tokens):
    labels = tokens.clone()
    labels[:-1] = tokens[1:]
    labels[-1] = 0
    mask = torch.ones(tokens.numel(), dtype=torch.float32)
    mask[-1] = 0
    return labels, mask


def adapt_native_item(item, sample_id, vocab_size, length, seed=1234):
    assert item["tokens"].device.type == "cpu" and item["tokens"].ndim == 1
    assert 0 < int(item["original_seq_len"]) <= int(item["padded_seq_len"]) <= length
    assert int(item["padded_seq_len"]) == item["tokens"].numel()
    tokens = uniform_text(length, sample_id, vocab_size, seed)
    labels, mask = labels_and_mask(tokens)
    return {**item, "tokens": tokens, "labels": labels, "loss_mask": mask,
            "position_ids": torch.arange(length, dtype=torch.int64),
            "original_seq_len": item["original_seq_len"].new_tensor([length]),
            "padded_seq_len": item["padded_seq_len"].new_tensor([length])}


def verify_mdp_item(dataset, sample_id, item):
    grids, chunks = dataset.scenarios[sample_id % len(dataset.scenarios)]
    generator = torch.Generator().manual_seed(dataset.seed + int(sample_id))
    excluded = (dataset.image_token_id, dataset.vision_start_token_id)
    pieces = [draw_text(chunks[0], dataset.vocab_size, generator, excluded)]
    for ordinal, grid in enumerate(grids):
        pieces.extend((torch.tensor([dataset.vision_start_token_id], dtype=torch.long),
                       torch.full((dataset.item_tokens(grid),), dataset.image_token_id, dtype=torch.long),
                       draw_text(chunks[ordinal + 1], dataset.vocab_size, generator, excluded)))
    expected = torch.cat(pieces)
    assert torch.equal(expected, item["input_ids"])
    labels, mask = labels_and_mask(expected)
    mask[expected == dataset.image_token_id] = 0
    mask[expected == dataset.vision_start_token_id] = 0
    assert torch.equal(labels, item["labels"]) and torch.equal(mask, item["loss_mask"])


def audit_once(mode, sample_id, item, vocab_size, seed=1234, dataset=None):
    key = (os.getpid(), mode, seed)
    if key in _audited:
        return
    tokens = item["tokens" if mode == "decoder" else "input_ids"]
    assert tokens.device.type == "cpu" and tokens.numel() == 16384
    if mode == "mdp":
        verify_mdp_item(dataset, sample_id, item)
    else:
        assert torch.equal(tokens, uniform_text(tokens.numel(), sample_id, vocab_size, seed))
        labels, mask = labels_and_mask(tokens)
        assert torch.equal(labels, item["labels"]) and torch.equal(mask, item["loss_mask"])
    _audited.add(key)
    record = {"mode": mode, "rule": RULE, "sample_id": int(sample_id), "seed": seed,
              "vocab_size": vocab_size, "length": tokens.numel(), "pid": os.getpid(),
              "reference_checks_pass": True, "final_label": int(item["labels"][-1]),
              "final_loss_mask": float(item["loss_mask"][-1]),
              "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest()}
    record["dataset_class"] = type(dataset).__name__
    if mode == "decoder":
        record["original_seq_len"] = int(item["original_seq_len"])
        record["padded_seq_len"] = int(item["padded_seq_len"])
    print("NUMERICAL_RULE_AUDIT " + json.dumps(record), flush=True)
