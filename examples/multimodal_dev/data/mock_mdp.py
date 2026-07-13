from __future__ import annotations

import random
from typing import Optional

from examples.multimodal_dev.data.dataset_utils import RawSample

class MockBackend:
    """Synthetic backend that exercises the blend dataset packing path."""

    def __init__(
        self,
        root: Optional[str] = None,
        max_samples: Optional[int] = None,
        split: str = "train",
        **_: object,
    ):
        self.max_samples = max(1, int(max_samples or 1))
        split_offsets = {
            "train": 0,
            "val": 100_000,
            "valid": 100_000,
            "test": 200_000,
        }
        self.seed = 1729 + split_offsets.get(split, 300_000)

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> RawSample:
        idx = int(idx) % self.max_samples
        rng = random.Random(self.seed + idx)

        num_images = rng.randint(1, 6)
        descriptors = []
        for image_idx in range(num_images):
            size = rng.randint(224, 512)
            descriptors.append(
                {
                    "kind": "mock_grid",
                    "materializer": "examples.multimodal_dev.data.mock",
                    "width": size,
                    "height": size,
                    "seed": self.seed * 1_000_003 + idx * 101 + image_idx,
                }
            )

        word_count = rng.randint(192, 768)
        token_ids = [100 + idx, 200 + num_images]
        token_ids.extend(300 + rng.randrange(10_000) for _ in range(word_count))
        text = " ".join(str(token_id) for token_id in token_ids)

        return RawSample(images=[], text=text, image_descriptors=descriptors)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    from examples.multimodal_dev.data.blend_dataset import (
        train_valid_test_datasets_provider as blend_provider,
    )
    from megatron.training import get_args

    args = get_args()
    missing = object()
    old_backend = getattr(args, "dataset_backend", missing)
    setattr(args, "dataset_backend", "mock")
    try:
        return blend_provider(train_val_test_num_samples)
    finally:
        if old_backend is missing:
            delattr(args, "dataset_backend")
        else:
            setattr(args, "dataset_backend", old_backend)
