# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Mock dataset for multimodal model testing.

Generates synthetic image + text data for end-to-end testing without
real datasets. Each sample has:
    - Random text tokens with image token placeholders
    - Random pixel values sized for the vision encoder
    - 3D MRoPE position IDs
    - Labels shifted from input_ids
"""

import torch
from torch.utils.data import Dataset

from multimodal.models.qwen35_vl import compute_mrope_position_ids


class MockQwen35VLDataset(Dataset):
    """Mock dataset that generates synthetic Qwen3.5-VL training samples.

    Args:
        num_samples: Number of samples in the dataset.
        seq_length: Total sequence length (text + image tokens).
        image_seq_length: Number of image tokens per sample.
        vocab_size: Vocabulary size for random text tokens.
        image_token_id: Token ID for image placeholders.
        image_size: Image height and width in pixels.
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal patch size.
        spatial_merge_size: Spatial merge factor.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 1024,
        image_seq_length: int = 256,
        vocab_size: int = 248320,
        image_token_id: int = 248056,
        image_size: int = 224,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.image_seq_length = image_seq_length
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size

        # Compute grid dimensions for a single image
        h_patches = image_size // patch_size
        w_patches = image_size // patch_size
        t_patches = temporal_patch_size  # for a single image, T = temporal_patch_size
        self.grid_thw = torch.tensor([[t_patches, h_patches, w_patches]])

        # Number of merged tokens = t * (h/merge) * (w/merge)
        self.num_merged_tokens = (
            t_patches
            * (h_patches // spatial_merge_size)
            * (w_patches // spatial_merge_size)
        )

        # Adjust image_seq_length to match actual merged token count
        self.image_seq_length = min(image_seq_length, self.num_merged_tokens)

        # Pixel values shape: we need C * T * patch_h * patch_w per patch
        # Total patches before merge: t_patches * h_patches * w_patches
        self.total_patches = t_patches * h_patches * w_patches

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random text tokens
        text_length = self.seq_length - self.image_seq_length
        text_tokens = torch.randint(
            1, self.vocab_size, (text_length,), dtype=torch.long
        )
        # Avoid generating the image token in text
        text_tokens[text_tokens == self.image_token_id] = 1

        # Create input_ids: [text_prefix, image_tokens, text_suffix]
        prefix_len = text_length // 2
        suffix_len = text_length - prefix_len
        input_ids = torch.cat([
            text_tokens[:prefix_len],
            torch.full((self.image_seq_length,), self.image_token_id, dtype=torch.long),
            text_tokens[prefix_len:prefix_len + suffix_len],
        ])

        # Labels: shifted input_ids (predict next token)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0  # padding

        # Loss mask: 1 for text tokens, 0 for image tokens and padding
        loss_mask = (input_ids != self.image_token_id).float()
        loss_mask[-1] = 0  # last token has no label

        # Pixel values: random [total_patches, C * temporal_patch * patch_h * patch_w]
        pixel_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size
        pixel_values = torch.randn(self.total_patches, pixel_dim)

        # Grid THW
        image_grid_thw = self.grid_thw.clone()

        # MRoPE position IDs: [3, S]
        position_ids = compute_mrope_position_ids(
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            image_token_id=self.image_token_id,
            spatial_merge_size=self.spatial_merge_size,
        ).squeeze(1)  # [3, S]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def mock_collate_fn(batch):
    """Collate function for mock dataset.

    Stacks tensors and handles the position_ids 3D shape.
    """
    result = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [sample[key] for sample in batch]
        if key == "position_ids":
            # [3, S] per sample -> [3, B, S]
            result[key] = torch.stack(tensors, dim=1)
        elif key == "image_grid_thw":
            # [1, 3] per sample -> [B, 3] (concatenate)
            result[key] = torch.cat(tensors, dim=0)
        elif key == "pixel_values":
            # Variable length -> concatenate along dim 0
            result[key] = torch.cat(tensors, dim=0)
        else:
            result[key] = torch.stack(tensors, dim=0)
    return result


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide mock datasets for training, validation, and test.

    Args:
        train_val_test_num_samples: Tuple of (train, val, test) sample counts.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from megatron.training import get_args

    args = get_args()

    kwargs = dict(
        seq_length=getattr(args, "total_seq_length", 1024),
        image_seq_length=getattr(args, "image_seq_length", 256),
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", 248056),
        image_size=getattr(args, "image_size", 224),
    )

    train_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[0], **kwargs
    )
    val_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[1], **kwargs
    )
    test_ds = MockQwen35VLDataset(
        num_samples=train_val_test_num_samples[2], **kwargs
    )

    return train_ds, val_ds, test_ds
