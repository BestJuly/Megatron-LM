# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP mock dataset: reproducible multi-image, variable-grid THD samples.

The fixed-grid single-image mock in ``mock.py`` is not sufficient to validate
MDP's dual-THD contract. This dataset generates deterministic samples covering:

* multiple images per sample with interleaved text,
* variable resolution (per-item ``grid_thw``),
* multi-frame video items (``t > 1``),
* text-only samples,
* variable sequence lengths (so true and padded ``cu_seqlens`` differ under an
  alignment multiple).

Every item's pixels are filled with a unique integer sentinel so pixel
dispatch, producer THD packing, endpoint reassembly, and gradient reverse
routing can be verified element by element.
"""

import math
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.data.mdp_scenarios import build_scenarios, scenario_totals
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)


def item_sentinel(sample_id: int, image_ordinal: int) -> int:
    """The integer written into every pixel of one vision item."""
    return 1000 * (sample_id + 1) + image_ordinal


# Per-sample scenarios, cycled by sample index. Grids are (t, h, w) in patch
# units; h and w must be divisible by the spatial merge size.
#
# The pool is the randomized one from ``mdp_scenarios``: 64 deterministic
# scenarios with sequence lengths in [1000, 2000] and ~67 distinct grids. The
# heterogeneity is the point -- variable lengths and grids are what exercise
# THD packing, planner load balancing, and the encoder's grid-keyed caches, so
# a run against this pool measures something. Pass ``scenarios=`` to
# ``MdpThdMockDataset`` to override.
_SCENARIOS: Sequence = build_scenarios()


class MdpThdMockDataset(Dataset):
    """Deterministic multi-image mock samples for the MDP dual-THD contract.

    Args:
        num_samples: Dataset length.
        vocab_size: Vocabulary for random text tokens.
        image_token_id: Placeholder token for vision rows.
        vision_start_token_id: Sentinel token before each vision block.
        patch_size: Spatial patch size (pixel payload width factor).
        temporal_patch_size: Temporal patch size (pixel payload width factor).
        spatial_merge_size: Post-merge factor ``m``; each item occupies
            ``t*(h/m)*(w/m)`` decoder token slots.
        seed: Base seed; sample ``i`` is generated from ``seed + i``.
        scenarios: Optional override of the scenario cycle, each entry
            ``(grids, text_chunk_lengths)`` with ``len(text_chunks) ==
            len(grids) + 1`` (text before/between/after vision blocks; the
            text-only scenario uses one chunk).
        length_config: Optional ``--mdp-mock-dataset-config-json`` payload
            controlling the per-sample *total* token length distribution
            (same schema as ``--varlen-mock-dataset-config-json``). Ignored
            when ``scenarios`` is given.
    """

    def __init__(
        self,
        num_samples: int = 64,
        vocab_size: int = 1024,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        seed: int = 1234,
        scenarios: Optional[Sequence] = None,
        length_config: Optional[dict] = None,
    ):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.seed = seed
        if scenarios is not None:
            self.scenarios = tuple(scenarios)
        elif length_config is not None:
            # Rebuilt independently on every rank; build_scenarios is seeded
            # from the fixed GENERATOR_SEED so the pools stay identical.
            self.scenarios = build_scenarios(length_config=length_config)
        else:
            self.scenarios = _SCENARIOS
        self.pixel_dim = 3 * temporal_patch_size * patch_size * patch_size
        for grids, text_chunks in self.scenarios:
            expected_chunks = len(grids) + 1 if grids else 1
            assert len(text_chunks) == expected_chunks, (
                f"scenario with {len(grids)} grids needs {expected_chunks} text chunks, "
                f"got {len(text_chunks)}"
            )
            for t, h, w in grids:
                assert h % spatial_merge_size == 0 and w % spatial_merge_size == 0, (
                    f"grid ({t},{h},{w}) not divisible by merge={spatial_merge_size}"
                )

    def __len__(self):
        return self.num_samples

    def item_rows(self, grid) -> int:
        """Pixel patch rows of one item."""
        t, h, w = grid
        return t * h * w

    def item_tokens(self, grid) -> int:
        """Decoder token slots of one item after spatial merge."""
        t, h, w = grid
        m = self.spatial_merge_size
        return t * (h // m) * (w // m)

    def __getitem__(self, idx):
        grids, text_chunks = self.scenarios[idx % len(self.scenarios)]
        generator = torch.Generator().manual_seed(self.seed + idx)

        def _text(length):
            tokens = torch.randint(
                1, self.vocab_size, (length,), dtype=torch.long, generator=generator
            )
            for special in (self.image_token_id, self.vision_start_token_id):
                tokens[tokens == special] = 1
            return tokens

        pieces = [_text(text_chunks[0])]
        for ordinal, grid in enumerate(grids):
            pieces.append(torch.tensor([self.vision_start_token_id], dtype=torch.long))
            pieces.append(
                torch.full((self.item_tokens(grid),), self.image_token_id, dtype=torch.long)
            )
            pieces.append(_text(text_chunks[ordinal + 1]))
        input_ids = torch.cat(pieces)

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0
        loss_mask = (input_ids != self.image_token_id).float()
        loss_mask[input_ids == self.vision_start_token_id] = 0.0
        loss_mask[-1] = 0.0

        if grids:
            pixel_chunks = []
            for ordinal, grid in enumerate(grids):
                pixel_chunks.append(
                    torch.full(
                        (self.item_rows(grid), self.pixel_dim),
                        float(item_sentinel(idx, ordinal)),
                        dtype=torch.float32,
                    )
                )
            pixel_values = torch.cat(pixel_chunks)
            image_grid_thw = torch.tensor(grids, dtype=torch.long)
        else:
            pixel_values = torch.empty(0, self.pixel_dim, dtype=torch.float32)
            image_grid_thw = torch.empty(0, 3, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


#: Extra margin on top of the computed greedy sample requirement. The mean
#: sample length only predicts the *average* bin occupancy; individual bins run
#: short or long, and the shortfall compounds across iterations.
GREEDY_SAMPLE_SAFETY = 1.5


def _greedy_sample_scale(args, scenarios):
    """Scale factor for the synthetic dataset length under greedy packing.

    Megatron sizes the dataset as ``train_iters * global_batch_size`` samples.
    Under ``--mdp-greedy-packing`` that is a *bin* count, not a sample count:
    each bin swallows roughly ``token_budget / mean_sample_len`` samples instead
    of exactly ``micro_batch_size``. Whenever the mean sample is shorter than
    ``token_budget / micro_batch_size`` the stream runs dry mid-run. The mock
    dataset is a pure function of the sample index, so enlarging it costs
    nothing and changes no sample's content.

    Returns 1.0 when greedy packing is off, so the default path is unchanged.
    """
    if not getattr(args, "mdp_greedy_packing", False):
        return 1.0
    budget = int(args.max_seqlen_per_dp_cp_rank) * int(args.context_parallel_size)
    cap = getattr(args, "thd_max_packed_sequences", None)
    mean_len = sum(scenario_totals(s)[0] for s in scenarios) / len(scenarios)
    samples_per_bin = budget / mean_len
    if cap:
        samples_per_bin = min(samples_per_bin, float(cap))
    return max(1.0, samples_per_bin / int(args.micro_batch_size)) * GREEDY_SAMPLE_SAFETY


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide MDP mock train / val / test datasets."""
    from megatron.training import get_args

    args = get_args()
    length_config = getattr(args, "mdp_mock_dataset_config_json", None)
    if length_config is not None:
        from megatron.training.datasets.utils import load_json_arg

        length_config = load_json_arg(length_config)
    # Built once and shared: the pool is a deterministic function of the length
    # config, and drawing it re-samples a million lognormal lengths.
    scenarios = build_scenarios(length_config=length_config)
    kwargs = dict(
        vocab_size=getattr(args, "padded_vocab_size", 1024),
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        scenarios=scenarios,
    )
    scale = _greedy_sample_scale(args, scenarios)
    return tuple(
        MdpThdMockDataset(num_samples=math.ceil(n * scale), seed=1234 + split, **kwargs)
        for split, n in enumerate(train_val_test_num_samples)
    )
