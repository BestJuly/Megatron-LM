"""Benchmark-only dataset adapter; production MCore files stay unchanged."""
import os
import runpy
import sys
from pathlib import Path

from dataset_alignment import adapt_native_item, audit_once


def install(mode):
    if mode == "decoder":
        from megatron.training import get_args
        from megatron.training.datasets.varlen_dataset import MockVarlenDataset

        original = MockVarlenDataset.__getitem__

        def aligned(self, sample_id):
            item = original(self, sample_id)
            split = getattr(self, "index_split", None)
            seed = 1234 + (int(split.value) if split is not None else 0)
            vocab_size = get_args().padded_vocab_size
            assert vocab_size == 248320
            assert get_args().use_varlen_dataset and not get_args().sft
            item = adapt_native_item(item, sample_id, vocab_size, self.config.sequence_length, seed)
            audit_once(mode, sample_id, item, vocab_size, seed, dataset=self)
            return item

        MockVarlenDataset.__getitem__ = aligned
    elif mode == "mdp":
        from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset

        original = MdpThdMockDataset.__getitem__

        def checked(self, sample_id):
            item = original(self, sample_id)
            assert self.vocab_size == 248320
            audit_once(mode, sample_id, item, self.vocab_size, self.seed, dataset=self)
            return item

        MdpThdMockDataset.__getitem__ = checked
    else:
        raise ValueError(mode)


def main():
    root = Path(__file__).resolve().parents[6]
    mode = os.environ["ALIGNED_NUMERICAL_MODE"]
    entries = {
        "decoder": root / "pretrain_gpt.py",
        "mdp": root / "examples/multimodal_dev/pretrain_multimodal.py",
    }
    if mode not in entries:
        raise ValueError(mode)
    entry = entries[mode]
    sys.path.insert(0, str(root))
    install(mode)
    sys.argv[0] = str(entry)
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
