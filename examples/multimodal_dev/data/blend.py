# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Registry and blend definitions for multimodal datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

from examples.multimodal_dev.data.finevision_max import FineVisionMaxBackend
from examples.multimodal_dev.data.m4_instruct import M4InstructBackend
from examples.multimodal_dev.data.mantis_instruct import MantisInstructBackend
from examples.multimodal_dev.data.mmmu import MmmuBackend
from examples.multimodal_dev.data.mock_mdp import MockBackend
from examples.multimodal_dev.data.muirbench import MuirbenchBackend
from examples.multimodal_dev.data.omnistar_rng import OmniStarRngBackend
from examples.multimodal_dev.data.pixmo_docs import PixmoDocsBackend

BACKEND_REGISTRY = {
    "mantis-instruct": MantisInstructBackend,
    "finevisionmax": FineVisionMaxBackend,
    "muirbench": MuirbenchBackend,
    "mmmu": MmmuBackend,
    "omnistar-rng": OmniStarRngBackend,
    "pixmo-docs": PixmoDocsBackend,
    "m4-instruct": M4InstructBackend,
    "mock": MockBackend,
}

DEFAULT_DATASET_SUBDIRS = {
    "mantis-instruct": "mantis/Mantis-Instruct",
    "finevisionmax": "highres-longvideo/finevisionmax",
    "muirbench": "muirbench",
    "mmmu": "mmmu",
    "pixmo-docs": "pixmo/pixmo-docs",
    "m4-instruct": "m4/M4-Instruct-Data",
    "omnistar-rng": "highres-longvideo/omnistar-rng",
    "mock": "",
}


class RawDatasetBlend:
    """Blend one or more raw dataset backends behind one index space."""

    def __init__(
        self,
        backend: str | Sequence[str],
        root: str | dict | None = None,
        max_samples: Optional[int] = None,
        subsets: Optional[Sequence[str]] = None,
        split: str = "train",
        seed: int = 0,
    ):
        backends = [backend] if isinstance(backend, str) else list(backend)
        roots = self._normalize_roots(backends, root)

        self.sources = []
        for name in backends:
            if name not in BACKEND_REGISTRY:
                raise ValueError(
                    f"Unknown backend {name!r}. Choices: "
                    f"{list(BACKEND_REGISTRY)}"
                )
            source = BACKEND_REGISTRY[name](
                **self._backend_kwargs(
                    name=name,
                    root=roots[name],
                    max_samples=max_samples,
                    subsets=subsets,
                    split=split,
                )
            )
            self.sources.append((name, source))

        self.index = []
        for bi, (_name, source) in enumerate(self.sources):
            for li in range(len(source)):
                self.index.append((bi, li))
        random.Random(seed).shuffle(self.index)

    @staticmethod
    def _normalize_roots(backends, root):
        if root is None:
            if backends == ["mock"]:
                return {"mock": ""}
            raise ValueError(
                "dataset root is required for non-mock blend datasets"
            )
        if isinstance(root, str):
            if len(backends) == 1:
                return {backends[0]: root}
            root_path = Path(root)
            return {
                name: str(root_path / DEFAULT_DATASET_SUBDIRS[name])
                for name in backends
            }
        return root

    @staticmethod
    def _backend_kwargs(name, root, max_samples, subsets, split):
        kwargs = dict(root=root, max_samples=max_samples)
        if name in ("mantis-instruct", "pixmo-docs", "finevisionmax"):
            kwargs["subsets"] = subsets
            kwargs["split"] = split
        elif name == "mock":
            kwargs["split"] = split
        elif name == "m4-instruct":
            kwargs["subsets"] = subsets
        return kwargs

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if not self.index:
            raise RuntimeError(
                "RawDatasetBlend index is empty; check dataset_root, "
                "subsets, split, and container mounts."
            )
        bi, li = self.index[int(idx) % len(self.index)]
        _name, source = self.sources[bi]
        return source[li]
