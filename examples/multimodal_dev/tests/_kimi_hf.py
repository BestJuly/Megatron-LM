# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Load the Kimi-K3 HF reference (Hub remote code) for parity tests.

The reference is ``moonshotai/Kimi-K3`` at a pinned revision. Only code,
config and preprocessing files are fetched -- never weight shards. The files
are imported as a throwaway package so their relative imports
(``from .configuration_kimi_k3 import ...``) resolve; nothing is registered
with ``transformers``' auto classes.

Set ``KIMI_K3_HF_DIR`` to a directory already holding these files to run
offline; otherwise they are downloaded into the HF cache. Tests skip when
neither works.

The remote code at this revision was written against transformers 4.56.x
(``config.json``: ``transformers_version``); it imports ``OutputRecorder`` from
``transformers.utils.generic``, which exists in 4.56 - 5.0 only. The training
image carries transformers 5.16, so these tests skip there unless a matching
transformers (with its ``huggingface_hub`` / ``tokenizers``) is put first on
``PYTHONPATH``, e.g. ``pip install --target <dir> --no-deps transformers==4.56.2
huggingface_hub==0.35.3 tokenizers==0.22.1``.
"""

import importlib
import os
import shutil
import sys
import tempfile

import pytest

from examples.multimodal_dev.models.kimi_k3_vl.configuration import KIMI_K3_HF_REVISION

HF_REPO = "moonshotai/Kimi-K3"
HF_FILES = (
    "config.json",
    "configuration_kimi_k3.py",
    "modeling_kimi_k3.py",
    "modeling_kimi_linear.py",
    "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py",
    "media_utils.py",
    "preprocessor_config.json",
)
_PACKAGE = "_kimi_k3_hf_reference"


#: transformers range the pinned remote code imports cleanly with.
HF_TRANSFORMERS_RANGE = ("4.56.0", "5.1.0")


def check_transformers_version() -> None:
    """``pytest.skip`` unless transformers is in :data:`HF_TRANSFORMERS_RANGE`."""
    try:
        import transformers
        from packaging.version import Version
    except ImportError:
        pytest.skip("transformers not installed")
    low, high = HF_TRANSFORMERS_RANGE
    if not Version(low) <= Version(transformers.__version__) < Version(high):
        pytest.skip(
            f"Kimi-K3 HF reference needs transformers in [{low}, {high}); found "
            f"{transformers.__version__} (see tests/_kimi_hf.py)"
        )


def hf_reference_dir() -> str:
    """Directory holding the pinned HF files, or ``pytest.skip``."""
    local = os.environ.get("KIMI_K3_HF_DIR")
    if local:
        missing = [f for f in HF_FILES if not os.path.exists(os.path.join(local, f))]
        if missing:
            pytest.skip(f"KIMI_K3_HF_DIR={local} lacks {missing}")
        return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        pytest.skip("huggingface_hub not installed and KIMI_K3_HF_DIR unset")
    try:
        paths = [hf_hub_download(HF_REPO, f, revision=KIMI_K3_HF_REVISION) for f in HF_FILES]
    except Exception as exc:  # network / auth failures -> skip, not error
        pytest.skip(f"cannot fetch {HF_REPO}@{KIMI_K3_HF_REVISION}: {exc}")
    return os.path.dirname(paths[0])


def import_hf_module(name: str):
    """Import ``<name>.py`` of the HF reference as part of a private package."""
    if _PACKAGE not in sys.modules:
        check_transformers_version()
        src = hf_reference_dir()
        root = tempfile.mkdtemp(prefix="kimi_k3_hf_")
        pkg = os.path.join(root, _PACKAGE)
        os.makedirs(pkg)
        for f in HF_FILES:
            shutil.copy(os.path.join(src, f), os.path.join(pkg, f))
        open(os.path.join(pkg, "__init__.py"), "w").close()
        sys.path.insert(0, root)
    return importlib.import_module(f"{_PACKAGE}.{name}")
