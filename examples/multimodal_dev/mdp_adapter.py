# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Registry dispatch for the MDP model adapter and the vision collate spec.

Both lookups go through ``MODEL_REGISTRY`` so adding an architecture never
touches the training entry point or the collator. Model-specific adapters live
in ``models/<arch>/mdp_adapter.py``.
"""

import weakref

from examples.multimodal_dev.models import registry_for


def build_mdp_adapter(args, language_config, vision_config=None):
    """Build the ``MdpModelAdapter`` registered for ``--model-arch``."""
    registry = registry_for(args)
    adapter_fn = registry.get("mdp_adapter_fn")
    if adapter_fn is None:
        raise ValueError(
            f"Model arch '{getattr(args, 'model_arch', None)}' has no 'mdp_adapter_fn' "
            "registry entry; --mdp-enable needs one."
        )
    return adapter_fn(args, language_config, vision_config)


#: Resolved specs, keyed by ``id(args)`` and guarded by a weak reference to that
#: ``args`` object (``argparse.Namespace`` is unhashable, so a WeakKeyDictionary
#: cannot hold it). The collator and the FLOPs accounting call
#: :func:`build_vision_collate_spec` once per micro-batch -- including on the
#: window-capture prefetch thread -- and the spec is a pure function of ``args``.
#: The cache must NOT live on ``args``: Megatron pickles ``args`` into every
#: checkpoint's common state, and a spec holds the model's row-arithmetic
#: callables (closures), which do not pickle.
_SPEC_CACHE: dict = {}


def _cached_spec(args):
    entry = _SPEC_CACHE.get(id(args))
    if entry is not None and entry[0]() is args:
        return entry[1]
    return None


def build_vision_collate_spec(args):
    """Build (once per ``args``) the ``VisionCollateSpec`` registered for ``--model-arch``.

    Used by the THD collator on every rank, including ranks that never build a
    vision encoder, so it must not require a model instance.
    """
    cached = _cached_spec(args)
    if cached is not None:
        return cached
    registry = registry_for(args)
    spec_fn = registry.get("vision_collate_spec_fn")
    if spec_fn is None:
        raise ValueError(
            f"Model arch '{getattr(args, 'model_arch', None)}' has no "
            "'vision_collate_spec_fn' registry entry; the vision sidecar needs one."
        )
    spec = spec_fn(args)
    try:
        _SPEC_CACHE[id(args)] = (weakref.ref(args), spec)
    except TypeError:  # not weak-referenceable (e.g. SimpleNamespace): no caching
        pass
    return spec
