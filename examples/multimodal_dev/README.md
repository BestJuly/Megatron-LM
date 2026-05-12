# multimodal_dev — Standalone Multimodal Training

Standalone, model-agnostic training entry point for multimodal
vision-language models built on Megatron-Core (FSDP + EP).

## Directory Structure

```
multimodal_dev/
├── pretrain_multimodal.py   # Training entry point (model-agnostic)
├── forward_step.py          # Forward step, TP broadcast, loss computation
├── arguments.py             # Multimodal CLI arguments
├── data/
│   └── mock.py              # Mock dataset for end-to-end testing
├── models/
│   ├── __init__.py          # MODEL_REGISTRY — central model registry
│   ├── base.py              # MultimodalModel base class (vision encoder + GPTModel)
│   └── qwen35_vl/           # Qwen3.5-VL architecture
│       ├── factory.py       # Factory functions for pretrain entry point
│       ├── model.py         # Qwen35VLModel (MRoPE, vision encoder wiring)
│       ├── configuration.py # TransformerConfig builders and constants
│       ├── specs.py         # Layer spec builders (hybrid attention, ViT)
│       ├── mrope.py         # 3D MRoPE position ID computation
│       └── vision_encoder.py# ViT encoder (patch embed, merger, RoPE)
└── scripts/                 # Launch scripts (torchrun, Slurm)
```

## Quick Start

```bash
torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider mock \
    ... # other Megatron args (--num-layers, --hidden-size, etc.)
```

Pipeline parallelism (vision encoder is fused into the first PP stage; the
language decoder spans all PP stages following standard Megatron PP layout
flags). FSDP and PP are mutually exclusive in Megatron's standard path —
the launcher script auto-disables FSDP when ``PP>1``:

```bash
PP=2 GPUS_PER_NODE=4 ./examples/multimodal_dev/scripts/run_qwen35_vl.sh
```

## Architecture

`pretrain_multimodal.py` is **model-agnostic**. All model-specific logic
is delegated to factory functions registered in `MODEL_REGISTRY`
(`models/__init__.py`). The entry point handles only generic concerns:

- Building `language_config` from Megatron CLI args
- Constructing `vision_config` via the registry
- Applying vision recompute and dtype propagation
- Routing to model and dataset factories

The `forward_step` is also model-agnostic — it uses the model's
`compute_position_ids()` method polymorphically and passes a standard
batch dict.

## Adding a New Model Architecture

Adding a new model (e.g. `llava_next`) requires **no changes** to
`pretrain_multimodal.py` or `forward_step.py`. Follow these steps:

### Step 1 — Create the model package

```
multimodal_dev/models/llava_next/
├── __init__.py
├── factory.py          # Required: factory functions
├── configuration.py    # Vision/language TransformerConfig builders
├── model.py            # Model class (subclass MultimodalModel)
├── specs.py            # Layer spec builders
└── vision_encoder.py   # Vision encoder (if custom)
```

### Step 2 — Implement factory functions

Create `factory.py` with up to three functions:

```python
# models/llava_next/factory.py

def post_language_config(language_config, args):
    """(Optional) Mutate language_config with model-specific fields."""
    # e.g. language_config.some_field = value
    pass

def set_vision_flops_metadata(args, language_config, vision_config):
    """(Optional) Set vision FLOPs metadata on args."""
    args.count_vision_model_flops = True
    args.vision_flops_variant = "llava_next"
    # ... set dimension fields for FLOPs calculation

def build_model(args, language_config, vision_config, **kwargs):
    """(Required) Build and return the complete model instance."""
    from .model import LlavaNextModel
    from .specs import get_llava_next_language_spec

    language_spec = get_llava_next_language_spec(
        config=language_config,
        vp_stage=kwargs.get("vp_stage", None),
        pp_rank=None,
    )
    return LlavaNextModel(
        language_config=language_config,
        language_spec=language_spec,
        vision_config=vision_config,
        # ... model-specific args
    )
```

### Step 3 — Register in `MODEL_REGISTRY`

Add an entry in `models/__init__.py`:

```python
from multimodal_dev.models.llava_next.configuration import (
    get_llava_next_vision_config,
)
from multimodal_dev.models.llava_next.factory import (
    build_model as _build_llava_next_model,
    post_language_config as _llava_next_post_language_config,
    set_vision_flops_metadata as _llava_next_vision_flops,
)

MODEL_REGISTRY["llava_next"] = {
    "model_factory_fn": _build_llava_next_model,           # required
    "vision_config_fn": get_llava_next_vision_config,      # required
    "post_language_config_fn": _llava_next_post_language_config,  # optional
    "vision_flops_fn": _llava_next_vision_flops,           # optional
    "dataset_providers": {                                  # optional
        "mock": "multimodal_dev.data.llava_mock.train_valid_test_datasets_provider",
    },
}
```

### Step 4 — (Optional) Add a dataset provider

Create a dataset module under `data/` if the model needs custom data
preprocessing. The provider function signature is:

```python
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Return (train_dataset, val_dataset, test_dataset)."""
    ...
```

Register it in the `dataset_providers` dict of the registry entry.
Providers can be either direct callables or dotted import path strings
(resolved lazily at runtime).

### Step 5 — Launch

```bash
torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \
    --model-arch llava_next \
    --dataset-provider mock \
    ...
```

## Registry Entry Reference

| Field | Required | Signature |
|-------|----------|-----------|
| `model_factory_fn` | Yes | `(args, language_config, vision_config, **kwargs) -> MegatronModule` |
| `vision_config_fn` | Yes | `(num_layers_override=None) -> TransformerConfig` |
| `post_language_config_fn` | No | `(language_config, args) -> None` |
| `vision_flops_fn` | No | `(args, language_config, vision_config) -> None` |
| `dataset_providers` | No | `Dict[str, str \| callable]` |
