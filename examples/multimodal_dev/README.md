# multimodal_dev — Standalone Multimodal Training

Standalone, model-agnostic training entry point for multimodal
vision-language models built on Megatron-Core (FSDP + EP).

## Directory Structure

```
multimodal_dev/
├── pretrain_multimodal.py   # Training entry point (model-agnostic)
├── forward_step.py          # Forward step, TP broadcast, loss computation
├── mdp_adapter.py           # Registry dispatch: MDP adapter + VisionCollateSpec
├── arguments.py             # Multimodal CLI arguments
├── data/
│   └── mock.py              # Mock dataset for end-to-end testing
├── models/
│   ├── __init__.py          # MODEL_REGISTRY — central model registry
│   ├── base.py              # MultimodalModel (vision encoder + built decoder), default decoder hook
│   ├── vision_base.py       # VisionCollateSpec, BaseGridVisionAdapter (shared MDP plumbing)
│   └── qwen35_vl/           # Qwen3.5-VL architecture
│       ├── factory.py       # Factory functions for pretrain entry point
│       ├── mdp_adapter.py   # Qwen3.5-VL MDP adapter + collate spec
│       ├── model.py         # Qwen35VLModel (MRoPE, vision encoder wiring)
│       ├── configuration.py # TransformerConfig builders and constants
│       ├── specs.py         # Vision layer spec builders
│       ├── mrope.py         # 3D MRoPE position ID computation
│       └── vision_encoder.py# ViT encoder (patch embed, merger, RoPE)
└── scripts/                 # Launch scripts (torchrun, Slurm)
```

## Quick Start

```bash
torchrun --nproc_per_node=8 multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider mock \
    --hybrid-layer-pattern GEGEGE*E/*E \
    --position-embedding-type mrope --mrope-section 11 11 10 \
    ... # other Megatron args (--hidden-size, etc.)
```

## HybridModel decoder

`multimodal_dev` uses `HybridModel` for the Qwen3.5-VL language decoder.
The unified `--hybrid-layer-pattern` is required. It replaces `--num-layers`
and the GPTModel-specific implicit in-layer attention/MLP ordering. The Qwen
launcher still declares `--experimental-attention-variant gated_delta_net` so
`TransformerConfig` validates the GDN dimensions (`--linear-key-head-dim`,
`--linear-num-value-heads`, ...) and related runtime constraints.

`--linear-attention-freq` is *not* passed. `TransformerConfig` skips its
assertion for hybrid models, and the only consumers are the GPTModel spec
builder, `TransformerBlock`, and the non-hybrid FLOPs branch — none of which
the HybridModel path reaches. The pattern is the single source of truth for
GDN / full-attention placement, and it is strictly more expressive than the
integer frequency (it matches the per-layer list form).

### Position embeddings

The decoder applies 3D MRoPE, so the recipe must say so: pass
`--position-embedding-type mrope` together with `--mrope-section 11 11 10`.
The sections are the temporal / height / width channel split and must sum to
half the rotary dimension (`kv_channels * rotary_percent / 2 = 32`); they must
also match `MROPE_SECTION` in `models/qwen35_vl/configuration.py`, which the
model factory enforces. `Qwen35VLModel` rejects any other
`--position-embedding-type`, so the parsed args, the constructed decoder, and
the checkpoint metadata can no longer disagree.

Each historical Qwen transformer block becomes two HybridModel layers:

- `G-` / `GE`: GatedDeltaNet followed by dense MLP / MoE.
- `*-` / `*E`: full attention followed by dense MLP / MoE.
- Qwen3.5's four-block cadence is `G-G-G-*-` for dense variants and
  `GEGEGE*E` for MoE variants.
- MTP patterns follow `/`; one Qwen MTP depth repeats the last block — `/*-`
  or `/*E` at the usual depths — because the former GPT path replicated the
  final transformer layer.

The supplied `scripts/run_qwen35_vl.sh` derives this pattern from
`NUM_LAYERS`, `NUM_EXPERTS`, and `MTP_NUM_LAYERS`. A `NUM_LAYERS` that is not
a multiple of 4 leaves trailing GatedDeltaNet blocks (and a `/G-` or `/GE` MTP
depth), which is what `--linear-attention-freq 4` produced on the GPT path, so
shallow proxy runs keep working.

### Token-indexed decoder features (Engram / PLE)

The decoder is called with `input_ids` on every pipeline stage, because every
stage collates its own batch. So Engram layers read token IDs directly, and the
decoder-only pre-schedule token broadcast is switched off for this entry
(`--no-engram-token-prefetch` is the parser default here). That broadcast expects
`batch["tokens"]` and would consume the MDP capture/replay iterator. Engram
hashes follow HF `qwen4_exp`: image/video placeholder IDs are hashed as-is, and
collate padding (`padding_mask`) is rewritten to the Engram boundary token
(EOS for the qwen variant) before the decoder sees `input_ids`. Packed documents
reset their n-gram windows at `cu_seqlens_q_padded`, where the rows actually
sit. Engram with `--use-packed-sequence` and `--context-parallel-size > 1` is
rejected at startup.

### Existing checkpoints

The decoder checkpoint layout changes when moving from GPTModel to
HybridModel: one GPT decoder layer is split across two Hybrid decoder layer
indices, the final norm key changes, and Hybrid MTP uses a nested HybridStack.
Therefore an existing GPTModel-format Qwen3.5-VL checkpoint is not directly
loadable by this path.

`tools/checkpoint/gpt_hybrid_conversion.py` is not currently sufficient for
Qwen3.5-VL because it explicitly rejects GDN (`G`) and MTP checkpoints. Use a
checkpoint exported directly for the HybridModel layout, or extend the
converter with Qwen-specific GDN and MTP key mapping and validate logits before
resuming training. `CKPT_LOAD` in the launch script must point to a
HybridModel-format checkpoint.

## Checkpoint Conversion (HF → Megatron-FSDP DTensor)

Convert a HuggingFace release to a Megatron-FSDP DTensor checkpoint via
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) before
pretraining from pretrained weights.

### Setup

Clone Bridge and pin its `3rdparty/Megatron-LM` submodule to this branch:

```bash
git clone --recurse-submodules https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge/3rdparty/Megatron-LM
git remote add wplf https://github.com/wplf/Megatron-LM.git
git fetch wplf feat/qwen35-vl-example
git checkout feat/qwen35-vl-example
cd ../..
```

### Convert

Single 8×GPU node, EP=8 / TP=CP=1; substitute any Qwen3.5 variant for
`--hf-model`:

```bash
PYTHONPATH=./src:./3rdparty/Megatron-LM/ \
  torchrun --nproc_per_node=8 \
  examples/conversion/mfsdp/convert_checkpoints_fsdp.py import \
  --hf-model Qwen/Qwen3.5-35B-A3B \
  --megatron-path ${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B-fsdp \
  --ckpt-format fsdp_dtensor \
  --ep 8
```

HF weights are auto-fetched on first run via `huggingface_hub`. Adjust
`--tp` / `--cp` / `--ep` to match the training topology (must satisfy
`WORLD_SIZE % (TP*CP*EP) == 0`).

### Output

```
${WORKSPACE}/models/Qwen/Qwen3.5-35B-A3B-fsdp/
├── iter_0000000/
│   ├── __0_0.distcp .. __7_0.distcp   # FSDP DTensor shards, one per rank (~18 GB each for 35B-A3B)
│   ├── .metadata
│   ├── run_config.yaml
│   └── train_state.pt
├── latest_checkpointed_iteration.txt
└── latest_train_state.pt
```

### Bridge dependency

Requires
[NVIDIA-NeMo/Megatron-Bridge#3987](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3987)
(skip tokenizer save). Without that fix the checkpoint is still written
correctly but the script exits non-zero after save with
`AttributeError: 'TokenizerConfig' object has no attribute 'make_vocab_size_divisible_by'`
against this branch's `megatron.core.tokenizers.utils.build_tokenizer`.

## Architecture

`pretrain_multimodal.py` is **model-agnostic**. All model-specific logic
is delegated to factory functions registered in `MODEL_REGISTRY`
(`models/__init__.py`). The entry point handles only generic concerns:

- Building `language_config` from Megatron CLI args
- Constructing `vision_config` via the registry
- Applying vision recompute and dtype propagation
- Binding the registry's decoder hook (`language_model_builder_fn`) to the
  stage placement and passing it to the model factory
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
    """(Optional) Set vision FLOPs metadata on args.

    ``num_floating_point_operations`` adds a vision-encoder term when
    ``count_vision_model_flops`` is set, so an arch that skips this hook
    reports language-model FLOPs only. All of the fields below are
    required once the flag is on:

        args.vision_num_layers, vision_hidden_size, vision_ffn_hidden_size,
        vision_num_attention_heads, vision_kv_channels, vision_in_channels,
        vision_patch_size, vision_temporal_patch_size, vision_out_hidden_size,
        vision_merge_dim

    plus the shape knobs ``vision_gated_mlp`` (default False) and
    ``vision_merger_hidden_size`` (default ``vision_merge_dim``). The
    per-batch row counts (patch rows, packed-attention ``L^2``, merged rows)
    are not derived from these fields: the collator measures them through
    the model's ``VisionCollateSpec.workload_of``. See
    ``megatron/training/training.py::vision_encoder_flops``.
    """
    args.count_vision_model_flops = True
    # ... set dimension fields for FLOPs calculation

def build_model(args, language_config, vision_config, language_model_builder, **kwargs):
    """(Required) Build and return the complete model instance.

    ``language_model_builder(**config_overrides)`` builds this stage's decoder
    (registry ``language_model_builder_fn``; default ``HybridModelBuilder``,
    so ``--spec`` and builder-side features apply). Never construct
    ``HybridModel(...)`` here -- it would silently drop whatever the builder
    adds. Pass only architectural constants as overrides.
    """
    from .model import LlavaNextModel

    return LlavaNextModel(
        language_config=language_config,
        # Build the vision encoder first, then call the builder: the order
        # fixes the parameter-initialization RNG stream.
        language_model_builder=lambda: language_model_builder(rotary_base=1_000_000),
        vision_config=vision_config,
        pre_process=kwargs.get("pre_process", True),
        vp_stage=kwargs.get("vp_stage"),
        # ... model-specific args
    )
```

`MultimodalModel(language_config, language_model, vision_encoder,
image_token_id, pre_process, vp_stage)` wraps an already-built decoder; any
decoder that accepts `decoder_input` / `input_ids` / `position_ids` /
`packed_seq_params` / `labels` and exposes `embedding` works (`HybridModel`,
`GPTModel`).

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
    "vision_collate_spec_fn": _llava_next_collate_spec,    # packed multimodal
    "mdp_adapter_fn": _llava_next_mdp_adapter,             # --mdp-enable
    "dataset_providers": {                                  # optional
        "mock": "multimodal_dev.data.llava_mock.train_valid_test_datasets_provider",
    },
}
```

### Step 3b — Vision collate spec and MDP adapter

The THD collator and MDP never read vision geometry from loose `args`
attributes; they ask the model. Provide, usually in
`models/<arch>/mdp_adapter.py`:

- `build_collate_spec(args) -> VisionCollateSpec`
  (`models/vision_base.py`): image-token id, `shape_key_width`,
  `payload_rows_of`, `output_rows_of`, `workload_of`, `grid_key`,
  `require_contiguous_slots`. It must be callable on every rank without a
  model instance.
- `build_mdp_adapter(args, language_config, vision_config)`: normally a
  `BaseGridVisionAdapter` subclass that only implements `build_encoder` and
  `encode`; `output_hidden_size` is the width of one encoder output row
  (the projector output), which MDP uses for its bridge buffers.

MDP core sees an item only as `payload_rows`, `output_rows`, an LPT cost and
an opaque `shape_key`; see `megatron/core/mdp/README.md`.

Shared code carries no architecture defaults: `--model-arch` is required and
`--image-token-id` defaults to `None`, meaning "the architecture's own id".
Resolve it once in the package (Qwen3.5-VL: `configuration.resolve_image_token_id`)
and use that everywhere the package needs the id (factory, collate spec,
datasets).

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
| `model_factory_fn` | Yes | `(args, language_config, vision_config, pre_process, post_process, language_model_builder, **kwargs) -> MegatronModule` |
| `language_model_builder_fn` | No | `(args, language_config, *, pre_process, post_process, vp_stage, pg_collection, **config_overrides) -> decoder`; default `models.base.build_language_model` |
| `vision_config_fn` | Yes | `(num_layers_override=None) -> TransformerConfig` |
| `post_language_config_fn` | No | `(language_config, args) -> None` |
| `vision_flops_fn` | No | `(args, language_config, vision_config) -> None` |
| `vision_collate_spec_fn` | For packed (THD) training | `(args) -> VisionCollateSpec` |
| `mdp_adapter_fn` | For `--mdp-enable` | `(args, language_config, vision_config) -> MdpModelAdapter` |
| `dataset_providers` | No | `Dict[str, str \| callable]` |
