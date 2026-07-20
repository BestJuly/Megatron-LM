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

## Qwen3.5-VL Energon Data

The `energon` provider packs source documents into fixed-length THD
sequences after inspecting image metadata. The standalone provider materializes
every selected image before returning the packed batch; later distributed
vision changes may partition that materialization without changing the data
contract.

Each CrudeWebDataset sample contains a `.json` member with `text`,
`conversation`, `conversations`, or `messages`. Its
`image_descriptors` follow image-token order and provide either `grid_thw`
or `width` and `height`. Two image storage forms are supported:

- JSON-only lazy descriptors name a `materializer` module and identify an
  image in a zip or parquet source. Mantis-Instruct, M4-Instruct, and PixMo
  descriptor materializers are included.
- A prepared `.jpgs` member contains a pickled list of raw image byte strings
  in descriptor order. Its restricted decoder rejects pickle globals and
  non-byte values.

Select the Qwen cooker in a metadataset entry:

```yaml
datasets:
  - path: /path/to/prepared-shards
    weight: 1.0
    subflavors:
      crude_type: qwen35
```

Launch with an external dataloader and micro-batch size one:

```bash
torchrun --nproc_per_node=8 examples/multimodal_dev/pretrain_multimodal.py \
    --model-arch qwen35_vl \
    --dataset-provider energon \
    --energon-path /path/to/energon-dataset \
    --dataloader-type external \
    --micro-batch-size 1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/Qwen3.5-VL \
    ...
```

`--image-min-pixels` and `--image-max-pixels` control aspect-preserving
resize by area; zero disables that bound. Packing is controlled by
`--energon-packing-buffer-size`,
`--energon-max-samples-per-sequence`, and the fixed
`--total-seq-length` container size.

### MDP vision prepartitioning

For pipeline-parallel packed training, select the PP x CP ownership group and
optionally fuse one optimization step's vision work into bounded packs:

```bash
--mdp-encoder-mode \
--mdp-inner-dp-scope pp_cp \
--mdp-fused-vision-window \
--mdp-vision-encoder-max-sequence-length 131072 \
--mdp-fused-vision-backward recompute
```

The maximum sequence length is a raw-patch pack limit, not an automatic GPU
memory limit. `131072` completed 50 iterations with 16 GB200 nodes, PP2 x CP2,
and the Mantis + M4 + PixMo blend. Revalidate the cap for other hardware,
topologies, or data distributions.

`recompute` rebuilds each vision pack during backward and uses less memory.
`retain` keeps the forward graph until its dependent text microbatches finish;
it can use substantially more memory. Repeated world-64 measurements did not
establish a consistent throughput winner between the two modes.

### GB200 world-64 performance reproduction

The scripts in `scripts/gb200_report_repro/` recreate the launch contract used
by the performance sections of stacked PRs #1, #2, #4, #5, and #6. They run on
this cluster's 16-node allocation with four GB200 GPUs per node, the ARM64
PyTorch 26.02 container, and the prepared Mantis + M4 + PixMo Energon blend.

| Report | Script | Cells | Topology |
|---|---|---|---|
| PR1 | `reproduce_pr1.sh` | PR1 + PR2 ordinary-loader baseline | PP1 x CP2 |
| PR2 | `reproduce_pr2.sh` | ordinary-loader baseline | PP1 x CP2 |
| PR4 | `reproduce_pr4.sh` | PR2 baseline, PR4 MDP off/on | PP1 x CP2 |
| PR5 | `reproduce_pr5.sh` | PP sidecar MDP off/on | PP2 x CP2 |
| PR6 | `reproduce_pr6.sh` | MDP off/on, fused retain/recompute at cap 131072 | PP2 x CP2 |

Each listed file is intentionally standalone. It contains its own pinned
commits, cell matrix, allocation validation, container-side runner, and result
summarizer; none of the PR scripts sources or invokes a shared reproduction
helper. Invoke only the `reproduce_pr*.sh` file for the report being measured.

PR1 does not contain the real Energon provider, so its PR body intentionally
reports the combined PR1 + PR2 boundary. `reproduce_pr1.sh` therefore runs the
PR2 feature SHA and labels the result `pr1_pr2_baseline`; it does not claim an
isolated PR1 data-path result.

Allocate from the login node. Do not enter a node with `srun --pty` before
starting the multi-node step.

```bash
salloc -A coreai_devtech_all \
  -J 'coreai_devtech_all-megatron:vlm' \
  -N 16 -p batch --qos=normal --gres=gpu:4 \
  --time=04:00:00 bash

examples/multimodal_dev/scripts/gb200_report_repro/reproduce_pr4.sh \
  --job-id "${SLURM_JOB_ID}"
```

Each standalone script prepares clean detached worktrees for its required PR
boundaries, runs one `srun` task per node, and lets the checked-out GB200
training launcher start four local workers. All report cells use world size 64, BF16 HybridEP,
`seq=8192`, `MBS=1`, `GBS=256`, `TP=1`, `CP=2`, `EP=8`, and `ETP=1`.
They execute five warmup plus 45 measured iterations. Historical PR1-PR5
references use the PR-body window (iterations 10-50); the PR6 safety report
uses iterations 6-50.

The default cluster paths are:

```text
container: /lustre/fsw/portfolios/coreai/users/dongjael/containers/mcore-moe-pytorch26.02-hybridep7febc6e-arm64.sqsh
energon:   /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/dongjael/Megatron-Energon-7.3.2
data:      /lustre/fsw/portfolios/coreai/users/dongjael/datasets/qwen35-mdp-data
blend:     /data/energon/blends/blend3.yaml
tokenizer: /data/tokenizer/Qwen3.5-35B-A3B
```

Override them with `CONTAINER_IMAGE`, `ENERGON_HOST`, `DATA_HOST`,
`RAW_DATA_HOST`, or `VENV_HOST`. Use `--dry-run` without an allocation to
prepare every exact checkout and validate the generated launcher commands:

```bash
examples/multimodal_dev/scripts/gb200_report_repro/reproduce_pr6.sh --dry-run
```

Every real run writes code revisions, the container and blend identity,
per-task logs, `manifest.tsv`, `summary.tsv`, and `summary.md` under its result
root. The manifest includes the number reported by the PR next to the observed
number; `summary.md` calculates the descriptive delta.

The PR1-PR5 reference numbers predate the final restack, while the scripts use
the final functional SHAs listed in those PRs. PR6's long-run references were
measured at `cf12b34e2`; its standalone script uses `e0450f1b`, which adds the
validated quality fix and completed a world-64 real-data smoke. The first complete run of
these scripts is therefore also the canonical long-run refresh for the final
code boundaries.

The bundled reference contract is:

| Cell | Statistic | Step | Padded tok/s | Peak |
|---|---|---:|---:|---:|
| PR1/PR2 ordinary loader | median | 20,619.3 ms | 101,708.21 | 101.0977 GiB reserved |
| PR4 MDP off | median | 19,745.3 ms | 106,210.19 | 101.0566 GiB reserved |
| PR4 MDP on | median | 18,910.4 ms | 110,899.40 | 76.6074 GiB reserved |
| PR5 MDP off | median | 37,782.3 ms | 55,506.20 | 111.7617 GiB reserved |
| PR5 MDP on | median | 29,419.9 ms | 71,283.45 | 138.6836 GiB reserved |
| PR6 MDP off | mean | 30,481.0 ms | 68,801.94 | 96,974.57 MB allocated |
| PR6 MDP on | mean | 24,086.0 ms | 87,069.33 | 127,735.80 MB allocated |
| PR6 fused retain, cap 131072 | mean | 13,030.0 ms | 160,947.97 | 128,383.62 MB allocated |
| PR6 fused recompute, cap 131072 | mean | 12,775.0 ms | 164,160.63 | 92,029.23 MB allocated |

PR6 uses cap `131072`, the value validated by the documented
GB200/world64/blend3/50-iteration safety run.

The PR reports are single-run diagnostics. They did not pin external payload
bytes or sample identity and have no repeat variance estimate. These scripts
pin the code boundary, topology, launcher arguments, measurement window,
container path, and current blend-file hash, but exact timing equality is not
guaranteed if node placement, payloads, or runtime state differ.

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
