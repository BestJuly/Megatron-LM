# Qwen3.5-family GB300 recipes

These recipes reproduce the best measured pretraining throughput of two
Qwen3.5-family MoE models on GB300, at 4K, 64K and 1M sequence length:

- **Qwen3.5-397B-A17B** on 128 GPUs, in [`gb300/`](gb300/).
- **Qwen3.8-2.4T-A95B** on 256 GPUs, in
  [`../qwen3_8_2_4t_a95b/gb300/`](../qwen3_8_2_4t_a95b/gb300/).

They use mock data with forced load balancing, so they are throughput
benchmarks, not convergence configurations. Both models share the container,
the launch procedure and most settings, so this guide covers both. Use the
Megatron-LM checkout that contains these recipes and `pretrain_gpt.py` as the
training entry point.

## Configurations

| Model | Recipe | GPUs | TP/PP/EP/CP/ETP | MBS/GBS/SL | Median TFLOP/s/GPU | Step time |
|---|---|---:|---|---|---:|---:|
| Qwen3.5-397B-A17B | [GB300 MXFP8 THD 4K](gb300/mxfp8_THD4K_128GPU_TP1PP2EP64_mok.yaml) | 128 | 1/2/64/1/1 | 1/2048/4096 | 768.1 | 9.32 s |
| Qwen3.5-397B-A17B | [GB300 MXFP8 THD 64K](gb300/mxfp8_THD64K_128GPU_TP1PP2EP64CP8_mok.yaml) | 128 | 1/2/64/8/1 | 1/256/65536 | 837.9 | 24.18 s |
| Qwen3.5-397B-A17B | [GB300 MXFP8 THD 1M](gb300/mxfp8_THD1M_128GPU_TP1PP2EP64CP64_mok.yaml) | 128 | 1/2/64/64/1 | 1/16/1048576 | 779.5 | 147.86 s |
| Qwen3.8-2.4T-A95B | [GB300 MXFP8 THD 4K](../qwen3_8_2_4t_a95b/gb300/mxfp8_THD4K_256GPU_TP1PP4EP64_mok.yaml) | 256 | 1/4/64/1/1 | 1/4096/4096 | 1064.9 | 36.23 s |
| Qwen3.8-2.4T-A95B | [GB300 MXFP8 THD 64K](../qwen3_8_2_4t_a95b/gb300/mxfp8_THD64K_256GPU_TP1PP4EP64CP16_mok.yaml) | 256 | 1/4/64/16/1 | 1/256/65536 | 902.4 | 52.86 s |
| Qwen3.8-2.4T-A95B | [GB300 MXFP8 THD 1M](../qwen3_8_2_4t_a95b/gb300/mxfp8_THD1M_256GPU_TP1PP4EP64CP64_mok.yaml) | 256 | 1/4/64/64/1 | 1/16/1048576 | 667.6 | 289.65 s |

A step processes 8.4M tokens for the 397B 4K recipe (2048 x 4096) and 16.8M
tokens for every other recipe.

### Models

Both models use the `qwen3_5_moe_text` architecture
(`Qwen3_5MoeForCausalLM`): every fourth layer is a full-attention GQA layer
with an output gate (`linear_attention_freq: 4`), and the rest are Gated
DeltaNet (GDN) layers. Both have 512 routed experts with top-10 routing, one
shared expert with the shared-expert gate, one MTP depth, head dimension 256,
a 248320-token vocabulary, and RoPE on 25% of the head dimension.

| | Qwen3.5-397B-A17B | Qwen3.8-2.4T-A95B |
|---|---:|---:|
| Layers (attention + GDN) | 60 (15 + 45) | 92 (23 + 69) |
| Hidden size | 4096 | 8192 |
| Attention heads / KV groups | 32 / 2 | 64 / 4 |
| GDN key / value heads | 16 / 64 | 16 / 128 |
| Expert and shared-expert FFN size | 1024 | 2048 |
| Total / active parameters | 397B / 17B | about 2.42T / 91B |

The dimensions follow the Hugging Face configs of
[Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json)
and [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json).
Qwen3.8-2.4T-A95B is a scaled-up Qwen3.5, not Qwen3.8-Flash-Next, which is a
different architecture.

### Common settings

All six recipes use MXFP8 GEMMs with MXFP8 parameter gather, the HybridEP flex
dispatcher, the Mixture-of-Kittens (MoK) MoE megakernel, pre-gated delta-rule
fusion, BF16 Adam moments with the precision-aware optimizer, TP1 and EP64,
and packed THD mock data in which every sample fills the whole sequence. The
GDN kernel is the cuDNN frontend backend (`gdn_gdr_backend: cudnn`), except at
1M, which uses the internal CuTe kernel.

The 2.4T recipes additionally offload the optimizer states to the CPU
(`offload_optimizer_states`), keep main gradients in BF16, enable sequence
parallelism, and always recompute activations: the 2.4T model does not fit on
256 GB300 GPUs without recompute.

| Model | Seq | PP layout | CP | Recompute | CUDA graph | Other | Peak device memory |
|---|---|---|---|---|---|---|---|
| 397B | 4K | `Et*16\|t*16\|t*16\|t*12,m,L` (PP2 x VPP2) | 1 | none | `[attn]` | `pipeline_p2p_fixed_shape` | 179.3 GiB ¹ |
| 397B | 64K | `Et*32\|t*28,m,L` (PP2, no VPP) | 8, headwise | none | `[attn]` | `pipeline_p2p_fixed_shape` | 240.5 GiB |
| 397B | 1M | `Et*16\|t*16\|t*16\|t*12,m,L` (PP2 x VPP2) | 64, chunkwise | full, block, 11 layers | none | internal GDR kernel | 261.8 GiB |
| 2.4T | 4K | `Et*24\|t*24\|t*24\|t*20,m,L` (PP4, no VPP) | 1 | selective `[moe_act, gdn]` | none | optimizer offload | 243.8 GiB |
| 2.4T | 64K | `Et*21\|t*25\|t*25\|t*21,m,L` (PP4, no VPP) | 16, headwise | selective `[moe_act, gdn]` | `[attn]` | optimizer offload | **273.3 GiB** |
| 2.4T | 1M | `Et*24\|t*24\|t*24\|t*20,m,L` (PP4, no VPP) | 64, chunkwise | full, uniform, 1 layer | none | optimizer offload, internal GDR kernel | 267.3 GiB |

GB300 has 276.6 GiB of device memory. In every recipe, PP stage 0 uses the
most memory. The 2.4T 64K recipe leaves only 3.3 GiB free on that stage; see
the tuning notes for a split with more headroom.

¹ Only 2 of 128 ranks reported memory in that run, so treat 179.3 GiB as
optimistic.

## Measured throughput

The numbers were measured from 2026-09-16 to 2026-09-18 on a GB300 NVL72
cluster with four GPUs per node (32 nodes for 397B, 64 nodes for 2.4T). TFLOP/s
is per GPU, and step time is the median over steady-state iterations.
Run-to-run noise at this scale is about ±2%.

- 397B 4K and 64K ran 20 iterations. The 64K figure is 836.8 under a
  different median window over the same run.
- 397B 1M stopped after 8 iterations; its median covers iterations 2-8
  (779.5-783.6 TFLOP/s/GPU).
- 2.4T 4K ran 28 clean iterations. 2.4T 64K was stopped at iteration 23,
  after the rate had plateaued.

All six numbers were measured at commit
[`2e8af47307c0`](https://github.com/BestJuly/Megatron-LM/commit/2e8af47307c03cc86af1cfa0a55ae3163aecf616)
of this branch. The commit that adds these recipes changes only files under
`examples/moe_recipes/`. The branch is `dev` @ `917e8e940c` plus commits that
these recipes require and that upstream `dev` does not have at that point:

- cuDNN frontend Gated DeltaNet backend (`gdn_gdr_backend: cudnn`);
- fixed-shape pipeline P2P for packed sequences (`pipeline_p2p_fixed_shape`,
  NVIDIA/Megatron-LM#7000);
- internal CuTe Gated DeltaNet kernel (`gdn_gdr_backend: internal`,
  NVIDIA/Megatron-LM#6511);
- MoK support for the shared-expert output gate (NVIDIA/Megatron-LM#7391);
- GDN activation recompute inside the FP8 autocast, and THD CUDA graph fixes
  for GDN layers.

## Build the container

All six YAMLs reference the shared [Dockerfile](Dockerfile) through
`DEPENDENCIES.dockerfile_path`, which is relative to the Megatron-LM repository
root. Build it on an ARM64 builder:

```bash
# Run from the Megatron-LM repository root.
docker build --platform=linux/arm64 \
    -f examples/moe_recipes/qwen3_5_397b/Dockerfile -t megatron-qwen3_5:gb300 .
```

The Dockerfile pins the components of the benchmark image:

| Component | Version / revision |
|---|---|
| NVIDIA PyTorch base | `nvcr.io/nvidia/pytorch:26.07-py3` (torch 2.13.0a0, CUDA 13.3, Python 3.12) |
| TransformerEngine | `5f6105b900778068144f1b24e25a2e84066983cb` (main) |
| cuDNN / cuDNN frontend | 9.25.1.1 / 1.28.0 |
| CUTLASS DSL / FlashAttention-4 | 4.7.0 / 4.0.0b29 |
| Flash Linear Attention | `6b6f546f9bd10a97be1e8cf2f0120de70cfca482` (0.6.0) |
| causal-conv1d | 1.7.0 |
| HybridEP (DeepEP `hybrid-ep`) | `58397eac493b075b0baea5861ac6e9793d8d7a21` |
| Mixture-of-Kittens | `05d7db7d74f77b6396f98df886f1f9bcda97ec51`, built for SM103 |
| Transformers / NVIDIA Resiliency Extension | 4.57.1 / 0.6.0 |

MoK is compiled for SM103 only, so the image runs on GB300 and not on GB200.
`gdn_pre_gated_delta_rule_fusion` needs causal-conv1d, and
`gdn_gdr_backend: cudnn` needs the `cudnn.fla` module from cuDNN frontend 1.28.

The benchmark image was not built from this Dockerfile. It was assembled as a
series of container overlays on the same NGC base, and it also had an apt copy
of cuDNN 9.25.0 next to the PyPI 9.25.1.1. That copy does not affect these
decoder-only recipes. The Dockerfile reproduces the recorded component versions,
but the recipes have not been re-measured on an image built from it.

## Prepare the launch

The YAML uses the recipe schema `DEPENDENCIES` / `ENTRY` / `ENV_VARS` / `ARGS`,
not the `--yaml-cfg` schema of the training entry point. Export `ENV_VARS` and
convert `ARGS` to command-line flags: lists become multiple arguments, `true`
becomes a bare flag, and `false` and `null` are omitted.

The following example writes a launch script from one recipe. Run it inside the
built container from the repository root with PyYAML available and `OUTPUT_PATH`
set to a writable directory that every node can see:

```bash
export RECIPE=examples/moe_recipes/qwen3_5_397b/gb300/mxfp8_THD64K_128GPU_TP1PP2EP64CP8_mok.yaml
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the run output directory}"
export OUTPUT_PATH
python - <<'PYTHON' > run-qwen3_5.sh
import os
import shlex
import yaml

with open(os.environ["RECIPE"]) as stream:
    recipe = yaml.safe_load(stream)
print("#!/usr/bin/env bash\nset -euo pipefail")
for key, value in recipe["ENV_VARS"].items():
    print(f"export {key}={shlex.quote(str(value))}")
args = []
for key, value in recipe["ARGS"].items():
    if value is None or value is False:
        continue
    args.append("--" + key.replace("_", "-"))
    if value is not True:
        values = value if isinstance(value, list) else [value]
        args.extend(os.path.expandvars(str(item)) for item in values)
launcher = (
    'exec python -m torch.distributed.run --nnodes="${NNODES:?}" '
    '--nproc-per-node="${GPUS_PER_NODE:-4}" --node-rank="${NODE_RANK:?}" '
    '--master-addr="${MASTER_ADDR:?}" --master-port="${MASTER_PORT:-29500}" '
    + recipe["ENTRY"]["script"] + " "
)
print(launcher + shlex.join(args) + ' "$@"')
PYTHON
```

Run `bash run-qwen3_5.sh` once on each node through your scheduler: 32 nodes
for the 397B recipes and 64 nodes for the 2.4T recipes. Provide `NNODES`, a
unique `NODE_RANK`, and the same `MASTER_ADDR` and `MASTER_PORT` on every node.

- **Placement.** Each pipeline stage holds 64 GPUs (one EP64 group), and MoK
  and HybridEP require that group to sit in one NVLink domain. Assign node ranks
  so that each block of 16 consecutive nodes (0-15, 16-31, ...) shares one
  NVL72 domain. The benchmark used SLURM `--segment=16` for this.
- **Tokenizer.** The recipes set `HF_HUB_OFFLINE=1` and load the tokenizer from
  Hugging Face (`Qwen/Qwen3.5-397B-A17B` or `Qwen/Qwen3.8-2.4T-A95B`). Download
  it into the Hugging Face cache first, or remove `HF_HUB_OFFLINE`.
- **Run length.** The 397B recipes stop after 20 iterations
  (`exit_interval: 20`), and their cosine schedule spans the full
  `train_samples`, so the learning rate stays near its peak. The 2.4T recipes
  run 50 iterations with a 50-iteration cosine schedule.
- **Evaluation.** The 2.4T recipes keep `eval_iters: 1`. With
  `use_varlen_dataset`, `eval_iters: 0` makes the validation data loader fail
  because it has no sample to consume.

## Tuning notes

- **MoK knobs depend on tokens per rank, which is `seq_length / CP`.** They do
  not depend on the model or on the sequence length alone. If you change CP,
  pick the row again:

  | Tokens/rank | Used by | fwd/bwd comm SMs | Minibatch | Macrobatch | Capacity multiplier |
  |---:|---|---|---:|---:|---:|
  | 4096 | 397B 4K, 2.4T 4K, 2.4T 64K | 64/64 | 8192 | 65536 | 0.0625 |
  | 8192 | 397B 64K | 72/60 | 8192 | 131072 | 0.0625 |
  | 16384 | 397B 1M, 2.4T 1M | 74/60 | 8192 | 262144 | 0.0625 |

  On 397B, these swept values beat a single hand-tuned setting by 2.0% (4K),
  7.6% (64K) and 4.6% (1M).
- **Do not add VPP to the 2.4T recipes.** On these hybrid models VPP keeps the
  per-rank layer counts but moves the attention/GDN mix between stages, and the
  selective recompute list covers only GDN. On 2.4T 4K, VPP4 measured 922
  TFLOP/s/GPU and hit an illegal memory access on stage 1; without VPP the
  result is 15.5% higher and stage 1 uses 42.5 GiB less. On 2.4T 1M, dropping
  VPP gained 10.1% (606.4 with VPP). On 397B, recheck memory per stage whenever
  you change the layout.
- **397B 64K drops VPP and recompute.** It gains 4.2% over PP2 x VPP2 with
  `moe_act` recompute (804.3). The uneven 32/28 split puts the embedding on
  stage 0 and the MTP layer and loss on stage 1; a 31/29 split measured 797.8.
- **`pipeline_p2p_fixed_shape`** skips the pipeline shape exchange, because every
  packed microbatch is padded to `max_seqlen_per_dp_cp_rank`. It is worth 2.2%
  on 397B 4K.
- **2.4T 64K: the attention CUDA graph is worth 24.6%** (about 724
  TFLOP/s/GPU without it), but its graph pool adds about 12 GiB per in-flight
  1F1B microbatch, which leaves PP stage 0 with 3.3 GiB free. If it runs out of
  memory, the layout `Et*19|t*26|t*26|t*21,m,L` leaves 29.8 GiB free on stage 0
  at a 1.7% cost (886.9).
- **Keep `FLA_DISABLE_TENSOR_CACHE=1`** when GDN layers run under CUDA graphs.
  Without it, the 2.4T 64K loss diverged silently and nothing crashed. All six
  recipes set it.
- **397B 1M: block recompute of 11 layers is the minimum that fits.** With 10
  layers, PP stage 0 runs out of memory. Each layer left un-recomputed costs
  about 11 GiB at 16384 tokens/rank. Do not carry this value over to other
  models or shapes. 2.4T 1M already recomputes every layer, so the remaining
  levers are the layout and offloading.
- **The internal GDR kernel (1M) needs `MCORE_GDN_MIN_PAD_GRANULARITY=64`.** Its
  CuTe kernels accept only packed lengths that are a multiple of 64. They are
  JIT-compiled in the first iteration, so the first step is slow. At CP64 GDN
  uses chunkwise CP, because headwise CP needs the 16 key heads to be divisible
  by TP x CP.
- The CUDA graph recipes capture only the attention modules
  (`cuda_graph_modules: [attn]`). MoK does not support capturing the router or
  the MoE preprocessing.
