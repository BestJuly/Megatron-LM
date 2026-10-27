# Kimi-K3 Blackwell recipes

These recipes describe the Kimi-K3 text backbone and a nine-decoder-block
proxy, including model parameters, runtime arguments and a shared ARM64
container build. Use the MCore checkout containing these recipes and
`pretrain_hybrid.py` as the training entry point.

**Status: all recipes now run the Mixture-of-Kittens (MoK) MoE megakernel and
fused FLA attention residuals with the output RMSNorm fused in, and they no
longer recompute layernorms. The full model is published as the packed THD
256K recipe on 256 GB300 GPUs and the 1M recipe on 512 GPUs; the 64K
configuration is the 256K recipe at CP2 and is documented below instead of
being shipped as a third file. Every length keeps the same local budget of
32768 tokens per CP rank (8192 per GPU). Recorded 2026-10-28 runs: 760
TFLOP/s/GPU at 64K, 813 at 256K and 813 at 1M for the full model; on the 9L
proxy at 512K this release's settings move the previous release's 907 to
978. See the measured-throughput and validation sections below.**

## Configurations

| Recipe | GPUs | TP/PP/EP/CP/ETP | MBS/GBS/SL | MLA CP communication | Decoder blocks / HybridModel entries | AttnRes block entries |
|---|---:|---|---|---|---|---:|
| [Kimi-K3 GB300, packed THD 256K](gb300/mxfp8_THD256K_256GPU_TP4PP4EP64CP8_muon.yaml) | 256 | 4/4/64/8/1 | 1/64/262144 | a2a8 | 93 / 186 | 24 |
| [Kimi-K3 GB300, packed THD 1M](gb300/mxfp8_THD1M_512GPU_TP4PP4EP64CP32_muon.yaml) | 512 | 4/4/64/32/1 | 1/32/1048576 | a2a8 + p2p4 | 93 / 186 | 24 |
| [Kimi-K3 9L proxy GB300, packed THD 512K](../kimi_k3_proxy_9layer/gb300/mxfp8_THD512K_64GPU_TP4PP1EP64CP16_muon.yaml) | 64 | 4/1/64/16/1 | 1/32/524288 | a2a8 + p2p2 | 9 / 18 | 2 |
| [Kimi-K3 9L proxy GB300, packed THD 64K](../kimi_k3_proxy_9layer/gb300/mxfp8_THD64K_64GPU_TP4PP1EP64CP2_muon.yaml) | 64 | 4/1/64/2/1 | 1/256/65536 | a2a2 | 9 / 18 | 2 |

Counts in the table exclude the one MTP prediction depth, which adds an
MLA+MoE block (`/+E`) to every configuration. The proxy runs without
pipeline parallelism on 64 GPUs: dense DP=1 (TP4 x CP16) at 512K and dense
DP=8 (TP4 x CP2 x DP8) at 64K, 16,777,216 tokens per optimizer step with 32
microbatches per DP rank in both cases.

### Full-model sequence lengths

The full model was measured at three sequence lengths with one parallel
layout (TP4 + sequence parallel, PP4, EP64, ETP1, MBS 1; four pipeline
stages of 64 GPUs on 256 GPUs and of 128 GPUs on 512 GPUs) and one local
budget of 32768 tokens per CP rank. Only CP, the batch size and the node
count change with the length:

| Sequence length | GPUs (nodes) | CP | `cp_comm_type` / `hierarchical_context_parallel_sizes` | Dense DP / expert DP | GBS | Microbatches per DP rank | Tokens per step |
|---|---:|---:|---|---|---:|---:|---:|
| 64K (65536) | 256 (64) | 2 | `a2a` / `[2, 1]` | 8 / 1 | 256 | 32 | 16.8M |
| 256K (262144) | 256 (64) | 8 | `a2a` / `[8, 1]` | 2 / 1 | 64 | 32 | 16.8M |
| 1M (1048576) | 512 (128) | 32 | `a2a+p2p` / `[8, 4]` | 1 / 2 | 32 | 32 | 33.6M |

The 64K configuration is the 256K recipe with `seq_length: 65536`, the mock
document length set to 65536 in `varlen_mock_dataset_config_json`,
`context_parallel_size: 2`, `hierarchical_context_parallel_sizes: [2, 1]`
and `global_batch_size: 256`; nothing else differs, which is why it is not
shipped as a separate file. The 1M recipe as published uses GBS 32
(33,554,432 tokens per step), the setting of the recorded run; GBS 16
restores the family's 16,777,216 tokens per step.

### Model and runtime settings

- Model dimensions: hidden size 7168, 96 attention heads, 896 routed experts,
  top-16 routing, MoE latent size 3584 and expert FFN size 3072.
- The proxy retains the full model's widths and expert count. Its main pattern
  contains seven KDA blocks, two MLA blocks, one dense MLP and eight MoE MLPs.
- The full model uses four pipeline stages with 24/24/24/21 decoder blocks
  (48/48/48/42 HybridModel entries); MTP runs on the last stage. The pipes in
  `hybrid_layer_pattern` define this layout.
- `attn_res_block_layers` counts individual HybridModel entries. An attention
  operation and its following MLP count as two entries.
- All use MXFP8 parameter gather, per-head Muon in duplicated TP mode,
  pre-GDR fusion, gated output-norm fusion and HybridEP dispatch.
- Attention residuals run on the FLA implementation with the block's output
  RMSNorm fused into the residual kernel (`attn_res_impl: fla`,
  `attn_res_norm_fusion: true`). This replaces the compiled AttnRes path of
  the earlier recipes; `DYNAMO_CACHE_SIZE_LIMIT` stays exported for
  compatibility.
- The MoE layers run the Mixture-of-Kittens megakernel
  (`moe_megakernel_backend: mok`, `moe_megakernel_backend_config:
  {shared_expert_mode: native}`): the routed experts' dispatch, grouped GEMMs
  and combine execute inside one persistent kernel, the shared experts stay
  on the native path. HybridEP remains the token dispatcher of record for the
  non-megakernel fallback.
- Selective recomputation covers `gdn`, `moe` and `mlp`; the layernorm
  recompute of the earlier recipes is dropped because the fused AttnRes keeps
  the same peak memory without it.
  Fine-grained activation offloading covers `core_attn` and `attn_proj`, with
  `NVTE_CPU_OFFLOAD_V1=1` selecting TE's activation-only CPU offload path.
  MLA up-projection and core-attention recomputation must remain disabled
  with these latent CP configurations.
- Chunked optimizer-state offload moves states to CPU between GPU optimizer
  computations. The chunk size is 256 MiB and the offload fraction is 1.0.
- The `dp_balanced` scheduler pads each buffer to
  `max_seqlen_per_dp_cp_rank` x CP: 32768 x 2 = 65536 tokens at 64K,
  32768 x 8 = 262144 at 256K, 32768 x 16 = 524288 at the proxy's 512K and
  32768 x 32 = 1048576 at 1M. The HybridEP chunk size is 128 tokens.
- The packed buffer uses a contiguous CP partition, matching KDA's native
  chunkwise layout. MLA handles the attention layout internally; the
  sequence-parallel conversion combines TP gathering and CP redistribution
  into one TP x CP all-to-all.
- `mla_latent_cp: true` exchanges compressed latent KV and reconstructs K/V
  for each attention step. CP grows with the sequence length so that every
  CP rank keeps 32768 tokens: the all-to-all group takes the first eight CP
  ranks and the latent ring the remainder. At 64K, `cp_comm_type: a2a` uses
  two CP ranks, 12 heads per rank and one causal attention step without a
  ring; at 256K, eight a2a ranks with three heads per rank and no ring; at
  the proxy's 512K, `cp_comm_type: a2a+p2p` with
  `hierarchical_context_parallel_sizes: [8, 2]` adds a two-step latent
  ring; at 1M, `[8, 4]` a four-step ring. The
  forward ring uses one stream (`mla_latent_cp_two_stream` is off). All use
  `attention_backend: flash` with FlashAttention-4 4.0.0b11.
- The full-model recipes and the proxy 512K recipe use fixed-length mock
  documents and forced balanced routing. The proxy 64K recipe uses
  LongAlign-10k and Quantile Balancing without forced routing. All use a
  50-step cosine schedule with a peak learning rate of 1e-5. MTP and per-head
  Muon remain experimental settings.

### NCCL memory settings

All recipes set `NCCL_LL128_BUFFSIZE=1048576` and `NCCL_PROTO=^LL128` to
reduce communication-buffer memory; the 614400-byte LL128 buffer used before
2026-10-08 breaks the RoCE transport's connection setup on GB300 NVL72
systems. The proxy keeps `NCCL_BUFFSIZE=1048576`; the pipeline-parallel
full-model recipes use `NCCL_BUFFSIZE=2097152`, because 1 MiB SIMPLE buffers
halve the inter-rack pipeline point-to-point bandwidth. All recipes also set
`strict_runtime_validation_frequency: once_per_microbatch`, which runs the
hybrid layers' strict runtime input checks on the first instance of each
layer type per microbatch instead of in every layer (the per-layer checks
cost about 3.8 ms of host work and four stream synchronizations per KDA
forward, 0.5% of the 512K step). The shared
[NCCL communicator configuration](nccl/tp_dp_cp_max_ctas16.yaml) sets
`max_ctas: 16` only for `tp_dp_cp`, the TP x CP all-to-all group
(eight ranks at 64K, 32 at 256K, 64 at the proxy's 512K and 128 at 1M).
Other process groups retain NCCL's default communicator settings.

`nccl_communicator_config_path` is relative to the Megatron-LM repository
root. Run the launch from that directory on every node. These settings were
measured with the TP x CP group inside one GB300 NVLink domain; reassess them
when changing the parallel layout or interconnect.

## Measured throughput

The recorded runs of 2026-10-28 used the benchmark image with MoK (NCCL
2.32.3, HybridEP 10d4dd7 with the RUNPATH fix, FlashAttention-4 4.0.0b11,
mixture-of-kittens d59188a) and this recipe family on mock data with forced
balanced routing. The 256-GPU configurations (the 256K recipe and its 64K
variant described above) ran on 64 GB300 nodes (four GPUs each) for 20
iterations, the 1M recipe on 128 nodes for 5, the proxy on 16 nodes for 20. The measurements override the recipes with `--train-iters N
--eval-iters 1`. Nsight Systems profiled the last two iterations, one rank
per pipeline stage for the full model and rank 0 for the proxy. Iteration
times are medians over the steady steps (2-18 of 20, 2-5 of 5); memory is
the maximum allocated over all ranks.

| Metric | Full model 64K, 256 GPUs | Full model 256K, 256 GPUs | Full model 1M, 512 GPUs | 9L proxy 512K, 64 GPUs |
|---|---:|---:|---:|---:|
| Median iteration time | 67.98 s | 100.11 s | 246.27 s | 60.44 s |
| Median throughput | 760 TFLOP/s/GPU | 813 TFLOP/s/GPU | 813 TFLOP/s/GPU | 978 TFLOP/s/GPU |
| Tokens per iteration | 16.8M (256 x 65536) | 16.8M (64 x 262144) | 33.6M (32 x 1048576) | 16.8M (32 x 524288) |
| First iteration | 1744 s | 1898 s | 2126 s | 635 s |
| Peak allocated GPU memory | 157.7 GB | 157.9 GB | 165.4 GB | 67.7 GB |
| Peak reserved GPU memory | 165.3 GB | 165.9 GB | 186.8 GB | 78.0 GB |

All four runs completed without errors and include one MTP depth. The first
iteration includes data construction, MoK CuTe-DSL compilation and
autotuning, and communication setup. The same recipes measured on
2026-10-27 with the pre-rebase code base gave 67.59 s / 765, 100.20 s / 812
and (at GBS 16) 132.51 s / 755 TFLOP/s/GPU with the same peak memory, so the
rebase onto Megatron Core `dev` left the step time and memory unchanged.

On the 9L proxy the new settings move the 512K point from 65.22 s / 907
TFLOP/s/GPU (previous release) to 60.44 s / 978 (−7.3 % step time; MoK
accounts for about 6.6 % of it and the fused AttnRes with the dropped
layernorm recompute for the rest). The proxy comparison isolates the effect
of this release's settings; the full-model columns are the published
throughput of the family at its three sequence lengths.

Validation scope. The fused-AttnRes change was checked on the 9L 64K
real-data protocol (LongAlign-10k, 32 continuation steps from a shared
checkpoint, router load balancing off, `alltoall` dispatcher): maximum
absolute LM-loss difference against the compiled reference 1.74e-4, MTP
2.35e-4, below the 1e-3 threshold (2026-10-13). The MoK megakernel was
compared with the native MoE path on the 9L 64K recipe from random
initialization over 32 steps: per-step LM-loss differences grow to 1.62e-3
at step 20 and re-converge to 8.0e-5 at step 32 with identical gradient-norm
trajectories (2026-10-20); a same-configuration rerun floor for that
from-scratch protocol has not been established, so this is a trajectory
check rather than a parity pass. The rebased code base matched the
pre-rebase one on the 9L 64K real-data protocol with a maximum LM-loss
difference of 1.9e-4 (2026-10-26). The earlier checks of the latent CP
attention path (2.69e-4 on 2026-10-13) remain valid. No full-model 64K
real-data result is claimed here; the mock-data throughput runs do not
establish long-context convergence.

Two regressions pulled in by the rebase onto `dev` were found with the
full-model runs and are fixed in this code base: the kernel warm-up left
16 GiB of cached pinned host memory per rank because `torch.accelerator
.memory.empty_host_cache` does not exist in the NGC 26.04 PyTorch
(`megatron/training/kernel_warmup.py` now falls back to
`torch._C._host_emptyCache`), and the eager element-wise MLA output gate
kept an fp32 sigmoid copy per layer and in-flight microbatch on the device
(+2.25 GiB per pipeline stage and microbatch; `attention_output_gate.py`
now saves only the gate inputs with bit-identical results). The 2026-10-28
numbers above include both fixes.

## Build the container

All four YAMLs reference the shared [Dockerfile](Dockerfile) through
`DEPENDENCIES.dockerfile_path`, relative to the Megatron-LM repository root.
One image serves all four recipes. Build on an ARM64 builder with Docker/BuildKit:

```bash
# Run from the Megatron-LM repository root.
docker build --platform=linux/arm64 \
    -f examples/moe_recipes/kimi_k3/Dockerfile -t megatron-kimi-k3:blackwell .
```

The public build uses the following fixed components:

| Component | Version / revision |
|---|---|
| NVIDIA PyTorch base | `nvcr.io/nvidia/pytorch:26.04-py3` |
| NCCL runtime and headers | `2.32.3-1+cuda13.4` (CUDA toolkit and PyTorch unchanged) |
| TransformerEngine | `0a2ebe942b9f6a66aa0292502966e73114a7b912` |
| cuDNN frontend | `5ba9432e652e0b4df3aaca85cf3f95fe6bb7375a` |
| Flash Linear Attention / fla-core | `0.5.2` |
| FlashAttention-4 | `4.0.0b11` (required by the MLA latent CP split forward/backward API) |
| Transformers | `5.16.1` |
| NVIDIA Resiliency Extension | `0.6.0` |
| Emerging-Optimizers | `a44d1f83a4950b445f2a77ca71177c5f033d45d5` |
| causal-conv1d | `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` (1.6.2.post1) |
| HybridEP | `10d4dd7377d5bce900fbb4b80cce863764892b95` (hybrid-ep branch) |
| Mixture-of-Kittens (MoK) | `d59188aab3c9d5f92dd197d9997226ba3a43c701` (ThunderKittens 1c3920d9 with the MXFP8 MN-major patch; built for sm_100a and sm_103a) |
| CUTLASS DSL | `4.5.2` (CUDA 13) |
| Apache TVM FFI | `0.1.11` |

The Dockerfile explicitly installs `flash-attn-4==4.0.0b11` and checks the
installed version. MLA latent CP uses its split forward/backward API and
rejects other versions. This matches the recorded benchmark image.

NCCL's runtime and development packages are pinned together. The Dockerfile
checks the library actually resolved by the dynamic linker with
`ncclGetVersion`; PyTorch's build-time NCCL version alone does not identify
the loaded runtime. The refreshed benchmark image passed single-node and
two-node GPU collective checks with NCCL 2.32.3.

HybridEP revisions before 17cfb81 can retain a recycled pinned-host token
count during blocking permute/unpermute, causing intermittent illegal memory
accesses at long sequence lengths. The pinned 10d4dd7 revision fixes that
lifetime issue. The public build sets `HYBRID_EP_MULTINODE=0`; place each EP64
group within one NVLink domain. The benchmark image was built with multinode
support enabled but used the same within-domain EP placement. Cross-domain
EP requires a suitable communication build and a separate placement review.

All recipes enable `gdn_pre_gated_delta_rule_fusion`, which requires the
C++ convolution backward from causal-conv1d. The Dockerfile builds it from
the pinned source revision against the NGC CUDA/PyTorch installation.
TE targets SM100a and SM103a; its optional NCCL EP extension stays disabled
because these recipes use HybridEP.

A `/opt/venv` environment inherits NGC's system packages and installs recipe
dependencies locally. The venv is first on `PATH`; `pip check` remains a
build requirement, and the build includes `pytest` for NGC's triton-kernels.
Run with the image's default Python environment.

The hybrid-ep build links libibverbs / libmlx5 from the NGC base image's
source build of rdma-core (`/workspace/rdma-core/build/lib`) and records that
directory as the extension's RUNPATH. Megatron imports `deep_ep` at module
level before NCCL creates its first communicator, so that rdma-core loads
first, NCCL's network plugin then loads the system rdma-core as well, its devx
HCA query fails and NCCL silently falls back to NET/Socket for every transfer
that leaves the NVLink domain (pipeline parallelism across racks ran at about
3 GB/s). The Dockerfile strips the RUNPATH after the HybridEP install and
asserts that no `rdma-core` entry remains. The benchmark image received the
same `patchelf --remove-rpath` step on 2026-10-08 (marker
`/etc/kimi_k3_container_patches.txt`); the 2026-10-09 measurements used that
image.

The original public ARM64 build and dependency checks passed in September.
The October measurements used the existing benchmark image after its NCCL
upgrade, the RUNPATH patch and the MoK installation (the 2026-10-20 image
adds mixture-of-kittens d59188a on top of the 2026-10-12 image; the
2026-10-28 measurements used it). The image carries the components pinned
in the Dockerfile, but the public Dockerfile has not been rebuilt from
scratch and compared with it as part of this recipe synchronization.

## Prepare the launch

The YAML has the recipe schema `DEPENDENCIES` / `ENV_VARS` / `ARGS`; it is
not the training entry point's `--yaml-cfg` schema. Export `ENV_VARS` and
translate `ARGS` into command-line flags. Lists become multiple arguments;
`true` becomes a flag; `false` and `null` are omitted.

The following example writes a launch script from any recipe. Run it
inside the built container with the checkout as the working directory,
Python/PyYAML available and `OUTPUT_PATH` set to a writable, container-visible
directory. The full-model recipes and the proxy 512K recipe use mock data
and need no checkpoint. The proxy 64K recipe loads
`zai-org/LongAlign-10k` with `SFTTokenizer`, uses a training-only split, and
cache data under `${OUTPUT_PATH}/data_cache/longalign_10k`. For a continuation
check, supply a matching checkpoint with `--load CHECKPOINT --finetune`.
The recipes enable one MTP depth: a checkpoint without MTP weights requires
removing `mtp_num_layers` and the trailing `/+E` from `hybrid_layer_pattern`
in the YAML before rendering the launcher.

```bash
export RECIPE=examples/moe_recipes/kimi_k3_proxy_9layer/gb300/mxfp8_THD512K_64GPU_TP4PP1EP64CP16_muon.yaml
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the run output directory}"
export OUTPUT_PATH
python - <<'PYTHON' > run-kimi-k3.sh
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
    'pretrain_hybrid.py '
)
print(launcher + shlex.join(args) + ' "$@"')
PYTHON
```

The launcher uses `python -m torch.distributed.run` to keep the worker
processes in the image's default Python environment.

Launch `bash run-kimi-k3.sh` once per node through your scheduler, with the
checkout and output directory mounted at the same paths on every node.
Provide `NNODES`, `GPUS_PER_NODE`, `NODE_RANK`, `MASTER_ADDR` and
`MASTER_PORT` from the allocation. With four GPUs per node, use 64 nodes for
the 256K recipe (and its 64K variant), 128 nodes for the 1M recipe and 16
nodes for the proxy.
Each node must use the same recipe, image and rendezvous address;
`NODE_RANK` must be unique.

`DYNAMO_CACHE_SIZE_LIMIT=64` accommodated the different depth-source counts
of the compiled AttnRes path; the FLA path does not depend on it, and the
recipes keep the export so earlier launch scripts remain valid.
