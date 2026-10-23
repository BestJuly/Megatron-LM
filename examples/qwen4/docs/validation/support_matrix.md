# Support matrix (measured on the whole model)

What the composed Qwen3.8-Flash-Next stack (GDN + QSA + MoE + gated residual + PLE + MTP) actually
does on this branch. Every row was produced by running the single-node proxy
([`../training/proxy_single_node.md`](../training/proxy_single_node.md)) in its `LAYOUT=bshd4k`
mode with flag overrides, 4 × GB300, 8 iterations, mock data, seq 4096 — **not** inferred from per-component status.

Re-run after any rebuild: the cases are cheap (one node) and they are the only thing that proves
the *combination* works.

## Parallelism and data layout

| Feature | Whole model | How it was run | Notes |
|---|:---:|---|---|
| **EP** | ✅ | `--expert-model-parallel-size 4` (default of the proxy) | 8 local experts per rank, as at EP64 on the full model |
| **TP + SP** | ✅ | `--tensor-model-parallel-size 2 --sequence-parallel` | GR warns without SP (replicated compute per TP rank) |
| **PP** | ✅ | pattern `GEGEGEQE\|GEGEGEQE/QE`, `--pipeline-model-parallel-size 2` | pipe separators define the stage boundaries |
| **VPP** | ✅ **with a caveat** | pattern with 4 `\|` segments, PP2 → 2 virtual stages per rank | **fails with `--overlap-param-gather`** (`assert self.param_gather_handle is None` in the distributed optimizer, no Qwen3.8 code in the traceback). Works with `--use-distributed-optimizer --overlap-grad-reduce`. Do not combine VPP with param-gather overlap |
| **CP** (unpacked BSHD) | ✅ | `--context-parallel-size 2 --qsa-use-sparse-attention --cp-comm-type all_gather` | QSA CP **requires the sparse kernel** (the dense-mask bridge is single-rank) and allgather comm; PLE uses its conv halo |
| **Packed rows (THD)** | ✅ **via `--sft` only** | `--sft --pad-packed-seq-alignment max --max-seqlen-per-dp-cp-rank 4096 --qsa-use-sparse-attention --eval-iters 1` | the **sequence-packing schedulers** selected by `--use-varlen-dataset` are rejected by the memory ("use the `--sft` packed path instead"). The SFT path still builds a validation loader, so `--eval-iters 0` fails with "no sample to consume" |
| **Packed rows + CP** | ⛔ | — | rejected at startup: upstream `get_thd_batch_on_this_cp_rank` passes `cu_seqlens_padded=None` to `thd_get_partitioned_indices` and fails with or without the memory. The memory's own packed-CP layout is implemented and module-verified |

## Memory, precision and capture

| Feature | Whole model | How it was run | Peak allocated (proxy) |
|---|:---:|---|---|
| **BF16** | ✅ | baseline | 71.9 GiB |
| **Activation recompute, full** | ✅ | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` | 50.7 GiB (−29 %) |
| **Activation recompute, selective** | ✅ | `--recompute-granularity selective --recompute-modules core_attn mhc` | 62.5 GiB (−13 %) |
| **MXFP8** | ✅ | `--fp8-format e4m3 --fp8-recipe mxfp8` | 72.4 GiB |
| **CUDA graphs, MoE scopes** | ✅ | `--cuda-graph-impl transformer_engine --cuda-graph-modules moe_router moe_preprocess --cuda-graph-warmup-steps 2` | 30 iterations, peak allocated unchanged (42.57 vs 42.59 GiB), loss identical to the eager run at iteration 1 (lm loss 1.073585E+01, grad norm 6.642). `_layer_is_graphable` leaves a hybrid layer eager unless its inner layer is MoE, so this captures the gated-residual aggregate plus the router and dispatch preprocessing and never touches GDN, QSA or the memory. `[moe_router]` alone measures the same — `moe_preprocess` adds nothing. Needed two fixes: the gated residual used to pack its always-`None` `h_res` slot as a graph output, and Engram rejected every `cuda_graph_impl` |
| **CUDA graphs, `attn` scope** | ✅ | `--cuda-graph-impl transformer_engine --cuda-graph-modules attn moe_router --cuda-graph-warmup-steps 2 --cuda-graph-capture-engram-host` | the proxy's default `LAYOUT=thd32k`: packed THD (`--sft`) at 32K per rank, CP1, EP4, with MoK, on GB300 ([`../training/proxy_single_node.md`](../training/proxy_single_node.md)). Engram-hosting layers stay eager unless `--cuda-graph-capture-engram-host` runs the Engram lookup eagerly in front of the layer's graph and captures the rest of the layer. Under THD `attn` capture GDN `internal` resolves to `fla` (use `--gdn-kernel-backend cudnn`) and QSA `cudnn` needs `--qsa-cudnn-bwd-impl qsa_api` |
| **CUDA graphs, full iteration** | ✅ **with MoK; packed rows at CP = 1** | `CUDA_GRAPH=full_iteration` (`--cuda-graph-impl full_iteration`) `--gdn-kernel-backend fla`, MoK `schedule_capacity_multiplier` 4.0 — [`../training/full_iteration_cuda_graph.md`](../training/full_iteration_cuda_graph.md) | the proxy's default `LAYOUT=thd32k`, Muon, 4 × GB300: **2844 ms vs 2880 ms eager and 2886 ms with `attn moe_router` capture (−1.3 %)**, 361.6 vs 357.0 TFLOP/s/GPU, peak allocated unchanged (128.0 GiB). Losses identical to eager through iteration 10. The memory's EP lookup runs on fixed-capacity buckets (`--engram-ep-capacity-factor`; an overflow is an error). The dropless alltoall MoE fails at capture; dynamic THD, CP > 1, PP > 1 and the packing schedulers are rejected |
| **CUDA graphs, other modes** | ⛔ | `--cuda-graph-impl local`, or `transformer_engine` without `--cuda-graph-modules` | rejected by Engram at config validation: these modes capture the Engram-hosting layer through per-layer graphs, whose dynamic hashed lookup and all-to-all have data-dependent shapes |
| **FSDP** | ⛔ | `--use-megatron-fsdp --ckpt-format fsdp_dtensor` | rejected: "Engram does not yet support FSDP" |

## Kernel backends (GB300)

Proxy, 4 × GB300, mock data, the image of [`../../Dockerfile.gb300`](../../Dockerfile.gb300)
(cuDNN 9.27.0, cudnn-frontend `lit/qsa`, MoK `lit/qwen4`). Flags and constraints:
[`../../README.md`](../../README.md#kernel-backends).

| Feature | Whole model | How it was run | Notes |
|---|:---:|---|---|
| **GDN `internal` (CuTe DSL), the default** | ✅ | no flag | CuTe DSL forward + fla backward at 48 value heads (logged). Under THD with `attn` captured it resolves to `fla` (one warning) |
| **GDN `fla`** | ✅ | `--gdn-kernel-backend fla` | the previous default |
| **GDN `cudnn`** | ✅ | `--gdn-kernel-backend cudnn` | the proxy's `thd32k` default: THD + `attn moe_router` capture + MoK, 32K per rank |
| **QSA `cute`** | ✅ | `--qsa-attn-backend cute` (= `--qsa-use-sparse-attention`) | |
| **QSA `cudnn` + `sparse_gqa` bwd** | ✅ | `--qsa-attn-backend cudnn` | rejected under THD CUDA-graph capture and for 12Q/1KV (TP2) |
| **QSA `cudnn` + `qsa_api` bwd** | ✅ | `--qsa-attn-backend cudnn --qsa-cudnn-bwd-impl qsa_api` | the one for 12Q/1KV and THD capture |
| **QSA `cudnn` + CP** | ⛔ | — | rejected at config time (no CP path in the cuDNN API) |
| **MoK** | ✅ | `--moe-megakernel-backend mok` | needs MoK `lit/qwen4` ≥ `2649f3b` (128-aligned *intermediate*, the 640 expert FFN) |

## Optimizer

| Feature | Whole model | How it was run | Notes |
|---|:---:|---|---|
| **Adam + distributed optimizer** | ✅ | `--optimizer adam` (the recorded 64-GPU run used it) | expert-parallel parameters — the routed experts and the memory's hashed tables — shard their optimizer state over the **expert** data-parallel group, dense parameters over the ordinary one |
| **Muon + distributed optimizer** | ✅ | the default of both recipes: `--optimizer muon --use-distributed-optimizer` (drop `--use-precision-aware-optimizer` and its four dtype flags; Muon rejects them) | Muon owns the 2-D matrices through the layer-wise optimizer; everything else goes to Adam siblings. **The memory's tables are expert-parallel *and* Adam-managed at once**, which used to be rejected at startup and forced the distributed optimizer off entirely, un-sharding every optimizer state. Measured on the proxy: peak allocated **−15.84 GiB (−23.7 %)** at expert-DP 2, −6.05 GiB at expert-DP 1 (dense side only), iteration-1 losses identical to the un-sharded arm, `torch_dist` round trip clean. Check the startup `[optimizer routing]` line: the tables must land in the scalar/embedding group, never in the matrix group |
| **Muon + expert tensor parallelism** | ❓ | — | untested: every run above used `--expert-tensor-parallel-size 1`. ETP does not change the sharding logic, it only shrinks the expert data-parallel group (`expert_data_parallel_size = world_size // (ETP × EP × PP)`), and the expert sibling is built with the same process groups the standard path uses — but this is reasoned from the code, not measured |
| **Muon + precision-aware optimizer** | ⛔ | `--optimizer muon --use-precision-aware-optimizer` | rejected: "only supported with adam" |

## Correctness checks that ran with the matrix

| Check | Result |
|---|---|
| PLE gradients in the composed stack | `--engram-verify-training`: every iteration `num_tables=64 zero_grad_tables=0 nonfinite_tables=0 changed_tables=64` (64 = 16 tables × EP4) |
| MTP on the hybrid path | `mtp_1 loss` finite and tracking `lm loss` in every case |
| Losses | finite and decreasing in all 11 passing cases |

## TODO — what this matrix says is missing

1. **CUDA graphs** — per-layer `attn moe_router` capture and full-iteration capture (with MoK,
   packed rows at CP = 1) both run; at `thd32k`'s 32K tokens per rank neither gains much, since
   the step is not launch-bound (full_iteration gave −26 % / −35 % on the earlier 4K
   configuration). Left: full_iteration with GDN `cudnn` (not run) and `internal` (not
   capturable), packed rows with CP > 1, and real-data sizing of `--engram-ep-capacity-factor`
   at EP64.
2. **Packed rows + CP** — blocked upstream (`cu_seqlens_padded=None`), not by this model. The
   last piece of long-context packed training.
3. **FSDP** — rejected by the memory; the only route to a non-EP memory layout.
4. **VPP + `--overlap-param-gather`** — a generic distributed-optimizer assertion, but it is the
   combination a user hits, so either fix it or keep it out of the recipes.
5. **Sequence-packing schedulers** (`--use-varlen-dataset`) — rejected by the memory; only the
   `--sft` fixed-capacity packing works today.
6. **EP32 sizing with recompute** — recompute saves 29 % on the proxy and EP32 was ~16 GiB short
   on the full model; worth re-measuring.

## What the matrix does not claim

Eight iterations on mock data at proxy scale. It shows each configuration *runs* on the composed
stack with finite losses and gradients reaching the memory — not that CP, packing or MXFP8 are
numerically equivalent to their single-rank / BF16 counterparts (that is the parity harness's job,
[`parity_and_acceptance.md`](parity_and_acceptance.md)), and not anything about throughput.
