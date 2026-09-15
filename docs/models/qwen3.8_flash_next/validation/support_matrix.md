# Support matrix (measured on the whole model)

What the composed Qwen3.8-Flash-Next stack (GDN + QSA + MoE + gated residual + PLE + MTP) actually
does on this branch. Every row was produced by running the single-node proxy
([`../training/proxy_single_node.md`](../training/proxy_single_node.md)) with flag overrides,
4 × GB300, 8 iterations, mock data, seq 4096 — **not** inferred from per-component status.

Measured 2026-09-15. Re-run after any rebuild: the cases are cheap (one node, ~40 min for all of
them) and they are the only thing that proves the *combination* works.

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
| **CUDA graphs** | ⛔ | `--cuda-graph-impl local` | rejected: the gated-residual variant returns `None` in the 4-tuple's `h_res` slot, which the partial-MoE capture path packs as a graph output. QSA's sparse path and the memory also reject capture |
| **FSDP** | ⛔ | `--use-megatron-fsdp --ckpt-format fsdp_dtensor` | rejected: "Engram does not yet support FSDP" |

## Correctness checks that ran with the matrix

| Check | Result |
|---|---|
| PLE gradients in the composed stack | `--engram-verify-training`: every iteration `num_tables=64 zero_grad_tables=0 nonfinite_tables=0 changed_tables=64` (64 = 16 tables × EP4) |
| MTP on the hybrid path | `mtp_1 loss` finite and tracking `lm loss` in every case |
| Losses | finite and decreasing in all 11 passing cases |

## TODO — what this matrix says is missing

1. **CUDA graphs** — all three new operators reject capture, for three different reasons (GR's
   `h_res` graph-output contract, QSA's host-synchronizing superset build, the memory's
   data-dependent lookup/all-to-all shapes). The single largest step-time opening.
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
