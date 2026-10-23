# Full-iteration CUDA graphs (THD + MoK)

`--cuda-graph-impl full_iteration` captures one training step — every microbatch's data read,
forward and backward, and the DDP gradient reduce — into a single CUDA graph and replays it on
later steps. The optimizer step stays outside the graph.

This page covers the whole model (GDN, QSA, MoE, gated residual, n-gram memory, MTP) with the
MoK megakernel as the MoE, on the proxy's default packed-THD layout, and the settings that have
to be right for it. What else the model supports is in
[`../validation/support_matrix.md`](../validation/support_matrix.md).

## Command (proxy, 4 GPUs, EP4, BF16, `LAYOUT=thd32k`)

In the image of [`../../Dockerfile.gb300`](../../Dockerfile.gb300):

```bash
CUDA_GRAPH=full_iteration ./examples/qwen4/train_proxy_1node.sh \
    --gdn-kernel-backend fla \
    --moe-megakernel-backend-config '{"fwd_num_comm_sms":48,"bwd_num_comm_sms":56,"minibatch_size":8192,"macrobatch_size":65536,"schedule_capacity_multiplier":4.0,"all_gather_top_experts_chunk_bytes":2048}'
```

`CUDA_GRAPH` selects the capture mode of the `thd32k` layout: `te` (default; Transformer Engine
per-layer graphs over attention and the MoE router), `full_iteration`, or `none` (eager). With
`full_iteration` the script passes `--cuda-graph-impl full_iteration --cuda-graph-warmup-steps 2
--te-rng-tracker` and drops the per-layer flags (`--cuda-graph-modules`,
`--cuda-graph-capture-engram-host`), which apply to `te` only. The two overrides above are the
GDN backend and the MoK scheduler capacity (below). Everything else is the layout's default:
Muon, packed THD with one 32K-token document per rank, the CuTe DSL sparse QSA path with the
fused indexer loss, the GDN pre-gated-delta-rule fusion.

Iterations 1–2 run eagerly, iteration 3 captures (a few seconds) and every later step replays.

## Settings that must be right

### GDN backend: `fla`

Use `--gdn-kernel-backend fla`. The `internal` (CuTe DSL) backend does not support CUDA graph
capture; `cudnn` has not been run under full_iteration. The fla in the image is installed from
source (main @ `a7a32d8`, newer than the last release tag); `FLA_DISABLE_TENSOR_CACHE=1`, which
the `thd32k` layout exports, is required: GDN refuses to capture packed inputs with fla's tensor
cache on.

### MoE: MoK, `schedule_capacity_multiplier` 4.0

MoK schedules on the device — the per-expert work and the split of large batches into
macrobatches never reach the host — so it captures as is and needs no paged stash. Its scheduler
has a fixed size, eager or captured, and the kernel traps (the process dies) when a step routes
more rows to a rank than it holds. Under full_iteration use `schedule_capacity_multiplier` 4.0;
the layout's default 0.0625 is a mean-load setting, and 1.0 has been seen to fail under capture.
The larger capacity sizes only small schedule arrays and launches idle clusters; it costs no
activation memory. Do not enable paged stash (`--moe-paged-stash`,
`--moe-expert-rank-capacity-factor`) with MoK. MoK needs EP ∈ {1, 4, 8, 16, 32, 64}, TP = 1, a
shared expert and SwiGLU, and a build that supports the 640-wide expert FFN (`lit/qwen4` ≥
`2649f3b`).

The proxy's own dropless alltoall MoE (used by `LAYOUT=bshd4k`) reads token counts on the host
and fails at capture: under full_iteration run the MoE as MoK.

### Packed rows: the static THD contract

A captured graph replays fixed shapes, so every microbatch is padded **before** the graph's data
read: tokens to `--max-seqlen-per-dp-cp-rank`, and `cu_seqlens` to
`--thd-max-packed-sequences + 1` entries; a padded tail becomes one dummy document.
`pretrain_hybrid.py` does this under full_iteration; the `thd32k` layout already passes the flags.

| Flag | What to set |
|---|---|
| `--sft` | packed rows go through the SFT path; the `--use-varlen-dataset` packing schedulers are rejected under full_iteration |
| `--pad-packed-seq-alignment max` | pads every row to the full capacity (required) |
| `--max-seqlen-per-dp-cp-rank` | the row capacity in tokens (32768 in `thd32k`) |
| `--thd-max-packed-sequences` | the most documents a row holds, **plus one** for a padding tail when rows are not full. `thd32k` has exactly one full-length document per row, hence 1. A row that needs more fails with a clear error (rows are never truncated) |
| `--thd-tail-padding-policy` | leave unset (`append_dummy_seq`) |

The padding tail is masked out of the LM / MTP losses, the MoE aux statistics, the QSA indexer
loss and the n-gram memory's lookup, but it still runs through the layers; TFLOP/s counts real
tokens only and the log prints `real tokens: N`. Only CP = 1 and PP = 1 are supported with packed
rows under full_iteration, as are dynamic CP and the sequence-packing schedulers.

### n-gram memory (Engram / PLE) at EP > 1

Under full_iteration the memory's EP-sharded table lookup switches to fixed-size buckets: each
(owner rank, table) bucket holds `ceil(--engram-ep-capacity-factor × tokens / EP)` requests
(default factor 2.0), so the exchange has static shapes.

- **A bucket that overflows is an error**, not a silent drop: the capacity is too small for the
  data. During the eager warmup steps it raises `ValueError` with the factor that would have fit
  the step; once the step is captured, a device-side assert aborts the replay with
  `Engram fixed-capacity EP lookup overflowed a bucket; raise --engram-ep-capacity-factor`.
- **A factor ≥ EP can never overflow.** At large EP that is expensive — the exchange grows with
  the factor — so size it from the real data's peak load per (owner, table) with headroom.
- At EP 1 the lookup is local and cannot overflow. `--deterministic-mode` is rejected with the
  memory under full_iteration.

### Other flags

| Flag | Note |
|---|---|
| `--no-check-for-nan-in-loss-and-grad` | required by full_iteration (set by `thd32k`) |
| `--cuda-graph-warmup-steps` | eager steps before capture (2 in `thd32k`) |
| `--moe-router-force-load-balancing` | mock-data timing aid; drop it with real data |

## What was measured

Proxy `LAYOUT=thd32k`, 1 × 4 GB300, EP4, BF16, Muon, mock data, with the two overrides above, the
three capture modes back to back on one node; mean of iterations 21–40 of 40:

| `CUDA_GRAPH` | Step time | TFLOP/s/GPU | Peak allocated |
|---|---|---|---|
| `none` (eager) | 2880 ms | 357.0 | 128.3 GiB |
| `te` (layout default) | 2886 ms | 356.3 | 128.3 GiB |
| `full_iteration` | **2844 ms (−1.3 %)** | **361.6** | 128.0 GiB |

At 32K tokens per rank the kernels are large and launch overhead is a small part of the step, so
capture gains little here (the per-layer `te` graphs gain nothing either). Correctness: lm loss,
MTP loss and grad norm of the `full_iteration` and the eager run are identical through
iteration 10 at the logged precision and differ by ~5e-6 relative at iteration 20 (run-to-run
non-determinism of the stack).

On the earlier 4K-token configuration (seq 4096, real vocabulary, Adam, MoK, fla GDN; measured on
`lit/qwen4_series`), where the step is launch-bound, full_iteration gave 727 → 535 ms (−26 %) on
plain rows (SBHD, bitwise identical to eager over 30 iterations) and 783 → 506 ms (−35 %) on
packed rows (THD, 24 % padding).
