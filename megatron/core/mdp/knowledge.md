# MDP Implementation Knowledge

This is the fast path for agents and developers working on Modality Decoupled
Parallelism (MDP). Read this file before tracing the implementation. The public
feature overview remains in [README.md](README.md); this document focuses on
code ownership, invariants, control flow, and safe extension points.

## Current baseline

- Development branch: `dev/mdp`.
- History baseline: `e0df73690`.
- The initial MDP implementation is reconstructed as eight cohesive commits.
- The implementation represented here stops at the functionality formerly
  contained through `a849b88a9`; later optimization branches are not part of
  this baseline.
- Owner-sharded pixel capture plus `all_to_all_single` is the only MDP data
  route. There is no endpoint-star pixel mode and no pixel-sharding
  compatibility switch.

When this file disagrees with code, code and tests win. Update this file in the
same commit whenever an invariant, phase, flag, support constraint, or primary
entry point changes.

## Mental model

MDP keeps the decoder's sample ownership and native PP/VPP/EP schedule intact,
but rebalances vision items across the `CP x PP` workers that belong to one
outer data-parallel decoder replica.

Each physical rank contains:

1. its normal decoder shard;
2. a complete vision encoder replica;
3. an MDP runtime coordinating data capture, planning, encoder execution, and
   cross-worker transport.

For every iteration, all workers see the same text tensors and vision metadata.
Pixel payload materialization is sharded by microbatch:

```text
pixel_owner_worker = microbatch_id % num_workers
```

Only that worker materializes the microbatch's pixels. The deterministic planner
may assign each vision item to any producer worker. One canonical bridge ledger
then drives three `all_to_all_single` phases:

```text
PIXEL:     pixel owner -> encoder producer
EMBEDDING: encoder producer -> decoder endpoint
GRADIENT:  decoder endpoint -> encoder producer
```

A *decoder endpoint* is a rank that runs `pre_process` and therefore consumes
vision rows: every pipeline-stage-0 rank of the planning group. At CP=1 that is
one rank; at CP>1 it is `cp` ranks, one per `cp_rank`. Do not confuse it with the
*descriptor source*, which is always exactly one rank (`group[0]`) because it
assigns `global_item_id` values and broadcasts the records.

Local routes are copied directly; remote routes are packed into collective
buffers. Every planning-group member enters every collective, including ranks
with zero-length splits.

## Non-negotiable invariants

Preserve these unless the feature design is intentionally changed:

- MDP-off must leave the native path unchanged.
- Decoder data ownership, sampler accounting, microbatch count, LR schedule,
  PP/VPP/EP schedule, and consumed-sample accounting remain native.
- Decoder THD and vision-encoder THD are separate layouts. Never reuse decoder
  `PackedSeqParams` for the vision encoder.
- `global_item_id` is stable and unique within one planning group and
  iteration.
- Descriptors and plans are deterministic and use integer arithmetic.
- The plan digest is checked before any bridge collective; a mismatch can
  otherwise become a distributed hang.
- The plan is the single source of truth for pixel, embedding, and gradient
  routes.
- Pixel ownership is always derived from the microbatch ID. It is not a user
  option.
- Every bridge phase uses `all_to_all_single`; do not add a second P2P
  transport.
- Empty workers and text-only microbatches still participate in group and WORLD
  collectives.
- Encoder and decoder parameter sets are disjoint.
- Encoder gradients are reduced over WORLD and normalized with the decoder
  finalizer's in-place-reduced global token count.
- Decoder DDP overlap stays inside the native decoder schedule. The encoder
  uses an independent synchronous DDP configuration for its P5/P6 lifecycle.
- The composite optimizer treats decoder and encoder overflow, norm clipping,
  and step success as one atomic decision.
- MDP-owned buffers must be allocated through `MdpBufferAllocator`.

## Phase machine

`MdpRuntime` has three externally visible states:

```text
EMPTY -> DECODER_READY -> DECODER_DONE -> EMPTY
```

The iteration phases are:

| Phase | Main implementation | Responsibility |
|---|---|---|
| P0 | `MdpRuntime.begin_iteration` | Reset iteration state and encoder gradients. |
| P1 | `window.py`, `groups.py`, `planner.py`, `bridge.py` | Capture the full iteration, shard pixel reads, broadcast descriptors, build/check the plan, and route pixels. |
| P2 | `runtime.py`, `activation.py`, model adapter | Pack producer chunks and run the vision encoder with autograd during training. |
| P3 | `bridge.py`, `storage.py` | Route detached vision embeddings to decoder endpoints and create endpoint leaves. |
| P4 | Native Megatron schedule | Replay captured microbatches through the unchanged decoder schedule, finish any native decoder gradient-reduce overlap, and capture global token count. |
| P5 | `runtime.py`, `activation.py`, `encoder.py` | Route leaf gradients back, run encoder backward, reduce WORLD gradients, and normalize them. |
| P6 | `optimizer.py` | Union overflow state, compute a combined norm, clip consistently, and step decoder plus encoder optimizers. |

Evaluation runs P0-P4, skips autograd/backward, releases retained state, and
returns to `EMPTY`.

## Quick code index

### Core package: `megatron/core/mdp/`

| File | Read when changing |
|---|---|
| `config.py` | CLI-derived configuration, validation, supported combinations, vision config overrides. |
| `errors.py` | MDP-specific failure classes. |
| `protocols.py` | Model adapter interface and capture/descriptor carrier types. |
| `rank_mapping.py` | Rank coordinates, outer-DP planning groups, logical workers, endpoint mapping. |
| `groups.py` | Process-group creation and fixed-width descriptor broadcast. |
| `plan.py` | Route/layout schema, row-capacity policy, chunk splitting, plan digest. |
| `planner.py` | Integer deterministic LPT assignment, pixel locality preference, consistency check. |
| `allocator.py` | The only allocation entry point for MDP-owned communication/storage buffers. |
| `storage.py` | Endpoint embedding leaves and lifecycle checks. |
| `bridge.py` | Canonical ledger and `all_to_all_single` transport for all three payload phases. |
| `window.py` | Whole-iteration capture, microbatch replay cursors, pixel ownership context. |
| `packing.py` | Greedy token-budget bin filling and the cross-iteration sample buffer (`--mdp-greedy-packing`). |
| `activation.py` | Encoder forward handle, chunk output retention, multi-tensor backward. |
| `encoder.py` | Encoder process groups, DDP/ZeRO-1 domain, gradient finalization. |
| `runtime.py` | P0-P5 orchestration, prefetch handoff, per-iteration state and metrics. |
| `schedule.py` | Native schedule and `finalize_model_grads_func` wrappers. |
| `optimizer.py` | Decoder/encoder composite optimizer and shared overflow/norm semantics. |
| `checkpoint.py` | `torch_dist` checkpoint facade for the vision model (save and load). |
| `integration.py` | Training-loop seams, adapter registration, runtime construction. |
| `observability.py` | MDP NVTX ranges and iteration metrics helpers. |

### Multimodal integration: `examples/multimodal_dev/`

| File | Responsibility |
|---|---|
| `arguments.py` | User-facing `--mdp-*` arguments. |
| `forward_step.py` | Dual-THD collation, sidecar creation, owner-aware pixel suppression, native and MDP forward steps. |
| `mdp_adapter.py` | Qwen3.5-VL implementation of `MdpModelAdapter`. |
| `data/mdp_mock.py` | Deterministic multi-image/video/text-only dataset with pixel sentinels. |
| `pretrain_multimodal.py` | Adapter registration, startup validation, schedule selection. |
| `models/base.py` | Native vision path and external `vision_embeddings` injection. |
| `models/qwen35_vl/vision_encoder.py` | Vision forward path and cached position metadata consumption. |
| `models/qwen35_vl/vision_pos_cache.py` | Grid-derived position/RoPE/cu-seqlens cache. |
| `observability.py` | Native multimodal NVTX ranges used for MDP-vs-native comparison. |
| `scripts/run_mdp_experiments.sh` | Reproducible reference launcher and profiling wrapper. |

### Megatron training seams

- `megatron/training/training.py`: creates the MDP domain and wraps train/eval
  schedules.
- `megatron/training/checkpointing.py`: injects MDP vision state into the
  distributed checkpoint on save and restores it on load.
- `megatron/training/arguments.py`: permits the validated TE
  cross-entropy-fusion baseline used by the reference launcher.

## Data contract

The collator builds normal decoder tensors plus an MDP vision sidecar:

- `vision_item_meta`: per-item sample, ordinal, `(t,h,w)`, and payload start;
- `vision_decoder_positions`: absolute image-token positions in the decoder's
  packed physical layout;
- `vision_item_meta` also carries each item's enclosing sample span
  (`sample_padded_start`, `sample_padded_len`) so the decoder-CP owner of every
  row is derivable in integer host arithmetic without touching the device;
- `pixel_values`: present only on the owner worker for that microbatch;
- `image_grid_thw`: present on all workers and used to derive item shapes;
- `flops_cu_seqlens`: present only under `--thd-static-packing`; the
  pre-tail-pad valid `cu_seqlens`, because the static tail is represented as an
  ordinary dummy sequence and would otherwise inflate the FLOPs accumulator.

Under `--thd-static-packing` the tail policy is always `append_dummy_seq`:
`ModelParallelConfig` rejects `extend_last` with static packing at every CP
size, and `build_static_thd_metadata` no longer implements it. `extend_last`
leaves `cu_seqlens_q` ending at the real token count while the tensors are
padded to the static target; TE then returns a shorter attention output than
the padded input. `pad_between_seqs` is derived
from the collator's row alignment (`divisible_by > 1`), not hardcoded to
`True`: at TP=CP=1 no sample is ever padded, so there is provably no gap, and
claiming otherwise makes FlashAttention ineligible and can drop TE onto its
unfused O(T^2) backend.

`MdpModelAdapter.get_batch` converts the model-specific batch into
`CapturedMicrobatch`. Core MDP treats `model_payload` as opaque and consumes
only the explicit vision carrier types.

Validation happens before distributed transport:

- pixels and grid metadata are consistent;
- payload rows equal `sum(t*h*w)`;
- decoder image-token slots equal post-merge vision output rows;
- item intervals do not overlap or exceed the flat pixel payload;
- decoder packed format is THD;
- item ordering is deterministic.

## Planning and routing

`MdpPlanner` sorts descriptors by descending integer cost and ascending item
ID, then assigns them with deterministic LPT. `--mdp-pixel-locality` changes
only the tie/preference inside the configured slack window; it must not violate
the load eligibility rule.

The plan contains:

- logical producer assignment;
- owner worker for the PIXEL source;
- endpoint rank for EMBEDDING/GRADIENT;
- producer encoder THD layouts;
- decoder microbatch leaf layouts;
- a 16-byte deterministic digest.

Capacity padding affects allocations only. Segment offsets accumulate valid
rows, and attention frame boundaries are derived from `grid_thw`.

## Runtime and prefetch

`--mdp-overlap-window-capture` captures the next training window on a
background thread and a dedicated CUDA stream. The consumer waits on a CUDA
event, not a host synchronization, and records captured tensors on the main
stream before use.

The prefetch path is keyed by iterator identity. Evaluation does not consume a
pending training window. Any change to captured tensor ownership must update the
`record_stream` traversal in `runtime.py`.

The collator also uses a pinned single-buffer path, and bridge receives can land
directly in final consumer views. Preserve those destination-view contracts
when changing payload shapes.

## Decoder context parallelism

At CP>1 the decoder shards its packed sequence with TransformerEngine's
per-sample zigzag: a sample's padded length `L` is cut into `2*cp` chunks of
`C = L // (2*cp)`, and `cp_rank r` takes chunk `r` followed by chunk
`2*cp-1-r`. The collator already pads every sample to a multiple of `2*cp`
(`pack_or_pad_batch`'s `divisible_by`, mirrored by `thd_row_alignment`).

A vision item owns a contiguous run of decoder positions, so chunk boundaries
cut that run into per-rank runs. `megatron/core/mdp/cp_partition.py` is the
integer inverse of TE's `thd_get_partitioned_indices` and produces the coarsest
legal decomposition:

- at most `2*cp - 1` runs per item (the `cp-1`/`cp` chunk pair is adjacent,
  same-rank and locally contiguous, so it fuses);
- at most **2** runs of one item on one rank (chunks `r` and `2*cp-1-r`) — this
  is why the routed unit is a slice with a `slice_id`, not an item;
- at `cp=1` the identity, so the CP=1 plan is bit-identical to the pre-CP one.

Consequences that are easy to get wrong:

- **Nothing is replicated and nothing is reduced.** Each vision row lands on
  exactly one endpoint, so its gradient exists on exactly one rank and the
  bridge's `copy_` stays correct. An accumulate mode would be wrong, and no
  CP-scoped process group is needed: slices ride the existing planning-group
  `all_to_all_single`.
- **PIXEL is per item, not per slice.** Pixels are CP-invariant; routing them
  per slice multiplies pixel traffic by up to `2*cp-1`.
- **An endpoint with zero rows for a vision-bearing microbatch is normal.**
  Roughly 5% of `(microbatch, cp_rank)` pairs are empty in the mock workload.
  No leaf is created for an empty shard, and `mdp_forward_step` decides whether
  to expect one from the plan, never from the microbatch-global `text_only`.
- **The scatter moves after the CP split.** The native path scatters the full
  vision output into the full sequence and then splits; an MDP endpoint holds
  only its own rows, so it splits first and scatters into the rank-local stream
  using the rank-local image-token mask. `masked_scatter` is pure data movement,
  so the reordering is bitwise neutral, and the leaf's rows are ordered by
  rank-local position, which is the order that mask enumerates.
- **The split is in the plan digest**, along with `cp_size`. Each member derives
  its slice table locally, so a divergence would otherwise produce identical
  digests and then a mismatched `all_to_all_single`.
- `install_mdp_process_groups` cross-checks the derived `(cp_rank, pp_rank)`
  against live MPU state once per job, because a rank map that is
  self-consistent but names the wrong physical ranks fails as a hang.

`cp_partition_mode` must be `zigzag`. Under `contiguous` the decoder would slice
its sequence differently from the plan and every embedding would land on the
wrong rank with no shape error.

## Optimizer and checkpoint semantics

The vision encoder is replicated over WORLD and uses its own DDP/ZeRO-1 domain.
The decoder retains its native dense/expert optimizer domains.

`MdpChainedOptimizer` coordinates all members:

- overflow is unioned with WORLD MAX before scaler updates;
- norm clipping uses one combined norm;
- all members either step or skip together;
- LR scheduler binding sees the composite optimizer.

The native decoder may enable `overlap_grad_reduce` and
`overlap_param_gather`. Its DDP hooks and pipeline schedule retain ownership of
those operations: decoder gradient communication is drained by the native P4
finalizer, and decoder parameter all-gathers are dispatched/waited by the
native forward path. The encoder DDP config is a copy with both overlap modes
disabled, so its WORLD gradient reduction and parameter synchronization remain
synchronous in P5/P6. Delayed gradient reduction and
`overlap_param_gather_with_optimizer_step` remain unsupported because they
cross that phase/domain boundary.

The current checkpoint support is intentionally narrow:

- synchronous global `torch_dist`;
- vision weights stored under the MDP vision key, saved and loaded through the
  MDP facade;
- composite optimizer state for both domains, with the encoder member under a
  fixed key so the decoder DP-CP and encoder WORLD sharding domains never
  collide (both otherwise compute `data_parallel_group_idx == 0`);
- unsupported save/load modes are rejected at startup.

Decoder and WORLD encoder optimizers do not share a DP sharding group; never
key or reshard them as if they did.

## Configuration quick reference

Primary flags:

- `--mdp-enable`
- `--mdp-encoder-cp` (currently must be 1)
- `--mdp-encoder-max-payload-rows`
- `--mdp-vision-config-override KEY=VALUE`
- `--mdp-locality-slack-permille`
- `--mdp-pixel-locality`
- `--mdp-row-alignment`
- `--mdp-plan-check-interval`
- `--mdp-overlap-window-capture`
- `--mdp-debug-plan-payload-check`
- `--mdp-greedy-packing`
- `--mdp-greedy-packing-approximate-resume`
- `--mdp-mock-dataset-config-json`

Packing flags MDP consumes from the core config (all optional, all off by
default):

- `--max-seqlen-per-dp-cp-rank` -- required by `--mdp-greedy-packing`; the
  greedy token budget is this times `context_parallel_size`.
- `--thd-max-packed-sequences` -- caps real sequences per bin, and fixes the
  `cu_seqlens` entry count under `--thd-static-packing`. The static padding tail
  occupies one of those slots, so it must exceed the real sequences a microbatch
  can hold: `greedy_max_real_sequences()` reserves the slot for greedy bins, and
  `validate_mdp_config` requires
  `>= max(micro_batch_size, eval_micro_batch_size) + 1` without greedy packing.
- `--thd-static-packing` -- the data path emits fixed-shape THD batches
  (`T == max_seqlen_per_dp_cp_rank * cp_size`, `cu_seqlens*` of
  `thd_max_packed_sequences + 1` entries). Requires
  `--pad-packed-seq-alignment max` and the `append_dummy_seq` tail policy
  (`extend_last` is rejected at every CP size). Independent of
  `--mdp-greedy-packing`: all four corners of the 2x2 are reachable.
- `--sequence-packing-scheduler` is **rejected** under MDP. It is not merely
  untested: `training.py` wraps the data iterator whenever it is set, and
  `DpBalancedScheduler.run` then asserts on GPT-only sample keys, deletes every
  key outside those six (dropping `pixel_values` / `image_grid_thw`), and
  reroutes samples across DP with an all-to-all that has no notion of
  variable-size pixel payloads. MDP owns its packing instead.

`--mdp-greedy-packing` **reinterprets** `--micro-batch-size` and
`--global-batch-size`: they no longer describe what goes into a microbatch,
only how many bins an iteration has (`N = GBS / (MBS x DP)`). Two consequences
to state in any comparison:

- GBS means "N x token budget", so loss curves are not iteration-by-iteration
  comparable against a fixed-GBS run;
- DP ranks consume different sample counts, so `consumed_train_samples` is
  computed from a real all-reduced count
  (`training._mdp_greedy_consumed_samples`) rather than the closed form. Samples
  are counted when the window built from them is installed for its iteration,
  not when they are drained -- under `--mdp-overlap-window-capture` the prefetch
  thread fills the next iteration's window during the current one, and the final
  prefetch is dropped unconsumed.
- Checkpointing is **rejected** with greedy packing unless
  `--mdp-greedy-packing-approximate-resume` is passed: the cross-iteration
  sample buffer is not checkpointed, and `MegatronPretrainingSampler` is
  positioned from one global `consumed_train_samples` that cannot express the
  per-DP-rank drain counts greedy packing produces, so a resume may skip or
  repeat samples. Greedy packing is a benchmarking path today.

The stream must be provisioned by **tokens**, not samples: an iteration eats
about `token_budget / mean_sample_len` samples per bin, so
`train_iters x GBS` under-provisions whenever the mean sample is shorter than
`token_budget / MBS`. The mock provider scales its dataset accordingly
(`mdp_mock._greedy_sample_scale`); a real dataset must be sized by the
operator.

There is deliberately no pixel-sharding flag. Pixel owner sharding is part of
the MDP definition in this baseline.

Current major constraints:

- Qwen3.5-VL adapter;
- TP=1;
- decoder CP>=1 with `cp_partition_mode=zigzag` (`contiguous` is rejected);
- encoder CP=1;
- distributed optimizer enabled;
- per-token loss enabled;
- bf16/fp16 mixed precision;
- synchronous global `torch_dist` checkpointing (exact resume, same world size);
- no FSDP/HSDP, FP8, full-iteration CUDA graph, CPU activation offload, or
  encoder communication overlap;
- no `--sequence-packing-scheduler`;
- native decoder `overlap_grad_reduce` and `overlap_param_gather` are supported,
  while delayed gradient reduction and parameter-gather overlap with the
  optimizer step are rejected by `validate_mdp_config`.

Always read `validate_mdp_config` before relaxing a constraint. A validation
change without corresponding runtime/test support is not an implementation.

## Metrics and observability

MDP NVTX ranges use the `mdp.` namespace; native multimodal comparison ranges
use `mm.`. The important coarse ranges correspond to capture, planning, pixel
dispatch, encoder forward/backward, embedding/gradient exchange, and leaf
assembly.

FLOP accounting is intentionally generic multimodal functionality, not MDP
functionality:

- real THD `cu_seqlens` supply packed token statistics;
- vision patch, attention, MLP, and merger FLOPs are added from replicated grid
  metadata;
- native and MDP paths publish equivalent statistics.

Do not put new generic multimodal metric code under an `mdp_enable` gate unless
the metric is genuinely MDP-specific.

## Verification entry points

Pure-compute and CPU-oriented tests:

```bash
python -m pytest -q \
  tests/unit_tests/mdp/test_config.py \
  tests/unit_tests/mdp/test_rank_mapping.py \
  tests/unit_tests/mdp/test_plan.py \
  tests/unit_tests/mdp/test_planner.py \
  tests/unit_tests/mdp/test_planner_cp.py \
  tests/unit_tests/mdp/test_cp_partition.py \
  tests/unit_tests/mdp/test_window.py
```

Distributed MDP transport/runtime tests:

```bash
torchrun --nproc_per_node=8 -m pytest -q \
  tests/unit_tests/mdp/test_groups.py \
  tests/unit_tests/mdp/test_bridge.py \
  tests/unit_tests/mdp/test_pixel_owner_shard.py \
  tests/unit_tests/mdp/test_runtime.py
```

Model-side contract and parity tests:

```bash
python -m pytest -q examples/multimodal_dev/tests/test_mdp_dataset.py
torchrun --nproc_per_node=8 -m pytest -q \
  examples/multimodal_dev/tests/test_mdp_parity.py
```

Reference launcher:

```bash
MDP=1 OVERLAP=1 PIXEL_LOCALITY=1 \
  bash examples/multimodal_dev/scripts/run_mdp_experiments.sh
```

Use `MDP=0` for the native comparison. Set `NSYS=1 OUT=<basename>` for an
NVTX/CUDA timeline. The launcher supports multi-node rendezvous through
`NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.

## Change routing guide

Start investigation here:

- data/collation or missing pixels: `forward_step.py` -> `mdp_adapter.py` ->
  `window.py`;
- rank/group bug: `rank_mapping.py` -> `groups.py`;
- imbalance or locality: `protocols.py` cost -> `planner.py` -> plan digest;
- wrong payload destination: `plan.py` -> `bridge.py` -> runtime destination
  views;
- autograd/loss mismatch: `storage.py` -> `runtime.py` P3/P5 ->
  `activation.py` -> `encoder.py`;
- skipped or inconsistent update: `optimizer.py`;
- startup rejection: `integration.py` snapshot -> `config.py`;
- checkpoint issue: `checkpoint.py` -> `training/checkpointing.py`;
- throughput metric issue: generic code in `forward_step.py` and
  `training.py`, not the MDP planner/bridge.

## Common failure modes

- A plan mismatch often appears as a collective hang if the digest check is
  bypassed. Never disable the check to get a run through.
- Every worker must consume the same number of microbatches during capture even
  though only owners materialize pixels.
- Descriptor schema changes require synchronized updates to serialization,
  deserialization, digest inputs, and tests.
- Floating-point planner costs can diverge across ranks. Planner inputs and
  comparisons must remain integer.
- A text-only microbatch has no vision descriptor, route, leaf, or encoder work,
  but it remains in decoder replay.
- Background capture must not enqueue work on the main compute stream.
- Host reads such as `.item()`, `.tolist()`, or implicit tensor formatting
  in the iteration hot path can serialize GPU work.
- An all-to-all rank with no local items still calls the collective with zero
  split sizes.
- Native-path instrumentation must not wrap or mutate tensors in a way that
  changes PP send-buffer or autograd contracts.

## Extension checklist

Before landing a new capability:

1. update the support matrix and remove only the validation that is now truly
   implemented;
2. identify affected rank, plan, carrier, and checkpoint schemas;
3. keep the native MDP-off path side-effect free;
4. add pure tests for deterministic transforms;
5. add distributed tests for every new collective/rank topology;
6. compare loss and gradient norm against the current reference;
7. report iteration time and all-rank peak allocated/reserved memory;
8. update this file and README when the mental model or entry points change.
