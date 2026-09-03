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
| P2 | `runtime.py`, `activation.py`, model adapter | Pack producer chunks. Default training runs the encoder with autograd; complete-encoder recompute runs it under `no_grad` and saves pixels/layouts/output metadata/RNG recipes. |
| P3 | `bridge.py`, `storage.py` | Route detached vision embeddings to decoder endpoints and create endpoint leaves. |
| P4 | Native Megatron schedule | Replay captured microbatches through the unchanged decoder schedule, finish any native decoder gradient-reduce overlap, and capture global token count. |
| P5 | `runtime.py`, `activation.py`, `encoder.py` | Route leaf gradients back; run retained-graph backward or replay complete encoder chunks with restored RNG before backward; reduce WORLD gradients and normalize them. |
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
| `activation.py` | Retained-graph and complete-replay encoder handles, RNG recipes, chunk backward. |
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
- `pixel_values`: present only on the owner worker for that microbatch;
- `image_grid_thw`: present on all workers and used to derive item shapes.

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

## FP8 and quantized-GEMM alignment

FP8 runs on one recipe for both sides. `args.fp8`/`args.fp8_recipe` configure
the decoder; `--encoder-fp8` (`MdpConfig.encoder_fp8`) makes the vision encoder
inherit them, copied onto the vision `TransformerConfig` by
`apply_encoder_fp8_config` from the compatibility snapshot
(`MdpCompatibilityOptions.decoder_fp8` / `decoder_fp8_recipe`), next to
`apply_encoder_recompute_config`. The encoder has no recipe of its own and FP8
attention is not offered for it.

`validate_mdp_config` leaves decoder-only FP8 alone. Once the encoder runs FP8
it requires decoder FP8 to be on (encoder-only FP8 is not supported: pure launch
overhead in measurement, and it would need its own recipe plumbing), requires
the shared recipe to be in `ENCODER_COMPATIBLE_FP8_RECIPES` (`delayed`: the
encoder's per-chunk, rank-dependent forward count -- doubled by P2/P5
whole-encoder replay -- would push the amax history unevenly, and TE all-reduces
its global amax buffer on every depth-0 `fp8_autocast` exit so the decoder's own
delayed buffers would see rank-dependent collective counts; `custom`:
unvalidated). Those gates run before the vision config
exists, so `validate_effective_vision_config` re-checks the built config
against the snapshot inside `build_encoder_domain`: an adapter that set `fp8`
on its own, or lost it, fails loudly rather than training an encoder the gates
never saw, and the built `ffn_hidden_size` -- whichever config it came from --
must be a multiple of the recipe's block size (Qwen3.5-VL's 4304 is not, for
MXFP8).

The FP8 context is opened per layer inside `TransformerBlock.forward` off the
config fields, so the wiring is complete once the fields are set: the P2
forward and, with `--encoder-recompute-granularity whole`, the P5 replay call
the same encoder and see the same recipe. Every compatible recipe derives its
scale from the current tensor and keeps no cross-forward state, which is what
makes the replay numerically equivalent.

Quantized GEMMs constrain the decoder's packed row count: `pack_or_pad_batch`
extends the last sample's padded region until the packed total is a multiple of
`get_fp8_align_size(fp8_recipe)` (32 for MXFP8, 16 otherwise). Every other
sample keeps its exact length, which is what lets that call site declare
`pad_between_seqs=False` and keep FlashAttention/FusedAttention eligible for
THD. Alignments it cannot derive fail loudly instead: with
`--use-packed-sequence`, `--fp4-format` and `--fp8-recipe custom` raise
`NotImplementedError`. Without `--use-packed-sequence` it contributes nothing:
BSHD collation is untouched by FP8. Under MDP that branch is unreachable (MDP
requires packed sequences); natively it is reachable and, exactly as on base,
unguarded -- a BSHD + FP8 run outside MDP gets no alignment from this stack.

### Decoder packed-row alignment: derivation

Where the requirement lands. Quantized GEMMs check the tensor as the layer
receives it, which is the rank-local one. CP has already taken its `1/cp_size`
slice. Under sequence parallelism a column-parallel layer is still holding its
`1/tp_size` shard when the check runs: `transformer_engine/pytorch/module/
layernorm_linear.py` calls `assert_dim_for_fp8_exec` on the input before the
sequence-parallel `gather_along_first_dim`, and quantizes that same shard
(MXFP8's `flat_first_dim % 32` check in `csrc/quantizer.cpp` sees it too). So
the collated multiple is `lcm(divisible_by, pad_to_multiple * cp_size *
(tp_size if SP else 1))`, where `divisible_by` is the parallelism-only
requirement (`tp_size` under SP, `cp_size * 2` under CP, their product with
both). The 8/16/32 sources: `assert_dim_for_fp8_exec` needs the leading dim
divisible by 8, the backward wgrad GEMM (`cublaslt_gemm.cu`
`CanonicalizeGemmInput`) needs 16, and MXFP8's `create_tensor` needs 32;
`megatron.core.fp8_utils.get_fp8_align_size` returns 32 for MXFP8 and 16
otherwise.

Why tail-only at `cp_size == 1`. Without CP the total is the only thing that
has to be aligned: SP scatters the `[T, 1, H]` tensor as a unit
(`scatter_to_sequence_parallel_region` in `models/base.py`), and the `tp_size`
factor above already lifted the quantized-GEMM requirement onto that same total.
Neither constrains per-sample boundaries. Rounding every sample up would open
literal inter-sample gaps (`[a, a, PAD, b, b, b, PAD, c, PAD]`) that TE's THD
attention detects as `pad_between_seqs=True` and answers by disabling
FlashAttention/FusedAttention in favour of `UnfusedDotProductAttention` -- a
~72% throughput regression measured on decoder FP8. So the samples are packed
back to back and the whole tail is absorbed once by extending the last sample's
padded region, the shape `megatron/core/packed_seq_params.py` calls
"extend_last". `cu_seqlens[:-1] == cu_seqlens_padded[:-1]` is asserted.

Why CP keeps per-sample alignment. TE partitions each sample on its own, so
per-sample alignment is mandatory under CP whether or not FP8 is on, and the
same `divisible_by` is applied to every sample. That is the tightest *uniform*
per-sample rule that forces every rank's local row count to be a multiple of
`pad_to_multiple` (a microbatch can hold a single sample). It is strict --
up to `divisible_by - 1` padded rows per sample, 256 at `cp_size=8` with MXFP8
and no SP. A total-only variant would be tighter and is not written because
MDP rejects CP > 1 outright.

Why `pad_between_seqs=False` is declared, and when it would be wrong. Declaring
it lets `TEDotProductAttention.forward()`
(`megatron/core/extensions/transformer_engine.py`) pass `cu_seqlens_q_padded`
as the effective attention boundary while `cu_seqlens_q` and `padding_mask`
stay exact for loss and routing. This deliberately diverges from
`pad_sequence_for_thd` in `packed_seq_params.py`, which reports `True` for the
identical "extend_last" layout -- the conservative answer a helper that cannot
see how its input was built has to give, and one the call site that built the
layout does not have to inherit. It is safe only because the decoder mask is
causal: the tail rows follow every real row of the last sequence, so no real
query reads them, and their own outputs are discarded by `loss_mask=0` /
`labels=-100`. The collate function never sees an `attn_mask_type`, so the
premise is owned by the decoder spec: `models/qwen35_vl/specs.py` delegates to
`get_transformer_block_with_experimental_attention_variant_spec`, whose
full-attention layers come from `get_gpt_layer_with_transformer_engine_spec`
with a hard-coded `attn_mask_type=AttnMaskType.causal` and whose
gated-delta-net layers are a causal recurrence; neither exposes a knob. Whoever
adds a non-causal decoder spec owns dropping the flag (or padding every sample).

## Configuration quick reference

Primary flags:

- `--mdp-enable`
- `--mdp-encoder-cp` (currently must be 1)
- `--mdp-encoder-max-payload-rows`
- `--encoder-recompute-granularity selective|full|whole`
- `--encoder-recompute-method uniform|block`
- `--encoder-recompute-num-layers`
- `--encoder-recompute-modules MODULE [MODULE ...]`
- `--encoder-fp8` (inherits the decoder's `--fp8-format`/`--fp8-recipe`;
  requires `--mdp-enable` and decoder FP8; recipe must be
  tensorwise|blockwise|mxfp8)
- `--mdp-locality-slack-permille`
- `--mdp-pixel-locality`
- `--mdp-row-alignment`
- `--mdp-plan-check-interval`
- `--mdp-overlap-window-capture`
- `--mdp-debug-plan-payload-check`

The typed encoder recompute arguments are shared by native `multimodal_dev` and
MDP training. Native training supports no recompute, `selective`, and `full`;
`whole` is MDP-only because it relies on the P2/P5 replay protocol. With no
encoder granularity, the native path keeps normal encoder activations and MDP
P2 retains the normal graph-connected encoder outputs.

`selective` and `full` use MCore's native Transformer checkpointing. The
typed encoder arguments are copied to the vision `TransformerConfig` through
`dataclasses.replace`, so MCore's own field and cross-field validation remains
authoritative. `selective` accepts `--encoder-recompute-modules`; `full`
uses `--encoder-recompute-method` and
`--encoder-recompute-num-layers`. Here `full` retains MCore's established
meaning: checkpoint complete Transformer layers or layer groups, not the
complete vision encoder.

`--encoder-recompute-granularity whole` follows the original MDP design: P2
runs patch embedding, positions/RoPE, all Transformer layers, and the patch
merger under `no_grad`; the producer retains valid packed pixels, immutable
chunk layouts, P2 output metadata, and CPU/CUDA/model-parallel RNG state per
chunk. In P5 it forks the ambient RNG, restores each chunk's P2 state, replays
the complete encoder with gradients enabled, immediately backpropagates the
routed output gradient, and finally restores the P5-entry RNG.
Chunk-at-a-time replay makes `encoder_max_payload_rows` bound the rebuilt
graph, but not all live state: every producer's packed pixels survive across
P4, and P5 materializes all routed chunk-output gradients before the first
replay. The initial P5 peak is therefore all retained pixels plus all routed
gradients plus one chunk's activation graph. Consumed pixel and gradient
references are dropped after each chunk backward, so their live storage
decreases through P5, but this does not reduce that initial peak. Smaller
chunks reduce only the rebuilt-graph term and add more serial replay/backward
launches.

Whole replay adds one full encoder forward: encoder forward FLOPs are
approximately doubled, while encoder backward still runs once. Prefer native
`selective` or `full` Transformer recompute when checkpointing Transformer
activations saves enough memory; use `whole` when the additional patch
embedding, position/RoPE, and patch-merger activation savings justify replaying
the complete encoder. `whole` rejects native Transformer recompute on the
effective vision config, including config supplied directly by an adapter,
because nesting the mechanisms would replay the vision Transformer twice in
P5.

There is deliberately no pixel-sharding flag. Pixel owner sharding is part of
the MDP definition in this baseline.

Current major constraints:

- Qwen3.5-VL adapter;
- TP=1;
- decoder CP=1;
- encoder CP=1;
- distributed optimizer enabled;
- per-token loss enabled;
- bf16/fp16 mixed precision;
- synchronous global `torch_dist` checkpointing (exact resume, same world size);
- decoder FP8 unrestricted on its own; with `--encoder-fp8` the shared recipe
  must be in `ENCODER_COMPATIBLE_FP8_RECIPES` (no `delayed`/`custom`);
- no FSDP/HSDP, full-iteration CUDA graph, CPU activation offload, or encoder
  communication overlap;
- native decoder `overlap_grad_reduce` and `overlap_param_gather` are supported,
  while delayed gradient reduction, parameter-gather overlap with the optimizer
  step, and MXFP8 grad-buffer reuse for the parameter all-gather are rejected by
  `validate_mdp_config`.

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
  tests/unit_tests/mdp/test_window.py \
  tests/unit_tests/mdp/test_quantized_alignment.py \
  tests/unit_tests/mdp/test_pinned_collate.py
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
# pack_or_pad_batch ends in a TP broadcast, so this one needs a rank:
torchrun --nproc_per_node=1 -m pytest -q \
  examples/multimodal_dev/tests/test_thd_e2e.py
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
