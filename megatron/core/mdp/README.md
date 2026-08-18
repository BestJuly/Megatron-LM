# MDP — Modality Decoupled Parallelism

MDP addresses GPU stalls caused by long-tail vision workloads in multimodal
training. It does not change sample ownership or decoder data-parallel
semantics: every physical rank co-locates a complete (replicated) vision
encoder with its language-decoder shard, and each iteration's visual items are
rebalanced across the `CP x PP` encoder workers inside each decoder replica.
The native decoder PP/VPP/EP schedule, sampler, microbatch, LR, and
consumed-sample accounting run unchanged.

Enable with `--mdp-enable` in a training entry point that registers an
`MdpModelAdapter` (see `examples/multimodal_dev`). With the flag absent, every
integration point is side-effect free and `finalize_model_grads_func` stays
unwrapped.

## Phase machine

The runtime exposes three states (`EMPTY -> DECODER_READY -> DECODER_DONE ->
EMPTY`) driving seven phases:

| Phase | Where | Action |
|---|---|---|
| P0 | `begin_iteration` | Zero encoder grads, reset iteration state |
| P1 | `begin_iteration` | Capture the iteration window, broadcast fixed-width descriptors from the PP0 endpoint, run deterministic LPT to logical workers, check the plan digest across the group, exchange pixels |
| P2 | `begin_iteration` | Grad-enabled chunked encoder forward on encoder THD (`no_grad` for evaluation); outputs retained as a list in the forward handle |
| P3 | `begin_iteration` | Exchange detached embeddings; endpoint assembles one detached leaf per vision-bearing microbatch |
| P4 | native schedule | Replay iterators feed the unmodified decoder schedule; the wrapped `finalize_model_grads_func` captures the in-place-reduced global token count |
| P5 | `end_iteration` | Exchange leaf gradients back, one multi-tensor backward per producer (native MCore recompute replays here), WORLD sum-reduce with prescale 1, scale by `1/clamp(T_global, 1)` |
| P6 | composite optimizer | WORLD MAX overflow union before any scaler update, combined-norm shared clipping, one atomic step for `[decoder_dense, decoder_expert?, encoder]` |

Key contracts: encoder and decoder THD packings are fully separate (linked
only by `global_item_id`, `(microbatch, sample, ordinal)` and exact row
counts, plus endpoint-local `decoder_positions`); one plan is the single
source of truth for pixel dispatch, embedding return, and reverse gradient
routing; pixels never enter the decoder; the encoder never enters the decoder
schedule model list.

## Module map

| File | Contents |
|---|---|
| `config.py` | `MdpConfig`, support-matrix validation, vision config override allowlist |
| `rank_mapping.py` | Pure-compute outer-DP planning groups and logical workers from `RankGenerator` coordinates |
| `groups.py` | Process-group installation, fixed-width descriptor broadcast |
| `plan.py` / `planner.py` | Minimal-sufficient plan data model, blake2b digest, deterministic integer LPT, group consistency check |
| `allocator.py` / `storage.py` | Single allocation point for MDP buffers; endpoint leaf storage |
| `bridge.py` | One ledger + transport for pixels/embeddings/gradients |
| `window.py` / `activation.py` | Iteration window with VPP replay cursors; forward handle, chunking, encoder THD params |
| `runtime.py` / `schedule.py` | Phase machine; schedule and finalizer wrappers |
| `encoder.py` / `optimizer.py` | Encoder DDP over WORLD + ZeRO-1; composite optimizer with WORLD overflow union |
| `checkpoint.py` | Weight-only torch_dist facade (`vision_model.*` with WORLD replica metadata) |
| `integration.py` / `observability.py` | Training-loop seams; iteration metrics and NVTX markers |

## Support matrix (v1)

Supported: Qwen3.5-VL (one vision encoder), `TP=1`, decoder `CP=1`,
`encoder_cp=1`, native PP/VPP/EP, fully replicated encoder with WORLD ZeRO-1,
`calculate_per_token_loss=True`, bf16 main path (fp16 covered by
overflow-union tests), THD packed sequences on both sides, native MCore vision
recompute (`None`/`selective`/`full`) via the override channel, text-only
microbatches, synchronous global `torch_dist` weight-only checkpoints,
`alignment_rows=1` (tests exercise 16).

Rejected at startup: FSDP/HSDP, FP8/MXFP8, full-iteration CUDA graphs, CPU
activation offload, comm overlap (`overlap_grad_reduce`,
`overlap_param_gather`, delayed reduction), multiple distributed-optimizer
instances, `calculate_per_token_loss=False`, non-`torch_dist` checkpoint
formats, non-weight-only save/load, invalid rank mappings.

Registered extension hooks (each exercised by a test at a non-degenerate
value): logical workers + `worker_ranks()` for encoder CP, single-valued
endpoints + multi-slice routes for decoder CP, the vision config override
allowlist + row-capacity policy for FP8, and the unified buffer allocator for
full-iteration CUDA graphs. The hooks guarantee no breaking schema change is
needed later; they do not mean the capability is implemented.
