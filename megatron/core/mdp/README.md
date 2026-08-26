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

For an agent-oriented implementation map, invariants, extension guide, and
verification commands, see [`knowledge.md`](knowledge.md).

## Phase machine

The runtime exposes three states (`EMPTY -> DECODER_READY -> DECODER_DONE ->
EMPTY`) driving seven phases:

| Phase | Where | Action |
|---|---|---|
| P0 | `begin_iteration` | Zero encoder grads, reset iteration state |
| P1 | `begin_iteration` | Capture the iteration window, broadcast fixed-width descriptors from the PP0 endpoint, run deterministic LPT to logical workers, check the plan digest across the group, exchange pixels |
| P2 | `begin_iteration` | Chunked encoder forward on encoder THD. Default training retains graph-connected outputs; `--mdp-encoder-recompute all` runs under `no_grad` and retains pixels/layouts/RNG recipes; evaluation retains neither graph nor recipe |
| P3 | `begin_iteration` | Exchange detached embeddings; endpoint assembles one detached leaf per vision-bearing microbatch |
| P4 | native schedule | Replay iterators feed the unmodified decoder schedule; the wrapped `finalize_model_grads_func` captures the in-place-reduced global token count |
| P5 | `end_iteration` | Exchange leaf gradients back; default mode runs one multi-tensor backward (native MCore Transformer recompute replays here), while `all` restores RNG and replays complete encoder chunks one by one before backward; WORLD sum-reduce with prescale 1, scale by `1/clamp(T_global, 1)` |
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
overflow-union tests), THD packed sequences on both sides, either native MCore
vision Transformer recompute (`None`/`selective`/`full`) via the override
channel or Design-Doc complete-encoder recompute (`--mdp-encoder-recompute
all`), text-only
microbatches, synchronous global `torch_dist` weight-only checkpoints,
`alignment_rows=1` (tests exercise 16), and native decoder DDP
`overlap_grad_reduce`/`overlap_param_gather`. Decoder overlap remains owned by
the native PP/VPP schedule; the separate encoder DDP domain stays synchronous
in P5/P6.

Rejected at startup: FSDP/HSDP, FP8/MXFP8, full-iteration CUDA graphs, CPU
activation offload, delayed gradient reduction,
`overlap_param_gather_with_optimizer_step`, multiple distributed-optimizer
instances, `calculate_per_token_loss=False`, non-`torch_dist` checkpoint formats,
non-weight-only save/load, invalid rank mappings.

Complete-encoder replay is deliberately exclusive with vision
`TransformerConfig` recompute overrides. Nesting the two would add a third
vision forward in P5 and obscure both the memory and compute contract.

`encoder_max_payload_rows` caps one rebuilt activation graph, not the complete
P5 footprint. Producers retain all packed pixels across P4, and P5 materializes
all routed chunk-output gradients before replay begins, so the initial peak is
all pixels plus all output gradients plus one chunk's activation graph.
Processed pixel and gradient references are dropped after each chunk backward;
smaller chunks reduce the graph term but add serial replay/backward launches.

Complete replay adds one full encoder forward, approximately doubling encoder
forward FLOPs while leaving encoder backward at one execution. Prefer native
`selective` or `full` Transformer recompute when its memory savings are enough;
use complete replay when saving patch embedding, position/RoPE, and patch-merger
activations justifies the extra complete forward.

Registered extension hooks (each exercised by a test at a non-degenerate
value): logical workers + `worker_ranks()` for encoder CP, single-valued
endpoints + multi-slice routes for decoder CP, the vision config override
allowlist + row-capacity policy for FP8, and the unified buffer allocator for
full-iteration CUDA graphs. The hooks guarantee no breaking schema change is
needed later; they do not mean the capability is implemented.
