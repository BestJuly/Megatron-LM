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
| `packing.py` | Greedy token-budget bin filling and the cross-iteration sample buffer |
| `runtime.py` / `schedule.py` | Phase machine; schedule and finalizer wrappers |
| `encoder.py` / `optimizer.py` | Encoder DDP over WORLD + ZeRO-1; composite optimizer with WORLD overflow union |
| `checkpoint.py` | torch_dist facade: `vision_model.*` save and load with WORLD replica metadata |
| `integration.py` / `observability.py` | Training-loop seams; iteration metrics and NVTX markers |

## Support matrix (v1)

Supported: Qwen3.5-VL (one vision encoder), `TP=1`, decoder `CP>=1`
(`cp_partition_mode=zigzag`), `encoder_cp=1`, native PP/VPP/EP, fully replicated encoder with WORLD ZeRO-1,
`calculate_per_token_loss=True`, bf16 main path (fp16 covered by
overflow-union tests), THD packed sequences on both sides, native MCore vision
recompute (`None`/`selective`/`full`) via the override channel, text-only
microbatches, synchronous global `torch_dist` checkpoints with exact resume
(model, optimizer, LR-scheduler and RNG state at the same world size),
`alignment_rows=1` (tests exercise 16), and native decoder DDP
`overlap_grad_reduce`/`overlap_param_gather`. Decoder overlap remains owned by
the native PP/VPP schedule; the separate encoder DDP domain stays synchronous
in P5/P6. Decoder-only EP A2A overlap via
`--overlap-moe-expert-parallel-comm --delay-wgrad-compute` is supported with
the native MCore requirements (`EP>1`, and VPP when `PP>1`); the vision encoder
remains outside that schedule.

Rejected at startup: FSDP/HSDP, FP8/MXFP8, full-iteration CUDA graphs, CPU
activation offload, delayed gradient reduction,
`overlap_param_gather_with_optimizer_step`, multiple distributed-optimizer
instances, `calculate_per_token_loss=False`, non-`torch_dist` checkpoint
formats, fully-parallel / asynchronous / non-persistent / constant-structure
checkpoint modes, invalid rank mappings.

### Checkpoint support matrix

Every "supported" row below was measured on 4x GB300 with the tiny MDP proxy
(4 decoder layers, 8 experts top-2, 2 vision layers, seq 1024, GBS 8, seed
1234), `--ckpt-format torch_dist --no-ckpt-fully-parallel-save`. Deltas are
against the trajectory the checkpoint was taken from; the run-to-run floor for
this shape is ~8.6e-2 in grad norm.

| Scenario | Status | Evidence |
|---|---|---|
| MDP -> MDP, identical parallel layout, full resume (model + optimizer + LR scheduler + RNG) | **Supported** | Resumed iterations reproduce the reference at d=0.000E+00 except one iteration at 4.6e-3, inside the floor |
| Cross-PP restart, weights only (`--no-load-optim`) | **Supported** | PP=2 save -> PP=4 load: the first resumed iteration matches the PP=2 source exactly in loss *and* grad norm; later iterations drift as expected without optimizer state |
| Cross-PP restart **with** optimizer state, saved with `--dist-ckpt-optim-fully-reshardable` | **Supported** | Same save/load pair tracks the source to 1e-3 grad norm and 1.2e-4 loss -- two orders tighter than weight-only, and tighter than the floor |
| Cross-PP restart with optimizer state, checkpoint saved with the defaults | **Rejected, by design** | Upstream raises before training starts (`distrib_optim_sharding_type == 'dp_reshardable'`). The flag is a **save-time** decision; a checkpoint already written without it can only be restarted weight-only. Note the upstream message names `--ckpt-fully-parallel-save`, which is a different flag and is itself rejected under MDP |
| Checkpoint missing encoder weights (e.g. a non-strict `--dist-ckpt-strictness` dropped them) | **Rejected, loudly** | `load_encoder_state` raises `MdpCheckpointError` instead of resuming from the random initialization |
| TransformerEngine `_extra_state` drift between the checkpoint and the running TE | **Tolerated** | The delegated load retries non-strictly, matching what `load_model_state_dict` gives every decoder chunk |
| Cross-TP / cross-EP / cross-CP restart, and changing the world size | **Untested** | Only the pipeline dimension was moved; no claim either way. Note that at `TP=PP=1, CP>1` the decoder and the WORLD encoder optimizer both shard over WORLD-sized groups and both compute `data_parallel_group_idx == 0`, which is exactly the collision the encoder's fixed key exists to prevent -- that topology has no end-to-end save/load coverage |
| native (non-MDP) checkpoint -> MDP, or MDP -> native | **Not supported** | Decoder keys line up, but the encoder is saved through its DDP wrapper and carries an extra `module.` level (`vision_model.module.<param>` vs `vision_model.<param>`) |
| Fully-parallel save/load, asynchronous, non-persistent, constant-structure caching, non-`torch_dist` formats | **Rejected at startup** | `assert_supported_checkpoint_config` and `validate_mdp_config` |

Decoder CP is implemented: an item's decoder rows are split across the
pipeline-stage-0 ranks with the integer inverse of TransformerEngine's zigzag
partition (`megatron/core/mdp/cp_partition.py`), and each endpoint receives only
its own rows. Nothing is replicated and no gradient is reduced across CP. See
the "Decoder context parallelism" section of `knowledge.md`.

Registered extension hooks (each exercised by a test at a non-degenerate
value): logical workers + `worker_ranks()` for encoder CP, the vision config
override allowlist + row-capacity policy for FP8, and the unified buffer
allocator for full-iteration CUDA graphs. The hooks guarantee no breaking
schema change is needed later; they do not mean the capability is implemented.
