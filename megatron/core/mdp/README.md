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
| P2 | `begin_iteration` | Chunked encoder forward on encoder THD. Default training retains graph-connected outputs; `--encoder-recompute-granularity whole` runs under `no_grad` and retains pixels/layouts/RNG recipes; evaluation retains neither graph nor recipe |
| P3 | `begin_iteration` | Exchange detached embeddings; endpoint assembles one detached leaf per vision-bearing microbatch |
| P4 | native schedule | Replay iterators feed the unmodified decoder schedule; the wrapped `finalize_model_grads_func` captures the in-place-reduced global token count |
| P5 | `end_iteration` | Exchange leaf gradients back; default mode runs one multi-tensor backward (native MCore Transformer recompute replays here), while `whole` restores RNG and replays complete encoder chunks one by one before backward; WORLD sum-reduce with prescale 1, scale by `1/clamp(T_global, 1)` |
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
| `config.py` | `MdpConfig`, support-matrix validation, typed encoder recompute configuration, vision config override allowlist |
| `protocols.py` | `MdpModelAdapter` and the carrier types; the only place model behavior enters core |
| `rank_mapping.py` | Pure-compute outer-DP planning groups and logical workers from `RankGenerator` coordinates |
| `groups.py` | Process-group installation, fixed-width descriptor broadcast |
| `plan.py` / `planner.py` | Minimal-sufficient plan data model, blake2b digest, deterministic integer LPT, group consistency check |
| `window.py` | Iteration capture with VPP replay cursors; delegates all model geometry to `adapter.validate_item` |
| `allocator.py` / `storage.py` | Single allocation point for MDP buffers; endpoint leaf storage |
| `bridge.py` | One ledger + transport for pixels/embeddings/gradients |
| `activation.py` | Forward handle (retained graph or whole-encoder replay recipe), chunk retention, multi-tensor backward |
| `packing.py` | Greedy token-budget bin filling and the cross-iteration sample buffer |
| `runtime.py` / `schedule.py` | Phase machine; schedule and finalizer wrappers |
| `encoder.py` / `optimizer.py` | Encoder DDP over WORLD + ZeRO-1; composite optimizer with WORLD overflow union |
| `checkpoint.py` | torch_dist facade: `vision_model.*` (native keys) save and load with WORLD replica metadata |
| `integration.py` / `observability.py` | Training-loop seams; iteration metrics and NVTX markers |

## Support matrix (v1)

Core is model-agnostic: an item reaches it as `payload_rows`, `output_rows`, an
integer LPT cost and an opaque fixed-width `shape_key` that only the adapter
interprets (`adapter.shape_key_width` slots). Row arithmetic, attention
segmentation and geometry validation belong to the adapter; core never assumes a
`(t, h, w)` patch grid or a spatial merge factor.

The decoder is equally opaque. Core touches it only through parameter
disjointness against the encoder, `len(model)` (VPP chunks), the wrapped native
schedule and the `finalize_model_grads` token capture. Which decoder runs is the
model package's choice: `examples/multimodal_dev` builds it through the
registry's `language_model_builder_fn`, by default `hybrid_config_from_args` +
`HybridModelBuilder` -- the same path as `pretrain_hybrid.py`, so `--spec` and
builder-side stack features apply.

| `--model-arch` | Vision encoder | Decoder build path | Status |
|---|---|---|---|
| `qwen35_vl` | Qwen3.5 ViT (grid shape key `(t, h, w)`, merge 2) | default hook (`HybridModelBuilder`), RoPE fraction/base pinned | supported |
| `qwen38_vl` | Qwen3.5 ViT, reused (HF `qwen4_exp` code identical; output width = decoder hidden 2560) | default hook; Qwen3.8 decoder (GDN + QSA + MoE, gated residual, Engram/PLE, MTP) from CLI args, RoPE fraction/base pinned | supported (proxy-validated) |

Supported: one vision encoder per model, `TP=1`, decoder `CP=1`,
`encoder_cp=1`, native PP/VPP/EP, fully replicated encoder with WORLD ZeRO-1,
`calculate_per_token_loss=True`, bf16 main path (fp16 covered by
overflow-union tests), THD packed sequences on both sides, native MCore vision
Transformer recompute (`selective`/`full`) or Design-Doc whole-encoder replay
(`--encoder-recompute-granularity whole`), text-only microbatches, synchronous
global `torch_dist` checkpoints with exact resume (model, optimizer,
LR-scheduler and RNG state at the same world size), `alignment_rows=1` (tests
exercise 16), and native decoder DDP
`overlap_grad_reduce`/`overlap_param_gather`. Decoder overlap remains owned by
the native PP/VPP schedule; the separate encoder DDP domain stays synchronous
in P5/P6. Decoder-only EP A2A overlap via
`--overlap-moe-expert-parallel-comm --delay-wgrad-compute` is supported with
the native MCore requirements (`EP>1`, and VPP when `PP>1`); the vision encoder
remains outside that schedule.

Decoder FP8 is supported. The decoder uses the native `--fp8`/`--fp8-recipe`
flags; the vision `TransformerConfig` is built separately by the model adapter
and never inherits them, so decoder FP8 leaves encoder model precision unchanged
(`validate_effective_vision_config` re-asserts that against the resolved vision
config inside `build_encoder_domain`). For packed decoder inputs, quantized
GEMMs need the packed row count to be a multiple of
`get_fp8_align_size(fp8_recipe)` (32 for MXFP8, 16 otherwise), which
`pack_or_pad_batch` in `examples/multimodal_dev/forward_step.py` supplies by
extending the last sample's padded region. Alignments that call site cannot
derive fail loudly instead: with `--use-packed-sequence`, `--fp4-format` and
`--fp8-recipe custom` raise `NotImplementedError`.

Decoder MXFP8 parameter gather may reuse its gradient buffer with
`--reuse-grad-buf-for-mxfp8-param-ag`. This requires BF16 training, decoder FP8,
`--fp8-recipe mxfp8`, and `--fp8-param-gather`.
The encoder keeps independent BF16 buffers and synchronous optimizer updates;
only decoder members participate in MXFP8 parameter staging. Both decoder
parameter-gather overlap modes support synchronous `torch_dist` checkpoint
save/load, including full resume and ordinary weight-only initialization
without `--load-main-params-from-ckpt`. For MXFP8 buffer reuse,
`fully_reshardable` optimizer checkpoints require encoder remainder storage
disabled at both save and load; the default `dp_reshardable` format supports
remainder storage. The optimizer checks the actual checkpoint format and
encoder remainder mode before saving or loading optimizer state.

Decoder optimizer: Adam (DistributedOptimizer) or an emerging optimizer such as
`--optimizer muon`, whose layer-wise distributed optimizer stays one member of the
composite (its layer-sharded parameter all-gather, grad norm and checkpoint state are
its own). The encoder is always an Adam ZeRO-1 domain over WORLD: under an emerging
decoder optimizer its optimizer config is projected to Adam with the decoder's learning
rate, schedule, weight decay and betas (`build_encoder_optimizer_config`).

Per-layer CUDA graphs on the decoder (`--cuda-graph-impl transformer_engine` or `local`)
are supported with static THD batches (`--thd-static-packing --pad-packed-seq-alignment
max --max-seqlen-per-dp-cp-rank N --thd-max-packed-sequences K`); the encoder, the
schedule and every MDP collective stay eager, and the graphs never see the bridge.
`MultimodalModel` forwards `decoder` / `mtp` / `rotary_pos_emb` /
`position_embedding_type` so `TECudaGraphHelper` finds the decoder layers. mRoPE (and
every rotary embedding but `rope`) is captured as a per-layer graph input, which needs
at least one eager step before capture (`--cuda-graph-warmup-steps >= 1`). Not
combinable with `--mdp-overlap-window-capture`.

Rejected at startup: FSDP/HSDP, encoder FP8, full-iteration CUDA graphs (a
full-iteration graph would capture P4 together with the host-planned phases around
it), CPU activation offload, delayed gradient reduction,
`overlap_param_gather_with_optimizer_step`,
multiple distributed-optimizer instances, replicated (non-distributed) decoder
optimizer state, `calculate_per_token_loss=False`,
non-`torch_dist` checkpoint formats, fully-parallel / asynchronous / non-persistent / constant-structure
checkpoint modes, invalid rank mappings.

### Checkpoint support matrix

The table below records the earlier MDP checkpoint validation, not the new
MXFP8 buffer-reuse combination. Reuse was validated separately with four-rank
DDP/optimizer checkpoint API round trips at fixed parallelism, with parameter
gather overlap off and on. Cross-parallelism and full Qwen3.5-VL 35B
training-entrypoint resumes remain unvalidated for the reuse combination.

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
| Cross-TP / cross-EP / cross-CP restart, and changing the world size | **Untested** | Only the pipeline dimension was moved; no claim either way |
| native (non-MDP) checkpoint -> MDP, or MDP -> native, weights (`--no-load-optim`) | **Supported** | The encoder is published from the module its DDP wraps, under the native keys `vision_model.<param>`. A native checkpoint loaded into MDP reproduces the native run's next-iteration loss on the same weights (Qwen3.8-VL proxy) |
| MDP checkpoint written with the DDP level in its encoder keys (`vision_model.module.<param>`) | **Supported** | `load_checkpoint` detects the layout from the checkpoint's tensor keys and builds a matching load skeleton |
| Fully-parallel save/load, asynchronous, non-persistent, constant-structure caching, non-`torch_dist` formats | **Rejected at startup** | `assert_supported_checkpoint_config` and `validate_mdp_config` |

Whole-encoder replay is deliberately exclusive with native vision
`TransformerConfig` recompute. Nesting the two would add a third vision
forward in P5 and obscure both the memory and compute contract.

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
endpoints + multi-slice routes for decoder CP, the typed encoder configuration
+ row-capacity policy for encoder FP8, and the unified buffer allocator
for full-iteration CUDA graphs. The hooks guarantee no breaking schema change is
needed later; they do not mean the capability is implemented.
