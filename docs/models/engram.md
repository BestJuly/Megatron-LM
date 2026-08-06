# Engram

Engram adds trainable, deterministic n-gram memory to selected GPT transformer layers. This
document defines the first supported MCore milestone. The numerical definition follows the
official DeepSeek Engram implementation at commit
`fb7f84a21f91223715394a33a1dc24bbfb7f788e` and Engram paper v2 (arXiv 2601.07372).

## Configuration and validation

Engram is enabled when `--engram-vocab-sizes` is present. The Engram namespace contains the
global table budget for every n-gram order, selected 1-based global layer IDs, maximum n-gram
order, hash-head count, memory dimension per n-gram order, convolution kernel size, hash seed,
raw tokenizer pad ID, tokenizer-map artifact, sparse-table LR multiplier, and sparse-table weight
decay.

Startup validates the following before model allocation:

- there is one positive global vocabulary budget for every order from 2 through the configured
  maximum order;
- layer IDs are unique and fall in `[1, num_layers]`;
- the memory dimension is divisible by the positive number of hash heads;
- the versioned tokenizer-map artifact matches the configured tokenizer vocabulary, pad ID,
  selected layers, maximum order, and hash seed;
- CP is one and packed sequences, VPP, activation recomputation, CUDA graphs, and FSDP are off.

EP is legal without MoE experts when Engram is enabled. If MoE and Engram coexist they share the
same EP dimension and `ProcessGroupCollection.ep`. DeepEP is not a dependency or backend.

## Offline tokenizer map and hashing

`tools/engram/generate_tokenizer_map.py` is the only component that constructs a Hugging Face
tokenizer. It scans raw token IDs in ascending order and reproduces the official normalization
sequence: NFKC, NFD, accent stripping, lowercase, whitespace collapse, preservation of the
single-space token, and surrounding-space stripping. Undecodable replacement-character tokens
use their tokenizer token string as the canonical key. The first occurrence of each key receives
the next compressed ID.

The artifact also contains the official NumPy-PCG64-generated odd multiplier for every selected
layer and n-gram position. Keeping these constants in the offline artifact lets model construction
register Torch buffers without importing NumPy or reproducing PCG64 in the training process.

For input `tokens[B,S]`, the model maps nonnegative token IDs through `remap[V]`, left-pads each
suffix with the compressed pad ID, and computes the official signed-int64 multiplicative-XOR mix.
For each order and head it applies the corresponding distinct prime modulus. Hashing finishes on
the full local input sequence before any SP selection, so n-grams at an SP boundary include tokens
from the preceding TP partition.

## Tensor layout and residual semantics

The canonical layouts are:

| Value | Standard residual | Native mHC |
| --- | --- | --- |
| Transformer input | `[S_local,B,H]` | `[S_local,B,N*H]` |
| Hash IDs before SP | `[B,S,(max_order-1)*K]` | same |
| Retrieved memory | `[S_local,B,D_mem]` | shared across branches |
| Gate | `[S_local,B,1]` | `[S_local,B,N,1]` |
| Engram output | `[S_local,B,H]` | `[S_local,B,N*H]` |

Each n-gram order has `K` prime-sized embedding tables. A head returns
`memory_dim / K` values; heads and orders are concatenated. One value projection and all sparse
tables are shared across mHC branches. Every branch has its own key projection, key RMSNorm,
query RMSNorm, scalar gate, convolution normalization, and convolution channels. The gate uses
the official signed-square-root transform before sigmoid.

The depthwise causal convolution has dilation equal to the maximum n-gram order. Its weight is
zero-initialized, so the convolution branch is initially zero. Engram output is added directly to
the residual tensor before attention. For native mHC this addition occurs before the layer's
`H_pre` mapping and preserves all real branches; Engram never creates temporary fake branches or
contracts them by averaging.

## EP ownership and communication

Every prime-sized table is independently divided into contiguous, possibly uneven row ranges.
For global row count `R`, EP rank `r` owns:

```text
base = R // EP
remainder = R % EP
start(r) = r * base + min(r, remainder)
rows(r) = base + (r < remainder)
```

No padding contributes to the logical shape. Each local weight is
`[rows(r), memory_dim / K]`; the full table is never allocated on an EP rank.

A module batches requests for all of its head tables into one routing exchange:

1. flatten `(table, row)` requests in token/head order;
2. calculate the owner and owner-local row, stable-sort by owner, exchange per-peer counts, and
   exchange integer requests with native `all_to_all_single`;
3. perform the appropriate local table lookups in received request order;
4. return embeddings with MCore's differentiable variable-split all-to-all;
5. invert the owner sort and restore `[B,S_local,num_tables,head_dim]`.

The protocol permits duplicate rows, unequal request counts, empty peer splits, uneven tables,
and EP=1. Backward through the return all-to-all sends gradients to the owning lookup operations,
where embedding accumulation combines duplicates. In deterministic mode, the owner sorts local row
IDs, sums each repeated-row gradient with a segmented reduction, and writes only unique rows. This
avoids CUDA embedding atomic-add ordering differences across checkpoint restarts without deduplicating
or changing the forward request protocol.

Sparse table weights have `allreduce=False`, so DDP synchronizes matching shards over expert-DP,
not across EP owners. The weights are replicated over TP in this milestone. With SP, their and all
dense Engram parameter gradients are summed across TP by the existing sequence-parallel finalizer.
All other Engram parameters use normal dense DP synchronization. The opt-in training verifier logs
per-rank ordered token checksums plus global sparse-table and FP64 full-model checksums before and
after every optimizer step, in addition to gradient, update, and peak-memory evidence. Normal
training does not compute these diagnostics.

## TP, SP, and PP data flow

TP does not shard Engram tables or dense Engram projections. SP hashes full `tokens[B,S]`, then
selects the same contiguous sequence interval used by MCore's sequence-parallel hidden state.

Before pipeline scheduling, the first PP stage broadcasts token IDs through the existing matching
PP group. A PP group fixes the data shard and TP/EP coordinates, so every stage receives the same
microbatch tokens without a model-forward collective or mutable module cache. Layers use their
global 1-based `layer_number`; therefore selected layers on middle and last stages work without
stage-specific layer specs or saved per-forward state.

## Optimizer policy

Only sparse table weights carry `is_engram_embedding=True`. A `ParamKey` override selects those
parameters, routes them to Adam, sets both maximum and minimum LR schedule endpoints to the model
endpoint multiplied by the configured factor (default 5), and applies the configured fixed weight
decay (default zero). Value/key projections, gates, norms, and convolution remain in the model's
selected optimizer and ordinary LR/weight-decay policy. The standard non-distributed optimizer path
splits mixed model/Engram policies into chained optimizer instances. Mixed optimizer types over
expert-parallel parameters are rejected with DistributedOptimizer, and mixed optimizer types are
rejected with Megatron FSDP, because those paths cannot safely split their shared gradient buffers.

## Distributed checkpointing

Each table weight is represented by an irregular `ShardedTensor` whose global shape is exactly
`[global_prime_rows, head_dim]`, whose local shape is `[owned_rows, head_dim]`, and whose global
offset is `[owned_start, 0]`. `axis_fragmentations=None` records a nonuniform grid. TP and
expert-DP copies are replicas; EP ranks are distinct global row slices. There is no EP-dependent
padding in checkpoint metadata.

Torch distributed checkpoint save/load therefore transfers intersecting slices directly between
the saved and target layouts when EP changes. Optimizer state uses the model parameter's same
sharded metadata, enabling Adam moment and master-weight resharding without gathering a full table
onto one rank. Optimizer restore also preserves occurrence order for parameter groups that share
the legacy weight-decay/expert identifier, so the Engram 5x LR group remains distinct from an
otherwise identical ordinary Adam expert group.

## Supported and deferred combinations

The first milestone supports BF16 GPT training with standard residuals or native mHC, EP, TP, PP,
SP, MoE coexistence, native all-to-all, and torch distributed checkpoints. CP greater than one,
packed THD inputs, VPP, activation recomputation, CUDA graphs, FSDP, inference serving, offload,
DeepEP, communication overlap, request deduplication, FP8 table storage, and fused Engram kernels
are intentionally deferred and rejected during startup.
