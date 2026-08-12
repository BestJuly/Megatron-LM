#!/bin/bash
# MDP-vs-native experiment launcher for a pre-allocated GPU node.
#
# Runs the Qwen3.5-VL light experiment shape (default: 8 decoder layers,
# 8 experts top-2, REAL GDN hybrid attention, 7 vision layers) on one node
# with TP1 x PP4 -> DP2, MBS=4 / GBS=128, THD packed, mdp_mock data.
#
# Every experiment dimension is an environment variable:
#
#   MDP=0|1          enable MDP (default 0 = native in-model encoder)
#   OVERLAP=0|1      window-capture prefetch on a background thread + side
#                    CUDA stream (--mdp-overlap-window-capture; ignored when
#                    MDP=0)
#   PIXEL_SHARD=0|1  owner-sharded pixel reading + all_to_all pixel exchange
#                    (--mdp-pixel-owner-shard; ignored when MDP=0)
#   PIXEL_LOCALITY=0|1  planner prefers assigning items to their pixel owner
#                    within the LPT slack (--mdp-pixel-locality; ignored when
#                    MDP=0)
#   GRID_CACHE=0|1   vision-encoder grid cache (default 1). 0 restores the
#                    original per-grid loop code (pre-optimization behavior;
#                    exported as QWEN35_VL_GRID_CACHE). Note: the TP=1
#                    collate broadcast short-circuit stays active either way
#                    (behavior-identical).
#   GDN=0|1          GDN hybrid attention (default 1). 0 falls back to
#                    standard attention (for containers without a working
#                    FLA; FLA git main + Triton>=3.7.1 or tilelang required
#                    for the GDN backward on Hopper, see FLA #640).
#   NSYS=0|1         wrap in nsys (default 0). Requires OUT=<basename>.
#                    Capture window: iterations PROF_START..PROF_END-1 via
#                    cudaProfilerApi (defaults 7..8), NVTX on all ranks.
#   ITERS=<n>        train iterations (default 10; use 50 for steady-state
#                    timing, 3 for a sanity run)
#   ENTRY=<path>     entry script (default: pretrain_multimodal.py). Point at
#                    agent_works/mdp-pp4-timeline/pretrain_wrapper.py to
#                    reproduce the randomized 1k-2k-token scenario pool.
#   FLA_PATH=<dir>   optional PYTHONPATH prepend for an out-of-container FLA
#   EXTRA="..."      extra args appended verbatim
#
# Shape overrides: PP TP EP CP MBS GBS SEQ_LEN NUM_LAYERS NUM_EXPERTS
# MOE_TOPK VISION_NUM_LAYERS SEED NPROC PROF_START PROF_END.
#
# Examples (inside the training container, on the compute node):
#   MDP=0 ITERS=50                            bash run_mdp_experiments.sh
#   MDP=1 PIXEL_SHARD=1 ITERS=50              bash run_mdp_experiments.sh
#   MDP=1 GRID_CACHE=0 ITERS=50               bash run_mdp_experiments.sh
#   MDP=1 PIXEL_SHARD=1 NSYS=1 OUT=/path/a4   bash run_mdp_experiments.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

MDP=${MDP:-0}
OVERLAP=${OVERLAP:-0}
PIXEL_SHARD=${PIXEL_SHARD:-0}
PIXEL_LOCALITY=${PIXEL_LOCALITY:-0}
GRID_CACHE=${GRID_CACHE:-1}
GDN=${GDN:-1}
NSYS=${NSYS:-0}
ITERS=${ITERS:-10}
PROF_START=${PROF_START:-7}
PROF_END=${PROF_END:-9}
PP=${PP:-4}
TP=${TP:-1}
EP=${EP:-1}
CP=${CP:-1}
MBS=${MBS:-4}
GBS=${GBS:-128}
SEQ_LEN=${SEQ_LEN:-8192}
NUM_LAYERS=${NUM_LAYERS:-8}
NUM_EXPERTS=${NUM_EXPERTS:-8}
MOE_TOPK=${MOE_TOPK:-2}
VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-7}
SEED=${SEED:-1234}
NPROC=${NPROC:-8}
ENTRY=${ENTRY:-$REPO_ROOT/examples/multimodal_dev/pretrain_multimodal.py}
EXTRA=${EXTRA:-}

export QWEN35_VL_GRID_CACHE=$GRID_CACHE
# The scenario-pool wrapper (see ENTRY docs above) locates the repo via WT.
export WT=$REPO_ROOT
export PYTHONPATH=${FLA_PATH:+$FLA_PATH:}$REPO_ROOT:${PYTHONPATH:-}
cd "$REPO_ROOT"

MDP_ARGS=()
if [ "$MDP" = "1" ]; then
    MDP_ARGS=( --mdp-enable )
    if [ "$OVERLAP" = "1" ]; then
        MDP_ARGS+=( --mdp-overlap-window-capture )
    fi
    if [ "$PIXEL_SHARD" = "1" ]; then
        MDP_ARGS+=( --mdp-pixel-owner-shard )
    fi
    if [ "$PIXEL_LOCALITY" = "1" ]; then
        MDP_ARGS+=( --mdp-pixel-locality )
    fi
fi

GDN_ARGS=()
if [ "$GDN" = "1" ]; then
    GDN_ARGS=( --experimental-attention-variant gated_delta_net
               --linear-attention-freq 4
               --linear-conv-kernel-dim 4
               --linear-key-head-dim 128
               --linear-value-head-dim 128
               --linear-num-key-heads 16
               --linear-num-value-heads 32 )
fi

PROF_ARGS=()
LAUNCH=( torchrun --nproc_per_node "$NPROC" )
if [ "$NSYS" = "1" ]; then
    OUT=${OUT:?NSYS=1 requires OUT=<nsys output basename, no extension>}
    RANKS=$(seq -s' ' 0 $((NPROC - 1)))
    PROF_ARGS=( --profile
                --profile-step-start "$PROF_START"
                --profile-step-end "$PROF_END"
                --profile-ranks $RANKS
                --nvtx-ranges )
    LAUNCH=( nsys profile
             -o "$OUT"
             --force-overwrite=true
             -t cuda,nvtx
             -s none
             --cpuctxsw=none
             --capture-range=cudaProfilerApi
             --capture-range-end=stop
             torchrun --nproc_per_node "$NPROC" )
fi

"${LAUNCH[@]}" "$ENTRY" \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b_light \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --use-packed-sequence \
    --image-token-id 248056 \
    --tokenizer-type NullTokenizer \
    --vocab-size 248320 \
    --tensor-model-parallel-size "$TP" \
    --pipeline-model-parallel-size "$PP" \
    --expert-model-parallel-size "$EP" \
    --context-parallel-size "$CP" \
    --use-distributed-optimizer \
    --micro-batch-size "$MBS" \
    --global-batch-size "$GBS" \
    --train-iters "$ITERS" \
    --lr 1e-4 --min-lr 1e-5 --lr-decay-style constant \
    --lr-warmup-iters 0 \
    --weight-decay 0.1 --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --bf16 \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --calculate-per-token-loss \
    --enable-experimental \
    --use-flash-attn \
    --mtp-num-layers 0 \
    --num-layers "$NUM_LAYERS" \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length "$SEQ_LEN" \
    --normalization RMSNorm --apply-layernorm-1p --norm-epsilon 1e-06 \
    --swiglu --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 0.25 --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --qk-layernorm --attention-output-gate \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --make-vocab-size-divisible-by 485 \
    --untie-embeddings-and-output-weights \
    --num-experts "$NUM_EXPERTS" \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk "$MOE_TOPK" \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-dtype fp32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --vision-num-layers "$VISION_NUM_LAYERS" \
    --log-interval 1 \
    --eval-interval 100000 \
    --eval-iters 2 \
    --seed "$SEED" \
    --distributed-timeout-minutes 10 \
    "${GDN_ARGS[@]}" \
    "${PROF_ARGS[@]}" \
    "${MDP_ARGS[@]}" \
    $EXTRA
