#!/bin/bash
# MDP-vs-native experiment launcher for a pre-allocated GPU node.
#
# Runs the Qwen3.5-VL experiment shape on one node. The defaults are the
# MDP_opt study shape: 20 decoder layers + 1 MTP layer, 128 experts top-8,
# REAL GDN hybrid attention, 13 vision layers, TP1 x CP1 x PP2 x EP2 -> DP2
# over NPROC=4 GB200, MBS=16 / GBS=256 (accum 8), 20 iters, THD packed,
# mdp_mock data. Every one of those is an env override (see below), so the
# older light shape is still reachable e.g. with
#   PP=4 EP=1 NPROC=8 NUM_LAYERS=8 NUM_EXPERTS=8 MOE_TOPK=2 \
#   VISION_NUM_LAYERS=7 MTP_NUM_LAYERS=0 MBS=4 GBS=128 ITERS=10
#
# Every experiment dimension is an environment variable:
#
#   MDP=0|1          enable MDP (default 0 = native in-model encoder)
#   OVERLAP=0|1      window-capture prefetch on a background thread + side
#                    CUDA stream (--mdp-overlap-window-capture; ignored when
#                    MDP=0)
#   EP_OVERLAP=0|1   native decoder 1F1B EP A2A overlap with delayed wgrad
#                    compute (default 0). This is independent of MDP window
#                    capture and requires EP>1 plus VPP>1 when PP>1.
#   VPP=<n>          virtual stages per PP rank (default 1 = disabled)
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
#   ITERS=<n>        train iterations (default 20; use 50 for steady-state
#                    timing, 3 for a sanity run)
#   ENTRY=<path>     entry script (default: pretrain_multimodal.py). Point at
#                    a wrapper to install a custom scenario pool.
#   NNODES=<n>       nodes in the job (default 1). With NNODES>1, NPROC is
#                    per-node and NODE_RANK / MASTER_ADDR / MASTER_PORT must be
#                    set per node -- e.g. 8 GPUs on GB200 (4 per node) is
#                    NNODES=2 NPROC=4. Defaults keep the single-node behavior
#                    byte-identical.
#   NODE_RANK=<n>    this node's index (default 0)
#   MASTER_ADDR/MASTER_PORT   rendezvous endpoint (default 127.0.0.1:29500)
#   FLA_PATH=<dir>   optional PYTHONPATH prepend for an out-of-container FLA
#   ROUTER_FUSION=0|1  fused MoE router (--moe-router-fusion, default 1). It
#                    changes top-k tie-breaking, so it shifts numerics
#                    slightly; 0 restores the unfused router for A/B runs.
#   CE_FUSION=te|native|off  cross-entropy implementation (default te).
#                    te     --cross-entropy-loss-fusion --cross-entropy-fusion-impl te
#                    native --cross-entropy-loss-fusion --cross-entropy-fusion-impl native
#                    off    no fusion args at all
#                    The TE path needs the assert in megatron/training/
#                    arguments.py (~1822) commented out; it is, on this branch.
#   EXTRA="..."      extra args appended verbatim
#
# Shape overrides: PP VPP TP EP CP MBS GBS SEQ_LEN NUM_LAYERS NUM_EXPERTS
# MOE_TOPK VISION_NUM_LAYERS MTP_NUM_LAYERS MTP_LOSS_SCALING_FACTOR SEED
# NPROC PROF_START PROF_END.
#
# Examples (inside the training container, on the compute node):
#   MDP=0 ITERS=50                            bash run_mdp_experiments.sh
#   MDP=1 ITERS=50                            bash run_mdp_experiments.sh
#   MDP=1 GRID_CACHE=0 ITERS=50               bash run_mdp_experiments.sh
#   MDP=1 EP_OVERLAP=1 VPP=2 ITERS=50         bash run_mdp_experiments.sh
#   MDP=1 NSYS=1 OUT=/path/a4                 bash run_mdp_experiments.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NVTE_FUSED_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

MDP=${MDP:-0}
OVERLAP=${OVERLAP:-0}
EP_OVERLAP=${EP_OVERLAP:-0}
PIXEL_LOCALITY=${PIXEL_LOCALITY:-0}
GRID_CACHE=${GRID_CACHE:-1}
GDN=${GDN:-1}
NSYS=${NSYS:-0}
ITERS=${ITERS:-20}
PROF_START=${PROF_START:-7}
PROF_END=${PROF_END:-9}
ROUTER_FUSION=${ROUTER_FUSION:-1}
CE_FUSION=${CE_FUSION:-te}
PP=${PP:-2}
VPP=${VPP:-1}
TP=${TP:-1}
EP=${EP:-2}
CP=${CP:-1}
MBS=${MBS:-16}
GBS=${GBS:-256}
SEQ_LEN=${SEQ_LEN:-8192}
NUM_LAYERS=${NUM_LAYERS:-20}
NUM_EXPERTS=${NUM_EXPERTS:-128}
MOE_TOPK=${MOE_TOPK:-8}
VISION_NUM_LAYERS=${VISION_NUM_LAYERS:-13}
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-1}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.1}
SEED=${SEED:-1234}
NPROC=${NPROC:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
ENTRY=${ENTRY:-$REPO_ROOT/examples/multimodal_dev/pretrain_multimodal.py}
EXTRA=${EXTRA:-}

if [ "$EP_OVERLAP" = "1" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
else
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
fi

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
    if [ "$PIXEL_LOCALITY" = "1" ]; then
        MDP_ARGS+=( --mdp-pixel-locality )
    fi
fi

if [ "$VPP" -lt 1 ]; then
    echo "ERROR: VPP must be >= 1, got '$VPP'" >&2
    exit 1
fi
VPP_ARGS=()
if [ "$VPP" -gt 1 ]; then
    VPP_ARGS=( --num-virtual-stages-per-pipeline-rank "$VPP" )
fi

EP_OVERLAP_ARGS=()
if [ "$EP_OVERLAP" = "1" ]; then
    if [ "$EP" -le 1 ]; then
        echo "ERROR: EP_OVERLAP=1 requires EP > 1" >&2
        exit 1
    fi
    if [ "$PP" -gt 1 ] && [ "$VPP" -le 1 ]; then
        echo "ERROR: EP_OVERLAP=1 with PP > 1 requires VPP > 1" >&2
        exit 1
    fi
    EP_OVERLAP_ARGS=(
        --overlap-moe-expert-parallel-comm
        --delay-wgrad-compute
    )
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

# MTP_NUM_LAYERS=0 omits the MTP args entirely (Megatron treats the arg's
# absence and 0 the same, but omitting keeps the command line honest).
MTP_ARGS=()
if [ "$MTP_NUM_LAYERS" -gt 0 ]; then
    MTP_ARGS=( --mtp-num-layers "$MTP_NUM_LAYERS"
               --mtp-loss-scaling-factor "$MTP_LOSS_SCALING_FACTOR" )
fi

ROUTER_FUSION_ARGS=()
if [ "$ROUTER_FUSION" = "1" ]; then
    ROUTER_FUSION_ARGS=( --moe-router-fusion )
fi

# TE is the default: the fused TE kernel is faster than the native one at this
# vocab size, and the stability bug that used to gate it is fixed in the
# container's TE build (the assert in megatron/training/arguments.py is
# commented out on this branch for exactly that reason).
CE_ARGS=()
case "$CE_FUSION" in
    te)     CE_ARGS=( --cross-entropy-loss-fusion --cross-entropy-fusion-impl te ) ;;
    native) CE_ARGS=( --cross-entropy-loss-fusion --cross-entropy-fusion-impl native ) ;;
    off)    CE_ARGS=() ;;
    *) echo "ERROR: CE_FUSION must be te|native|off, got '$CE_FUSION'" >&2; exit 1 ;;
esac

PROF_ARGS=()
TORCHRUN=( torchrun
           --nnodes "$NNODES"
           --node_rank "$NODE_RANK"
           --master_addr "$MASTER_ADDR"
           --master_port "$MASTER_PORT"
           --nproc_per_node "$NPROC" )
LAUNCH=( "${TORCHRUN[@]}" )
if [ "$NSYS" = "1" ]; then
    OUT=${OUT:?NSYS=1 requires OUT=<nsys output basename, no extension>}
    # Profile-rank ids are GLOBAL, so span every rank in the job, not just
    # this node's share.
    RANKS=$(seq -s' ' 0 $((NPROC * NNODES - 1)))
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
             "${TORCHRUN[@]}" )
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
    "${VPP_ARGS[@]}" \
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
    --vision-num-layers "$VISION_NUM_LAYERS" \
    --log-interval 1 \
    --eval-interval 100000 \
    --eval-iters 2 \
    --seed "$SEED" \
    --distributed-timeout-minutes 10 \
    "${MTP_ARGS[@]}" \
    "${ROUTER_FUSION_ARGS[@]}" \
    "${CE_ARGS[@]}" \
    "${GDN_ARGS[@]}" \
    "${EP_OVERLAP_ARGS[@]}" \
    "${PROF_ARGS[@]}" \
    "${MDP_ARGS[@]}" \
    $EXTRA
