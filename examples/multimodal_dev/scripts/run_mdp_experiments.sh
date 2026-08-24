#!/bin/bash
# MDP-vs-native experiment launcher for a pre-allocated GPU node.
#
# Runs the Qwen3.5-VL experiment shape on one node. The defaults are the
# MDP_opt study shape: 20 decoder layers + 1 MTP layer, 128 experts top-8,
# REAL GDN hybrid attention, 13 vision layers, TP1 x CP1 x PP2 x EP2 -> DP2
# over NPROC=4 GB200, MBS=16 / GBS=256 (accum 8), 20 iters, THD packed,
# mdp_mock data, flex/hybridep MoE dispatcher, vision-encoder full recompute,
# precision-aware optimizer, forced uniform routing, GDN fusion and manual GC
# (the last four match the GB200 EP8 reference run; decoder recompute, which
# that reference also used, stays OFF here and is opt-in per experiment).
# Every one of those is an env override (see below), so the
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
#   DISPATCHER=flex|alltoall  MoE token dispatcher (default flex). flex adds
#                    --moe-flex-dispatcher-backend $FLEX_BACKEND, and for
#                    hybridep also --moe-hybridep-num-sms $HYBRIDEP_NUM_SMS.
#                    This matches the GB200/GB300 benchmark recipes; the
#                    previous hardcoded default was alltoall, so results
#                    taken before this change are not comparable.
#   FLEX_BACKEND=<name>       flex dispatcher backend (default hybridep)
#   HYBRIDEP_NUM_SMS=<n>      hybridep SM budget (default 32)
#   VISION_RECOMPUTE=0|1      full activation recompute for the vision
#                    encoder (default 1). The switch differs by path: the
#                    native path takes --recompute-vision, while MDP rejects
#                    that flag outright and requires the override channel
#                    (--mdp-vision-config-override, see
#                    pretrain_multimodal.py:197). This knob picks the right
#                    one for the current MDP setting. Note it is independent
#                    of the decoder --recompute-* flags (DECODER_RECOMPUTE).
#   DECODER_RECOMPUTE=0|1     decoder activation recompute (default 0 = off).
#                    Turn on per experiment when the shape would otherwise
#                    OOM, or to match a reference run that used it. Values
#                    via DECODER_RECOMPUTE_GRANULARITY (full),
#                    DECODER_RECOMPUTE_METHOD (uniform) and
#                    DECODER_RECOMPUTE_NUM_LAYERS (1).
#   PRECISION_AWARE_OPT=0|1   fp32 master grads/params + bf16 Adam moments
#                    (default 1). Saves 4 bytes per parameter against the
#                    all-fp32 default.
#   FORCE_LOAD_BALANCING=0|1  --moe-router-force-load-balancing (default 1).
#                    Uniform routing removes expert-load jitter from
#                    iteration timing. MUST be 0 for any convergence /
#                    fine-tuning run: it freezes data-dependent routing.
#   GDN_FUSION=0|1   fused streamed pre-gated-delta-rule (default 1, requires
#                    GDN=1 and causal-conv1d in the container).
#   MANUAL_GC=0|1    disable automatic GC, collect every MANUAL_GC_INTERVAL
#                    iterations instead (default 1 / 50). Removes random GC
#                    pauses from steady-state timing.
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
DISPATCHER=${DISPATCHER:-flex}
FLEX_BACKEND=${FLEX_BACKEND:-hybridep}
HYBRIDEP_NUM_SMS=${HYBRIDEP_NUM_SMS:-32}
VISION_RECOMPUTE=${VISION_RECOMPUTE:-1}
DECODER_RECOMPUTE=${DECODER_RECOMPUTE:-0}
DECODER_RECOMPUTE_GRANULARITY=${DECODER_RECOMPUTE_GRANULARITY:-full}
DECODER_RECOMPUTE_METHOD=${DECODER_RECOMPUTE_METHOD:-uniform}
DECODER_RECOMPUTE_NUM_LAYERS=${DECODER_RECOMPUTE_NUM_LAYERS:-1}
PRECISION_AWARE_OPT=${PRECISION_AWARE_OPT:-1}
FORCE_LOAD_BALANCING=${FORCE_LOAD_BALANCING:-1}
# Tracked separately so GDN=0 can silently drop the fusion when it was merely
# left at its default, but still fail loudly when the caller asked for both.
GDN_FUSION_EXPLICIT=${GDN_FUSION+set}
GDN_FUSION=${GDN_FUSION:-1}
MANUAL_GC=${MANUAL_GC:-1}
MANUAL_GC_INTERVAL=${MANUAL_GC_INTERVAL:-50}
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

DISPATCHER_ARGS=( --moe-token-dispatcher-type "$DISPATCHER" )
case "$DISPATCHER" in
    flex)
        DISPATCHER_ARGS+=( --moe-flex-dispatcher-backend "$FLEX_BACKEND" )
        if [ "$FLEX_BACKEND" = "hybridep" ]; then
            DISPATCHER_ARGS+=( --moe-hybridep-num-sms "$HYBRIDEP_NUM_SMS" )
        fi
        ;;
    alltoall) ;;
    *) echo "ERROR: DISPATCHER must be flex|alltoall, got '$DISPATCHER'" >&2; exit 1 ;;
esac

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

# Vision-encoder full recompute. --recompute-vision sets granularity=full,
# method=uniform, num_layers=1 on the vision config (pretrain_multimodal.py:88);
# MDP raises on that flag and takes the same three values through its own
# allowlisted override channel, so spell them out to keep both paths identical.
VISION_RECOMPUTE_ARGS=()
if [ "$VISION_RECOMPUTE" = "1" ]; then
    if [ "$MDP" = "1" ]; then
        VISION_RECOMPUTE_ARGS=( --mdp-vision-config-override recompute_granularity=full
                                --mdp-vision-config-override recompute_method=uniform
                                --mdp-vision-config-override recompute_num_layers=1 )
    else
        VISION_RECOMPUTE_ARGS=( --recompute-vision )
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
    # Fused streamed pre-gated-delta-rule. transformer_config.py rejects it
    # unless the attention variant is gated_delta_net, so it lives inside this
    # branch; it also needs causal-conv1d in the container
    # (megatron/core/fusions/fused_pre_gated_delta_rule.py).
    if [ "$GDN_FUSION" = "1" ]; then
        GDN_ARGS+=( --gdn-pre-gated-delta-rule-fusion )
    fi
elif [ "$GDN_FUSION" = "1" ]; then
    if [ -n "$GDN_FUSION_EXPLICIT" ]; then
        echo "ERROR: GDN_FUSION=1 requires GDN=1 (the fusion is only supported with" >&2
        echo "       experimental_attention_variant=gated_delta_net)." >&2
        exit 1
    fi
    # Default value only -- GDN=0 is the documented fallback for containers
    # without a working FLA, so drop the fusion instead of blocking it.
    GDN_FUSION=0
fi

# Decoder activation recompute. Off by default -- turn on per experiment when
# the shape would otherwise OOM, or to match a reference run that used it.
DECODER_RECOMPUTE_ARGS=()
if [ "$DECODER_RECOMPUTE" = "1" ]; then
    DECODER_RECOMPUTE_ARGS=( --recompute-granularity "$DECODER_RECOMPUTE_GRANULARITY"
                             --recompute-method "$DECODER_RECOMPUTE_METHOD"
                             --recompute-num-layers "$DECODER_RECOMPUTE_NUM_LAYERS" )
fi

# Precision-aware optimizer: fp32 master grads/params with bf16 Adam moments,
# saving 4 bytes per parameter against the all-fp32 default.
PRECISION_AWARE_OPT_ARGS=()
if [ "$PRECISION_AWARE_OPT" = "1" ]; then
    PRECISION_AWARE_OPT_ARGS=( --use-precision-aware-optimizer
                               --main-grads-dtype fp32
                               --main-params-dtype fp32
                               --exp-avg-dtype bf16
                               --exp-avg-sq-dtype bf16 )
fi

# Uniform routing for throughput measurement. It freezes data-dependent
# routing, so it must be off for any convergence / fine-tuning run.
FORCE_LOAD_BALANCING_ARGS=()
if [ "$FORCE_LOAD_BALANCING" = "1" ]; then
    FORCE_LOAD_BALANCING_ARGS=( --moe-router-force-load-balancing )
fi

# Disable Python's automatic GC and collect on a fixed interval instead, so
# random collection pauses stop polluting steady-state iteration timing.
MANUAL_GC_ARGS=()
if [ "$MANUAL_GC" = "1" ]; then
    MANUAL_GC_ARGS=( --manual-gc --manual-gc-interval "$MANUAL_GC_INTERVAL" )
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
    "${DISPATCHER_ARGS[@]}" \
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
    "${VISION_RECOMPUTE_ARGS[@]}" \
    "${DECODER_RECOMPUTE_ARGS[@]}" \
    "${PRECISION_AWARE_OPT_ARGS[@]}" \
    "${FORCE_LOAD_BALANCING_ARGS[@]}" \
    "${MANUAL_GC_ARGS[@]}" \
    $EXTRA
