#!/bin/bash

# Launch script for Qwen3.5-VL training with pure mcore implementation.
#
# Usage (from Megatron-LM repo root):
#   ./multimodal/scripts/run_qwen35_vl.sh
#
# Environment variables:
#   MODEL_VARIANT: proxy (default), 9b, 397b_a17b
#   TP, EP, PP: parallelism sizes
#   MBS, GBS: micro/global batch sizes
#   NUM_LAYERS, NUM_EXPERTS: override for proxy testing
#   PROFILE: set to 1 to enable Nsight Systems profiling (default: 0)
#   PROFILE_STEP_START/PROFILE_STEP_END: profiled iteration window (default: 4-5)

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export NVTE_FUSED_ATTN=1

DRY_RUN=${DRY_RUN:-false}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NNODES:-1}
PROFILE=${PROFILE:-0}
PROFILE_STEP_START=${PROFILE_STEP_START:-4}
PROFILE_STEP_END=${PROFILE_STEP_END:-5}
PROFILE_RANKS=${PROFILE_RANKS:-0}

MODEL_VARIANT=${MODEL_VARIANT:-proxy}

# Batch sizes
MBS=${MBS:-1}
GBS=${GBS:-64}

# Parallelism
TP=${TP:-1}
EP=${EP:-2}
PP=${PP:-1}

# Variant-aware defaults for NUM_LAYERS and NUM_EXPERTS.
# The model provider builds configs from the variant dict, but Megatron also uses
# these CLI args internally (PP splits, param counting). They must match the variant.
case "$MODEL_VARIANT" in
    proxy)
        NUM_LAYERS=${NUM_LAYERS:-4}
        NUM_EXPERTS=${NUM_EXPERTS:-4}
        ;;
    9b)
        NUM_LAYERS=${NUM_LAYERS:-60}
        NUM_EXPERTS=${NUM_EXPERTS:-0}  # dense (no MoE)
        ;;
    397b_a17b)
        NUM_LAYERS=${NUM_LAYERS:-60}
        NUM_EXPERTS=${NUM_EXPERTS:-512}
        ;;
    *)
        # Unknown variant — fall back to env vars, fail if not set
        : "${NUM_LAYERS:?NUM_LAYERS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        : "${NUM_EXPERTS:?NUM_EXPERTS must be set for MODEL_VARIANT=$MODEL_VARIANT}"
        ;;
esac
SEQ_LEN=${SEQ_LEN:-1024}

WANDB_PROJECT='multimodal-qwen35-vl'
EXP_NAME="qwen35vl_${MODEL_VARIANT}_tp${TP}_ep${EP}_pp${PP}"

ROOT_DIR='./local/'
CHECKPOINT_STORE_PATH="${ROOT_DIR}${EXP_NAME}"
mkdir -p "$CHECKPOINT_STORE_PATH"

TENSORBOARD_LOGS_PATH='./logs'
mkdir -p "$TENSORBOARD_LOGS_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
)

if [ "$NUM_NODES" -gt 1 ]; then
    DISTRIBUTED_ARGS+=(
        --master_addr "${MASTER_ADDR:-localhost}"
        --master_port "${MASTER_PORT:-6000}"
    )
fi

# --- Parallelism ---
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --expert-model-parallel-size "$EP"
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

# --- Training ---
TRAINING_ARGS=(
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --train-iters 100
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 1.2e-4
    --min-lr 1.2e-5
    --lr-decay-style cosine
    --lr-warmup-iters 100
    --lr-decay-iters 2000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --use-mcore-models
    --use-flash-attn
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --enable-experimental
    --manual-gc
    --manual-gc-interval 5
)

PROFILE_ARGS=()
NSYS_CMD=()
if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
        --profile
        --profile-step-start "$PROFILE_STEP_START"
        --profile-step-end "$PROFILE_STEP_END"
        --profile-ranks "$PROFILE_RANKS"
    )

    NSYS_OUTPUT_DIR="${CHECKPOINT_STORE_PATH}/nsys"
    mkdir -p "$NSYS_OUTPUT_DIR"
    NSYS_CMD=(
        nsys profile
        --sample=none
        --cpuctxsw=none
        --trace=cuda,nvtx,cublas,cudnn
        --force-overwrite=true
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        -o "${NSYS_OUTPUT_DIR}/${EXP_NAME}_rank%q{RANK}"
    )
fi

# --- Logging & Checkpointing ---
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 500
    --eval-interval 500
    --save "$CHECKPOINT_STORE_PATH"
    --eval-iters 10
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$EXP_NAME"
    --wandb-save-dir "$CHECKPOINT_STORE_PATH"
    --log-throughput
)

# --- Tokenizer ---
# For mock-data runs: NullTokenizer with the real Qwen3.5 vocab size avoids
# requiring the HF tokenizer weights to be downloaded locally.
# Switch to HuggingFaceTokenizer + tokenizer-model for real-data runs.
TOKENIZER_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 248320
)

# --- Multimodal-specific ---
MULTIMODAL_ARGS=(
    --model-arch qwen35_vl
    --model-variant "$MODEL_VARIANT"
    --dataset-provider mock
    --image-token-id 248056
    --image-size 224
    --total-seq-length "$SEQ_LEN"
    --image-seq-length 256
)

# --- Qwen3-Next Decoder Architecture ---
GPT_MODEL_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size 4096
    --ffn-hidden-size 10240
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --max-position-embeddings 262144
    --seq-length "$SEQ_LEN"
    --normalization RMSNorm
    --apply-layernorm-1p
    --norm-epsilon 1e-06
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-percent 0.25
    --rotary-base 10000000
    --rotary-seq-len-interpolation-factor 1
    --qk-layernorm
    --attention-output-gate
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 64
    --make-vocab-size-divisible-by 485
)

# --- MoE args (MoE variants only) ---
MOE_ARGS=()
case "$MODEL_VARIANT" in
    proxy|397b_a17b)
        # topk: proxy=2, 397b=10 — must match the variant config in qwen35_vl.py
        MOE_TOPK=2
        [ "$MODEL_VARIANT" = "397b_a17b" ] && MOE_TOPK=10
        MOE_ARGS=(
            --num-experts "$NUM_EXPERTS"
            --moe-ffn-hidden-size 1024
            --moe-shared-expert-intermediate-size 1024
            --moe-shared-expert-gate
            --moe-router-load-balancing-type aux_loss
            --moe-router-topk "$MOE_TOPK"
            --moe-grouped-gemm
            --moe-aux-loss-coeff 1e-3
            --moe-token-dispatcher-type alltoall
            --moe-router-dtype fp32
        )
        ;;
    9b)
        # Dense model — no MoE args
        ;;
esac

# --- Recompute ---
RECOMPUTE_ARGS=(
    --recompute-granularity selective
    --recompute-modules moe_act shared_experts layernorm
)

echo "================================================================"
echo "Qwen3.5-VL Multimodal Training (mcore)"
echo "  Variant:       $MODEL_VARIANT"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Num nodes:     $NUM_NODES"
echo "  TP=$TP  EP=$EP  PP=$PP  CP=1"
echo "  MBS=$MBS  GBS=$GBS"
echo "  PROFILE:       $PROFILE"
if [ "$PROFILE" = "1" ]; then
    echo "  Profile steps: ${PROFILE_STEP_START}-${PROFILE_STEP_END}"
    echo "  Profile ranks: $PROFILE_RANKS"
fi
echo "================================================================"

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN ==="
    echo "${NSYS_CMD[@]} torchrun ${DISTRIBUTED_ARGS[@]} multimodal/pretrain_multimodal.py" \
        "${TRAINING_ARGS[@]}" \
        "${PROFILE_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${TOKENIZER_ARGS[@]}" \
        "${MULTIMODAL_ARGS[@]}" \
        "${GPT_MODEL_ARGS[@]}" \
        "${MOE_ARGS[@]}" \
        "${RECOMPUTE_ARGS[@]}"
    echo "=== End of DRY RUN ==="
else
    "${NSYS_CMD[@]}" torchrun "${DISTRIBUTED_ARGS[@]}" multimodal/pretrain_multimodal.py \
        "${TRAINING_ARGS[@]}" \
        "${PROFILE_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${TOKENIZER_ARGS[@]}" \
        "${MULTIMODAL_ARGS[@]}" \
        "${GPT_MODEL_ARGS[@]}" \
        "${MOE_ARGS[@]}" \
        "${RECOMPUTE_ARGS[@]}"
fi
