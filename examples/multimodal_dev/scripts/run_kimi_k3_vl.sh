#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Kimi-K3 VL (MoonViT + Kimi-K3 hybrid decoder) on examples/multimodal_dev.
#
# Default: the single-node H100 proxy -- every Kimi-K3 width, 4 decoder blocks
# (K-KEKE+E) + 1 MTP depth, 32 experts top-8, 4-layer MoonViT -- on mdp_mock
# data, TP1 EP8, MDP on. H100 runs keep the Blackwell-only Kimi-K3 features
# off (MoK, MXFP8, FlashAttention-4, MLA latent CP).
#
#   MDP=0 bash examples/multimodal_dev/scripts/run_kimi_k3_vl.sh        # native path
#   MODEL_VARIANT=full bash ...                                          # 27-layer MoonViT
#   DATASET=cord_v2 bash ...                                             # real data
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MODEL_VARIANT=${MODEL_VARIANT:-proxy}
DATASET=${DATASET:-mdp_mock}
MDP=${MDP:-1}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
MBS=${MBS:-1}
GBS=${GBS:-16}
TRAIN_ITERS=${TRAIN_ITERS:-50}
NUM_EXPERTS=${NUM_EXPERTS:-32}
TOPK=${TOPK:-8}
PATTERN=${PATTERN:-"K-KEKE+E/+E"}

ARGS=(
    --model-arch kimi_k3_vl --model-variant "$MODEL_VARIANT"
    --dataset-provider "$DATASET" --use-packed-sequence --use-vanilla-collate-fn
    --tokenizer-type NullTokenizer --vocab-size 163840 --make-vocab-size-divisible-by 1280
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
    --enable-experimental
    --num-layers "$(echo "${PATTERN%%/*}" | tr -d '|' | wc -c | awk '{print $1-1}')"
    --hybrid-layer-pattern "$PATTERN"
    --experimental-attention-variant kda
    --hidden-size 7168 --ffn-hidden-size 33792 --num-attention-heads 96 --kv-channels 64
    --max-position-embeddings 1048576 --seq-length 8192 --init-method-std 0.02
    --linear-conv-kernel-dim 4 --linear-key-head-dim 128 --linear-value-head-dim 128
    --linear-num-key-heads 96 --linear-num-value-heads 96
    --kda-f-lora-rank 128 --kda-safe-gate --kda-lower-bound -5.0
    --multi-latent-attention --q-lora-rank 1536 --kv-lora-rank 512
    --qk-head-dim 128 --qk-pos-emb-head-dim 64 --v-head-dim 128
    --qk-layernorm --attention-latent-norm-epsilon 1e-6
    --attention-output-gate --gated-attention-proj-granularity elementwise
    --no-rope-freq 1 --position-embedding-type none
    --enable-attention-residuals --attn-res-block-layers 2
    --normalization RMSNorm --norm-epsilon 1e-5
    --situ-glu --situ-glu-beta1 4.0 --situ-glu-beta2 25.0
    --disable-bias-linear --untie-embeddings-and-output-weights
    --num-experts "$NUM_EXPERTS" --moe-ffn-hidden-size 3072
    --moe-latent-size 3584 --moe-latent-up-projection-rmsnorm
    --moe-shared-expert-intermediate-size 6144 --moe-router-topk "$TOPK"
    --moe-router-score-function sigmoid --moe-router-dtype fp32
    --moe-router-topk-scaling-factor 1.0
    --moe-router-load-balancing-type quantile_balancing --moe-aux-loss-coeff 0.0
    --moe-router-quantile-balancing-estimation-scope global_batch --moe-router-qb-num-bins 1000
    --moe-router-force-load-balancing
    --moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-permute-fusion
    --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1
    --attention-dropout 0.0 --hidden-dropout 0.0
    --tensor-model-parallel-size "$TP" --pipeline-model-parallel-size "$PP"
    --expert-model-parallel-size "$EP" --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --micro-batch-size "$MBS" --global-batch-size "$GBS" --train-iters "$TRAIN_ITERS"
    --recompute-granularity selective --recompute-modules moe mlp
    --lr 1e-4 --min-lr 1e-5 --lr-warmup-iters 5 --lr-decay-style cosine
    --lr-decay-iters "$TRAIN_ITERS" --weight-decay 0.1 --clip-grad 1.0
    --bf16 --transformer-impl transformer_engine --calculate-per-token-loss
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native
    --log-interval 1 --log-throughput --eval-iters 0 --eval-interval 100000
)
if [ "$TP" -gt 1 ]; then ARGS+=(--sequence-parallel); fi
if [ "$MDP" = 1 ]; then ARGS+=(--mdp-enable); fi
if [ "$DATASET" = cord_v2 ]; then ARGS+=(--hf-processor-path moonshotai/Kimi-K3 --total-seq-length 8192); fi

cd "$(dirname "$0")/../../.."
torchrun --nproc_per_node "$GPUS_PER_NODE" --nnodes "${NNODES:-1}" \
    examples/multimodal_dev/pretrain_multimodal.py "${ARGS[@]}" "$@"
