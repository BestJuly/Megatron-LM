#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Launcher-ready command from job 711351; measured code: 4476895424269ad168e181cfd69beaec0701c71a.
# Run INSIDE an allocated GPU container. This script does not allocate nodes.
# Pass a launcher prefix, e.g. python under a per-rank srun, or torchrun with all topology arguments.
set -euo pipefail
if [[ $# -eq 0 ]]; then
    printf 'Usage: bash %s <launcher> [launcher arguments...]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi
REPRO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MCORE_ROOT="$(cd -- "$REPRO_DIR/../../../../../.." && pwd)"
MCORE_BENCH_OUTPUT="${MCORE_BENCH_OUTPUT:-$MCORE_ROOT/runtime/mdp_rebench/s16k_pp2ep4_mdp_rebench_cudnn_gdn}"
export MCORE_ROOT MCORE_BENCH_OUTPUT
export PYTHONPATH="$MCORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDNN_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn
export FLA_DISABLE_TENSOR_CACHE=1
export MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS=1
export NCCL_GRAPH_REGISTER=0
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API=128
export NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API=128
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd "$MCORE_ROOT"
exec "$@" "$MCORE_ROOT/examples/multimodal_dev/pretrain_multimodal.py" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --model-arch qwen35_vl \
    --model-variant 35b_a3b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --num-layers 40 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --seq-length 16384 \
    --qk-layernorm \
    --attention-output-gate \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --make-vocab-size-divisible-by 485 \
    --num-experts 256 \
    --moe-ffn-hidden-size 512 \
    --moe-shared-expert-intermediate-size 512 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type flex \
    --moe-router-dtype fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --use-packed-sequence \
    --moe-router-force-load-balancing \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --hf-processor-path Qwen/Qwen3.5-35B-A3B \
    --sft \
    --enable-experimental \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --pipeline-model-parallel-layout 'Et*23|t*17mL' \
    --expert-model-parallel-size 4 \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --cp-comm-type a2a \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --total-seq-length 16384 \
    --train-iters 20 \
    --bf16 \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --mdp-greedy-packing \
    --thd-static-packing \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 16384 \
    --thd-max-packed-sequences 32 \
    --mdp-mock-dataset-config-json "${MCORE_ROOT}/examples/multimodal_dev/doc/mdp/recipes/data/mock_lognormal.json" \
    --cuda-graph-impl transformer_engine \
    --cuda-graph-warmup-steps 2 \
    --cuda-graph-modules attn moe_router moe_preprocess \
    --mdp-enable \
    --mdp-overlap-window-capture \
    --encoder-recompute-granularity whole \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-permute-fusion \
    --moe-router-fusion \
    --gdn-pre-gated-delta-rule-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --calculate-per-token-loss \
    --optimizer adam \
    --use-distributed-optimizer \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --manual-gc \
    --manual-gc-interval 10 \
    --eval-iters 1 \
    --eval-interval 100000 \
    --log-throughput \
    --log-interval 1 \
    --log-device-memory-used \
    --log-memory-interval 1 \
    --distributed-timeout-minutes 20 \
    --gdn-kernel-backend cudnn
