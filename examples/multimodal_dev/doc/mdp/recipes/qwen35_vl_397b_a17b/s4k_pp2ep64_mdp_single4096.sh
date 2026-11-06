#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Launcher-ready command from job 717519; measured code: 6a975c1aab457ae88b16247582674c3dc81a1300.
# Run INSIDE an allocated GPU container. This script does not allocate nodes.
# Pass a launcher prefix, e.g. python under a per-rank srun, or torchrun with all topology arguments.
set -euo pipefail
if [[ $# -eq 0 ]]; then
    printf 'Usage: bash %s <launcher> [launcher arguments...]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi
REPRO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MCORE_ROOT="$(cd -- "$REPRO_DIR/../../../../../.." && pwd)"
MCORE_BENCH_OUTPUT="${MCORE_BENCH_OUTPUT:-$MCORE_ROOT/runtime/mdp_rebench/s4k_pp2ep64_mdp_single4096}"
export MCORE_ROOT MCORE_BENCH_OUTPUT
export PYTHONPATH="$MCORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDNN_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn
export FLA_DISABLE_TENSOR_CACHE=1
export HF_HUB_OFFLINE=1
export MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS=1
export NCCL_GRAPH_REGISTER=0
export NCCL_NVLS_ENABLE=0
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API=128
export NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API=128
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
cd "$MCORE_ROOT"
exec "$@" "$MCORE_ROOT/examples/multimodal_dev/pretrain_multimodal.py" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3.5-397B-A17B \
    --num-layers 60 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --kv-channels 256 \
    --max-position-embeddings 262144 \
    --group-query-attention \
    --num-query-groups 2 \
    --qk-layernorm \
    --attention-output-gate \
    --normalization RMSNorm \
    --apply-layernorm-1p \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 0.25 \
    --rotary-base 10000000 \
    --make-vocab-size-divisible-by 1940 \
    --experimental-attention-variant gated_delta_net \
    --linear-attention-freq 4 \
    --linear-conv-kernel-dim 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 64 \
    --num-experts 512 \
    --moe-ffn-hidden-size 1024 \
    --moe-shared-expert-intermediate-size 1024 \
    --moe-shared-expert-gate \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 10 \
    --moe-aux-loss-coeff 1e-3 \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-weight-decay-cond-type apply_wd_to_qk_layernorm \
    --dataset-provider mdp_mock \
    --use-vanilla-collate-fn \
    --use-packed-sequence \
    --moe-router-force-load-balancing \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 64 \
    --pipeline-model-parallel-layout 'Et*16|t*16|t*16|t*12,m,L' \
    --context-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --moe-token-dispatcher-type flex \
    --moe-flex-dispatcher-backend hybridep \
    --moe-flex-dispatcher-num-sms 32 \
    --moe-grouped-gemm \
    --moe-permute-fusion \
    --moe-router-fusion \
    --moe-router-dtype fp32 \
    --cuda-graph-impl transformer_engine \
    --cuda-graph-modules attn moe_router moe_preprocess \
    --te-rng-tracker \
    --cuda-graph-warmup-steps 2 \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --gdn-pre-gated-delta-rule-fusion \
    --gdn-kernel-backend cudnn \
    --micro-batch-size 1 \
    --global-batch-size 2048 \
    --seq-length 4096 \
    --train-samples 585937500 \
    --exit-duration-in-mins 25 \
    --exit-interval 20 \
    --no-save-optim \
    --no-check-for-nan-in-loss-and-grad \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl te \
    --no-create-attention-mask-in-dataloader \
    --manual-gc \
    --manual-gc-interval 5 \
    --rerun-mode disabled \
    --num-workers 6 \
    --calculate-per-token-loss \
    --pad-packed-seq-alignment max \
    --max-seqlen-per-dp-cp-rank 4096 \
    --thd-max-packed-sequences 8 \
    --thd-tail-padding-policy append_dummy_seq \
    --use-precision-aware-optimizer \
    --main-grads-dtype fp32 \
    --main-params-dtype fp32 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --lr 0.00012 \
    --min-lr 1.2e-05 \
    --lr-decay-style cosine \
    --lr-decay-samples 584765624 \
    --lr-warmup-iters 0 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --fp8-recipe mxfp8 \
    --fp8-format e4m3 \
    --fp8-param-gather \
    --reuse-grad-buf-for-mxfp8-param-ag \
    --moe-router-padding-for-quantization \
    --init-method-std 0.02 \
    --seed 1234 \
    --eval-iters 32 \
    --eval-interval 1000 \
    --log-throughput \
    --log-interval 1 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-memory-interval 5 \
    --log-device-memory-used \
    --logging-level 40 \
    --tensorboard-dir "${MCORE_BENCH_OUTPUT}/tensorboard" \
    --enable-experimental \
    --model-arch qwen35_vl \
    --model-variant 397b_a17b \
    --image-token-id 248056 \
    --vision-num-layers 27 \
    --hf-processor-path Qwen/Qwen3.5-397B-A17B \
    --sft \
    --total-seq-length 4096 \
    --thd-static-packing \
    --mdp-greedy-packing \
    --mdp-mock-dataset-config-json "${MCORE_ROOT}/examples/multimodal_dev/doc/mdp/recipes/data/mock_fixed_4096.json" \
    --mdp-enable \
    --mdp-overlap-window-capture \
    --encoder-recompute-granularity whole \
    --mdp-encoder-max-payload-rows 65536
