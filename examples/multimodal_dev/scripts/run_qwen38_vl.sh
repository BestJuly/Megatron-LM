#!/bin/bash
# Qwen3.8-Flash-Next VL (HF Qwen4ExpForConditionalGeneration) through multimodal_dev, one node.
#
# The decoder is the Qwen3.8-Flash-Next proxy of examples/qwen3.8_flash_next/train_proxy_1node.sh
# (GDN + QSA + MoE on every layer, gated residual over 4 streams, Engram/PLE on layer 3, MTP),
# switched to 3-D mRoPE; the vision tower is the Qwen3.5-VL ViT. Runs on mdp_mock data with
# NullTokenizer at the real vocabulary, so nothing has to be downloaded.
#
# Every dimension is an environment variable:
#   MDP=0|1            MDP (default 0 = in-model encoder). MDP needs the packed (THD) layout.
#   PACKED=1|0         --use-packed-sequence + mdp_mock (default 1); 0 = padded BSHD on the
#                      `mock` provider (non-MDP only)
#   GREEDY=0|1         --mdp-greedy-packing (MDP only); token budget per microbatch per rank
#                      MAX_SEQLEN (default SEQ_LEN x MBS)
#   PP EP              parallel sizes (defaults 1, 4); TP=CP=1 (MDP limits)
#   MBS GBS SEQ_LEN ITERS SEED NPROC
#   VARIANT=proxy|full vision depth default (4 / 27) and decoder depth (8 / 48 Qwen layers)
#   NUM_QWEN_LAYERS    decoder depth in Qwen layers, cadence GGGQ (default per variant)
#   NUM_EXPERTS MOE_TOPK  (defaults 32 / 10 for proxy, 512 / 10 for full)
#   VISION_NUM_LAYERS  ViT depth override
#   MTP_NUM_LAYERS     0 or 1 (default 1)
#   ENGRAM_BASE        n-gram table base per order (proxy 1000000, full 20000000); 0 = no Engram
#   QSA_LOSS_COEFF     indexer KL coefficient (default 0.01)
#   VERIFY_ENGRAM=0|1  --engram-verify-training (non-MDP only: it is incompatible with the
#                      distributed optimizer, which MDP requires)
#   ENTRY=<path>       entry script (default pretrain_multimodal.py)
#   EXTRA="..."        extra args appended verbatim
#
# Examples (inside the training container, on the compute node):
#   MDP=0 ITERS=10 bash examples/multimodal_dev/scripts/run_qwen38_vl.sh
#   MDP=1 ITERS=10 bash examples/multimodal_dev/scripts/run_qwen38_vl.sh
#   MDP=1 PP=2 EP=2 ITERS=10 bash examples/multimodal_dev/scripts/run_qwen38_vl.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NVTE_FUSED_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HF_HUB_OFFLINE=1

MDP=${MDP:-0}
PACKED=${PACKED:-1}
GREEDY=${GREEDY:-0}
PP=${PP:-1}
EP=${EP:-4}
MBS=${MBS:-2}
GBS=${GBS:-16}
SEQ_LEN=${SEQ_LEN:-4096}
ITERS=${ITERS:-10}
SEED=${SEED:-1234}
NPROC=${NPROC:-4}
VARIANT=${VARIANT:-proxy}
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-1}
QSA_LOSS_COEFF=${QSA_LOSS_COEFF:-0.01}
VERIFY_ENGRAM=${VERIFY_ENGRAM:-0}
MASTER_PORT=${MASTER_PORT:-29500}
ENTRY=${ENTRY:-$REPO_ROOT/examples/multimodal_dev/pretrain_multimodal.py}
EXTRA=${EXTRA:-}

case "$VARIANT" in
    proxy)
        NUM_QWEN_LAYERS=${NUM_QWEN_LAYERS:-8}
        NUM_EXPERTS=${NUM_EXPERTS:-32}
        ENGRAM_BASE=${ENGRAM_BASE:-1000000}
        ;;
    full)
        NUM_QWEN_LAYERS=${NUM_QWEN_LAYERS:-48}
        NUM_EXPERTS=${NUM_EXPERTS:-512}
        ENGRAM_BASE=${ENGRAM_BASE:-20000000}
        ;;
    *) echo "ERROR: VARIANT must be proxy|full, got '$VARIANT'" >&2; exit 1 ;;
esac
MOE_TOPK=${MOE_TOPK:-10}

if [ "$MDP" = "1" ] && [ "$PACKED" != "1" ]; then
    echo "ERROR: MDP=1 needs PACKED=1 (MDP packs the decoder batch into THD)" >&2; exit 1
fi
if [ "$MDP" = "1" ] && [ "$VERIFY_ENGRAM" = "1" ]; then
    echo "ERROR: VERIFY_ENGRAM=1 needs MDP=0 (it excludes the distributed optimizer)" >&2; exit 1
fi

# Qwen layer i is one attention symbol (G = GDN, Q = QSA; cadence GGGQ) plus one MoE symbol.
# The single MTP depth is a QSA + MoE layer, as in the released checkpoint.
PATTERN=""
for ((i = 0; i < NUM_QWEN_LAYERS; i++)); do
    if [ $(((i + 1) % 4)) -eq 0 ]; then PATTERN+="QE"; else PATTERN+="GE"; fi
done
MTP_ARGS=()
if [ "$MTP_NUM_LAYERS" -gt 0 ]; then
    PATTERN+="/QE"
    MTP_ARGS=( --mtp-num-layers "$MTP_NUM_LAYERS" --mtp-loss-scaling-factor 0.1 )
fi

DATA_ARGS=()
if [ "$PACKED" = "1" ]; then
    DATA_ARGS=( --dataset-provider mdp_mock --use-packed-sequence --qsa-use-sparse-attention )
else
    DATA_ARGS=( --dataset-provider mock --total-seq-length "$SEQ_LEN" )
fi

MDP_ARGS=()
if [ "$MDP" = "1" ]; then
    MDP_ARGS=( --mdp-enable --no-ckpt-fully-parallel-save )
    if [ "$GREEDY" = "1" ]; then
        MDP_ARGS+=( --mdp-greedy-packing --max-seqlen-per-dp-cp-rank "${MAX_SEQLEN:-$((SEQ_LEN * MBS))}" )
    fi
fi

ENGRAM_ARGS=()
if [ "$ENGRAM_BASE" -gt 0 ]; then
    ENGRAM_ARGS=( --engram-variant qwen --engram-layer-ids 3
                  --engram-vocab-sizes "$ENGRAM_BASE" "$ENGRAM_BASE"
                  --engram-max-ngram-order 3 --engram-num-hash-heads 8 --engram-memory-dim 1280
                  --engram-kernel-size 4 --engram-unigram-vocab-size 248320
                  --engram-hash-seed 1234 --engram-eos-token-id 248044 )
fi

OPT_ARGS=( --use-distributed-optimizer )
if [ "$VERIFY_ENGRAM" = "1" ]; then
    OPT_ARGS=( --engram-verify-training )
fi

export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}
cd "$REPO_ROOT"   # DeepEP JIT trap: never start from a directory with a specs/ sibling

torchrun --nnodes 1 --nproc_per_node "$NPROC" --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
    "$ENTRY" \
    --model-arch qwen38_vl \
    --model-variant "$VARIANT" \
    ${VISION_NUM_LAYERS:+--vision-num-layers "$VISION_NUM_LAYERS"} \
    --use-vanilla-collate-fn \
    "${DATA_ARGS[@]}" \
    --tokenizer-type NullTokenizer --vocab-size 248320 --make-vocab-size-divisible-by 1940 \
    --hybrid-layer-pattern "$PATTERN" \
    --spec megatron.core.models.hybrid.hybrid_layer_specs gated_residual_hybrid_stack_spec \
    --hidden-size 2560 --num-attention-heads 24 --kv-channels 256 --max-position-embeddings 262144 \
    --group-query-attention --num-query-groups 2 --qk-layernorm --attention-output-gate \
    --position-embedding-type mrope --mrope-section 11 11 10 \
    --rotary-percent 0.25 --rotary-base 10000000 \
    --qsa-indexer-n-heads 4 --qsa-indexer-head-dim 128 --qsa-indexer-budget 2048 \
    --qsa-indexer-compress-ratio 4 --qsa-indexer-loss-coeff "$QSA_LOSS_COEFF" \
    --linear-conv-kernel-dim 4 --linear-key-head-dim 128 --linear-value-head-dim 128 \
    --linear-num-key-heads 16 --linear-num-value-heads 48 --gdn-output-gate-activation sigmoid \
    --enable-hyper-connections --hyper-connection-variant gated_residual \
    --num-residual-streams 4 --hc-lowrank 320 \
    --num-experts "$NUM_EXPERTS" --moe-ffn-hidden-size 640 --moe-shared-expert-intermediate-size 640 \
    --moe-shared-expert-gate --moe-router-topk "$MOE_TOPK" --moe-router-score-function softmax \
    --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3 --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-permute-fusion \
    --moe-router-force-load-balancing \
    "${ENGRAM_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    --normalization RMSNorm --apply-layernorm-1p --norm-epsilon 1e-6 --swiglu \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --no-weight-decay-cond-type apply_wd_to_qk_layernorm \
    --seq-length "$SEQ_LEN" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size "$PP" \
    --expert-model-parallel-size "$EP" --context-parallel-size 1 --expert-tensor-parallel-size 1 \
    "${OPT_ARGS[@]}" \
    --micro-batch-size "$MBS" --global-batch-size "$GBS" \
    --bf16 --transformer-impl transformer_engine --use-mcore-models --enable-experimental \
    --use-flash-attn --calculate-per-token-loss \
    --cross-entropy-loss-fusion --cross-entropy-fusion-impl native \
    --train-iters "$ITERS" --lr 1e-4 --min-lr 1e-5 --lr-decay-style constant --lr-warmup-iters 0 \
    --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
    --init-method-std 0.02 --seed "$SEED" \
    --log-interval 1 --log-throughput --eval-iters 2 --eval-interval 100000 \
    --distributed-timeout-minutes 10 \
    "${MDP_ARGS[@]}" \
    $EXTRA
