#!/bin/bash
# Parity test: compare Kimi K2.5 VL training between Bridge and MIMO.
#
# Runs both frameworks for a few iterations with matching architecture,
# same seed, and mock data, then compares loss values.
#
# Usage:
#   bash examples/mimo/scripts/parity_test_kimi_k25.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMO_REPO="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_mcore/users/jinliangl/repos/Megatron-LM"
BRIDGE_REPO="/lustre/fs1/portfolios/coreai/users/jinliangl/repos/Megatron-Bridge"

GPUS=8
TP=4
EP=2
SEQ=2048
MBS=1
GBS=8
ITERS=5
SEED=1234

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export NVTE_FUSED_ATTN=1
export WANDB_MODE=disabled

BRIDGE_LOG="/tmp/kimi_k25_bridge_loss.log"
MIMO_LOG="/tmp/kimi_k25_mimo_loss.log"

echo "================================================================"
echo "  Kimi K2.5 VL Parity Test: Bridge vs MIMO"
echo "  GPUs=$GPUS  TP=$TP  EP=$EP  SEQ=$SEQ  MBS=$MBS  GBS=$GBS"
echo "  Iterations=$ITERS  Seed=$SEED"
echo "================================================================"

# ---------------------------------------------------------------
# Step 1: Run Bridge
# ---------------------------------------------------------------
echo ""
echo ">>> [1/2] Running Bridge training..."
echo ""

# Install Emerging-Optimizers (for Muon) if needed
if [ -d "$HOME2/repos/Emerging-Optimizers" ]; then
    (cd "$HOME2/repos/Emerging-Optimizers" && pip install -q . 2>/dev/null) || true
fi

export PYTHONPATH="${BRIDGE_REPO}/src:${BRIDGE_REPO}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

cd "${BRIDGE_REPO}/examples"

torchrun --nproc_per_node=$GPUS scripts/training/run_recipe.py \
    --recipe kimi_k25_vl_sft_config \
    --step_func vlm_step \
    --hf_path moonshotai/Kimi-K2.5 \
    model.seq_length=$SEQ \
    model.tensor_model_parallel_size=$TP \
    model.sequence_parallel=true \
    model.pipeline_model_parallel_size=1 \
    model.expert_model_parallel_size=$EP \
    model.pipeline_model_parallel_layout=null \
    model.hf_model_path=moonshotai/Kimi-K2.5 \
    model.freeze_vision_model=true \
    model.freeze_vision_projection=true \
    model.calculate_per_token_loss=true \
    model.cross_entropy_loss_fusion=false \
    train.train_iters=$ITERS \
    train.micro_batch_size=$MBS \
    train.global_batch_size=$GBS \
    dataset.maker_name=make_cord_v2_dataset \
    dataset.pack_sequences_in_batch=false \
    dataset.seq_length=$SEQ \
    checkpoint.save=null \
    checkpoint.load=null \
    checkpoint.pretrained_checkpoint=null \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.wandb_project=null \
    ddp.average_in_collective=false \
    model.hidden_size=7168 \
    model.ffn_hidden_size=1024 \
    model.num_moe_experts=16 \
    model.moe_ffn_hidden_size=64 \
    rng.seed=$SEED \
    2>&1 | tee "$BRIDGE_LOG"

echo ""
echo ">>> Bridge done. Extracting loss values..."
grep "lm loss" "$BRIDGE_LOG" | head -$ITERS

# ---------------------------------------------------------------
# Step 2: Run MIMO
# ---------------------------------------------------------------
echo ""
echo ">>> [2/2] Running MIMO training..."
echo ""

# Reset PYTHONPATH for MIMO
export PYTHONPATH="${MIMO_REPO}:${PYTHONPATH:-}"

cd "$MIMO_REPO"

torchrun --nproc_per_node=$GPUS examples/mimo/train.py \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-iters $ITERS \
    --seed $SEED \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --lr 1.2e-4 \
    --min-lr 1.2e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --lr-decay-iters $ITERS \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --model-provider kimi_k25_vlm \
    --model-variant proxy \
    --bf16 \
    --use-mcore-models \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --enable-experimental \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size $EP \
    --context-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --log-interval 1 \
    --eval-interval 10000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
    --tokenizer-type NullTokenizer \
    --vocab-size 163840 \
    --num-layers 4 \
    --hidden-size 7168 \
    --ffn-hidden-size 1024 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 64 \
    --max-position-embeddings 262144 \
    --seq-length $SEQ \
    --normalization RMSNorm \
    --norm-epsilon 1e-05 \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --qk-layernorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --num-experts 16 \
    --moe-ffn-hidden-size 64 \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-dtype fp32 \
    --make-vocab-size-divisible-by 1280 \
    --dataset-provider kimi_k25_vlm \
    --image-token-id 163605 \
    --total-seq-length $SEQ \
    --log-throughput \
    2>&1 | tee "$MIMO_LOG"

echo ""
echo ">>> MIMO done. Extracting loss values..."
grep "lm loss" "$MIMO_LOG" | head -$ITERS

# ---------------------------------------------------------------
# Step 3: Compare
# ---------------------------------------------------------------
echo ""
echo "================================================================"
echo "  COMPARISON"
echo "================================================================"
echo ""
echo "Bridge losses:"
grep -oP "lm loss: \S+" "$BRIDGE_LOG" | head -$ITERS
echo ""
echo "MIMO losses:"
grep -oP "lm loss: \S+" "$MIMO_LOG" | head -$ITERS
echo ""
echo "Done. Log files:"
echo "  Bridge: $BRIDGE_LOG"
echo "  MIMO:   $MIMO_LOG"
