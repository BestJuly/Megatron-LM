#!/bin/bash

# =============================================================================
# Qwen3-30B-A3B (text-only MoE) GB200 launcher
# =============================================================================
#
# Single-run launcher for the text-only Qwen3-30B-A3B model.
#
# Key differences vs dev_gb200.sh:
#   * --model-arch qwen3 (text-only registry entry)
#   * No vision tower, no MRoPE, no MTP by default
#   * 1D RoPE rotary_percent=1.0 (vs 0.25 for VL)
#   * Architecture: 48 layers, hidden=2048, ffn=6144, attn_heads=32,
#     num_query_groups=4 (GQA factor 8), kv_channels=128 (head_dim=128 per
#     official Qwen3-30B-A3B HF config — non-standard: total Q dim = 32×128 =
#     4096, projected to hidden=2048; do NOT compute as hidden/heads).
#   * MoE: 128 experts × top-8, moe_ffn=768, no shared expert
#   * vocab=151_936
#   * GBS=512 default (vs 256 on the qwen35_vl 32-GPU baseline)
#   * Always --text-only (the model has no vision tower)
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_LM_PATH="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVLINK_DOMAIN_SIZE=72
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
: ${NCCL_DEBUG:=VERSION}
export NCCL_DEBUG
: ${NCCL_GRAPH_REGISTER:=0}
export NCCL_GRAPH_REGISTER


GPUS_PER_NODE=4
NUM_NODES=16
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_NODES))

TRAIN_ITERS=45
WARMUP_ITERS=5
ACTUAL_ITERS=$((TRAIN_ITERS + WARMUP_ITERS))

RESULTS_DIR="${MEGATRON_LM_PATH}/benchmark_results/qwen3_gb200"
DRY_RUN=0
PROF_LEVEL=0
PROFILE_RANKS="0 1"
PROFILE_STEP_START=""
PROFILE_STEP_END=""
WANDB_PROJECT=""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_DEV_BENCH_NAME="qwen3_dev"

usage() {
    cat <<'EOF'
Qwen3-30B-A3B (text-only MoE) GB200 launcher.

Usage:
  dev_qwen3_gb200.sh [options] [bench_name] [key=value ...]

  bench_name     Run label (tensorboard/wandb/log dir). If omitted, uses "qwen3_dev".
  key=value      Passed to run_benchmark (tp, ep, pp, cp, etp, mbs, gbs,
                 seq_len, dispatcher_backend, a2a_overlap, use_fsdp,
                 cuda_graph_scope, recompute, mtp, fp8, ...).

Global options:
  -h, --help                Show this help and exit
  --dry-run                 Set DRY_RUN=1 (print command, do not execute)
  --gpus N                  GPUS_PER_NODE
  --nnodes N                NNODES
  --train-iters N           TRAIN_ITERS
  --warmup-iters N          WARMUP_ITERS
  --results-dir DIR         RESULTS_DIR
  --prof-level N            PROF_LEVEL (0-3)
  --profile-ranks S         PROFILE_RANKS (e.g. "0" or "0 1")
  --profile-step-start N    PROFILE_STEP_START
  --profile-step-end N      PROFILE_STEP_END
  --wandb-project P         WANDB_PROJECT

Examples:
  dev_qwen3_gb200.sh exp040
  dev_qwen3_gb200.sh exp041 cp=2
  dev_qwen3_gb200.sh longseq cp=2 seq_len=16384
  dev_qwen3_gb200.sh --dry-run --gpus 4 --nnodes 1
EOF
}

_cli_bench_name=""
_cli_bench_overrides=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --nnodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --train-iters)
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --warmup-iters)
            WARMUP_ITERS="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --prof-level)
            PROF_LEVEL="$2"
            shift 2
            ;;
        --profile-ranks)
            PROFILE_RANKS="$2"
            shift 2
            ;;
        --profile-step-start)
            PROFILE_STEP_START="$2"
            shift 2
            ;;
        --profile-step-end)
            PROFILE_STEP_END="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        *)
            if [[ "$1" == *"="* ]]; then
                _cli_bench_overrides+=("$1")
            elif [[ -z "$_cli_bench_name" ]]; then
                _cli_bench_name="$1"
            else
                echo "ERROR: unexpected argument '$1' (expected key=value after bench name)" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

TOTAL_GPUS=$((GPUS_PER_NODE * NUM_NODES))
ACTUAL_ITERS=$((TRAIN_ITERS + WARMUP_ITERS))

PROFILE_STEP_START=${PROFILE_STEP_START:-$((WARMUP_ITERS + 15))}
PROFILE_STEP_END=${PROFILE_STEP_END:-$((WARMUP_ITERS + 17))}

_dev_merged_overrides=(${_cli_bench_overrides[@]+"${_cli_bench_overrides[@]}"})

DEV_BENCH_NAME="${_cli_bench_name:-$DEFAULT_DEV_BENCH_NAME}"

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Helper: launch a single benchmark run
# ---------------------------------------------------------------------------
run_benchmark() {
    local name="$1"
    shift

    # --- Defaults (Qwen3-30B-A3B, 64 GB200) ---
    local tp=1 ep=8 pp=1 cp=1 etp=1 vpp=1
    local mbs=1 gbs=512
    local seq_len=4096
    local dispatcher="alltoall"
    local dispatcher_backend="hybridep"
    local mtp=0
    local a2a_overlap=0
    local use_fsdp=0
    local recompute=0
    local recompute_modules="moe_act shared_experts layernorm"
    local recompute_full=0
    local cuda_graph_scope=""
    local moe_grouped_gemm=1
    local cross_entropy_impl="te"
    local calculate_per_token_loss=0
    local fp8=""
    local fp8_recipe=""
    local extra_args=""
    local num_nodes_override=""
    local gpus_override=""
    local prof_level="$PROF_LEVEL"
    local profile_step_start="${PROFILE_STEP_START}"
    local profile_step_end="${PROFILE_STEP_END}"

    # Parse overrides
    for kv in "$@"; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        case "$key" in
            tp)                  tp="$val" ;;
            ep)                  ep="$val" ;;
            pp)                  pp="$val" ;;
            vpp)                 vpp="$val" ;;
            cp)                  cp="$val" ;;
            etp)                 etp="$val" ;;
            mbs)                 mbs="$val" ;;
            gbs)                 gbs="$val" ;;
            seq_len)             seq_len="$val" ;;
            dispatcher)          dispatcher="$val" ;;
            dispatcher_backend)  dispatcher_backend="$val" ;;
            a2a_overlap)         a2a_overlap="$val" ;;
            mtp)                 mtp="$val" ;;
            use_fsdp)            use_fsdp="$val" ;;
            recompute)           recompute="$val" ;;
            recompute_modules)   recompute_modules="$val" ;;
            recompute_full)      recompute_full="$val" ;;
            cuda_graph_scope)    cuda_graph_scope="$val" ;;
            moe_grouped_gemm)    moe_grouped_gemm="$val" ;;
            cross_entropy_impl)  cross_entropy_impl="$val" ;;
            calculate_per_token_loss) calculate_per_token_loss="$val" ;;
            fp8)                 fp8="$val" ;;
            fp8_recipe)          fp8_recipe="$val" ;;
            extra_args)          extra_args="$val" ;;
            num_nodes)           num_nodes_override="$val" ;;
            gpus)                gpus_override="$val" ;;
            prof_level)          prof_level="$val" ;;
            profile_step_start)  profile_step_start="$val" ;;
            profile_step_end)    profile_step_end="$val" ;;
            *) echo "ERROR: unknown override key '$key'" >&2; return 2 ;;
        esac
    done

    local run_nodes="${num_nodes_override:-$NUM_NODES}"
    local run_gpus_per_node="${gpus_override:-$GPUS_PER_NODE}"
    local run_total_gpus=$((run_gpus_per_node * run_nodes))

    local exp_name="${name}"
    local run_dir="${RESULTS_DIR}/${exp_name}"
    mkdir -p "$run_dir"

    local precision_label="BF16"
    if [ -n "$fp8" ]; then
        precision_label="FP8(${fp8_recipe:-blockwise})"
    fi

    local prof_label="L${prof_level}"
    case "$prof_level" in
        0) prof_label="L0:throughput" ;;
        1) prof_label="L1:fine-timers" ;;
        2) prof_label="L2:nsys+nvtx" ;;
        3) prof_label="L3:full-diag" ;;
    esac

    echo ""
    echo "================================================================"
    echo " BENCHMARK: $name  [GB200 192GB, text-only]"
    echo "   Model:      Qwen3-30B-A3B (text-only, 48L / 128 experts × top-8)"
    echo "   GPUs:       ${run_total_gpus} (${run_nodes}N × ${run_gpus_per_node}G)"
    echo "   Parallel:   TP=${tp} EP=${ep} PP=${pp} CP=${cp} ETP=${etp}"
    echo "   Batch:      MBS=${mbs} GBS=${gbs}"
    echo "   Seq:        ${seq_len}"
    echo "   Precision:  ${precision_label}"
    echo "   Dispatcher: ${dispatcher} ${dispatcher_backend}"
    echo "   A2A overlap: ${a2a_overlap}"
    echo "   CUDA Graph: ${cuda_graph_scope:-none}"
    echo "   Recompute:  ${recompute} (modules=${recompute_modules:-full-uniform})"
    echo "   MTP:        ${mtp}"
    echo "   Profiling:  ${prof_label}"
    if [ "$prof_level" -ge 2 ]; then
        echo "   Nsys steps: --profile-step-start ${profile_step_start} --profile-step-end ${profile_step_end}"
    fi
    echo "================================================================"

    # --- Distributed args ---
    local dist_args=(
        --nproc_per_node "$run_gpus_per_node"
        --nnodes "$run_nodes"
    )
    if [ "$run_nodes" -gt 1 ]; then
        dist_args+=(
            --master_addr "${MASTER_ADDR:-localhost}"
            --master_port "${MASTER_PORT:-6000}"
            --node_rank ${SLURM_NODEID:-0}
        )
    fi

    # --- Parallelism ---
    local parallel_args=(
        --tensor-model-parallel-size "$tp"
        --pipeline-model-parallel-size "$pp"
        --expert-model-parallel-size "$ep"
        --context-parallel-size "$cp"
        --expert-tensor-parallel-size "$etp"
        --use-distributed-optimizer
        --distributed-timeout-minutes 60
        --sequence-parallel
        --overlap-grad-reduce
        --overlap-param-gather
        --no-create-attention-mask-in-dataloader
    )
    if [ "$pp" -gt 1 ] && [ -n "$vpp" ] && [ "$vpp" != "0" ]; then
        parallel_args+=( --num-virtual-stages-per-pipeline-rank "$vpp" )
    fi

    # --- Training ---
    local training_args=(
        --micro-batch-size "$mbs"
        --global-batch-size "$gbs"
        --train-iters "$ACTUAL_ITERS"
        --adam-beta1 0.9
        --adam-beta2 0.95
        --lr 1.2e-4
        --min-lr 1.2e-5
        --lr-decay-style cosine
        --lr-warmup-iters "$WARMUP_ITERS"
        --lr-decay-iters 2000
        --weight-decay 0.1
        --clip-grad 1.0
        --bf16
        --use-mcore-models
        --use-flash-attn
        --transformer-impl transformer_engine
        --cross-entropy-loss-fusion
        --cross-entropy-fusion-impl "$cross_entropy_impl"
        --enable-experimental
        --manual-gc
        --manual-gc-interval 5
        --use-precision-aware-optimizer
        --main-grads-dtype fp32
        --main-params-dtype fp32
        --exp-avg-dtype bf16
        --exp-avg-sq-dtype bf16
    )

    local mtp_args=()
    if [ "$mtp" -eq 1 ]; then
        mtp_args=(
            --mtp-num-layers 1
            --mtp-loss-scaling-factor 0.1
        )
    fi

    # --- FP8 ---
    local fp8_args=()
    if [ -n "$fp8" ]; then
        fp8_args=(
            --fp8-format "$fp8"
            --fp8-recipe "${fp8_recipe:-blockwise}"
            --fp8-amax-history-len 1
            --fp8-amax-compute-algo most_recent
        )
    fi

    # --- MoE (Qwen3-30B-A3B: 128 experts × top-8, no shared expert) ---
    local moe_args=(
        --num-experts 128
        --moe-ffn-hidden-size 768
        --moe-router-load-balancing-type aux_loss
        --moe-router-topk 8
        --moe-aux-loss-coeff 1e-3
        --moe-token-dispatcher-type "$dispatcher"
        --moe-router-dtype fp32
        --moe-router-force-load-balancing
        --moe-router-fusion
        --moe-permute-fusion
    )

    if [ "$moe_grouped_gemm" -eq 1 ]; then
        moe_args+=( --moe-grouped-gemm )
    fi
    if [[ ${dispatcher_backend} == "alltoall" ]]; then
        moe_args+=( --moe-token-dispatcher-type alltoall )
    elif [[ ${dispatcher_backend} == "deepep" ]]; then
        moe_args+=( --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep --moe-deepep-num-sms 32 )
    elif [[ ${dispatcher_backend} == "hybridep" ]]; then
        moe_args+=( --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --moe-hybridep-num-sms 32 )
    fi

    # --- 1F1B A2A overlap ---
    local a2a_args=()
    if [ "$a2a_overlap" -eq 1 ]; then
        export NVTE_FWD_LAYERNORM_SM_MARGIN=24
        export NVTE_BWD_LAYERNORM_SM_MARGIN=24
        a2a_args+=( --delay-wgrad-compute --overlap-moe-expert-parallel-comm )
    fi

    # --- CUDA Graphs ---
    if [ -n "$cuda_graph_scope" ]; then
        moe_args+=( --cuda-graph-scope $cuda_graph_scope --cuda-graph-impl transformer_engine --te-rng-tracker )
    fi

    # --- Model architecture (Qwen3-30B-A3B decoder) ---
    # NOTE: kv_channels=128 is the published Qwen3 head_dim — non-standard
    # (hidden=2048, num_heads=32, but Q dim = 32*128=4096 projected back to
    # hidden). Do NOT compute as hidden_size/num_heads.
    local gpt_args=(
        --num-layers 48
        --hidden-size 2048
        --ffn-hidden-size 6144
        --num-attention-heads 32
        --num-query-groups 4
        --kv-channels 128
        --max-position-embeddings 40960
        --seq-length "$seq_len"
        --normalization RMSNorm
        --norm-epsilon 1e-06
        --swiglu
        --disable-bias-linear
        --untie-embeddings-and-output-weights
        --position-embedding-type rope
        --rotary-percent 1.0
        --rotary-base 1000000
        --rotary-seq-len-interpolation-factor 1
        --qk-layernorm
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --group-query-attention
        --make-vocab-size-divisible-by 128
    )

    # --- Shared multimodal entry-point contract ---
    # The shared entry point still expects multimodal-shaped batch keys.
    # --text-only keeps those keys while emitting no vision payload.
    local multimodal_args=(
        --model-arch qwen3
        --dataset-provider mock
        --total-seq-length "$seq_len"
        --image-token-id 1
        --image-size 224
        --text-only
        --use-packed-sequence
    )
    if [ "$cp" -gt 1 ] || [ "$calculate_per_token_loss" = "1" ]; then
        multimodal_args+=( --calculate-per-token-loss )
    fi

    # --- Tokenizer (Qwen3 vocab = 151_936) ---
    local tokenizer_args=(
        --tokenizer-type NullMultimodalTokenizer
        --vocab-size 151936
    )

    # --- Recompute ---
    local recompute_args=()
    if [ "$recompute" -eq 1 ]; then
        if [ "$recompute_full" -eq 1 ]; then
            recompute_args=(
                --recompute-granularity full
                --recompute-method uniform
                --recompute-num-layers 1
            )
        elif [ -n "$recompute_modules" ]; then
            recompute_args=(
                --recompute-granularity selective
                --recompute-modules $recompute_modules
            )
        else
            recompute_args=(
                --recompute-granularity full
                --recompute-method uniform
                --recompute-num-layers 1
            )
        fi
    fi

    # --- FSDP ---
    local fsdp_args=()
    if [ "$use_fsdp" -eq 1 ]; then
        fsdp_args=(
            --use-megatron-fsdp
            --data-parallel-sharding-strategy optim_grads_params
            --no-gradient-accumulation-fusion
            --init-model-with-meta-device
            --use-distributed-optimizer
            --ckpt-format fsdp_dtensor
        )
    fi

    # =====================================================================
    # PROFILING — layered by prof_level (0-3)
    # =====================================================================
    local logging_args=(
        --log-interval 1
        --save-interval 99999
        --eval-interval 99999
        --eval-iters 1
        --tensorboard-dir "$run_dir"
        --log-throughput
        --log-timers-to-tensorboard
        --log-memory-to-tensorboard
        --log-memory-interval 5
    )
    if [[ -n "$WANDB_PROJECT" ]]; then
        logging_args+=(
            --wandb-project "$WANDB_PROJECT"
            --wandb-exp-name "$exp_name"
            --wandb-save-dir "$run_dir"
        )
    fi

    local profile_args=()
    local nsys_cmd=()
    local timing_level=0

    if [ "$prof_level" -ge 1 ]; then
        timing_level=2
        logging_args+=( --timing-log-option minmax )
    fi

    if [ "$prof_level" -ge 2 ]; then
        profile_args+=(
            --profile
            --profile-step-start "$profile_step_start"
            --profile-step-end "$profile_step_end"
            --profile-ranks $PROFILE_RANKS
        )
        local nsys_dir="${run_dir}/nsys"
        mkdir -p "$nsys_dir"
        nsys_cmd=(
            nsys profile
            --sample=none
            --cpuctxsw=none
            --trace=cuda,nvtx
            --force-overwrite=true
            --cuda-memory-usage=true
            --capture-range=cudaProfilerApi
            --capture-range-end=stop
            --stats=true
            --output "${nsys_dir}/${exp_name}_$(date +%Y%m%d_%H%M%S)"
        )
    fi

    if [ "$prof_level" -ge 3 ]; then
        local pytorch_prof_dir="${run_dir}/pytorch_profiler"
        mkdir -p "$pytorch_prof_dir"
        profile_args+=(
            --use-pytorch-profiler
            --pytorch-profiler-collect-shapes
            --pytorch-profiler-collect-callstack
            --record-memory-history
            --memory-snapshot-path "${run_dir}/memory_snapshot.pickle"
        )
    fi

    logging_args+=( --timing-log-level "$timing_level" )

    # --- Assemble & run ---
    local cmd=(
        ${nsys_cmd[@]+"${nsys_cmd[@]}"}
        torchrun "${dist_args[@]}"
        "$MEGATRON_LM_PATH/examples/multimodal_dev/pretrain_multimodal.py"
        "${training_args[@]}"
        ${fp8_args[@]+"${fp8_args[@]}"}
        ${profile_args[@]+"${profile_args[@]}"}
        "${parallel_args[@]}"
        "${logging_args[@]}"
        "${tokenizer_args[@]}"
        "${multimodal_args[@]}"
        "${gpt_args[@]}"
        "${moe_args[@]}"
        ${a2a_args[@]+"${a2a_args[@]}"}
        ${mtp_args[@]+"${mtp_args[@]}"}
        ${recompute_args[@]+"${recompute_args[@]}"}
        ${fsdp_args[@]+"${fsdp_args[@]}"}
    )

    if [ -n "$extra_args" ]; then
        cmd+=( $extra_args )
    fi

    echo ""
    echo "CMD: ${cmd[*]}"
    echo ""

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "=== DRY RUN — skipping execution ==="
        echo ""
    else
        "${cmd[@]}" 2>&1 | tee "${run_dir}/output.log"
        echo ""
        echo "Results saved to: $run_dir"

        if [ "$prof_level" -ge 2 ]; then
            local nsys_dir="${run_dir}/nsys"
            for nsys_file in "$nsys_dir"/*.nsys-rep; do
                [ -f "$nsys_file" ] || continue
                echo "  Generating nsys stats: ${nsys_file}"
                nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
                    --format csv \
                    --output "${nsys_file%.nsys-rep}" \
                    "$nsys_file" 2>/dev/null || true
            done
        fi
        echo ""
    fi
}


run_benchmark "$DEV_BENCH_NAME" ${_dev_merged_overrides[@]+"${_dev_merged_overrides[@]}"}
