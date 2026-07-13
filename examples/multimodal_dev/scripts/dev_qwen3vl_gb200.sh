#!/bin/bash

# =============================================================================
# Qwen3-VL hybrid (qwen35-VL vision + qwen3 LLM decoder) - GB200 launcher
# =============================================================================
#
# Vision encoder, MRoPE, and multimodal vocab are carried from qwen35-VL;
# the LLM decoder uses the qwen3 model path.
#
# Key choices vs the two parent launchers:
#   * --model-arch qwen3vl
#   * Vision tower: 27 layers, image_size variable (mock data)
#   * LLM decoder: qwen3 hyperparams (48 layers, hidden=2048, ffn=6144,
#     32 heads, num_query_groups=4, kv_channels=128, qk_layernorm)
#   * MoE: 128 experts x top-8, moe_ffn=768, NO shared expert
#   * MRoPE: rotary_percent=0.5 (was 0.25 on qwen35-VL with kv_channels=256;
#     adapted to qwen3's head_dim=128 to keep mrope_section=[11,11,10] valid)
#   * vocab=248320 (preserves image_token_id=248056 + multimodal block)
#   * --use-packed-sequence always (CP=1 no-op; CP>1 forces qkv_format=thd)
#
# Defaults below are the small image, single-layout CP=1 anchor stack.
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

RESULTS_DIR="${MEGATRON_LM_PATH}/benchmark_results/qwen3vl_gb200"
DRY_RUN=0
PROF_LEVEL=0
PROFILE_RANKS="0 1"
PROFILE_STEP_START=""
PROFILE_STEP_END=""
WANDB_PROJECT=""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_DEV_BENCH_NAME="qwen3vl_dev"

usage() {
    cat <<'EOF'
Qwen3-VL hybrid (qwen35-VL vision + qwen3 decoder) GB200 launcher.

Usage:
  dev_qwen3vl_gb200.sh [options] [bench_name] [key=value ...]

  bench_name     Run label.
  key=value      run_benchmark overrides (tp, ep, pp, cp, etp, mbs, gbs,
                 seq_len, image_size, image_size_w, image_sizes_h,
                 image_sizes_w, num_images, mock_layout,
                 mock_pack_num_docs, text_only, vision_num_layers,
                 dataset_provider, dataset_backend, dataset_subsets,
                 dataset_root, dataset_split, pack_samples_per_item,
                 pack_scan_multiplier, image_size_max,
                 dispatcher_backend, a2a_overlap, use_fsdp, cuda_graph_scope,
                 recompute, mtp, fp8, fp8_recipe).

Global options: --dry-run --gpus N --nnodes N --train-iters N --warmup-iters N
                --results-dir DIR --prof-level N --profile-ranks "N N"
                --profile-step-start N --profile-step-end N --wandb-project P

Examples:
  dev_qwen3vl_gb200.sh exp000
  dev_qwen3vl_gb200.sh exp001 cp=2 image_size=224
  dev_qwen3vl_gb200.sh longseq cp=2 seq_len=16384 image_size=224
  dev_qwen3vl_gb200.sh --dry-run --gpus 4 --nnodes 1
EOF
}

_cli_bench_name=""
_cli_bench_overrides=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --gpus) GPUS_PER_NODE="$2"; shift 2 ;;
        --nnodes) NUM_NODES="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --prof-level) PROF_LEVEL="$2"; shift 2 ;;
        --profile-ranks) PROFILE_RANKS="$2"; shift 2 ;;
        --profile-step-start) PROFILE_STEP_START="$2"; shift 2 ;;
        --profile-step-end) PROFILE_STEP_END="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        *)
            if [[ "$1" == *"="* ]]; then _cli_bench_overrides+=("$1")
            elif [[ -z "$_cli_bench_name" ]]; then _cli_bench_name="$1"
            else echo "ERROR: unexpected '$1'" >&2; usage >&2; exit 1; fi
            shift ;;
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
# run_benchmark - single dispatch
# ---------------------------------------------------------------------------
run_benchmark() {
    local name="$1"
    shift

    # --- Defaults ---
    local tp=1 ep=8 pp=1 cp=1 etp=1 vpp=1
    local mbs=1 gbs=512
    local seq_len=4096
    local image_size=224
    local image_size_max=""
    local image_size_w=""
    local image_sizes_h=""
    local image_sizes_w=""
    local num_images=1
    local mock_layout="single"
    local mock_pack_num_docs=1
    local text_only=0
    local vision_num_layers=27
    local dispatcher="alltoall"
    local dispatcher_backend="hybridep"
    local mtp=0
    local a2a_overlap=0
    local use_fsdp=0
    local recompute=0
    local recompute_vision=0
    local recompute_modules="moe_act shared_experts layernorm"
    local recompute_full=0
    local cuda_graph_scope=""
    local moe_grouped_gemm=1
    local cross_entropy_impl="te"
    local calculate_per_token_loss=0
    local fp8=""
    local fp8_recipe=""
    local dataset_provider="mock"
    local dataset_backend=""
    local dataset_subsets=""
    local dataset_root=""
    local dataset_split="train"
    local pack_samples_per_item=1
    local pack_scan_multiplier=1
    local extra_args=""
    local num_nodes_override=""
    local gpus_override=""
    local prof_level="$PROF_LEVEL"
    local profile_step_start="${PROFILE_STEP_START}"
    local profile_step_end="${PROFILE_STEP_END}"
    local use_packed_sequence=1

    # Parse overrides
    for kv in "$@"; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        case "$key" in
            tp) tp="$val" ;; ep) ep="$val" ;; pp) pp="$val" ;; vpp) vpp="$val" ;;
            cp) cp="$val" ;; etp) etp="$val" ;;
            mbs) mbs="$val" ;; gbs) gbs="$val" ;;
            seq_len) seq_len="$val" ;;
            image_size) image_size="$val" ;;
            image_size_max) image_size_max="$val" ;;
            image_size_w) image_size_w="$val" ;;
            image_sizes_h) image_sizes_h="$val" ;;
            image_sizes_w) image_sizes_w="$val" ;;
            num_images) num_images="$val" ;;
            mock_layout) mock_layout="$val" ;;
            mock_pack_num_docs) mock_pack_num_docs="$val" ;;
            text_only) text_only="$val" ;;
            vision_num_layers) vision_num_layers="$val" ;;
            dispatcher) dispatcher="$val" ;;
            dispatcher_backend) dispatcher_backend="$val" ;;
            a2a_overlap) a2a_overlap="$val" ;;
            mtp) mtp="$val" ;;
            use_fsdp) use_fsdp="$val" ;;
            recompute) recompute="$val" ;;
            recompute_vision) recompute_vision="$val" ;;
            recompute_modules) recompute_modules="$val" ;;
            recompute_full) recompute_full="$val" ;;
            cuda_graph_scope) cuda_graph_scope="$val" ;;
            moe_grouped_gemm) moe_grouped_gemm="$val" ;;
            cross_entropy_impl) cross_entropy_impl="$val" ;;
            calculate_per_token_loss) calculate_per_token_loss="$val" ;;
            fp8) fp8="$val" ;;
            fp8_recipe) fp8_recipe="$val" ;;
            dataset_provider) dataset_provider="$val" ;;
            dataset_backend) dataset_backend="$val" ;;
            dataset_subsets) dataset_subsets="$val" ;;
            dataset_root) dataset_root="$val" ;;
            dataset_split) dataset_split="$val" ;;
            pack_samples_per_item) pack_samples_per_item="$val" ;;
            pack_scan_multiplier) pack_scan_multiplier="$val" ;;
            extra_args) extra_args="$val" ;;
            num_nodes) num_nodes_override="$val" ;;
            gpus) gpus_override="$val" ;;
            prof_level) prof_level="$val" ;;
            profile_step_start) profile_step_start="$val" ;;
            profile_step_end) profile_step_end="$val" ;;
            use_packed_sequence) use_packed_sequence="$val" ;;
            *) echo "ERROR: unknown override '$key'" >&2; return 2 ;;
        esac
    done

    local run_nodes="${num_nodes_override:-$NUM_NODES}"
    local run_gpus_per_node="${gpus_override:-$GPUS_PER_NODE}"
    local run_total_gpus=$((run_gpus_per_node * run_nodes))

    local exp_name="${name}"
    local run_dir="${RESULTS_DIR}/${exp_name}"
    mkdir -p "$run_dir"

    local precision_label="BF16"
    [ -n "$fp8" ] && precision_label="FP8(${fp8_recipe:-blockwise})"

    echo ""
    echo "================================================================"
    echo " BENCHMARK: $name  [GB200 192GB, qwen3vl hybrid]"
    echo "   Vision:   qwen35-VL ViT, layers=${vision_num_layers}, image_size=${image_size}"
    echo "   LLM:      qwen3 MoE decoder"
    echo "   GPUs:     ${run_total_gpus} (${run_nodes}N x ${run_gpus_per_node}G)"
    echo "   Parallel: TP=${tp} EP=${ep} PP=${pp} CP=${cp} ETP=${etp}"
    echo "   Batch:    MBS=${mbs} GBS=${gbs}"
    echo "   Seq:      ${seq_len}  Images: ${num_images} (${mock_layout})  Pack: ${mock_pack_num_docs}"
    echo "   Precision:${precision_label}  Dispatcher: ${dispatcher_backend}"
    echo "   A2A overlap: ${a2a_overlap}  CUDA Graph: ${cuda_graph_scope:-none}"
    echo "   Recompute: ${recompute} (modules=${recompute_modules:-full-uniform})  vision_recompute=${recompute_vision}"
    echo "   MTP: ${mtp}"
    echo "================================================================"

    # --- Distributed args ---
    local dist_args=( --nproc_per_node "$run_gpus_per_node" --nnodes "$run_nodes" )
    if [ "$run_nodes" -gt 1 ]; then
        dist_args+=( --master_addr "${MASTER_ADDR:-localhost}" --master_port "${MASTER_PORT:-6000}" --node_rank ${SLURM_NODEID:-0} )
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
    if [ -n "$vpp" ] && [ "$vpp" != "0" ]; then
        parallel_args+=( --num-virtual-stages-per-pipeline-rank "$vpp" )
    fi

    # --- Training ---
    local training_args=(
        --micro-batch-size "$mbs"
        --global-batch-size "$gbs"
        --train-iters "$ACTUAL_ITERS"
        --adam-beta1 0.9 --adam-beta2 0.95
        --lr 1.2e-4 --min-lr 1.2e-5
        --lr-decay-style cosine
        --lr-warmup-iters "$WARMUP_ITERS"
        --lr-decay-iters 2000
        --weight-decay 0.1 --clip-grad 1.0
        --bf16
        --use-mcore-models
        --use-flash-attn
        --transformer-impl transformer_engine
        --cross-entropy-loss-fusion
        --cross-entropy-fusion-impl "$cross_entropy_impl"
        --enable-experimental
        --use-precision-aware-optimizer
        --main-grads-dtype fp32 --main-params-dtype fp32
        --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16
    )
    training_args+=( --manual-gc --manual-gc-interval 5 )

    local mtp_args=()
    [ "$mtp" -eq 1 ] && mtp_args=( --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 )

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

    # --- MoE (qwen3: 128 x top-8, no shared expert) ---
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
    [ "$moe_grouped_gemm" -eq 1 ] && moe_args+=( --moe-grouped-gemm )
    if [[ ${dispatcher_backend} == "alltoall" ]]; then
        moe_args+=( --moe-token-dispatcher-type alltoall )
    elif [[ ${dispatcher_backend} == "deepep" ]]; then
        moe_args+=( --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep --moe-deepep-num-sms 32 )
    elif [[ ${dispatcher_backend} == "hybridep" ]]; then
        moe_args+=( --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --moe-hybridep-num-sms 32 )
    fi

    # --- A2A overlap ---
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

    # --- Model architecture (qwen3 LLM decoder; vision config from qwen35_vl factory) ---
    # kv_channels=128 is qwen3's head_dim. qwen3 head_dim=128 needs
    # rotary_percent=0.5 to keep 32 RoPE pairs for mrope_section=[11,11,10].
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
        --rotary-percent 0.5
        --rotary-base 10000000
        --rotary-seq-len-interpolation-factor 1
        --qk-layernorm
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --group-query-attention
        --make-vocab-size-divisible-by 128
    )

    # --- Multimodal (image-token-id=248056 inherited from qwen35-VL) ---
    # use_packed_sequence default 1: forces qkv_format=thd at CP>1.
    # Override to 0 to test the sbhd path.
    local multimodal_args=(
        --model-arch qwen3vl
        --dataset-provider "$dataset_provider"
        --dataset-split "$dataset_split"
        --pack-samples-per-item "$pack_samples_per_item"
        --pack-scan-multiplier "$pack_scan_multiplier"
        --image-token-id 248056
        --image-size "$image_size"
        --total-seq-length "$seq_len"
        --vision-num-layers "$vision_num_layers"
        --num-images "$num_images"
        --mock-layout "$mock_layout"
        --mock-pack-num-docs "$mock_pack_num_docs"
    )
    [ -n "$dataset_backend" ] && multimodal_args+=( --dataset-backend "$dataset_backend" )
    [ -n "$dataset_subsets" ] && multimodal_args+=( --dataset-subsets "$dataset_subsets" )
    [ -n "$dataset_root" ] && multimodal_args+=( --dataset-root "$dataset_root" )
    [ -n "$image_size_max" ] && multimodal_args+=( --image-size-max "$image_size_max" )
    [ -n "$image_size_w" ] && multimodal_args+=( --image-size-w "$image_size_w" )
    [ -n "$image_sizes_h" ] && multimodal_args+=( --image-sizes-h "$image_sizes_h" )
    [ -n "$image_sizes_w" ] && multimodal_args+=( --image-sizes-w "$image_sizes_w" )
    if [ "$cp" -gt 1 ] || [ "$calculate_per_token_loss" = "1" ]; then
        multimodal_args+=( --calculate-per-token-loss )
    fi
    [ "$use_packed_sequence" -eq 1 ] && multimodal_args+=( --use-packed-sequence )
    [ "$text_only" -eq 1 ] && multimodal_args+=( --text-only )

    # --- Tokenizer (qwen35-VL vocab to retain image_token_id) ---
    local tokenizer_args=(
        --tokenizer-type NullMultimodalTokenizer
        --vocab-size 248320
    )

    # --- Recompute ---
    local recompute_args=()
    if [ "$recompute" -eq 1 ]; then
        if [ "$recompute_full" -eq 1 ]; then
            recompute_args=( --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 )
        elif [ -n "$recompute_modules" ]; then
            recompute_args=( --recompute-granularity selective --recompute-modules $recompute_modules )
        else
            recompute_args=( --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 )
        fi
    fi
    [ "$recompute_vision" -eq 1 ] && recompute_args+=( --recompute-vision )

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

    # --- Logging / profiling ---
    local logging_args=(
        --log-interval 1
        --save-interval 99999 --eval-interval 99999 --eval-iters 1
        --tensorboard-dir "$run_dir"
        --log-throughput
        --log-timers-to-tensorboard
    )
    if [[ -n "$WANDB_PROJECT" ]]; then
        logging_args+=(
            --wandb-project "$WANDB_PROJECT"
            --wandb-exp-name "$exp_name"
            --wandb-save-dir "$run_dir"
        )
    fi
    logging_args+=( --log-memory-to-tensorboard --log-memory-interval 5 )
    local profile_args=()
    local nsys_cmd=()
    local timing_level=0
    if [ "$prof_level" -ge 1 ]; then
        timing_level=2
        logging_args+=( --timing-log-option minmax )
    fi
    if [ "$prof_level" -ge 2 ]; then
        profile_args+=( --profile --profile-step-start "$profile_step_start" --profile-step-end "$profile_step_end" --profile-ranks $PROFILE_RANKS )
        local nsys_dir="${run_dir}/nsys"
        mkdir -p "$nsys_dir"
        nsys_cmd=( nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --force-overwrite=true --cuda-memory-usage=true --capture-range=cudaProfilerApi --capture-range-end=stop --stats=true --output "${nsys_dir}/${exp_name}_$(date +%Y%m%d_%H%M%S)" )
    fi
    if [ "$prof_level" -ge 3 ]; then
        local pytorch_prof_dir="${run_dir}/pytorch_profiler"
        mkdir -p "$pytorch_prof_dir"
        profile_args+=( --use-pytorch-profiler --pytorch-profiler-collect-shapes --pytorch-profiler-collect-callstack --record-memory-history --memory-snapshot-path "${run_dir}/memory_snapshot.pickle" )
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
    [ -n "$extra_args" ] && cmd+=( $extra_args )

    echo ""
    echo "CMD: ${cmd[*]}"
    echo ""
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "=== DRY RUN - skipping execution ==="
    else
        "${cmd[@]}" 2>&1 | tee "${run_dir}/output.log"
        echo "Results saved to: $run_dir"
        if [ "$prof_level" -ge 2 ]; then
            local nsys_dir="${run_dir}/nsys"
            for nsys_file in "$nsys_dir"/*.nsys-rep; do
                [ -f "$nsys_file" ] || continue
                nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum --format csv --output "${nsys_file%.nsys-rep}" "$nsys_file" 2>/dev/null || true
            done
        fi
    fi
}

run_benchmark "$DEV_BENCH_NAME" ${_dev_merged_overrides[@]+"${_dev_merged_overrides[@]}"}
