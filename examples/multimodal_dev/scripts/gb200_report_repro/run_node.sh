#!/usr/bin/env bash

set -euo pipefail

: "${CELL:?CELL is required}"
: "${STACK_LEVEL:?STACK_LEVEL is required}"
: "${PP_SIZE:?PP_SIZE is required}"
: "${MDP_MODE:?MDP_MODE is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

target_repo="${TARGET_REPO:-/workspace/Megatron-LM}"
result_root="${RESULT_ROOT:-/workspace/results}"
energon_root="${ENERGON_ROOT:-/workspace/Megatron-Energon-7.3.2}"
venv_root="${VENV_ROOT:-/workspace/venv}"

if [[ ! -x "${target_repo}/examples/multimodal_dev/scripts/dev_qwen3vl_gb200.sh" ]]; then
    echo "Missing Qwen3-VL launcher in ${target_repo}" >&2
    exit 2
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    export PATH="${venv_root}/bin:${PATH}"
    export PYTHONPATH="${venv_root}/lib/python3.12/site-packages:${target_repo}:${energon_root}/src"
    export LD_LIBRARY_PATH="/usr/local/cuda-13.1/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONDONTWRITEBYTECODE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MDP_HYBRID_EP_PER_RANK_CACHE=1
export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset NCCL_NVLS_ENABLE
unset TORCH_FR_BUFFER_SIZE
unset TORCH_NCCL_TRACE_BUFFER_SIZE
unset TORCH_NCCL_DUMP_ON_TIMEOUT
unset TORCH_NCCL_DEBUG_INFO_TEMP_FILE
unset TORCH_NCCL_DESYNC_DEBUG
unset TORCH_NCCL_TRACE_CPP_STACK

nnodes="${NNODES:-16}"
train_iters="${TRAIN_ITERS:-45}"
warmup_iters="${WARMUP_ITERS:-5}"
fused_backward="${FUSED_BACKWARD:-recompute}"
vision_cap="${VISION_MAX_SEQUENCE_LENGTH:-0}"

case "${fused_backward}" in
    none|retain|recompute) ;;
    *) echo "Unknown FUSED_BACKWARD=${fused_backward}" >&2; exit 2 ;;
esac

common_extra=(
    --dataloader-type external
    --energon-path /data/energon/blends/blend3.yaml
    --tokenizer-model /data/tokenizer/Qwen3.5-35B-A3B
    --image-min-pixels 0
    --image-max-pixels 327680
    --energon-packing-buffer-size 128
    --energon-shuffle-buffer-size 128
    --energon-max-samples-per-sequence 16
    --energon-prefetch-factor 1
    --num-workers 1
    --dataloader-sequence-packing
    --eval-iters 0
)
if [[ -n "${DIST_TIMEOUT_MINUTES:-}" ]]; then
    common_extra+=(--distributed-timeout-minutes "${DIST_TIMEOUT_MINUTES}")
fi

case "${STACK_LEVEL}" in
    pr2) ;;
    pr4) common_extra+=(--mdp-inner-dp-scope cp) ;;
    pr5) common_extra+=(--mdp-inner-dp-scope pp_cp) ;;
    pr6)
        common_extra+=(
            --mdp-inner-dp-scope pp_cp
            --mdp-loader-prepartition-prefetch-windows 1
        )
        ;;
    *) echo "Unknown STACK_LEVEL=${STACK_LEVEL}" >&2; exit 2 ;;
esac

mode_extra=()
case "${MDP_MODE}" in
    ordinary)
        if [[ "${STACK_LEVEL}" != "pr2" ]]; then
            echo "ordinary mode is only valid at the PR2 boundary" >&2
            exit 2
        fi
        ;;
    mdp_off)
        mode_extra+=(--no-mdp-encoder-mode)
        if [[ "${STACK_LEVEL}" == "pr6" ]]; then
            mode_extra+=(
                --no-mdp-fused-vision-window
                --mdp-vision-encoder-max-sequence-length 0
                --mdp-fused-vision-backward recompute
            )
        fi
        ;;
    mdp_on)
        mode_extra+=(--mdp-encoder-mode)
        if [[ "${STACK_LEVEL}" == "pr6" ]]; then
            mode_extra+=(
                --no-mdp-fused-vision-window
                --mdp-vision-encoder-max-sequence-length 0
                --mdp-fused-vision-backward recompute
            )
        fi
        ;;
    mdp_fused)
        if [[ "${STACK_LEVEL}" != "pr6" ]]; then
            echo "mdp_fused requires the PR6 boundary" >&2
            exit 2
        fi
        if [[ "${vision_cap}" -le 0 ]]; then
            echo "A positive VISION_MAX_SEQUENCE_LENGTH is required for fused mode" >&2
            exit 2
        fi
        mode_extra+=(
            --mdp-encoder-mode
            --mdp-fused-vision-window
            --mdp-vision-encoder-max-sequence-length "${vision_cap}"
            --mdp-fused-vision-backward "${fused_backward}"
        )
        ;;
    *) echo "Unknown MDP_MODE=${MDP_MODE}" >&2; exit 2 ;;
esac

dry_run_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    dry_run_args+=(--dry-run)
fi

extra_args="${common_extra[*]} ${mode_extra[*]}"
command=(
    bash "${target_repo}/examples/multimodal_dev/scripts/dev_qwen3vl_gb200.sh"
    "${dry_run_args[@]}"
    --gpus 4
    --nnodes "${nnodes}"
    --train-iters "${train_iters}"
    --warmup-iters "${warmup_iters}"
    --results-dir "${result_root}/${CELL}/node${SLURM_NODEID:-0}"
    "${CELL}_pp${PP_SIZE}cp2_gbs256"
    tp=1
    ep=8
    pp="${PP_SIZE}"
    cp=2
    etp=1
    vpp=0
    mbs=1
    gbs=256
    seq_len=8192
    image_size=448
    dispatcher_backend=hybridep
    a2a_overlap=0
    recompute=0
    recompute_vision=0
    mtp=0
    use_packed_sequence=1
    dataset_provider=energon
    "extra_args=${extra_args}"
)

if [[ -n "${CELL_TIMEOUT_SECONDS:-}" && "${DRY_RUN:-0}" != "1" ]]; then
    exec timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SECONDS}" "${command[@]}"
fi
exec "${command[@]}"
