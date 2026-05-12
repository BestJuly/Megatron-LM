#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# End-to-end driver for the PP support added on
# `lit/qwen35_dev_pp_support`.  Exercises four scenarios on the proxy
# variant of Qwen3.5-VL:
#
#   1. unit:    examples/multimodal_dev/tests/test_pp_construction.py (PP=1, PP=2)
#   2. pp1:     PP=1 + FSDP+EP regression  (must keep working as before)
#   3. pp2:     PP=2 + EP=2 functionality   (the new feature)
#   4. resume:  PP=2 ckpt save → ckpt load round-trip (correctness)
#
# Usage::
#
#     # From the Megatron-LM-qwen35-pp/ repo root:
#     bash examples/multimodal_dev/tests/run_pp_e2e.sh
#
# Override via env:
#
#     GPUS_PER_NODE   8        GPUs to use
#     TRAIN_ITERS     10       iters per phase
#     SKIP_UNIT       0        skip unit-test phase if 1
#     SKIP_PP1        0        skip PP=1 FSDP regression if 1
#     SKIP_PP2        0        skip PP=2 functionality if 1
#     SKIP_RESUME     0        skip ckpt save+load if 1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TRAIN_ITERS=${TRAIN_ITERS:-10}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/local/pp_e2e_logs}

SKIP_UNIT=${SKIP_UNIT:-0}
SKIP_PP1=${SKIP_PP1:-0}
SKIP_PP2=${SKIP_PP2:-0}
SKIP_RESUME=${SKIP_RESUME:-0}

mkdir -p "${LOG_DIR}"

banner() {
    echo
    echo "================================================================"
    echo "  $*"
    echo "================================================================"
}

# Common env shared across phases.
export DRY_RUN=0
export MODEL_VARIANT=proxy
export USE_PACKED_SEQUENCE=${USE_PACKED_SEQUENCE:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}
# Disable wandb args entirely — the wandb package in this container
# pulls a pydantic_core.from_json that may be missing, crashing the
# post-save artifact writer.  PP correctness is independent of wandb.
export USE_WANDB=${USE_WANDB:-0}
export GPUS_PER_NODE
export TRAIN_ITERS
# Mock dataset path avoids any HF download dependency for the E2E run.
export DATASET_PROVIDER=${DATASET_PROVIDER:-mock}

# Bookkeeping for ckpt paths produced by run_qwen35_vl.sh.
PP1_CKPT="${REPO_ROOT}/local/qwen35vl_proxy_tp1_ep2_pp1_cp1"
PP2_CKPT="${REPO_ROOT}/local/qwen35vl_proxy_tp1_ep2_pp2_cp1"

# ----------------------------------------------------------------
# Phase 0 — unit test
# ----------------------------------------------------------------
if [ "${SKIP_UNIT}" -eq 0 ]; then
    banner "[unit] test_pp_construction.py PP=1 (1 GPU)"
    log="${LOG_DIR}/unit_pp1.log"
    torchrun --nproc_per_node 1 \
        examples/multimodal_dev/tests/test_pp_construction.py --pp 1 \
        2>&1 | tee "${log}"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[unit:pp=1] FAIL"; exit 1
    fi

    if [ "${GPUS_PER_NODE}" -ge 2 ]; then
        banner "[unit] test_pp_construction.py PP=2 (2 GPUs)"
        log="${LOG_DIR}/unit_pp2.log"
        torchrun --nproc_per_node 2 \
            examples/multimodal_dev/tests/test_pp_construction.py --pp 2 \
            2>&1 | tee "${log}"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo "[unit:pp=2] FAIL"; exit 1
        fi
    fi
    echo "[unit] PASS"
fi

# ----------------------------------------------------------------
# Phase 1 — PP=1 + FSDP+EP regression
# ----------------------------------------------------------------
if [ "${SKIP_PP1}" -eq 0 ]; then
    banner "[pp1_fsdp] PP=1 USE_FSDP=1 EP=2 train_iters=${TRAIN_ITERS}"
    rm -rf "${PP1_CKPT}" || true
    log="${LOG_DIR}/pp1_fsdp.log"
    PP=1 USE_FSDP=1 EP=2 TP=1 CP=1 MBS=2 GBS=8 \
        TRAIN_ITERS="${TRAIN_ITERS}" \
        SAVE_INTERVAL="${TRAIN_ITERS}" \
        bash examples/multimodal_dev/scripts/run_qwen35_vl.sh \
        2>&1 | tee "${log}"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[pp1_fsdp] FAIL"; exit 1
    fi
    echo "[pp1_fsdp] PASS — last loss lines:"
    grep -E "iteration .* +loss" "${log}" | tail -3 || true
fi

# ----------------------------------------------------------------
# Phase 2 — PP=2 + EP=2 functionality (no FSDP)
# ----------------------------------------------------------------
# Non-FSDP path needs an explicit ckpt-format because:
#   * The default ``torch_dist`` triggers the NVRX async checkpointing
#     path which depends on ``nvidia_resiliency_ext`` symbols that are
#     not always available in this container.
#   * ``torch`` (legacy per-rank) avoids the NVRX path entirely and
#     round-trips PP=2 cleanly for our purposes.
PP_NOFSDP_CKPT_FORMAT=${PP_NOFSDP_CKPT_FORMAT:-torch}

if [ "${SKIP_PP2}" -eq 0 ]; then
    banner "[pp2] PP=2 USE_FSDP=0 EP=2 train_iters=${TRAIN_ITERS}"
    rm -rf "${PP2_CKPT}" || true
    log="${LOG_DIR}/pp2.log"
    PP=2 USE_FSDP=0 EP=2 TP=1 CP=1 MBS=2 GBS=8 \
        TRAIN_ITERS="${TRAIN_ITERS}" \
        SAVE_INTERVAL="${TRAIN_ITERS}" \
        CKPT_FORMAT="${PP_NOFSDP_CKPT_FORMAT}" \
        bash examples/multimodal_dev/scripts/run_qwen35_vl.sh \
        2>&1 | tee "${log}"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[pp2] FAIL"; exit 1
    fi
    echo "[pp2] PASS — last loss lines:"
    grep -E "iteration .* +loss" "${log}" | tail -3 || true

    # Confirm a checkpoint was actually written.
    if ! ls -d "${PP2_CKPT}/iter_"* >/dev/null 2>&1; then
        echo "[pp2] FAIL — no checkpoint produced at ${PP2_CKPT}"
        exit 1
    fi
    echo "[pp2] checkpoint(s) at ${PP2_CKPT}:"
    ls -d "${PP2_CKPT}/iter_"* | tail -3
fi

# ----------------------------------------------------------------
# Phase 3 — PP=2 ckpt save → load round-trip
# ----------------------------------------------------------------
if [ "${SKIP_RESUME}" -eq 0 ]; then
    HALF=$(( TRAIN_ITERS > 1 ? TRAIN_ITERS / 2 : 1 ))
    rm -rf "${PP2_CKPT}" || true

    banner "[resume A] PP=2 train ${HALF} iters, save"
    log_a="${LOG_DIR}/resume_a.log"
    PP=2 USE_FSDP=0 EP=2 TP=1 CP=1 MBS=2 GBS=8 \
        TRAIN_ITERS="${HALF}" SAVE_INTERVAL="${HALF}" \
        CKPT_FORMAT="${PP_NOFSDP_CKPT_FORMAT}" \
        bash examples/multimodal_dev/scripts/run_qwen35_vl.sh \
        2>&1 | tee "${log_a}"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[resume A] FAIL"; exit 1
    fi
    if ! ls -d "${PP2_CKPT}/iter_"* >/dev/null 2>&1; then
        echo "[resume A] FAIL — checkpoint missing"
        exit 1
    fi

    banner "[resume B] PP=2 load → run to ${TRAIN_ITERS} iters"
    log_b="${LOG_DIR}/resume_b.log"
    # CKPT_OVERRIDE_SCHEDULER=1 is needed because TRAIN_ITERS differs
    # between save & load (HALF vs TRAIN_ITERS), which changes the
    # weight-decay-iterations the scheduler computes.
    PP=2 USE_FSDP=0 EP=2 TP=1 CP=1 MBS=2 GBS=8 \
        TRAIN_ITERS="${TRAIN_ITERS}" \
        SAVE_INTERVAL=$((TRAIN_ITERS + 1)) \
        CKPT_LOAD="${PP2_CKPT}" CKPT_RESUME=1 \
        CKPT_FORMAT="${PP_NOFSDP_CKPT_FORMAT}" \
        CKPT_OVERRIDE_SCHEDULER=1 \
        bash examples/multimodal_dev/scripts/run_qwen35_vl.sh \
        2>&1 | tee "${log_b}"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[resume B] FAIL"; exit 1
    fi

    # The resumed run must start its iteration counter at HALF+1 (or any
    # iter > HALF), proving it actually loaded the checkpoint.
    if grep -qE "iteration +$((HALF + 1))/" "${log_b}"; then
        echo "[resume] PASS — resumed at iter $((HALF + 1))"
    elif grep -qE "set iteration to +${HALF}" "${log_b}"; then
        echo "[resume] PASS — loader reports iteration set to ${HALF}"
    else
        echo "[resume] FAIL — no evidence of correct resume in ${log_b}"
        echo "  (looking for 'iteration $((HALF + 1))/' or 'set iteration to ${HALF}')"
        tail -20 "${log_b}"
        exit 1
    fi
fi

banner "ALL PP E2E PHASES PASSED"
