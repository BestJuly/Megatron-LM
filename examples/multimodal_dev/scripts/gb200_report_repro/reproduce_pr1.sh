#!/usr/bin/env bash

set -euo pipefail

REPORT="pr1"
PR1_SHA="1c99e8d550e530c7d04564b4ae2061f23b7cdcee"
PR2_SHA="d7b3bb7f2df4e48a8ea0bf3c78f0791a415db54c"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CELL_ROWS=(
    "pr1_pr2_baseline|${PR2_SHA}|pr2|1|ordinary|none|0|10|median|20619.3|45.9244|101708.21|reserved=101.0977_GiB|PR1 plus PR2 prior-chain report; PR1 has no standalone real-data result"
)

usage() {
    cat <<'EOF'
Reproduce the standalone PR1 GB200 performance report.

Usage:
  reproduce_pr1.sh [options]

Options:
  --job-id ID                 Running 16-node salloc job ID.
  --result-root DIR           Host output directory on shared storage.
  --source-repo DIR           Git repository used for the pinned checkout.
  --checkout-root DIR         Shared cache for detached worktrees.
  --cells CSV                 Run only the listed PR1 cells.
  --dry-run                   Prepare checkouts and print launcher commands.
  --prepare-only              Prepare checkouts and metadata, then stop.
  -h, --help                  Show this help.

This file contains its host runner, container-side runner, PR1 cell matrix,
and result summarizer. It has no shared reproduction-script dependency.

PR1 predates the real Energon provider. Its reported real-data boundary is
therefore the combined PR1 plus PR2 cell pinned to the PR2 SHA.
EOF
}

run_node() {
    : "${CELL:?CELL is required}"
    : "${STACK_LEVEL:?STACK_LEVEL is required}"
    : "${PP_SIZE:?PP_SIZE is required}"
    : "${MDP_MODE:?MDP_MODE is required}"
    : "${MASTER_ADDR:?MASTER_ADDR is required}"
    : "${MASTER_PORT:?MASTER_PORT is required}"

    local target_repo="${TARGET_REPO:-/workspace/Megatron-LM}"
    local result_root="${RESULT_ROOT:-/workspace/results}"
    local energon_root="${ENERGON_ROOT:-/workspace/Megatron-Energon-7.3.2}"
    local venv_root="${VENV_ROOT:-/workspace/venv}"
    local nnodes="${NNODES:-16}"
    local train_iters="${TRAIN_ITERS:-45}"
    local warmup_iters="${WARMUP_ITERS:-5}"
    local fused_backward="${FUSED_BACKWARD:-recompute}"
    local vision_cap="${VISION_MAX_SEQUENCE_LENGTH:-0}"

    if [[ ! -x "${target_repo}/examples/multimodal_dev/scripts/dev_qwen3vl_gb200.sh" ]]; then
        echo "Missing Qwen3-VL launcher in ${target_repo}" >&2
        exit 2
    fi
    if [[ "${STACK_LEVEL}" != "pr2" ]]; then
        echo "PR1 real-data runner requires the PR1 plus PR2 boundary" >&2
        exit 2
    fi
    case "${fused_backward}" in
        none|retain|recompute) ;;
        *) echo "Unknown FUSED_BACKWARD=${fused_backward}" >&2; exit 2 ;;
    esac

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

    local common_extra=(
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

    local mode_extra=()
    case "${MDP_MODE}" in
        ordinary) ;;
        *) echo "Unknown MDP_MODE=${MDP_MODE}" >&2; exit 2 ;;
    esac

    local dry_run_args=()
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        dry_run_args+=(--dry-run)
    fi
    local extra_args="${common_extra[*]} ${mode_extra[*]}"
    local command=(
        bash "${target_repo}/examples/multimodal_dev/scripts/dev_qwen3vl_gb200.sh"
        "${dry_run_args[@]}"
        --gpus 4
        --nnodes "${nnodes}"
        --train-iters "${train_iters}"
        --warmup-iters "${warmup_iters}"
        --results-dir "${result_root}/${CELL}/node${SLURM_NODEID:-0}"
        "${CELL}_pp${PP_SIZE}cp2_gbs256"
        tp=1 ep=8 pp="${PP_SIZE}" cp=2 etp=1 vpp=0
        mbs=1 gbs=256 seq_len=8192 image_size=448
        dispatcher_backend=hybridep a2a_overlap=0
        recompute=0 recompute_vision=0 mtp=0
        use_packed_sequence=1 dataset_provider=energon
        "extra_args=${extra_args}"
    )

    if [[ -n "${CELL_TIMEOUT_SECONDS:-}" && "${DRY_RUN:-0}" != "1" ]]; then
        exec timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SECONDS}" "${command[@]}"
    fi
    exec "${command[@]}"
}

summarize_results() {
    python3 - "$1" <<'PY'
import csv
import math
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
iteration_re = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([0-9.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)"
)
skipped_re = re.compile(r"number of skipped iterations:\s*(\d+)")
nan_re = re.compile(r"number of nan iterations:\s*(\d+)")
memory_re = re.compile(r"max allocated:\s*([0-9.]+).*?max reserved:\s*([0-9.]+)")

with (root / "manifest.tsv").open(newline="") as stream:
    manifest = list(csv.DictReader(stream, delimiter="\t"))

rows = []
for source in manifest:
    best = {}
    peak_allocated = 0.0
    peak_reserved = 0.0
    for log_path in sorted((root / source["cell"]).rglob("*.log")):
        parsed = {}
        with log_path.open(errors="replace") as stream:
            for line in stream:
                for match in memory_re.finditer(line):
                    peak_allocated = max(peak_allocated, float(match.group(1)))
                    peak_reserved = max(peak_reserved, float(match.group(2)))
                match = iteration_re.search(line)
                if match:
                    parsed[int(match.group(1))] = {
                        "total": int(match.group(2)),
                        "step": float(match.group(3)),
                        "tflops": float(match.group(4)),
                        "skipped": int(skipped_re.search(line).group(1)) if skipped_re.search(line) else 0,
                        "nan": int(nan_re.search(line).group(1)) if nan_re.search(line) else 0,
                    }
        if len(parsed) > len(best):
            best = parsed

    measured = [value for key, value in sorted(best.items()) if key >= int(source["measure_from"])]
    steps = [value["step"] for value in measured]
    tflops = [value["tflops"] for value in measured]
    mean_step = statistics.fmean(steps) if steps else None
    median_step = statistics.median(steps) if steps else None
    p95_step = sorted(steps)[max(0, math.ceil(0.95 * len(steps)) - 1)] if steps else None
    mean_tflops = statistics.fmean(tflops) if tflops else None
    completed = max(best, default=0)
    total = max((value["total"] for value in best.values()), default=0)
    skipped = max((value["skipped"] for value in best.values()), default=0)
    nan = max((value["nan"] for value in best.values()), default=0)
    cell_dir = root / source["cell"]
    status = "FAILED" if (cell_dir / "FAILED").exists() else "PASS" if (cell_dir / "SUCCESS").exists() else "UNKNOWN"
    if status == "PASS" and (not total or completed != total or len(best) != total or skipped or nan):
        status = "INVALID_RESULT"
    observed = median_step if source["reference_stat"] == "median" else mean_step
    delta = None
    if observed is not None and source["reference_step_ms"] not in {"", "NA"}:
        expected = float(source["reference_step_ms"])
        delta = 100.0 * (observed - expected) / expected

    def fmt(value):
        return "NA" if value is None else f"{value:.2f}"

    rows.append({
        **source,
        "status": status,
        "completed": f"{completed}/{total}" if total else "0/0",
        "samples": str(len(measured)),
        "mean_step_ms": fmt(mean_step),
        "median_step_ms": fmt(median_step),
        "p95_step_ms": fmt(p95_step),
        "mean_tflops": fmt(mean_tflops),
        "padded_tps_median": fmt(256 * 8192 * 1000 / median_step if median_step else None),
        "padded_tps_mean": fmt(256 * 8192 * 1000 / mean_step if mean_step else None),
        "peak_allocated_gib": fmt(peak_allocated / 1024 if peak_allocated else None),
        "peak_reserved_gib": fmt(peak_reserved / 1024 if peak_reserved else None),
        "skipped": str(skipped),
        "nan": str(nan),
        "reference_delta_pct": fmt(delta),
    })

fields = [
    "cell", "sha", "status", "completed", "samples", "mean_step_ms", "median_step_ms",
    "p95_step_ms", "mean_tflops", "padded_tps_median", "padded_tps_mean",
    "peak_allocated_gib", "peak_reserved_gib", "skipped", "nan", "reference_stat",
    "reference_step_ms", "reference_delta_pct", "reference_tflops", "reference_padded_tps",
    "reference_peak", "reference_note",
]
with (root / "summary.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)

lines = [
    "# GB200 PR1 reproduction summary", "",
    "| Cell | Status | Done | Mean ms | Median ms | Mean TFLOPs | Padded tok/s | Peak alloc GiB | Reference delta |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    delta = "NA" if row["reference_delta_pct"] == "NA" else f'{row["reference_delta_pct"]}%'
    lines.append(
        f'| {row["cell"]} | {row["status"]} | {row["completed"]} | {row["mean_step_ms"]} | '
        f'{row["median_step_ms"]} | {row["mean_tflops"]} | {row["padded_tps_median"]} | '
        f'{row["peak_allocated_gib"]} | {delta} |'
    )
(root / "summary.md").write_text("\n".join(lines) + "\n")
print((root / "summary.md").read_text(), end="")
if any(row["status"] in {"FAILED", "INVALID_RESULT", "UNKNOWN"} for row in rows):
    raise SystemExit(1)
PY
}

cell_selected() {
    local label="$1"
    [[ -z "${selected_cells}" || ",${selected_cells}," == *",${label},"* ]]
}

prepare_checkout() {
    local source_repo="$1"
    local checkout_root="$2"
    local sha="$3"
    local checkout="${checkout_root}/${sha:0:12}"
    if ! git -C "${source_repo}" cat-file -e "${sha}^{commit}"; then
        echo "Commit is not available in ${source_repo}: ${sha}" >&2
        exit 2
    fi
    if [[ ! -e "${checkout}/.git" ]]; then
        git -C "${source_repo}" worktree add --detach "${checkout}" "${sha}" >/dev/null
    fi
    local actual
    actual="$(git -C "${checkout}" rev-parse HEAD)"
    if [[ "${actual}" != "${sha}" ]]; then
        echo "Checkout mismatch: ${checkout} is ${actual}, expected ${sha}" >&2
        exit 2
    fi
    if [[ -n "$(git -C "${checkout}" status --porcelain --untracked-files=no)" ]]; then
        echo "Checkout has tracked changes: ${checkout}" >&2
        exit 2
    fi
    printf '%s\n' "${checkout}"
}

if [[ "${1:-}" == "__run_node" ]]; then
    shift
    run_node "$@"
fi
if [[ "${1:-}" == "__summarize" ]]; then
    shift
    summarize_results "${1:?result root is required}"
    exit 0
fi

job_id=""
result_root=""
source_repo="${SOURCE_REPO:-${REPO_ROOT}}"
checkout_root="${MDP_REPRO_CHECKOUT_ROOT:-${HOME}/.cache/megatron-mdp-report/checkouts}"
selected_cells=""
dry_run=0
prepare_only=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id) job_id="${2:?missing value for --job-id}"; shift 2 ;;
        --result-root) result_root="${2:?missing value for --result-root}"; shift 2 ;;
        --source-repo) source_repo="${2:?missing value for --source-repo}"; shift 2 ;;
        --checkout-root) checkout_root="${2:?missing value for --checkout-root}"; shift 2 ;;
        --cells) selected_cells="${2:?missing value for --cells}"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --prepare-only) prepare_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${dry_run}" != "1" && "${prepare_only}" != "1" && -z "${job_id}" ]]; then
    echo "--job-id is required unless --dry-run or --prepare-only is used" >&2
    exit 2
fi

container="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/dongjael/containers/mcore-moe-pytorch26.02-hybridep7febc6e-arm64.sqsh}"
energon_host="${ENERGON_HOST:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/dongjael/Megatron-Energon-7.3.2}"
data_host="${DATA_HOST:-/lustre/fsw/portfolios/coreai/users/dongjael/datasets/qwen35-mdp-data}"
raw_host="${RAW_DATA_HOST:-/lustre/fsw/portfolios/coreai/users/dongjael/datasets}"
venv_host="${VENV_HOST:-/home/dongjael/autoresearch/.runtime/mdp-stack-venv-20260717}"
blend_host="${data_host}/energon/blends/blend3.yaml"
stamp="$(date +%Y%m%d_%H%M%S)"

if [[ -z "${result_root}" ]]; then
    if [[ "${dry_run}" == "1" || "${prepare_only}" == "1" ]]; then
        result_root="/tmp/mdp_${REPORT}_report_repro_${stamp}"
    else
        result_root="/lustre/fsw/portfolios/coreai/users/dongjael/megatron-lm/benchmark_results/mdp_${REPORT}_report_repro_${stamp}_job${job_id}"
    fi
fi

for path in "${source_repo}" "${energon_host}" "${data_host}" "${raw_host}" "${venv_host}" "${container}" "${blend_host}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done
git -C "${source_repo}" rev-parse --git-dir >/dev/null
mkdir -p "${checkout_root}" "${result_root}"

manifest="${result_root}/manifest.tsv"
printf 'cell\tsha\tstack_level\tpp\tmode\tbackward\tcap\tmeasure_from\treference_stat\treference_step_ms\treference_tflops\treference_padded_tps\treference_peak\treference_note\n' > "${manifest}"
selected_rows=()
checkouts=()
for row in "${CELL_ROWS[@]}"; do
    IFS='|' read -r label sha level pp mode backward cap measure_from reference_stat reference_step reference_tflops reference_tps reference_peak reference_note <<< "${row}"
    if ! cell_selected "${label}"; then
        continue
    fi
    checkout="$(prepare_checkout "${source_repo}" "${checkout_root}" "${sha}")"
    selected_rows+=("${row}")
    checkouts+=("${checkout}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${sha}" "${level}" "${pp}" "${mode}" "${backward}" "${cap}" \
        "${measure_from}" "${reference_stat}" "${reference_step}" "${reference_tflops}" \
        "${reference_tps}" "${reference_peak}" "${reference_note}" >> "${manifest}"
done
if [[ "${#selected_rows[@]}" -eq 0 ]]; then
    echo "No cells selected" >&2
    exit 2
fi

git -C "${REPO_ROOT}" rev-parse HEAD > "${result_root}/harness_revision.txt"
git -C "${REPO_ROOT}" diff -- "${SCRIPT_PATH#${REPO_ROOT}/}" examples/multimodal_dev/README.md > "${result_root}/harness_working_tree.patch"
printf '%s\n' "${REPORT}" > "${result_root}/report.txt"
printf '%s\n' "${container}" > "${result_root}/container.txt"
printf '%s\n' "${blend_host}" > "${result_root}/blend_path.txt"
sha256sum "${blend_host}" > "${result_root}/blend_sha256.txt"
printf '%s\n' "${PR1_SHA}" "${PR2_SHA}" > "${result_root}/stack_code_revisions.txt"

if [[ "${prepare_only}" == "1" ]]; then
    echo "Prepared ${#selected_rows[@]} PR1 cells under ${checkout_root}"
    echo "RESULT_ROOT=${result_root}"
    exit 0
fi

master_addr="dry-run"
nodes="dry-run"
if [[ "${dry_run}" != "1" ]]; then
    job_info="$(scontrol show job "${job_id}" -o)"
    for required in "JobState=RUNNING" "Account=coreai_devtech_all" "Partition=batch" "QOS=normal" "NumNodes=16"; do
        if [[ " ${job_info} " != *" ${required} "* ]]; then
            echo "Allocation ${job_id} does not satisfy ${required}: ${job_info}" >&2
            exit 2
        fi
    done
    if [[ " ${job_info} " != *" JobName=coreai_devtech_all-megatron:vlm "* && "${ALLOW_NONCANONICAL_ALLOCATION:-0}" != "1" ]]; then
        echo "Allocation job name must be coreai_devtech_all-megatron:vlm" >&2
        exit 2
    fi
    nodes="$(squeue -h -j "${job_id}" -o '%N')"
    master_addr="$(scontrol show hostnames "${nodes}" | sed -n '1p')"
    printf '%s\n' "${job_info}" > "${result_root}/allocation.txt"
fi
printf '%s\n' "${nodes}" > "${result_root}/nodes.txt"
printf '%s\n' "${master_addr}" > "${result_root}/master_addr.txt"

overall_rc=0
for index in "${!selected_rows[@]}"; do
    row="${selected_rows[$index]}"
    checkout="${checkouts[$index]}"
    IFS='|' read -r label sha level pp mode backward cap _rest <<< "${row}"
    cell_host="${result_root}/${label}"
    if [[ -e "${cell_host}" ]]; then
        echo "Cell result directory already exists: ${cell_host}" >&2
        exit 2
    fi
    mkdir -p "${cell_host}"
    port=$(( ${MASTER_PORT_BASE:-29600} + index ))
    echo "Starting ${label}: sha=${sha} pp=${pp} mode=${mode}"

    if [[ "${dry_run}" == "1" ]]; then
        CELL="${label}" STACK_LEVEL="${level}" PP_SIZE="${pp}" MDP_MODE="${mode}" \
            FUSED_BACKWARD="${backward}" VISION_MAX_SEQUENCE_LENGTH="${cap}" \
            MASTER_ADDR=localhost MASTER_PORT="${port}" NNODES=16 SLURM_NODEID=0 \
            DRY_RUN=1 TARGET_REPO="${checkout}" RESULT_ROOT="${result_root}" \
            bash "${SCRIPT_PATH}" __run_node | tee "${cell_host}/dry_run.log"
        touch "${cell_host}/SUCCESS"
        continue
    fi

    mounts="${checkout}:/workspace/Megatron-LM"
    mounts+=",${SCRIPT_DIR}:/workspace/repro"
    mounts+=",${energon_host}:/workspace/Megatron-Energon-7.3.2"
    mounts+=",${data_host}:/data,${raw_host}:/raw"
    mounts+=",${venv_host}:/workspace/venv,${result_root}:/workspace/results"
    cell_rc=0
    CELL="${label}" STACK_LEVEL="${level}" PP_SIZE="${pp}" MDP_MODE="${mode}" \
        FUSED_BACKWARD="${backward}" VISION_MAX_SEQUENCE_LENGTH="${cap}" \
        MASTER_ADDR="${master_addr}" MASTER_PORT="${port}" NNODES=16 \
        TRAIN_ITERS="${TRAIN_ITERS:-45}" WARMUP_ITERS="${WARMUP_ITERS:-5}" \
        DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}" \
        CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-5400}" \
        srun --jobid="${job_id}" --overlap --nodes=16 --ntasks=16 --ntasks-per-node=1 \
        --kill-on-bad-exit=1 --mpi=pmix \
        --output="${cell_host}/launcher_%t.log" --error="${cell_host}/launcher_%t.err" \
        --container-image="${container}" --container-mounts="${mounts}" \
        --container-workdir=/workspace/Megatron-LM \
        bash "/workspace/repro/${SCRIPT_NAME}" __run_node || cell_rc=$?
    if [[ "${cell_rc}" -eq 0 ]]; then
        touch "${cell_host}/SUCCESS"
    else
        touch "${cell_host}/FAILED"
        overall_rc=1
        echo "Cell failed: ${label} (rc=${cell_rc})" >&2
    fi
done

if [[ "${dry_run}" == "1" ]]; then
    echo "Dry-run commands are under ${result_root}"
else
    summarize_results "${result_root}" || overall_rc=1
fi
echo "RESULT_ROOT=${result_root}"
exit "${overall_rc}"
