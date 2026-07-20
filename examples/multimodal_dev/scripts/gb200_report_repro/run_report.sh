#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

PR1_SHA="1c99e8d550e530c7d04564b4ae2061f23b7cdcee"
PR2_SHA="d7b3bb7f2df4e48a8ea0bf3c78f0791a415db54c"
PR4_SHA="7696cc537cbd946a075c5595dc2b2d25269d1553"
PR5_SHA="e755e9f47b342ab73538ff7f637a0b817e7ff634"
# This is the PR6 code tip before this documentation-only reproduction commit.
PR6_CODE_SHA="e0450f1b948ed53932a9059a3cefc09d0a0a2371"

usage() {
    cat <<'EOF'
Reproduce one MDP stack PR performance report on the GB200 cluster.

Usage:
  run_report.sh <pr1|pr2|pr4|pr5|pr6> [options]

Options:
  --job-id ID                 Running 16-node salloc job ID.
  --result-root DIR           Host output directory on shared storage.
  --source-repo DIR           Git repository used to create exact checkouts.
  --checkout-root DIR         Shared cache for detached PR worktrees.
  --cells CSV                 Run only the listed cell labels.
  --dry-run                   Build every launcher command without srun/torch.
  --prepare-only              Prepare checkouts and metadata, then stop.
  -h, --help                  Show this help.

The PR-specific reproduce_pr*.sh wrappers supply the first argument.
EOF
}

report="${1:-}"
case "${report}" in
    pr1|pr2|pr4|pr5|pr6) shift ;;
    -h|--help|"") usage; exit 0 ;;
    *) echo "Unknown report: ${report}" >&2; usage >&2; exit 2 ;;
esac

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
        result_root="/tmp/mdp_${report}_report_repro_${stamp}"
    else
        result_root="/lustre/fsw/portfolios/coreai/users/dongjael/megatron-lm/benchmark_results/mdp_${report}_report_repro_${stamp}_job${job_id:-unset}"
    fi
fi

labels=()
shas=()
levels=()
pps=()
modes=()
backward_modes=()
caps=()
measure_from=()
reference_stats=()
reference_steps_ms=()
reference_tflops=()
reference_padded_tps=()
reference_peaks=()
reference_notes=()

add_cell() {
    labels+=("$1")
    shas+=("$2")
    levels+=("$3")
    pps+=("$4")
    modes+=("$5")
    backward_modes+=("$6")
    caps+=("$7")
    measure_from+=("$8")
    reference_stats+=("$9")
    reference_steps_ms+=("${10}")
    reference_tflops+=("${11}")
    reference_padded_tps+=("${12}")
    reference_peaks+=("${13}")
    reference_notes+=("${14}")
}

case "${report}" in
    pr1)
        # PR1's body explicitly reports the PR1+PR2 boundary because real
        # Energon data support first exists in PR2.
        add_cell pr1_pr2_baseline "${PR2_SHA}" pr2 1 ordinary none 0 10 \
            median 20619.3 45.9244 101708.21 reserved=101.0977_GiB \
            "PR1+PR2 prior-chain report; PR1 has no standalone real-data result"
        ;;
    pr2)
        add_cell pr2_baseline "${PR2_SHA}" pr2 1 ordinary none 0 10 \
            median 20619.3 45.9244 101708.21 reserved=101.0977_GiB \
            "PR2 prior-chain ordinary-loader report"
        ;;
    pr4)
        add_cell pr2_baseline "${PR2_SHA}" pr2 1 ordinary none 0 10 \
            median 20619.3 45.9244 101708.21 reserved=101.0977_GiB \
            "PR2 comparison boundary"
        add_cell pr4_mdp_off "${PR4_SHA}" pr4 1 mdp_off none 0 10 \
            median 19745.3 NA 106210.19 reserved=101.0566_GiB \
            "PR4 default path with MDP disabled"
        add_cell pr4_mdp_on "${PR4_SHA}" pr4 1 mdp_on none 0 10 \
            median 18910.4 NA 110899.40 reserved=76.6074_GiB \
            "PR4 CP-local MDP"
        ;;
    pr5)
        add_cell pr5_mdp_off "${PR5_SHA}" pr5 2 mdp_off none 0 10 \
            median 37782.3 NA 55506.20 reserved=111.7617_GiB \
            "PR5 generic PP sidecar, MDP disabled"
        add_cell pr5_mdp_on "${PR5_SHA}" pr5 2 mdp_on none 0 10 \
            median 29419.9 NA 71283.45 reserved=138.6836_GiB \
            "PR5 replicated PPxCP vision"
        ;;
    pr6)
        # The default set is the safe world64 report. The reference numbers
        # were measured at cf12; 4403202 adds the validated PR6 quality fix.
        add_cell pr6_mdp_off "${PR6_CODE_SHA}" pr6 2 mdp_off recompute 0 6 \
            mean 30481.0 31.26 68801.94 allocated=96974.57_MB \
            "cf12 exact-tip world64 rerun reference"
        add_cell pr6_mdp_on "${PR6_CODE_SHA}" pr6 2 mdp_on recompute 0 6 \
            mean 24086.0 39.71 87069.33 allocated=127735.80_MB \
            "cf12 exact-tip world64 rerun reference"
        add_cell pr6_fused_retain_131072 "${PR6_CODE_SHA}" pr6 2 mdp_fused retain 131072 6 \
            mean 13030.0 74.25 160947.97 allocated=128383.62_MB \
            "safe-cap exact-tip world64 rerun reference"
        add_cell pr6_fused_recompute_131072 "${PR6_CODE_SHA}" pr6 2 mdp_fused recompute 131072 6 \
            mean 12775.0 74.79 164160.63 allocated=92029.23_MB \
            "safe-cap exact-tip world64 rerun reference"
        ;;
esac

cell_selected() {
    local label="$1"
    [[ -z "${selected_cells}" ]] && return 0
    [[ ",${selected_cells}," == *",${label},"* ]]
}

required_paths=("${source_repo}" "${energon_host}" "${data_host}" "${raw_host}" "${venv_host}" "${container}" "${blend_host}")
for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done
git -C "${source_repo}" rev-parse --git-dir >/dev/null

mkdir -p "${checkout_root}" "${result_root}"

prepare_checkout() {
    local sha="$1"
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

manifest="${result_root}/manifest.tsv"
printf 'cell\tsha\tstack_level\tpp\tmode\tbackward\tcap\tmeasure_from\treference_stat\treference_step_ms\treference_tflops\treference_padded_tps\treference_peak\treference_note\n' > "${manifest}"

selected_count=0
declare -a checkouts=()
for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    if ! cell_selected "${label}"; then
        checkouts+=("")
        continue
    fi
    checkout="$(prepare_checkout "${shas[$index]}")"
    checkouts+=("${checkout}")
    selected_count=$((selected_count + 1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${shas[$index]}" "${levels[$index]}" "${pps[$index]}" \
        "${modes[$index]}" "${backward_modes[$index]}" "${caps[$index]}" \
        "${measure_from[$index]}" "${reference_stats[$index]}" "${reference_steps_ms[$index]}" \
        "${reference_tflops[$index]}" "${reference_padded_tps[$index]}" \
        "${reference_peaks[$index]}" "${reference_notes[$index]}" >> "${manifest}"
done
if [[ "${selected_count}" -eq 0 ]]; then
    echo "No cells selected. Available: ${labels[*]}" >&2
    exit 2
fi

git -C "${REPO_ROOT}" rev-parse HEAD > "${result_root}/harness_revision.txt"
git -C "${REPO_ROOT}" diff -- examples/multimodal_dev/scripts/gb200_report_repro \
    examples/multimodal_dev/README.md > "${result_root}/harness_working_tree.patch"
printf '%s\n' "${report}" > "${result_root}/report.txt"
printf '%s\n' "${container}" > "${result_root}/container.txt"
printf '%s\n' "${blend_host}" > "${result_root}/blend_path.txt"
sha256sum "${blend_host}" > "${result_root}/blend_sha256.txt"
printf '%s\n' "${PR1_SHA}" "${PR2_SHA}" "${PR4_SHA}" "${PR5_SHA}" "${PR6_CODE_SHA}" > "${result_root}/stack_code_revisions.txt"

if [[ "${prepare_only}" == "1" ]]; then
    echo "Prepared ${selected_count} cells under ${checkout_root}"
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
cell_number=0
for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    if ! cell_selected "${label}"; then
        continue
    fi
    checkout="${checkouts[$index]}"
    cell_host="${result_root}/${label}"
    mkdir -p "${cell_host}"
    rm -f "${cell_host}/SUCCESS" "${cell_host}/FAILED"
    port=$(( ${MASTER_PORT_BASE:-29600} + cell_number ))
    cell_number=$((cell_number + 1))

    echo "Starting ${label}: sha=${shas[$index]} pp=${pps[$index]} mode=${modes[$index]}"
    if [[ "${dry_run}" == "1" ]]; then
        CELL="${label}" STACK_LEVEL="${levels[$index]}" PP_SIZE="${pps[$index]}" \
            MDP_MODE="${modes[$index]}" FUSED_BACKWARD="${backward_modes[$index]}" \
            VISION_MAX_SEQUENCE_LENGTH="${caps[$index]}" MASTER_ADDR=localhost \
            MASTER_PORT="${port}" NNODES=16 SLURM_NODEID=0 DRY_RUN=1 \
            TARGET_REPO="${checkout}" RESULT_ROOT="${result_root}" \
            bash "${SCRIPT_DIR}/run_node.sh" | tee "${cell_host}/dry_run.log"
        touch "${cell_host}/SUCCESS"
        continue
    fi

    mounts="${checkout}:/workspace/Megatron-LM"
    mounts+=",${SCRIPT_DIR}:/workspace/repro"
    mounts+=",${energon_host}:/workspace/Megatron-Energon-7.3.2"
    mounts+=",${data_host}:/data"
    mounts+=",${raw_host}:/raw"
    mounts+=",${venv_host}:/workspace/venv"
    mounts+=",${result_root}:/workspace/results"

    cell_rc=0
    CELL="${label}" STACK_LEVEL="${levels[$index]}" PP_SIZE="${pps[$index]}" \
        MDP_MODE="${modes[$index]}" FUSED_BACKWARD="${backward_modes[$index]}" \
        VISION_MAX_SEQUENCE_LENGTH="${caps[$index]}" MASTER_ADDR="${master_addr}" \
        MASTER_PORT="${port}" NNODES=16 TRAIN_ITERS="${TRAIN_ITERS:-45}" \
        WARMUP_ITERS="${WARMUP_ITERS:-5}" DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}" \
        CELL_TIMEOUT_SECONDS="${CELL_TIMEOUT_SECONDS:-5400}" \
        srun --jobid="${job_id}" --overlap \
        --nodes=16 --ntasks=16 --ntasks-per-node=1 \
        --kill-on-bad-exit=1 --mpi=pmix \
        --output="${cell_host}/launcher_%t.log" \
        --error="${cell_host}/launcher_%t.err" \
        --container-image="${container}" \
        --container-mounts="${mounts}" \
        --container-workdir=/workspace/Megatron-LM \
        bash /workspace/repro/run_node.sh || cell_rc=$?

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
    python3 "${SCRIPT_DIR}/summarize_report.py" "${result_root}"
fi
echo "RESULT_ROOT=${result_root}"
exit "${overall_rc}"
