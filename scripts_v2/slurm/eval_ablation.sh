#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/eval_ablation"
RESULTS_DIR="${REPO_ROOT}/outputs/eval_ablation"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${NUM_GPUS} GPU(s)"

# ── Grid: 6 trained models × 2 ensemble settings = 12 runs ──
TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DataCollection-v0"
NUM_ENVS=64
NUM_TRAJ=200

configs=(
    "train_mlp_sim2real_state_workspace"
    "train_transformer_sim2real_state_workspace"
    "train_diffusion_sim2real_state_workspace"
)
action_steps=(8 16)

jobs_config=()
jobs_act=()
jobs_ens_flag=()
jobs_ens_label=()

for config in "${configs[@]}"; do
    for n_act in "${action_steps[@]}"; do
        for ens in "none" "temporal"; do
            jobs_config+=("${config}")
            jobs_act+=("${n_act}")
            if [ "${ens}" = "temporal" ]; then
                jobs_ens_flag+=("--temporal_ensemble")
                jobs_ens_label+=("temporal_ensemble")
            else
                jobs_ens_flag+=("")
                jobs_ens_label+=("no_ensemble")
            fi
        done
    done
done

NUM_JOBS=${#jobs_config[@]}

run_eval() {
    local idx=$1
    local gpu=$2
    local config=${jobs_config[$idx]}
    local n_act=${jobs_act[$idx]}
    local ens_flag=${jobs_ens_flag[$idx]}
    local ens_label=${jobs_ens_label[$idx]}
    local exp="h${n_act}"
    local run_name="${config}_${exp}_${ens_label}"

    local ckpt_dir="${REPO_ROOT}/outputs/train_ablation/${config}/${exp}/checkpoints"
    local ckpt
    ckpt=$(ls -t "${ckpt_dir}"/latest.ckpt 2>/dev/null | head -1)
    if [ -z "${ckpt}" ]; then
        ckpt=$(ls -t "${ckpt_dir}"/*.ckpt 2>/dev/null | head -1)
    fi
    if [ -z "${ckpt}" ]; then
        echo "[Job ${idx}] SKIP: no checkpoint in ${ckpt_dir}"
        return 1
    fi

    local run_dir="${RESULTS_DIR}/${run_name}"
    local log="${LOG_DIR}/${run_name}.log"
    mkdir -p "${run_dir}"

    echo "[Job ${idx}] GPU=${gpu}  ${run_name}  ckpt=${ckpt}"

    # Run from per-run dir so --save_video writes there
    cd "${run_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${REPO_ROOT}/scripts_v2/tools/eval_distilled_policy.py" \
        --task "${TASK}" \
        --checkpoint "${ckpt}" \
        --num_envs "${NUM_ENVS}" \
        --num_trajectories "${NUM_TRAJ}" \
        --seed 42 \
        --save_video \
        --headless \
        ${ens_flag} \
        > "${log}" 2>&1
    cd "${REPO_ROOT}"
}

pids=()

for ((i=0; i<NUM_JOBS; i++)); do
    gpu=$((i % NUM_GPUS))

    if [ ${#pids[@]} -ge ${NUM_GPUS} ]; then
        wait_idx=$((i - NUM_GPUS))
        echo "Waiting for Job ${wait_idx} (PID ${pids[$wait_idx]})..."
        wait "${pids[$wait_idx]}" || true
        echo "Job ${wait_idx} done."
    fi

    run_eval "${i}" "${gpu}" &
    pids+=($!)
done

echo "Waiting for remaining jobs..."
for ((i=0; i<NUM_JOBS; i++)); do
    wait "${pids[$i]}" 2>/dev/null || true
done

# ── Collect summary stats ──
echo ""
echo "============================================"
echo "RESULTS SUMMARY"
echo "============================================"
SUMMARY="${RESULTS_DIR}/summary.txt"
: > "${SUMMARY}"
for log in "${LOG_DIR}"/*.log; do
    name=$(basename "${log}" .log)
    echo "--- ${name} ---" >> "${SUMMARY}"
    grep -E "Success rate|Successful trajectories|Total trajectories|Average Metrics" "${log}" >> "${SUMMARY}" 2>/dev/null || echo "  (no results found)" >> "${SUMMARY}"
    echo "" >> "${SUMMARY}"
done
cat "${SUMMARY}"

echo ""
echo "All ${NUM_JOBS} eval runs complete."
echo "Logs:    ${LOG_DIR}/"
echo "Videos:  ${RESULTS_DIR}/<run_name>/outputs/videos/"
echo "Summary: ${SUMMARY}"
