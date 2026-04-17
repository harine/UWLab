#!/bin/bash
set -euo pipefail

# ============================================================
# Eval sweep: 3 architectures × 2 horizons on peg insertion
#
# For each checkpoint, runs 100 rollouts with:
#   execute_horizon: 1, n_action_steps/4, n_action_steps/2
#   temporal_ensemble: on, off
# = 6 configs per model, 36 runs total
# ============================================================

EVAL_SCRIPT="scripts_v2/tools/eval_distilled_policy.py"
TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0"
NUM_ENVS=32
NUM_TRAJ=100
CKPT_BASE="/home/harine/UWLab/diffusion_policy/outputs/train_ablation"
RESULTS_BASE="outputs/eval_results"

MODELS=(mlp_h8 mlp_h16 transformer_h8 transformer_h16 unet_h8 unet_h16)
declare -A ACTION_STEPS=( [h8]=8 [h16]=16 )

run_eval() {
    local tag="$1"
    local ckpt="$2"
    local eh="$3"
    local te="$4"
    local out_dir="${RESULTS_BASE}/${tag}"

    if [[ -f "${out_dir}/stats.json" ]]; then
        echo "SKIP ${tag}: stats.json already exists at ${out_dir}"
        return 0
    fi

    mkdir -p "${out_dir}"

    echo ""
    echo "========================================"
    echo "RUN: ${tag}"
    echo "  checkpoint       : ${ckpt}"
    echo "  execute_horizon  : ${eh}"
    echo "  temporal_ensemble: ${te}"
    echo "========================================"

    local extra_args=()
    if [[ "${eh}" != "none" ]]; then
        extra_args+=(--execute_horizon "${eh}")
    fi
    if [[ "${te}" == "true" ]]; then
        extra_args+=(--temporal_ensemble)
    fi

    python "${EVAL_SCRIPT}" \
        --task "${TASK}" \
        --checkpoint "${ckpt}" \
        --num_envs "${NUM_ENVS}" \
        --num_trajectories "${NUM_TRAJ}" \
        --headless \
        --enable_cameras \
        --save_video \
        --output_dir "${out_dir}" \
        "${extra_args[@]}" \
        env.scene.insertive_object=peg \
        env.scene.receptive_object=peghole \
        2>&1 | tee "${out_dir}/eval.log"

    echo "---- ${tag} done ----"
}

for model in "${MODELS[@]}"; do
    ckpt="${CKPT_BASE}/${model}/checkpoints/latest.ckpt"
    if [[ ! -f "${ckpt}" ]]; then
        echo "SKIPPING ${model}: ${ckpt} not found"
        continue
    fi

    hsuffix="${model##*_}"
    n_act="${ACTION_STEPS[${hsuffix}]}"
    half=$(( n_act / 2 ))
    quarter=$(( n_act / 4 ))

    # without temporal ensemble
    run_eval "${model}_no_te_eh1"         "${ckpt}" "1"         "false"
    run_eval "${model}_no_te_eh${quarter}" "${ckpt}" "${quarter}" "false"
    run_eval "${model}_no_te_eh${half}"   "${ckpt}" "${half}"   "false"

    # with temporal ensemble
    run_eval "${model}_te_eh1"         "${ckpt}" "1"         "true"
    run_eval "${model}_te_eh${quarter}" "${ckpt}" "${quarter}" "true"
    run_eval "${model}_te_eh${half}"   "${ckpt}" "${half}"   "true"
done

# ── Print results table ──
echo ""
echo "============================================"
echo "ALL EVAL RUNS COMPLETE"
echo "============================================"
printf "\n%-40s %8s %8s %10s\n" "Run" "Success" "Total" "Rate(%)"
printf "%-40s %8s %8s %10s\n"  "---" "-------" "-----" "-------"
for d in $(ls -d "${RESULTS_BASE}"/*/ 2>/dev/null | sort); do
    tag="$(basename "${d}")"
    json="${d}stats.json"
    if [[ -f "${json}" ]]; then
        succ=$(python3 -c "import json; d=json.load(open('${json}')); print(d.get('successful_trajectories','?'))")
        total=$(python3 -c "import json; d=json.load(open('${json}')); print(d.get('total_trajectories','?'))")
        rate=$(python3 -c "import json; d=json.load(open('${json}')); print(f\"{d.get('success_rate',0):.2f}\")")
        printf "%-40s %8s %8s %10s\n" "${tag}" "${succ}" "${total}" "${rate}"
    else
        printf "%-40s %8s\n" "${tag}" "NO STATS"
    fi
done
