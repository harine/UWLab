#!/usr/bin/env bash
set -euo pipefail

# Horizons to test
HORIZONS=(1 2 4 8)

# Directory for logs
LOG_DIR="outputs/evals"
mkdir -p "$LOG_DIR"

# Optional run identifier so files from different runs do not collide
RUN_ID="$(date +%Y%m%d_%H%M%S)"

for H in "${HORIZONS[@]}"; do
    OUT_FILE="${LOG_DIR}/eval_execute_horizon_${H}_${RUN_ID}.log"

    echo "Running execute_horizon=${H}"
    echo "Writing output to ${OUT_FILE}"

    python scripts_v2/tools/eval_distilled_policy.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
        --checkpoint diffusion_policy/data/outputs/2026.04.02/13.45.16_train_mlp_image_aux_sim2real_state_privledged/checkpoints/step_0050000.ckpt \
        --num_envs 32 \
        --num_trajectories 100 \
        --headless \
        env.scene.insertive_object=peg \
        env.scene.receptive_object=peghole \
        --execute_horizon "$H" \
        > "$OUT_FILE" 2>&1
done

echo "All runs finished."