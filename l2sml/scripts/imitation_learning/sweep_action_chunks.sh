#!/usr/bin/env bash
set -euo pipefail

# Runs diffusion-policy training sweeps over action chunk sizes for two datasets:
#   - data/peg_expert
#   - data/peg_expert_10k_noise005
#
# Usage:
#   bash l2sml/scripts/imitation_learning/sweep_action_chunks.sh
#
# Optional:
#   CHUNK_SIZES="1,3,8,16" bash l2sml/scripts/imitation_learning/sweep_action_chunks.sh
#   bash l2sml/scripts/imitation_learning/sweep_action_chunks.sh training.num_epochs=1000
#
# Any extra arguments are forwarded to train.py as Hydra overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DP_ROOT="${REPO_ROOT}/l2sml/diffusion_policy"

CHUNK_SIZES="${CHUNK_SIZES:-1,3,8,16}"

DATASETS=(
  "peg_expert:${REPO_ROOT}/data/peg_expert"
  "peg_expert_10k_noise005:${REPO_ROOT}/data/peg_expert_10k_noise005"
)

if [[ ! -d "${DP_ROOT}" ]]; then
  echo "Error: diffusion_policy root not found at ${DP_ROOT}" >&2
  exit 1
fi

cd "${DP_ROOT}"

for entry in "${DATASETS[@]}"; do
  dataset_name="${entry%%:*}"
  dataset_dir="${entry#*:}"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "Skipping ${dataset_name}: dataset directory not found at ${dataset_dir}" >&2
    continue
  fi

  echo
  echo "============================================================"
  echo "Dataset: ${dataset_name}"
  echo "Path:    ${dataset_dir}"
  echo "Chunks:  ${CHUNK_SIZES}"
  echo "============================================================"

  python train.py --config-name=train_diffusion_unet_uwlab_state_workspace -m \
    task.dataset_dir="${dataset_dir}" \
    n_action_steps="${CHUNK_SIZES}" \
    exp_name="chunk_sweep_${dataset_name}" \
    "$@"
done
