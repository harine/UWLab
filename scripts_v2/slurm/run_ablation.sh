#!/bin/bash
# Master script: runs training then evaluation for peg insertion ablation.
# Usage: bash scripts_v2/slurm/run_ablation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========== TRAINING =========="
bash "${SCRIPT_DIR}/train_ablation.sh"

echo ""
echo "========== EVALUATION =========="
bash "${SCRIPT_DIR}/eval_ablation.sh"
