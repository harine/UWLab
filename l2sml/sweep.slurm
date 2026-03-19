#!/bin/bash
#SBATCH --job-name=sweep_action_chunks
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate l2sml_diffusion

export CHUNK_SIZES="1,3,8,16"

bash /gscratch/weirdlab/sidhraja/isaac_projects/l2sml/UWLab/l2sml/scripts/imitation_learning/sweep_action_chunks.sh
