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
  "peg_expert_state:/gscratch/scrubbed/sidhraja/datasets/peg_expert_state"
#  "peg_expert_10k_noise005:${REPO_ROOT}/data/peg_expert_10k_noise005"
)
if [[ ! -d "${DP_ROOT}" ]]; then
	  echo "Error: diffusion_policy root not found at ${DP_ROOT}" >&2
	    exit 1
    fi

    IFS=',' read -r -a CHUNK_SIZE_ARRAY <<< "${CHUNK_SIZES}"

    if [[ ${#CHUNK_SIZE_ARRAY[@]} -eq 0 ]]; then
	      echo "Error: no chunk sizes provided in CHUNK_SIZES" >&2
	        exit 1
	fi

	# Choose a single chunk size for this run.
	if [[ -n "${ACTION_CHUNK:-}" ]]; then
		  SELECTED_CHUNK_SIZE="${ACTION_CHUNK}"
	  elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
		    if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CHUNK_SIZE_ARRAY[@]} )); then
			        echo "Error: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range for CHUNK_SIZES=${CHUNK_SIZES}" >&2
				    exit 1
				      fi
				        SELECTED_CHUNK_SIZE="${CHUNK_SIZE_ARRAY[$SLURM_ARRAY_TASK_ID]}"
				else
					  echo "Error: set ACTION_CHUNK or run under a Slurm array with SLURM_ARRAY_TASK_ID" >&2
					    echo "Example: CHUNK_SIZES=\"1,3,8,16\" SLURM_ARRAY_TASK_ID=0 bash $0" >&2
					      exit 1
				      fi

				      echo "Selected action chunk size: ${SELECTED_CHUNK_SIZE}"

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
									        echo "Chunk:   ${SELECTED_CHUNK_SIZE}"
										  echo "============================================================"

										    python train.py --config-name=train_diffusion_unet_uwlab_state_workspace \
											        task.dataset_dir="${dataset_dir}" \
												    n_action_steps="${SELECTED_CHUNK_SIZE}" \
												        exp_name="chunk_sweep_${dataset_name}_chunk${SELECTED_CHUNK_SIZE}" \
													    "$@"
									    done
