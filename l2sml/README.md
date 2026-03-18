# l2sml — Learning from Simulation: Machine Learning

This directory contains the full pipeline for training and evaluating imitation learning policies (specifically diffusion policy) on UWLab tasks, using expert data collected from RL policies.

## Pipeline Overview

```
1. Collect expert data  →  2. Analyze / inspect data  →  3. Train diffusion policy  →  4. Evaluate policy
```

---

## 1. Collecting Expert Data

Expert trajectories are collected by rolling out a pre-trained RL policy in simulation.

### Basic usage

```bash
python l2sml/scripts/collect_expert_data.py
```

This uses the defaults from `l2sml/configs/collect_expert_data.yaml`:
- **Task:** `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Collect-v0`
- **Checkpoint:** `peg_state_rl_expert.pt`
- **Output:** `data/peg_expert/`
- **Trajectories:** 1000, with 10 parallel environments

### Common overrides

```bash
# Collect 500 trajectories using 20 parallel environments
python l2sml/scripts/collect_expert_data.py \
    --num_trajectories 500 \
    --num_envs 20 \
    --output_dir data/peg_expert_500

# Collect with rendered camera images (needed for image-based policy)
python l2sml/scripts/collect_expert_data.py \
    --capture_rendered_images \
    --output_dir data/peg_expert_with_images
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Collect-v0` | Gym task name |
| `--checkpoint` | `peg_state_rl_expert.pt` | RL expert checkpoint |
| `--output_dir` | `data/peg_expert` | Where to save trajectories |
| `--num_trajectories` | `1000` | Total trajectories to collect |
| `--horizon` | `800` | Max steps per episode |
| `--num_envs` | `10` | Parallel environments |
| `--capture_rendered_images` | `false` | Save camera RGB frames |
| `--headless` | `true` | Run without GUI |

### Output format

Each collected episode is saved as `data/peg_expert/trajectories/traj_XXXXXX.pt` containing:

| Key | Description |
|---|---|
| `actions` | `[T, 7]` — Cartesian OSC commands |
| `obs_flat` | `[T, obs_dim]` — flattened observation vector |
| `obs_proprio` | Dict of proprioceptive terms (`prev_actions`, `joint_pos`, `end_effector_pose`) |
| `obs_assets` | Dict of object-state terms (`insertive_asset_pose`, etc.) |
| `obs_images` | Dict of camera observations (if `--capture_rendered_images`) |
| `rewards` | `[T]` — per-step reward |
| `terminated` | `[T]` — episode termination flags |

A `manifest.json` file is written alongside the trajectories summarizing the dataset.

---

## 2. Analyzing and Viewing the Data

### Analyze dataset statistics and generate plots

```bash
python l2sml/scripts/analyze_expert_trajectories.py \
    --dataset_dir data/peg_expert \
    --output_dir data/peg_expert/analysis
```

This produces:
- `summary.json` — aggregate statistics (mean/std/min/max of episode length, return, done rate)
- `trajectory_table.csv` — per-trajectory breakdown
- `plots/steps_hist.png` — episode length distribution
- `plots/return_hist.png` — return distribution
- `plots/action_l2_hist.png` — action magnitude distribution
- `plots/action_dim_stats.png` — per-dimension action mean/std
- `plots/reward_per_step.png` — per-step reward curves
- `plots/cumulative_reward.png` — cumulative reward curves

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | `data/peg_expert_with_images` | Path to dataset |
| `--output_dir` | `<dataset_dir>/analysis` | Where to write outputs |
| `--num_videos` | `10` | Number of trajectory videos to create |
| `--video_fps` | `20` | Video frame rate |
| `--max_trajs` | `0` (all) | Cap on trajectories (debug) |

### Inspect trajectory file structure

To print the shapes, dtypes, and nested keys of a single `.pt` file:

```bash
python l2sml/scripts/data_collect/inspect_uwlab_obs_images.py \
    --dataset_dir data/peg_expert \
    --traj 0
```

### Create multi-camera videos

To tile `front_rgb`, `side_rgb`, and `wrist_rgb` side-by-side into `.mp4` files:

```bash
python l2sml/scripts/data_collect/make_traj_video.py \
    --dataset_dir data/peg_expert_with_images \
    --num_trajs 10 \
    --fps 15 \
    --out_dir data/peg_expert_with_images/videos
```

This requires that the dataset was collected with `--capture_rendered_images`.

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | `data/peg_expert_100` | Path to dataset |
| `--num_trajs` | `5` | How many trajectories to render |
| `--fps` | `10` | Video frame rate |
| `--camera_keys` | `front_rgb side_rgb wrist_rgb` | Which cameras to include |

---

## 3. Setting Up and Training the Diffusion Policy

### Setup

The diffusion policy library lives at `l2sml/diffusion_policy/`. Install it in your active conda environment:

```bash
cd l2sml/diffusion_policy
pip install -e .
```

You will also need the standard diffusion policy dependencies (PyTorch, hydra-core, wandb, etc.). Refer to the upstream [diffusion policy README](diffusion_policy/README.md) for a full dependency list.

### Training a state-based policy

From the `l2sml/diffusion_policy/` directory, run:

```bash
cd l2sml/diffusion_policy
python train.py --config-name=train_diffusion_unet_uwlab_state_workspace
```

The config at `diffusion_policy/config/train_diffusion_unet_uwlab_state_workspace.yaml` picks up the task config from `diffusion_policy/config/task/uwlab_peg_state.yaml`. Key parameters:

| Parameter | Value | Description |
|---|---|---|
| `obs_dim` | 225 | Flattened obs vector (5-step history) |
| `action_dim` | 7 | 7-DOF Cartesian OSC commands |
| `horizon` | 32 | Prediction horizon |
| `n_obs_steps` | 2 | Obs history fed to policy |
| `n_action_steps` | 16 | Action chunk executed per inference |
| `batch_size` | 256 | Training batch size |
| `num_epochs` | 3000 | Training epochs |
| `checkpoint_every` | 50 | Save checkpoint frequency |

The obs vector (225-dim) is built from a 5-step history of:
- `prev_actions` (7-dim) × 5 = 35
- `joint_pos` (14-dim) × 5 = 70
- `end_effector_pose` (6-dim) × 5 = 30
- `insertive_asset_pose` (6-dim) × 5 = 30
- `receptive_asset_pose` (6-dim) × 5 = 30
- `insertive_asset_in_receptive_asset_frame` (6-dim) × 5 = 30

### Overriding config values

Hydra allows inline config overrides:

```bash
python train.py --config-name=train_diffusion_unet_uwlab_state_workspace \
    task.dataset_path=data/peg_expert_500 \
    training.num_epochs=1000 \
    training.batch_size=512
```

### Training outputs

Runs are saved to `data/outputs/YYYY.MM.DD/HH.MM.SS_<name>/`:

```
data/outputs/
└── 2026.03.17/
    └── 10.10.56_train_diffusion_unet_uwlab_state_uwlab_peg_state/
        ├── checkpoints/
        │   ├── epoch=0050-train_loss=0.000.ckpt   # top-k by train loss
        │   └── latest.ckpt
        ├── .hydra/config.yaml                      # full resolved config
        └── wandb/                                  # W&B logs
```

Training metrics are logged to [Weights & Biases](https://wandb.ai) under the project `uwlab_diffusion_policy`.

---

## 4. Evaluating the Policy

Roll out a trained checkpoint in the UWLab simulation environment:

```bash
python l2sml/scripts/imitation_learning/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Collect-v0 \
    --checkpoint l2sml/diffusion_policy/data/outputs/2026.03.17/10.10.56_train_diffusion_unet_uwlab_state_uwlab_peg_state/checkpoints/latest.ckpt \
    --num_rollouts 50 \
    --headless
```

To evaluate every checkpoint in a folder (useful for sweeping over training epochs):

```bash
python l2sml/scripts/imitation_learning/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Collect-v0 \
    --checkpoint l2sml/diffusion_policy/data/outputs/.../checkpoints/ \
    --num_rollouts 20 \
    --headless
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | — | Gym task name (required) |
| `--checkpoint` | — | Path to `.ckpt` file or directory of checkpoints |
| `--num_envs` | `1` | Parallel environments |
| `--num_rollouts` | `10` | Episodes to evaluate |
| `--output_dir` | `data/eval_results` | Where to save evaluation trajectories |
| `--horizon` | `800` | Max steps per episode |
| `--use_ema` / `--no_ema` | `true` | Use EMA weights if available |
| `--headless` | `false` | Run without GUI |
| `--seed` | `42` | RNG seed |

### Outputs

Results are saved to `data/eval_results/<checkpoint_name>/`:
- `manifest.json` — success rate and per-episode stats
- `traj_XXXXXX.pt` — per-episode trajectory files

The success criterion is based on the `progress_context` reward term reporting task completion (peg fully inserted).
