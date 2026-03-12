"""
Evaluate Q-function robustness by adding Gaussian noise to trajectory actions
and plotting mean Q value vs. noise level (sigma).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from models import QFunction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to trajectory actions and plot Q value vs. noise level.",
    )
    parser.add_argument("--q_checkpoint", type=str, required=True, help="Path to trained Q-function checkpoint (.pt).")
    parser.add_argument(
        "--trajectory_path",
        type=str,
        required=True,
        help="Path to dataset root (directory containing trajectories/ and manifest.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output plot. Defaults to <trajectory_path>/q_noise_eval.",
    )
    parser.add_argument(
        "--noise_levels",
        type=str,
        default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0",
        help="Comma-separated noise std values (sigma) for Gaussian noise on actions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference (e.g. cuda or cpu).",
    )
    parser.add_argument("--max_trajs", type=int, default=0, help="Cap number of trajectories. 0 = all.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise.")
    return parser.parse_args()


def _resolve_dataset_paths(dataset_path: Path) -> tuple[Path, Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise ValueError(f"Expected dataset directory, got: {dataset_path}")
    traj_dir = dataset_path / "trajectories"
    manifest_path = dataset_path / "manifest.json"
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"Expected trajectories directory at: {traj_dir}")
    return traj_dir, manifest_path


def _resolve_trajectory_files(traj_dir: Path) -> list[Path]:
    traj_files = sorted(traj_dir.glob("traj_*.pt"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files matching traj_*.pt under {traj_dir}")
    return traj_files


def _to_float_tensor_2d(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        raise ValueError(f"{name} must have a time dimension, got scalar.")
    if value.ndim == 1:
        return value.view(-1, 1).to(dtype=torch.float32)
    return value.reshape(value.shape[0], -1).to(dtype=torch.float32)


def _load_q_model(checkpoint_path: Path, device: torch.device) -> tuple[QFunction, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict at {checkpoint_path}, got {type(checkpoint)}.")
    for key in ("model_state_dict", "state_dim", "action_chunk_dim", "hidden_dims"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key '{key}': {checkpoint_path}")

    model = QFunction(
        state_dim=int(checkpoint["state_dim"]),
        action_chunk_dim=int(checkpoint["action_chunk_dim"]),
        hidden_dims=checkpoint["hidden_dims"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _infer_chunk_size(action_chunk_dim: int, action_dim: int) -> int:
    if action_dim <= 0:
        raise ValueError(f"action_dim must be > 0, got {action_dim}")
    if action_chunk_dim % action_dim != 0:
        raise ValueError(
            f"Incompatible action dims: action_chunk_dim={action_chunk_dim} is not divisible by action_dim={action_dim}"
        )
    chunk_size = action_chunk_dim // action_dim
    if chunk_size <= 0:
        raise ValueError(f"Inferred non-positive chunk_size={chunk_size}")
    return chunk_size


def _prepare_traj_tensors(
    data: dict[str, Any],
    expected_state_dim: int,
    action_chunk_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract aligned states and actions and chunk_size. Raises on mismatch or missing keys."""
    if "obs_flat" not in data or "actions" not in data:
        raise KeyError("Trajectory must contain both 'obs_flat' and 'actions'.")
    obs_flat = _to_float_tensor_2d(data["obs_flat"], "obs_flat")
    actions = _to_float_tensor_2d(data["actions"], "actions")

    aligned_steps = min(obs_flat.shape[0], actions.shape[0])
    if aligned_steps <= 0:
        raise ValueError("No aligned timesteps.")
    obs_flat = obs_flat[:aligned_steps]
    actions = actions[:aligned_steps]

    if obs_flat.shape[1] != expected_state_dim:
        raise ValueError(
            f"State dim mismatch: traj={obs_flat.shape[1]} checkpoint={expected_state_dim}"
        )

    action_dim = int(actions.shape[1])
    chunk_size = _infer_chunk_size(action_chunk_dim=action_chunk_dim, action_dim=action_dim)
    valid_steps = aligned_steps - chunk_size + 1
    if valid_steps <= 0:
        raise ValueError(
            f"Trajectory too short for chunk_size={chunk_size}: aligned_steps={aligned_steps}"
        )

    states = obs_flat[:valid_steps]
    return states, actions, chunk_size


def _compute_mean_q_for_traj_at_sigmas(
    model: QFunction,
    states: torch.Tensor,
    actions: torch.Tensor,
    chunk_size: int,
    action_chunk_dim: int,
    device: torch.device,
    noise_levels: list[float],
    rng: np.random.Generator,
) -> list[float]:
    """For one trajectory, compute mean Q at each sigma (Gaussian noise on actions). Returns one mean Q per sigma."""
    valid_steps = states.shape[0]
    results: list[float] = []
    for sigma in noise_levels:
        if sigma > 0:
            noise = torch.randn(actions.shape[1]) * sigma
            actions_noisy = actions + noise[None, :]
        else:
            actions_noisy = actions
        action_chunks = actions_noisy.unfold(0, chunk_size, 1).reshape(valid_steps, -1)
        if action_chunks.shape[1] != action_chunk_dim:
            raise ValueError(
                f"Action chunk dim mismatch: got {action_chunks.shape[1]}, expected {action_chunk_dim}"
            )
        with torch.inference_mode():
            q_values = (
                model(states.to(device), action_chunks.to(device))
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
        results.append(float(np.mean(q_values)))
    return results


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    checkpoint_path = Path(args.q_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Q checkpoint not found: {checkpoint_path}")

    dataset_path = Path(args.trajectory_path)
    traj_dir, _ = _resolve_dataset_paths(dataset_path)
    traj_files = _resolve_trajectory_files(traj_dir)
    if args.max_trajs > 0:
        traj_files = traj_files[: args.max_trajs]

    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]
    if not noise_levels:
        raise ValueError("At least one noise level required in --noise_levels")

    if args.output_dir is None:
        output_dir = dataset_path / "q_noise_eval"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint = _load_q_model(checkpoint_path, device)
    expected_state_dim = int(checkpoint["state_dim"])
    action_chunk_dim = int(checkpoint["action_chunk_dim"])

    # For each sigma, collect mean Q from every trajectory (one value per traj)
    mean_q_per_sigma: list[list[float]] = [[] for _ in noise_levels]

    for traj_path in tqdm(traj_files, desc="Trajectories"):
        try:
            data = torch.load(traj_path, map_location="cpu", weights_only=False)
            if not isinstance(data, dict):
                continue
            states, actions, chunk_size = _prepare_traj_tensors(
                data=data,
                expected_state_dim=expected_state_dim,
                action_chunk_dim=action_chunk_dim,
            )
            mean_q_list = _compute_mean_q_for_traj_at_sigmas(
                model=model,
                states=states,
                actions=actions,
                chunk_size=chunk_size,
                action_chunk_dim=action_chunk_dim,
                device=device,
                noise_levels=noise_levels,
                rng=rng,
            )
            for sigma_idx, mean_q in enumerate(mean_q_list):
                mean_q_per_sigma[sigma_idx].append(mean_q)
        except Exception:
            continue

    mean_q_per_noise = np.array(
        [np.mean(q_list) if q_list else np.nan for q_list in mean_q_per_sigma]
    )
    std_q_per_noise = np.array(
        [np.std(q_list) if q_list else np.nan for q_list in mean_q_per_sigma]
    )
    noise_levels_arr = np.array(noise_levels)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        noise_levels_arr,
        mean_q_per_noise,
        yerr=std_q_per_noise,
        capsize=4,
        capthick=1,
        marker="o",
        markersize=6,
        linestyle="-",
        linewidth=1.5,
    )
    plt.xlabel("Noise level (Gaussian std σ on actions)")
    plt.ylabel("Mean Q value")
    plt.title("Q value vs. action noise level")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "q_vs_noise.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to {out_path}")
    print(f"[INFO] Noise levels: {noise_levels}")
    print(f"[INFO] Mean Q per noise: {mean_q_per_noise.tolist()}")


if __name__ == "__main__":
    main()
