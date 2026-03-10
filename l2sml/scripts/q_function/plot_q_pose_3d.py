from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from models import QFunction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 3D trajectories for end effector / insertive / receptive poses, "
            "with end-effector line color determined by action-chunk Q values."
        )
    )
    parser.add_argument("--q_checkpoint", type=str, required=True, help="Path to trained Q-function checkpoint (.pt).")
    parser.add_argument("--trajectory_path", type=str, required=True, help="Path to dataset root directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output plots. Defaults to <dataset_dir>/q_pose_3d_plots.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for model inference, e.g. cpu or cuda.")
    parser.add_argument("--max_trajs", type=int, default=0, help="Optional cap for debugging. 0 means all.")
    parser.add_argument("--dpi", type=int, default=150, help="Output plot DPI.")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap for Q-colored EE line.")
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional color scale minimum for Q values. If omitted, use per-trajectory min.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional color scale maximum for Q values. If omitted, use per-trajectory max.",
    )
    return parser.parse_args()


def _extract_traj_id(path: Path) -> int:
    stem = path.stem
    if "_" not in stem:
        return -1
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return -1


def _resolve_dataset_paths(dataset_path: Path) -> tuple[Path, Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise ValueError(f"Expected dataset directory, got: {dataset_path}")

    traj_dir = dataset_path / "trajectories"
    manifest_path = dataset_path / "manifest.json"
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"Expected trajectories directory at: {traj_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected manifest file at: {manifest_path}")
    return traj_dir, manifest_path


def _resolve_trajectory_files(traj_dir: Path) -> list[Path]:
    traj_files = sorted(traj_dir.glob("traj_*.pt"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files matching traj_*.pt found under {traj_dir}")
    return traj_files


def _load_manifest_success_map(manifest_path: Path) -> dict[int, str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest object at {manifest_path}")
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise ValueError(f"Manifest 'files' must be a list: {manifest_path}")

    status_map: dict[int, str] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        traj_id_raw = item.get("trajectory_id")
        if traj_id_raw is None:
            continue
        try:
            traj_id = int(traj_id_raw)
        except (TypeError, ValueError):
            continue

        success_raw = item.get("success")
        if success_raw is None:
            status_map[traj_id] = "UNKNOWN"
            continue
        try:
            success_val = float(success_raw)
        except (TypeError, ValueError):
            status_map[traj_id] = "UNKNOWN"
            continue
        status_map[traj_id] = "SUCCESS" if success_val > 0.5 else "FAILURE"
    return status_map


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
    for key in ("model_state_dict", "state_dim", "action_chunk_dim", "hidden_dim"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key '{key}': {checkpoint_path}")

    model = QFunction(
        state_dim=int(checkpoint["state_dim"]),
        action_chunk_dim=int(checkpoint["action_chunk_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _extract_xyz_from_pose_tensor(pose_tensor: Any, name: str) -> torch.Tensor:
    pose = _to_float_tensor_2d(pose_tensor, name)
    if pose.shape[1] < 3:
        raise ValueError(f"{name} must have at least 3 dims, got shape={tuple(pose.shape)}.")
    return pose[:, :3]


def _extract_xyz_tracks(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preferred_keys = ("ee_positions", "insertive_positions", "receptive_positions")
    if all(k in data for k in preferred_keys):
        return (
            _extract_xyz_from_pose_tensor(data["ee_positions"], "ee_positions"),
            _extract_xyz_from_pose_tensor(data["insertive_positions"], "insertive_positions"),
            _extract_xyz_from_pose_tensor(data["receptive_positions"], "receptive_positions"),
        )

    obs_proprio = data.get("obs_proprio")
    obs_assets = data.get("obs_assets")
    if not isinstance(obs_proprio, dict) or not isinstance(obs_assets, dict):
        raise KeyError(
            "Trajectory missing both preferred xyz keys and fallback obs dicts "
            "(obs_proprio / obs_assets)."
        )

    ee_key = "end_effector_pose"
    insertive_key = "insertive_asset_pose" if "insertive_asset_pose" in obs_assets else "insertive_object_pose"
    receptive_key = "receptive_asset_pose" if "receptive_asset_pose" in obs_assets else "receptive_object_pose"

    if ee_key not in obs_proprio:
        raise KeyError("obs_proprio is missing 'end_effector_pose'.")
    if insertive_key not in obs_assets:
        raise KeyError("obs_assets is missing insertive pose key (asset/object).")
    if receptive_key not in obs_assets:
        raise KeyError("obs_assets is missing receptive pose key (asset/object).")

    return (
        _extract_xyz_from_pose_tensor(obs_proprio[ee_key], f"obs_proprio.{ee_key}"),
        _extract_xyz_from_pose_tensor(obs_assets[insertive_key], f"obs_assets.{insertive_key}"),
        _extract_xyz_from_pose_tensor(obs_assets[receptive_key], f"obs_assets.{receptive_key}"),
    )


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


def _compute_q_values_for_traj(
    model: QFunction,
    data: dict[str, Any],
    expected_state_dim: int,
    action_chunk_dim: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "obs_flat" not in data or "actions" not in data:
        raise KeyError("Trajectory must contain both 'obs_flat' and 'actions'.")
    obs_flat = _to_float_tensor_2d(data["obs_flat"], "obs_flat")
    actions = _to_float_tensor_2d(data["actions"], "actions")
    ee_xyz, ins_xyz, rec_xyz = _extract_xyz_tracks(data)

    aligned_steps = min(obs_flat.shape[0], actions.shape[0], ee_xyz.shape[0], ins_xyz.shape[0], rec_xyz.shape[0])
    if aligned_steps <= 0:
        raise ValueError("No aligned timesteps available.")

    obs_flat = obs_flat[:aligned_steps]
    actions = actions[:aligned_steps]
    ee_xyz = ee_xyz[:aligned_steps]
    ins_xyz = ins_xyz[:aligned_steps]
    rec_xyz = rec_xyz[:aligned_steps]

    if obs_flat.shape[1] != expected_state_dim:
        raise ValueError(f"State dim mismatch: traj={obs_flat.shape[1]} checkpoint={expected_state_dim}")

    action_dim = int(actions.shape[1])
    chunk_size = _infer_chunk_size(action_chunk_dim=action_chunk_dim, action_dim=action_dim)
    valid_steps = aligned_steps - chunk_size + 1
    if valid_steps <= 0:
        raise ValueError(f"Trajectory too short for chunk_size={chunk_size}: aligned_steps={aligned_steps}")

    states = obs_flat[:valid_steps]
    action_chunks = actions.unfold(0, chunk_size, 1).reshape(valid_steps, -1)
    if action_chunks.shape[1] != action_chunk_dim:
        raise ValueError(
            f"Action chunk dim mismatch after unfold: got {action_chunks.shape[1]}, expected {action_chunk_dim}"
        )

    with torch.inference_mode():
        q_values = model(states.to(device), action_chunks.to(device)).squeeze(-1).detach().cpu().numpy()

    ee_np = ee_xyz[:valid_steps].detach().cpu().numpy()
    ins_np = ins_xyz[:valid_steps].detach().cpu().numpy()
    rec_np = rec_xyz[:valid_steps].detach().cpu().numpy()
    return q_values, ee_np, ins_np, rec_np


def _plot_traj_q_colored_3d(
    traj_id: int,
    status_label: str,
    q_values: np.ndarray,
    ee_xyz: np.ndarray,
    ins_xyz: np.ndarray,
    rec_xyz: np.ndarray,
    out_path: Path,
    cmap_name: str,
    dpi: int,
    vmin: float | None,
    vmax: float | None,
) -> None:
    if q_values.shape[0] < 1:
        raise ValueError("q_values is empty.")
    if ee_xyz.shape[0] != q_values.shape[0]:
        raise ValueError(f"ee_xyz and q_values length mismatch: {ee_xyz.shape[0]} vs {q_values.shape[0]}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if ee_xyz.shape[0] >= 2:
        ee_segments = np.stack([ee_xyz[:-1], ee_xyz[1:]], axis=1)
        cmap = plt.get_cmap(cmap_name)
        q_for_segments = q_values[:-1]
        q_min = float(np.min(q_values)) if vmin is None else float(vmin)
        q_max = float(np.max(q_values)) if vmax is None else float(vmax)
        if np.isclose(q_min, q_max):
            q_max = q_min + 1e-6
        norm = mcolors.Normalize(vmin=q_min, vmax=q_max)

        ee_collection = Line3DCollection(ee_segments, cmap=cmap, norm=norm, linewidth=2.0, alpha=0.95)
        ee_collection.set_array(q_for_segments)
        ax.add_collection(ee_collection)
        cbar = fig.colorbar(ee_collection, ax=ax, fraction=0.03, pad=0.1)
        cbar.set_label("Q(s_t, a_{t:t+k-1})")
    else:
        ax.scatter(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], c=q_values, cmap=cmap_name, s=20, label="end_effector")

    ax.plot(ins_xyz[:, 0], ins_xyz[:, 1], ins_xyz[:, 2], color="tab:orange", alpha=0.8, linewidth=1.4, label="insertive")
    ax.plot(rec_xyz[:, 0], rec_xyz[:, 1], rec_xyz[:, 2], color="tab:green", alpha=0.8, linewidth=1.4, label="receptive")

    ax.scatter(ee_xyz[0, 0], ee_xyz[0, 1], ee_xyz[0, 2], color="tab:blue", s=16, alpha=0.9)
    ax.scatter(ins_xyz[0, 0], ins_xyz[0, 1], ins_xyz[0, 2], color="tab:orange", s=16, alpha=0.9)
    ax.scatter(rec_xyz[0, 0], rec_xyz[0, 1], rec_xyz[0, 2], color="tab:green", s=16, alpha=0.9)

    xyz_all = np.concatenate([ee_xyz, ins_xyz, rec_xyz], axis=0)
    mins = xyz_all.min(axis=0)
    maxs = xyz_all.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Q-Colored 3D Pose Trajectory (traj {traj_id:06d}, {status_label})")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.max_trajs < 0:
        raise ValueError(f"max_trajs must be >= 0, got {args.max_trajs}")
    if args.dpi <= 0:
        raise ValueError(f"dpi must be > 0, got {args.dpi}")
    if args.vmin is not None and args.vmax is not None and args.vmin >= args.vmax:
        raise ValueError(f"Expected vmin < vmax, got vmin={args.vmin}, vmax={args.vmax}")

    checkpoint_path = Path(args.q_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Q checkpoint not found: {checkpoint_path}")

    dataset_path = Path(args.trajectory_path)
    traj_dir, manifest_path = _resolve_dataset_paths(dataset_path)
    traj_files = _resolve_trajectory_files(traj_dir)
    if args.max_trajs > 0:
        traj_files = traj_files[: args.max_trajs]
    if not traj_files:
        raise FileNotFoundError("No trajectory files selected after applying max_trajs.")
    status_map = _load_manifest_success_map(manifest_path)

    if args.output_dir is None:
        output_dir = dataset_path / "q_pose_3d_plots"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint = _load_q_model(checkpoint_path, device)
    expected_state_dim = int(checkpoint["state_dim"])
    action_chunk_dim = int(checkpoint["action_chunk_dim"])

    num_processed = 0
    skipped: list[tuple[str, str]] = []
    for traj_path in traj_files:
        try:
            data = torch.load(traj_path, map_location="cpu", weights_only=False)
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict trajectory, got {type(data)}")

            q_values, ee_xyz, ins_xyz, rec_xyz = _compute_q_values_for_traj(
                model=model,
                data=data,
                expected_state_dim=expected_state_dim,
                action_chunk_dim=action_chunk_dim,
                device=device,
            )
            traj_id = _extract_traj_id(traj_path)
            status_label = status_map.get(traj_id, "UNKNOWN")
            out_name = f"q_pose_3d_traj_{traj_id:06d}.png" if traj_id >= 0 else f"q_pose_3d_{traj_path.stem}.png"
            _plot_traj_q_colored_3d(
                traj_id=traj_id if traj_id >= 0 else 0,
                status_label=status_label,
                q_values=q_values,
                ee_xyz=ee_xyz,
                ins_xyz=ins_xyz,
                rec_xyz=rec_xyz,
                out_path=output_dir / out_name,
                cmap_name=args.cmap,
                dpi=args.dpi,
                vmin=args.vmin,
                vmax=args.vmax,
            )
            num_processed += 1
        except Exception as exc:
            skipped.append((str(traj_path), str(exc)))
            print(f"[WARN] Skipping {traj_path}: {exc}")

    if num_processed == 0:
        reasons = "\n".join(f"- {p}: {r}" for p, r in skipped[:10])
        raise RuntimeError(
            "No trajectories were successfully processed.\n"
            f"Attempted: {len(traj_files)}\n"
            f"Examples:\n{reasons}"
        )

    print(f"[INFO] Processed trajectories: {num_processed}/{len(traj_files)}")
    print(f"[INFO] Skipped trajectories: {len(skipped)}")
    print(f"[INFO] Saved Q-colored 3D plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
