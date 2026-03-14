from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection


@dataclass
class TrajSummary:
    traj_id: int
    steps: int
    done: bool
    return_sum: float
    reward_mean: float
    action_l2_mean: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze expert trajectories and generate plots/videos.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/peg_expert_test",
        help=(
            "Path to a dataset directory, a monolithic trajectories.pt file, "
            "or a legacy traj_*.pt file."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for analysis outputs. Defaults to <dataset_dir>/analysis.",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=0,
        help="Optional cap on saved videos. 0 means save videos for all trajectories with images.",
    )
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150, help="Output plot DPI.")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap for return-colored EE line.")
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional color scale minimum for return-to-go values. If omitted, use per-trajectory min.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional color scale maximum for return-to-go values. If omitted, use per-trajectory max.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trajs", type=int, default=0, help="Optional cap for debugging. 0 means all.")
    return parser.parse_args()


def _extract_traj_id(path: Path) -> int:
    stem = path.stem
    if "_" not in stem:
        return -1
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return -1


def _load_traj(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_mapping_like(value: Any) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _is_trajectory_record(data: Any) -> bool:
    if not _is_mapping_like(data):
        return False
    keys = set(data.keys())
    return {"actions", "rewards"}.issubset(keys)


def _is_q_data_record(data: Any) -> bool:
    if not _is_mapping_like(data):
        return False
    keys = set(data.keys())
    return {"states", "action_chunks", "next_states", "returns"}.issubset(keys)


def _resolve_trajectory_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix != ".pt":
            raise ValueError(f"Expected a .pt file, got: {dataset_path}")
        return [dataset_path]

    if dataset_path.is_dir():
        traj_dir = dataset_path / "trajectories"
        monolithic_candidates = [dataset_path / "trajectories.pt", traj_dir / "trajectories.pt"]
        for candidate in monolithic_candidates:
            if candidate.is_file():
                return [candidate]

        search_dirs = [traj_dir, dataset_path] if traj_dir.is_dir() else [dataset_path]
        traj_files: list[Path] = []
        for search_dir in search_dirs:
            traj_files.extend(sorted(search_dir.glob("traj_*.pt")))
        if traj_files:
            return sorted(set(traj_files))

        raise FileNotFoundError(
            f"No trajectory data found under {dataset_path}. "
            "Expected trajectories.pt or traj_*.pt."
        )

    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")


def _iter_trajectory_entries(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    data = _load_traj(path)
    if _is_trajectory_record(data):
        yield str(path), data
        return

    if _is_q_data_record(data):
        raw_hint = path.with_name("trajectories.pt")
        hint_msg = f" Use the raw trajectory dataset instead, e.g. {raw_hint}." if raw_hint != path else ""
        raise ValueError(
            f"{path} looks like processed Q-data (states/action_chunks/next_states/returns), "
            f"not raw expert trajectories.{hint_msg}"
        )

    if _is_mapping_like(data):
        traj_keys = [key for key in sorted(data.keys(), key=str) if str(key).startswith("traj_")]
        if traj_keys:
            for traj_key in traj_keys:
                traj_data = data[traj_key]
                if not _is_trajectory_record(traj_data):
                    raise KeyError(
                        f"Trajectory entry {traj_key!r} in {path} is missing required keys such as actions/rewards."
                    )
                yield f"{path}::{traj_key}", traj_data
            return

    raise ValueError(
        f"Trajectory data at {path} must be either a single trajectory record or a mapping of traj_* entries."
    )


def _extract_traj_id_from_source(source: str) -> int:
    suffix = source.split("::")[-1]
    return _extract_traj_id(Path(suffix))


# Camera keys for side-by-side video (order = left to right). Labels shown above each panel.
CAMERA_VIDEO_KEYS = ["front_rgb", "side_rgb", "wrist_rgb"]
CAMERA_LABELS = {"front_rgb": "Front", "side_rgb": "Side", "wrist_rgb": "Wrist"}


def _stack_frames_with_labels(frames_list: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Stack frames horizontally and draw label text above each panel. All frames must be HWC."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "opencv-python (cv2) is required for side-by-side camera videos with labels. "
            "Install with: pip install opencv-python"
        ) from e
    padded = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    pad_h = 28
    for fr, label in zip(frames_list, labels):
        h, w = fr.shape[:2]
        panel = np.ones((h + pad_h, w, 3), dtype=np.uint8) * 40
        panel[pad_h:] = fr
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        x = (w - tw) // 2
        y = 20
        cv2.putText(panel, label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        padded.append(panel)
    return np.concatenate(padded, axis=1)


def _get_composite_frames_from_obs_images(obs_images: Any) -> np.ndarray | None:
    """Build side-by-side (front | side | wrist) frames with labels from obs_images."""
    available = [k for k in CAMERA_VIDEO_KEYS if k in obs_images and obs_images[k] is not None]
    if not available:
        return None

    lengths = []
    for k in available:
        v = obs_images[k]
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().numpy()
        else:
            a = np.asarray(v)
        if a.ndim == 3:
            a = a[np.newaxis, ...]
        if a.ndim == 4 and a.shape[0] > 0:
            lengths.append(a.shape[0])
    if not lengths:
        return None
    T = min(lengths)

    composite_frames = []
    for t in range(T):
        panels = []
        used_labels = []
        for k in CAMERA_VIDEO_KEYS:
            if k not in obs_images or obs_images[k] is None:
                continue
            v = obs_images[k]
            if isinstance(v, torch.Tensor):
                a = v.detach().cpu().numpy()
            else:
                a = np.asarray(v)
            if a.ndim == 3:
                a = a[np.newaxis, ...]
            if a.ndim == 4 and a.shape[0] > t:
                frame = a[t]
            else:
                continue
            if frame.shape[-1] in (1, 3, 4):
                f = frame.copy()
            elif frame.shape[0] in (1, 3, 4):
                f = np.transpose(frame, (1, 2, 0)).copy()
            else:
                continue
            f = _ensure_uint8(f)
            if f.shape[-1] == 1:
                f = np.repeat(f, 3, axis=-1)
            elif f.shape[-1] == 4:
                f = f[..., :3]
            panels.append(f)
            used_labels.append(CAMERA_LABELS.get(k, k))
        if not panels:
            continue
        composite_frames.append(_stack_frames_with_labels(panels, used_labels))
    if not composite_frames:
        return None
    return np.stack(composite_frames, axis=0)


def _get_frames_from_traj(data: dict[str, Any]) -> np.ndarray | None:
    obs_images = data.get("obs_images", {})
    if _is_mapping_like(obs_images) and obs_images is not None:
        composite = _get_composite_frames_from_obs_images(obs_images)
        if composite is not None:
            return composite

    rendered = data.get("rendered_images")
    if isinstance(rendered, torch.Tensor):
        rendered = rendered.detach().cpu().numpy()
    if isinstance(rendered, np.ndarray) and rendered.ndim == 4 and rendered.shape[0] > 0:
        return rendered

    if not _is_mapping_like(obs_images) or not obs_images:
        return None

    key = sorted(obs_images.keys())[0]
    value = obs_images[key]
    if isinstance(value, torch.Tensor):
        arr = value.cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        return None

    if arr.ndim != 4 or arr.shape[0] == 0:
        return None

    # Accept THWC or TCHW
    if arr.shape[-1] in (1, 3, 4):
        out = arr
    elif arr.shape[1] in (1, 3, 4):
        out = np.transpose(arr, (0, 2, 3, 1))
    else:
        return None
    return out


def _ensure_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    clipped = np.clip(frames, 0, 255)
    if clipped.max() <= 1.0:
        clipped = clipped * 255.0
    return clipped.astype(np.uint8)


def _save_videos(video_items: list[tuple[int, str, np.ndarray]], out_dir: Path, fps: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to write videos. Install it in your env.") from exc

    for traj_id, _traj_source, frames in video_items:
        safe_frames = _ensure_uint8(frames)
        out_path = out_dir / f"traj_{traj_id:06d}.mp4"
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264")
        for frame in safe_frames:
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            writer.append_data(frame)
        writer.close()


def _plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 40) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_reward_curves(reward_curves: list[tuple[int, np.ndarray]], out_path: Path, max_curves: int = 50) -> None:
    plt.figure(figsize=(10, 6))
    alpha = max(0.1, min(1.0, 10.0 / len(reward_curves)))
    for traj_id, curve in reward_curves[:max_curves]:
        plt.plot(curve, alpha=alpha, linewidth=0.8)

    if reward_curves:
        max_len = max(len(c) for _, c in reward_curves)
        padded = np.full((len(reward_curves), max_len), np.nan)
        for i, (_, c) in enumerate(reward_curves):
            padded[i, : len(c)] = c
        mean_curve = np.nanmean(padded, axis=0)
        plt.plot(mean_curve, color="black", linewidth=2, label="mean")
        plt.legend()

    plt.title("Per-Step Reward Curves")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_cumulative_reward_curves(reward_curves: list[tuple[int, np.ndarray]], out_path: Path, max_curves: int = 50) -> None:
    plt.figure(figsize=(10, 6))
    alpha = max(0.1, min(1.0, 10.0 / len(reward_curves)))
    all_cumulative = []
    for traj_id, curve in reward_curves[:max_curves]:
        cumulative = np.cumsum(curve)
        plt.plot(cumulative, alpha=alpha, linewidth=0.8)
        all_cumulative.append(cumulative)

    if all_cumulative:
        max_len = max(len(c) for c in all_cumulative)
        padded = np.full((len(all_cumulative), max_len), np.nan)
        for i, c in enumerate(all_cumulative):
            padded[i, : len(c)] = c
        mean_curve = np.nanmean(padded, axis=0)
        plt.plot(mean_curve, color="black", linewidth=2, label="mean")
        plt.legend()

    plt.title("Cumulative Reward Curves")
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_action_stats(action_sums: np.ndarray, action_sq_sums: np.ndarray, action_count: int, out_path: Path) -> None:
    means = action_sums / max(action_count, 1)
    vars_ = np.maximum(action_sq_sums / max(action_count, 1) - np.square(means), 0.0)
    stds = np.sqrt(vars_)

    idx = np.arange(len(means))
    plt.figure(figsize=(9, 5))
    plt.bar(idx - 0.2, means, width=0.4, label="mean")
    plt.bar(idx + 0.2, stds, width=0.4, label="std")
    plt.title("Action Dimension Statistics")
    plt.xlabel("action dim")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _compute_return_to_go(rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    returns = np.empty_like(rewards, dtype=np.float32)
    running = 0.0
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = float(rewards[t]) + running
        returns[t] = running
    return returns


def _extract_positions_xyz(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    required_keys = ("ee_positions", "insertive_positions", "receptive_positions")
    if not all(key in data for key in required_keys):
        return None
    ee_xyz = data["ee_positions"]
    ins_xyz = data["insertive_positions"]
    rec_xyz = data["receptive_positions"]
    ee_xyz = _to_numpy(ee_xyz)
    ins_xyz = _to_numpy(ins_xyz)
    rec_xyz = _to_numpy(rec_xyz)
    return ee_xyz, ins_xyz, rec_xyz


def _plot_pose_3d_single_traj(
    traj_id: int,
    ee_xyz: np.ndarray,
    ins_xyz: np.ndarray,
    rec_xyz: np.ndarray,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        ee_xyz[:, 0],
        ee_xyz[:, 1],
        ee_xyz[:, 2],
        color="tab:blue",
        alpha=0.75,
        linewidth=1.2,
        label="end_effector_position",
    )
    ax.plot(
        ins_xyz[:, 0],
        ins_xyz[:, 1],
        ins_xyz[:, 2],
        color="tab:orange",
        alpha=0.75,
        linewidth=1.2,
        label="insertive_position",
    )
    ax.plot(
        rec_xyz[:, 0],
        rec_xyz[:, 1],
        rec_xyz[:, 2],
        color="tab:green",
        alpha=0.75,
        linewidth=1.2,
        label="receptive_position",
    )
    ax.scatter(ee_xyz[0, 0], ee_xyz[0, 1], ee_xyz[0, 2], color="tab:blue", s=12, alpha=0.8)
    ax.scatter(ins_xyz[0, 0], ins_xyz[0, 1], ins_xyz[0, 2], color="tab:orange", s=12, alpha=0.8)
    ax.scatter(rec_xyz[0, 0], rec_xyz[0, 1], rec_xyz[0, 2], color="tab:green", s=12, alpha=0.8)

    ax.set_title(f"3D Position Trajectory (traj {traj_id:06d})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_traj_return_colored_3d(
    traj_id: int,
    returns_to_go: np.ndarray,
    ee_xyz: np.ndarray,
    ins_xyz: np.ndarray,
    rec_xyz: np.ndarray,
    out_path: Path,
    cmap_name: str,
    dpi: int,
    vmin: float | None,
    vmax: float | None,
) -> None:
    if returns_to_go.shape[0] < 1:
        raise ValueError("returns_to_go is empty.")
    if ee_xyz.shape[0] != returns_to_go.shape[0]:
        raise ValueError(f"ee_xyz and returns_to_go length mismatch: {ee_xyz.shape[0]} vs {returns_to_go.shape[0]}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if ee_xyz.shape[0] >= 2:
        ee_segments = np.stack([ee_xyz[:-1], ee_xyz[1:]], axis=1)
        cmap = plt.get_cmap(cmap_name)
        returns_for_segments = returns_to_go[:-1]
        returns_min = float(np.min(returns_to_go)) if vmin is None else float(vmin)
        returns_max = float(np.max(returns_to_go)) if vmax is None else float(vmax)
        if np.isclose(returns_min, returns_max):
            returns_max = returns_min + 1e-6
        norm = mcolors.Normalize(vmin=returns_min, vmax=returns_max)

        ee_collection = Line3DCollection(ee_segments, cmap=cmap, norm=norm, linewidth=2.0, alpha=0.95)
        ee_collection.set_array(returns_for_segments)
        ax.add_collection(ee_collection)
        cbar = fig.colorbar(ee_collection, ax=ax, fraction=0.03, pad=0.1)
        cbar.set_label("Return-to-go")
    else:
        ax.scatter(
            ee_xyz[:, 0],
            ee_xyz[:, 1],
            ee_xyz[:, 2],
            c=returns_to_go,
            cmap=cmap_name,
            s=20,
            label="end_effector",
        )

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
    ax.set_title(f"Return-Colored 3D Pose Trajectory (traj {traj_id:06d})")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    if args.max_trajs < 0:
        raise ValueError(f"max_trajs must be >= 0, got {args.max_trajs}")
    if args.dpi <= 0:
        raise ValueError(f"dpi must be > 0, got {args.dpi}")
    if args.vmin is not None and args.vmax is not None and args.vmin >= args.vmax:
        raise ValueError(f"Expected vmin < vmax, got vmin={args.vmin}, vmax={args.vmax}")

    dataset_path = Path(args.dataset_dir)
    traj_files = _resolve_trajectory_files(dataset_path)
    if args.max_trajs > 0:
        traj_files = traj_files[: args.max_trajs]
    if not traj_files:
        raise FileNotFoundError("No trajectory files selected after applying max_trajs.")

    default_output_root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    out_dir = Path(args.output_dir) if args.output_dir is not None else default_output_root / "analysis"
    plots_dir = out_dir / "plots"
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[TrajSummary] = []
    action_dim = None
    action_sums = None
    action_sq_sums = None
    action_count = 0

    candidates_for_video: list[tuple[int, str, np.ndarray]] = []
    all_reward_curves: list[tuple[int, np.ndarray]] = []
    pose_items: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    return_pose_items: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for traj_path in traj_files:
        for traj_source, data in _iter_trajectory_entries(traj_path):
            traj_id = _extract_traj_id_from_source(traj_source)

            actions = data["actions"]
            rewards = data["rewards"]
            terminated = data.get("terminated", torch.zeros_like(torch.as_tensor(rewards), dtype=torch.bool))
            truncated = data.get("truncated", torch.zeros_like(torch.as_tensor(rewards), dtype=torch.bool))

            if isinstance(actions, torch.Tensor):
                actions_np = actions.numpy()
            else:
                actions_np = np.asarray(actions)
            if isinstance(rewards, torch.Tensor):
                rewards_np = rewards.numpy().reshape(-1)
            else:
                rewards_np = np.asarray(rewards).reshape(-1)
            if isinstance(terminated, torch.Tensor):
                terminated_np = terminated.numpy().reshape(-1)
            else:
                terminated_np = np.asarray(terminated).reshape(-1)
            if isinstance(truncated, torch.Tensor):
                truncated_np = truncated.numpy().reshape(-1)
            else:
                truncated_np = np.asarray(truncated).reshape(-1)

            steps = int(actions_np.shape[0])
            done = bool(np.any(terminated_np) or np.any(truncated_np))
            return_sum = float(rewards_np.sum()) if rewards_np.size else 0.0
            reward_mean = float(rewards_np.mean()) if rewards_np.size else 0.0
            action_l2_mean = float(np.linalg.norm(actions_np, axis=1).mean()) if actions_np.size else 0.0
            summaries.append(
                TrajSummary(
                    traj_id=traj_id,
                    steps=steps,
                    done=done,
                    return_sum=return_sum,
                    reward_mean=reward_mean,
                    action_l2_mean=action_l2_mean,
                )
            )

            if actions_np.size:
                if action_dim is None:
                    action_dim = int(actions_np.shape[1])
                    action_sums = np.zeros(action_dim, dtype=np.float64)
                    action_sq_sums = np.zeros(action_dim, dtype=np.float64)
                action_sums += actions_np.sum(axis=0)
                action_sq_sums += np.square(actions_np).sum(axis=0)
                action_count += int(actions_np.shape[0])

            all_reward_curves.append((traj_id, rewards_np.copy()))

            frames = _get_frames_from_traj(data)
            if frames is not None:
                candidates_for_video.append((traj_id, traj_source, frames))

            positions_xyz = _extract_positions_xyz(data)
            if positions_xyz is not None:
                ee_xyz, ins_xyz, rec_xyz = positions_xyz
                returns_to_go = _compute_return_to_go(rewards_np)
                min_len = min(ee_xyz.shape[0], ins_xyz.shape[0], rec_xyz.shape[0], returns_to_go.shape[0])
                if min_len > 0:
                    ee_xyz = ee_xyz[:min_len]
                    ins_xyz = ins_xyz[:min_len]
                    rec_xyz = rec_xyz[:min_len]
                    returns_to_go = returns_to_go[:min_len]

                    pose_items.append(
                        (traj_id, ee_xyz, ins_xyz, rec_xyz)
                    )
                    return_pose_items.append((traj_id, returns_to_go, ee_xyz, ins_xyz, rec_xyz))

    summary_json = {
        "dataset_dir": str(dataset_path.resolve()),
        "num_trajectories": len(summaries),
        "num_with_images": len(candidates_for_video),
        "steps": {
            "mean": float(np.mean([s.steps for s in summaries])),
            "std": float(np.std([s.steps for s in summaries])),
            "min": int(np.min([s.steps for s in summaries])),
            "max": int(np.max([s.steps for s in summaries])),
        },
        "returns": {
            "mean": float(np.mean([s.return_sum for s in summaries])),
            "std": float(np.std([s.return_sum for s in summaries])),
            "min": float(np.min([s.return_sum for s in summaries])),
            "max": float(np.max([s.return_sum for s in summaries])),
        },
        "done_rate": float(np.mean([1.0 if s.done else 0.0 for s in summaries])),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    with open(out_dir / "trajectory_table.csv", "w", encoding="utf-8") as f:
        f.write("traj_id,steps,done,return_sum,reward_mean,action_l2_mean\n")
        for s in summaries:
            f.write(
                f"{s.traj_id},{s.steps},{int(s.done)},{s.return_sum:.8f},{s.reward_mean:.8f},{s.action_l2_mean:.8f}\n"
            )

    _plot_hist(np.array([s.steps for s in summaries]), "Trajectory Length Distribution", "steps", plots_dir / "steps_hist.png")
    _plot_hist(
        np.array([s.return_sum for s in summaries]),
        "Trajectory Return Distribution",
        "return",
        plots_dir / "return_hist.png",
    )
    _plot_hist(
        np.array([s.action_l2_mean for s in summaries]),
        "Mean Action L2 Distribution",
        "mean ||action||_2",
        plots_dir / "action_l2_hist.png",
    )

    if action_dim is not None and action_sums is not None and action_sq_sums is not None:
        _plot_action_stats(action_sums, action_sq_sums, action_count, plots_dir / "action_dim_stats.png")

    if all_reward_curves:
        _plot_reward_curves(all_reward_curves, plots_dir / "reward_per_step.png")
        _plot_cumulative_reward_curves(all_reward_curves, plots_dir / "cumulative_reward.png")
    if pose_items:
        pose_plots_dir = plots_dir / "poses_3d"
        pose_plots_dir.mkdir(parents=True, exist_ok=True)
        for traj_id, ee_xyz, ins_xyz, rec_xyz in pose_items:
            _plot_pose_3d_single_traj(
                traj_id,
                ee_xyz,
                ins_xyz,
                rec_xyz,
                pose_plots_dir / f"poses_3d_traj_{traj_id:06d}.png",
            )
    if return_pose_items:
        return_pose_plots_dir = plots_dir / "return_pose_3d"
        return_pose_plots_dir.mkdir(parents=True, exist_ok=True)
        for traj_id, returns_to_go, ee_xyz, ins_xyz, rec_xyz in return_pose_items:
            _plot_traj_return_colored_3d(
                traj_id=traj_id,
                returns_to_go=returns_to_go,
                ee_xyz=ee_xyz,
                ins_xyz=ins_xyz,
                rec_xyz=rec_xyz,
                out_path=return_pose_plots_dir / f"return_pose_3d_traj_{traj_id:06d}.png",
                cmap_name=args.cmap,
                dpi=args.dpi,
                vmin=args.vmin,
                vmax=args.vmax,
            )

    if candidates_for_video:
        if args.num_videos > 0:
            random.shuffle(candidates_for_video)
            selected = candidates_for_video[: min(args.num_videos, len(candidates_for_video))]
        else:
            selected = sorted(candidates_for_video, key=lambda item: item[0])
        _save_videos(selected, videos_dir, args.video_fps)
        print(f"[INFO] Saved {len(selected)} trajectory videos to {videos_dir}")
    else:
        print("[INFO] No image streams found in trajectories. No videos were created.")

    print(f"[INFO] Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
