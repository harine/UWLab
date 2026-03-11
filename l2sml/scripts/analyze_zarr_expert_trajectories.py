from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrajSummary:
    traj_id: int
    steps: int
    done: bool
    return_sum: float
    reward_mean: float
    action_l2_mean: float
    success: bool  # True if reward at last timestep > 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze zarr expert trajectories and generate plots/videos.")
    parser.add_argument(
        "--dataset_dir",
        "--dataset_path",
        dest="dataset_path",
        type=str,
        default="datasets/peg_expert_test2/dataset.zarr",
    )
    parser.add_argument("--output_dir", type=str, default="", help="Defaults to <dataset parent>/analysis.")
    parser.add_argument("--num_videos", type=int, default=100)
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trajs", type=int, default=0, help="Optional cap for debugging. 0 means all.")
    return parser.parse_args()


def _import_zarr() -> Any:
    try:
        import zarr
    except Exception as exc:
        raise RuntimeError(
            "Failed to import zarr. Ensure the active environment has zarr and its runtime dependencies installed."
        ) from exc
    return zarr


# Camera keys for side-by-side video (order = left to right). Labels shown above each panel.
CAMERA_VIDEO_KEYS = ["front_rgb", "side_rgb", "wrist_rgb"]
CAMERA_LABELS = {"front_rgb": "Front", "side_rgb": "Side", "wrist_rgb": "Wrist"}


def _stack_frames_with_labels(frames_list: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Stack frames horizontally and draw label text above each panel. All frames must be HWC."""
    try:
        import cv2  # type: ignore[import-not-found]
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


def _get_composite_frames_from_camera(camera_obs: dict[str, Any]) -> np.ndarray | None:
    """Build side-by-side (front | side | wrist) frames with labels from camera arrays."""
    available = [k for k in CAMERA_VIDEO_KEYS if k in camera_obs and camera_obs[k] is not None]
    if not available:
        return None

    lengths = []
    for k in available:
        a = np.asarray(camera_obs[k])
        if a.ndim == 3:
            a = a[np.newaxis, ...]
        if a.ndim == 4 and a.shape[0] > 0:
            lengths.append(a.shape[0])
    if not lengths:
        return None
    t_max = min(lengths)

    composite_frames = []
    for t in range(t_max):
        panels = []
        used_labels = []
        for k in CAMERA_VIDEO_KEYS:
            if k not in camera_obs or camera_obs[k] is None:
                continue
            a = np.asarray(camera_obs[k])
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
    camera = data.get("camera", {})
    if isinstance(camera, dict) and camera:
        composite = _get_composite_frames_from_camera(camera)
        if composite is not None:
            return composite

    rendered = data.get("rendered_images")
    if isinstance(rendered, np.ndarray) and rendered.ndim == 4 and rendered.shape[0] > 0:
        return rendered

    return None


def _ensure_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    clipped = np.clip(frames, 0, 255)
    if clipped.size and clipped.max() <= 1.0:
        clipped = clipped * 255.0
    return clipped.astype(np.uint8)


def _save_videos(video_items: list[tuple[int, np.ndarray]], out_dir: Path, fps: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to write videos. Install it in your env.") from exc

    for traj_id, frames in video_items:
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
    for _, curve in reward_curves[:max_curves]:
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
    for _, curve in reward_curves[:max_curves]:
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


def _extract_xyz_from_pose_array(value: Any) -> np.ndarray | None:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim < 2 or arr.shape[-1] < 3:
        return None
    return arr[..., :3]


def _extract_positions_xyz(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    obs = data.get("positions", {})
    # if isinstance(obs, dict):
    #     ee_xyz = _extract_xyz_from_pose_array(obs.get("end_effector_pose"))
    #     ins_xyz = _extract_xyz_from_pose_array(obs.get("insertive_asset_pose"))
    #     rec_xyz = _extract_xyz_from_pose_array(obs.get("receptive_asset_pose"))
    #     if ee_xyz is not None and ins_xyz is not None and rec_xyz is not None:
    #         return ee_xyz, ins_xyz, rec_xyz

    positions = data.get("positions")
    if positions is None:
        return None

    positions_np = np.asarray(positions)
    if positions_np.ndim == 1:
        positions_np = positions_np[np.newaxis, :]
    if positions_np.ndim < 2 or positions_np.shape[-1] < 9:
        return None

    block_size = positions_np.shape[-1] // 3
    if block_size < 3:
        return None

    ee_xyz = positions_np[..., 0:3]
    ins_xyz = positions_np[..., block_size : block_size + 3]
    rec_xyz = positions_np[..., 2 * block_size : 2 * block_size + 3]
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


def _load_optional_slice(group: Any, key: str, start: int, end: int) -> np.ndarray | None:
    if key not in group:
        return None
    return np.asarray(group[key][start:end])


def _load_episode_data(root: Any, start: int, end: int) -> dict[str, Any]:
    data_group = root["data"]
    out: dict[str, Any] = {}

    for key in ("actions", "rewards", "dones", "terminated", "truncated", "positions", "rendered_images"):
        value = _load_optional_slice(data_group, key, start, end)
        if value is not None:
            out[key] = value

    if "camera" in data_group:
        camera_group = data_group["camera"]
        camera_data = {
            key: _load_optional_slice(camera_group, key, start, end)
            for key in CAMERA_VIDEO_KEYS
            if key in camera_group
        }
        camera_data = {key: value for key, value in camera_data.items() if value is not None}
        if camera_data:
            out["camera"] = camera_data

    if "obs" in data_group:
        obs_group = data_group["obs"]
        obs_keys = [
            "end_effector_pose",
            "insertive_asset_pose",
            "receptive_asset_pose",
            "expert_action",
            "joint_pos",
            "prev_actions",
            "insertive_asset_in_receptive_asset_frame",
        ]
        obs_data = {
            key: _load_optional_slice(obs_group, key, start, end)
            for key in obs_keys
            if key in obs_group
        }
        obs_data = {key: value for key, value in obs_data.items() if value is not None}
        if obs_data:
            out["obs"] = obs_data

    return out


def _get_episode_ranges(root: Any) -> list[tuple[int, int]]:
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64).reshape(-1)
    if episode_ends.size == 0:
        return []
    starts = np.concatenate([np.array([0], dtype=np.int64), episode_ends[:-1]])
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends, strict=False)]


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    zarr = _import_zarr()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    out_dir = Path(args.output_dir) if args.output_dir else dataset_path.parent / "analysis"
    plots_dir = out_dir / "plots"
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(dataset_path), mode="r")
    traj_ranges = _get_episode_ranges(root)
    if args.max_trajs > 0:
        traj_ranges = traj_ranges[: args.max_trajs]

    if not traj_ranges:
        raise FileNotFoundError(f"No trajectories found in {dataset_path}")

    summaries: list[TrajSummary] = []
    action_dim = None
    action_sums = None
    action_sq_sums = None
    action_count = 0

    candidates_for_video: list[tuple[int, np.ndarray]] = []
    all_reward_curves: list[tuple[int, np.ndarray]] = []
    pose_items_90_100: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    for traj_id, (start, end) in enumerate(traj_ranges):
        data = _load_episode_data(root, start, end)
        actions_np = np.asarray(data.get("actions", np.empty((0, 0), dtype=np.float32)))
        rewards_np = np.asarray(data.get("rewards", np.empty((0,), dtype=np.float32))).reshape(-1)

        dones_np = None
        if "dones" in data:
            dones_np = np.asarray(data["dones"]).reshape(-1)
        elif "terminated" in data or "truncated" in data:
            terminated_np = np.asarray(data.get("terminated", np.empty((0,), dtype=bool))).reshape(-1)
            truncated_np = np.asarray(data.get("truncated", np.empty((0,), dtype=bool))).reshape(-1)
            max_len = max(len(terminated_np), len(truncated_np))
            dones_np = np.zeros(max_len, dtype=bool)
            if len(terminated_np):
                dones_np[: len(terminated_np)] |= terminated_np.astype(bool)
            if len(truncated_np):
                dones_np[: len(truncated_np)] |= truncated_np.astype(bool)

        steps = int(actions_np.shape[0])
        done = bool(np.any(dones_np)) if dones_np is not None and dones_np.size else True
        return_sum = float(rewards_np.sum()) if rewards_np.size else 0.0
        reward_mean = float(rewards_np.mean()) if rewards_np.size else 0.0
        action_l2_mean = float(np.linalg.norm(actions_np, axis=1).mean()) if actions_np.size else 0.0
        last_reward = float(rewards_np[-1]) if rewards_np.size else 0.0
        success = last_reward > 0.05
        summaries.append(
            TrajSummary(
                traj_id=traj_id,
                steps=steps,
                done=done,
                return_sum=return_sum,
                reward_mean=reward_mean,
                action_l2_mean=action_l2_mean,
                success=success,
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

        if rewards_np.size:
            all_reward_curves.append((traj_id, rewards_np.copy()))

        frames = _get_frames_from_traj(data)
        if frames is not None:
            candidates_for_video.append((traj_id, frames))

        if 0 <= traj_id <= 100:
            positions_xyz = _extract_positions_xyz(data)
            if positions_xyz is not None:
                ee_xyz, ins_xyz, rec_xyz = positions_xyz
                min_len = min(ee_xyz.shape[0], ins_xyz.shape[0], rec_xyz.shape[0])
                if min_len > 0:
                    pose_items_90_100.append(
                        (
                            traj_id,
                            ee_xyz[:min_len],
                            ins_xyz[:min_len],
                            rec_xyz[:min_len],
                        )
                    )

    n_success = sum(1 for s in summaries if s.success)
    n_fail = len(summaries) - n_success
    proportion_success = n_success / len(summaries) if summaries else 0.0
    proportion_fail = n_fail / len(summaries) if summaries else 0.0

    summary_json = {
        "dataset_path": str(dataset_path.resolve()),
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
        "success_by_last_reward": {
            "threshold": 0.05,
            "num_success": n_success,
            "num_fail": n_fail,
            "proportion_success": proportion_success,
            "proportion_fail": proportion_fail,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    with open(out_dir / "trajectory_table.csv", "w", encoding="utf-8") as f:
        f.write("traj_id,steps,done,return_sum,reward_mean,action_l2_mean,success\n")
        for s in summaries:
            f.write(
                f"{s.traj_id},{s.steps},{int(s.done)},{s.return_sum:.8f},{s.reward_mean:.8f},{s.action_l2_mean:.8f},{int(s.success)}\n"
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

    if pose_items_90_100:
        pose_plots_dir = plots_dir / "poses_3d_traj_90_100"
        pose_plots_dir.mkdir(parents=True, exist_ok=True)
        for traj_id, ee_xyz, ins_xyz, rec_xyz in pose_items_90_100:
            _plot_pose_3d_single_traj(
                traj_id,
                ee_xyz,
                ins_xyz,
                rec_xyz,
                pose_plots_dir / f"poses_3d_traj_{traj_id:06d}.png",
            )

    if candidates_for_video:
        random.shuffle(candidates_for_video)
        selected = candidates_for_video[: min(args.num_videos, len(candidates_for_video))]
        _save_videos(selected, videos_dir, args.video_fps)
        print(f"[INFO] Saved {len(selected)} trajectory videos to {videos_dir}")
    else:
        print("[INFO] No image streams found in trajectories. No videos were created.")

    print(f"[INFO] Success/fail (last reward > 0.05): {n_success} succeed ({proportion_success:.2%}), {n_fail} fail ({proportion_fail:.2%})")
    print(f"[INFO] Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
