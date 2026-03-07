from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


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
    parser.add_argument("--dataset_dir", type=str, default="data/peg_expert_with_images")
    parser.add_argument("--output_dir", type=str, default="data/peg_expert_with_images/analysis")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--video_fps", type=int, default=20)
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


def _get_frames_from_traj(data: dict[str, Any]) -> np.ndarray | None:
    rendered = data.get("rendered_images")
    if isinstance(rendered, np.ndarray) and rendered.ndim == 4 and rendered.shape[0] > 0:
        return rendered

    obs_images = data.get("obs_images", {})
    if not isinstance(obs_images, dict) or not obs_images:
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


def _save_videos(video_items: list[tuple[int, Path, np.ndarray]], out_dir: Path, fps: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to write videos. Install it in your env.") from exc

    for traj_id, traj_path, frames in video_items:
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


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    traj_dir = dataset_dir / "trajectories"
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(traj_dir.glob("traj_*.pt"))
    if args.max_trajs > 0:
        traj_files = traj_files[: args.max_trajs]

    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found under {traj_dir}")

    summaries: list[TrajSummary] = []
    action_dim = None
    action_sums = None
    action_sq_sums = None
    action_count = 0

    candidates_for_video: list[tuple[int, Path, np.ndarray]] = []
    all_reward_curves: list[tuple[int, np.ndarray]] = []

    for traj_path in traj_files:
        data = _load_traj(traj_path)
        traj_id = _extract_traj_id(traj_path)

        actions = data["actions"]
        rewards = data["rewards"]
        terminated = data["terminated"]
        truncated = data["truncated"]

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
            candidates_for_video.append((traj_id, traj_path, frames))

    summary_json = {
        "dataset_dir": str(dataset_dir.resolve()),
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

    if candidates_for_video:
        random.shuffle(candidates_for_video)
        selected = candidates_for_video[: min(args.num_videos, len(candidates_for_video))]
        _save_videos(selected, videos_dir, args.video_fps)
        print(f"[INFO] Saved {len(selected)} trajectory videos to {videos_dir}")
    else:
        print("[INFO] No image streams found in trajectories. No videos were created.")

    print(f"[INFO] Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
