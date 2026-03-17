"""Create side-by-side camera videos from collected expert trajectories.

Usage:
    python l2sml/scripts/data_collect/make_traj_video.py \
        --dataset_dir data/peg_expert_100 --num_trajs 5
"""

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import imageio.v3 as iio

    _USE_IMAGEIO = True
except ImportError:
    _USE_IMAGEIO = False
    import cv2


CAMERA_KEYS = ["front_rgb", "side_rgb", "wrist_rgb"]


def _load_traj(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _make_video_frames(traj: dict, camera_keys: list[str]) -> np.ndarray:
    """Tile camera views horizontally for each timestep -> (T, H, W_total, 3)."""
    panels = []
    for key in camera_keys:
        imgs = traj["obs_images"].get(key)
        if imgs is None:
            continue
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.numpy()
        if imgs.dtype != np.uint8:
            imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)
        # squeeze any leading (T,1,...) dimension from single-env slice
        if imgs.ndim == 5 and imgs.shape[1] == 1:
            imgs = imgs[:, 0]
        panels.append(imgs)
    if not panels:
        raise RuntimeError(f"No camera images found. Available keys: {list(traj['obs_images'].keys())}")
    return np.concatenate(panels, axis=2)  # tile along width


def write_video(frames: np.ndarray, out_path: Path, fps: int = 10) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _USE_IMAGEIO:
        iio.imwrite(str(out_path), frames, fps=fps)
    else:
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Create videos from collected trajectory camera images.")
    parser.add_argument("--dataset_dir", type=str, default="data/peg_expert_100")
    parser.add_argument("--num_trajs", type=int, default=5, help="Number of trajectories to render.")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <dataset_dir>/videos).")
    parser.add_argument("--camera_keys", nargs="*", default=CAMERA_KEYS, help="Camera image keys to include.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    traj_dir = dataset_dir / "trajectories"
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(traj_dir.glob("traj_*.pt"))[: args.num_trajs]
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {traj_dir}")

    print(f"Creating videos for {len(traj_files)} trajectories -> {out_dir}")
    for traj_file in traj_files:
        traj = _load_traj(traj_file)
        frames = _make_video_frames(traj, args.camera_keys)
        vid_name = traj_file.stem + ".mp4"
        vid_path = out_dir / vid_name
        write_video(frames, vid_path, fps=args.fps)
        print(f"  {vid_name}: {frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]}")

    print("Done.")


if __name__ == "__main__":
    main()
