from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
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


@dataclass
class TrajQResult:
    q_values: np.ndarray
    frame_q_values: np.ndarray
    ee_xyz: np.ndarray
    ins_xyz: np.ndarray
    rec_xyz: np.ndarray
    chunk_starts: np.ndarray
    chunk_ends_exclusive: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 3D trajectories for end effector / insertive / receptive poses, "
            "with end-effector line color determined by action-chunk Q values."
        )
    )
    parser.add_argument("--q_checkpoint", type=str, required=True, help="Path to trained Q-function checkpoint (.pt).")
    parser.add_argument(
        "--trajectory_path",
        type=str,
        required=True,
        help=(
            "Path to a single trajectory .pt file, a monolithic trajectories.pt dataset file, "
            "or a directory containing trajectories.pt / traj_*.pt files."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output plots. Defaults to <dataset_dir>/q_pose_3d_plots.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for model inference, e.g. cpu or cuda.")
    parser.add_argument("--max_trajs", type=int, default=0, help="Optional cap for debugging. 0 means all.")
    parser.add_argument("--dpi", type=int, default=150, help="Output plot DPI.")
    parser.add_argument("--video_fps", type=int, default=20, help="FPS for Q-overlay trajectory videos.")
    parser.add_argument("--q_stride", type=int, default=5, help="Evaluate Q only every N frames.")
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


def _load_traj(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_mapping_like(value: Any) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _is_trajectory_record(data: Any) -> bool:
    if not _is_mapping_like(data):
        return False
    keys = set(data.keys())
    return {"obs_flat", "actions"}.issubset(keys)


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
                        f"Trajectory entry {traj_key!r} in {path} is missing required keys such as obs_flat/actions."
                    )
                yield f"{path}::{traj_key}", traj_data
            return

    raise ValueError(
        f"Trajectory data at {path} must be either a single trajectory record or a mapping of traj_* entries."
    )


def _extract_traj_id_from_source(source: str) -> int:
    suffix = source.split("::")[-1]
    return _extract_traj_id(Path(suffix))


CAMERA_VIDEO_KEYS = ["front_rgb", "side_rgb", "wrist_rgb"]
CAMERA_LABELS = {"front_rgb": "Front", "side_rgb": "Side", "wrist_rgb": "Wrist"}


def _resolve_manifest_path(dataset_path: Path) -> Path | None:
    candidate_paths: list[Path] = []
    if dataset_path.is_dir():
        candidate_paths.append(dataset_path / "manifest.json")
    elif dataset_path.is_file():
        candidate_paths.append(dataset_path.with_name("manifest.json"))
        if dataset_path.parent.name == "trajectories":
            candidate_paths.append(dataset_path.parent.parent / "manifest.json")

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate
    return None


def _load_manifest_success_map(manifest_path: Path | None) -> dict[int, str]:
    if manifest_path is None:
        return {}
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


def _ensure_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    clipped = np.clip(frames, 0, 255)
    if clipped.size > 0 and clipped.max() <= 1.0:
        clipped = clipped * 255.0
    return clipped.astype(np.uint8)


def _ensure_frame_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected HWC frame, got shape {frame.shape}")
    if frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        return frame[..., :3]
    if frame.shape[-1] == 3:
        return frame
    raise ValueError(f"Expected frame with 1, 3, or 4 channels, got shape {frame.shape}")


def _stack_frames_with_labels(frames_list: list[np.ndarray], labels: list[str]) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python (cv2) is required for side-by-side camera videos with labels. "
            "Install with: pip install opencv-python"
        ) from exc

    padded = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    pad_h = 28
    for frame, label in zip(frames_list, labels):
        h, w = frame.shape[:2]
        panel = np.ones((h + pad_h, w, 3), dtype=np.uint8) * 40
        panel[pad_h:] = frame
        (text_w, _text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        x = (w - text_w) // 2
        y = 20
        cv2.putText(panel, label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        padded.append(panel)
    return np.concatenate(padded, axis=1)


def _get_composite_frames_from_obs_images(obs_images: Any) -> np.ndarray | None:
    available = [k for k in CAMERA_VIDEO_KEYS if k in obs_images and obs_images[k] is not None]
    if not available:
        return None

    lengths = []
    for key in available:
        value = obs_images[key]
        arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        if arr.ndim == 4 and arr.shape[0] > 0:
            lengths.append(arr.shape[0])
    if not lengths:
        return None

    num_frames = min(lengths)
    composite_frames = []
    for t in range(num_frames):
        panels = []
        used_labels = []
        for key in CAMERA_VIDEO_KEYS:
            if key not in obs_images or obs_images[key] is None:
                continue
            value = obs_images[key]
            arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            if arr.ndim != 4 or arr.shape[0] <= t:
                continue

            frame = arr[t]
            if frame.shape[-1] in (1, 3, 4):
                panel = frame.copy()
            elif frame.shape[0] in (1, 3, 4):
                panel = np.transpose(frame, (1, 2, 0)).copy()
            else:
                continue

            panel = _ensure_uint8(panel)
            if panel.shape[-1] == 1:
                panel = np.repeat(panel, 3, axis=-1)
            elif panel.shape[-1] == 4:
                panel = panel[..., :3]
            panels.append(panel)
            used_labels.append(CAMERA_LABELS.get(key, key))
        if panels:
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
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        return None

    if arr.ndim != 4 or arr.shape[0] == 0:
        return None

    if arr.shape[-1] in (1, 3, 4):
        return arr
    if arr.shape[1] in (1, 3, 4):
        return np.transpose(arr, (0, 2, 3, 1))
    return None


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
    q_stride: int,
) -> TrajQResult:
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
    if q_stride <= 0:
        raise ValueError(f"q_stride must be > 0, got {q_stride}")

    states = obs_flat[:valid_steps]
    action_chunks = actions.unfold(0, chunk_size, 1).reshape(valid_steps, -1)
    if action_chunks.shape[1] != action_chunk_dim:
        raise ValueError(
            f"Action chunk dim mismatch after unfold: got {action_chunks.shape[1]}, expected {action_chunk_dim}"
        )

    chunk_starts = np.arange(0, valid_steps, q_stride, dtype=np.int64)
    sampled_states = states[chunk_starts]
    sampled_action_chunks = action_chunks[chunk_starts]
    with torch.inference_mode():
        q_values = model(sampled_states.to(device), sampled_action_chunks.to(device)).squeeze(-1).detach().cpu().numpy()

    ee_np = ee_xyz.detach().cpu().numpy()
    ins_np = ins_xyz.detach().cpu().numpy()
    rec_np = rec_xyz.detach().cpu().numpy()
    chunk_ends_exclusive = np.minimum(chunk_starts + chunk_size, aligned_steps)
    frame_q_values = np.full(aligned_steps, np.nan, dtype=np.float32)
    for q_value, start_idx, end_idx in zip(q_values, chunk_starts, chunk_ends_exclusive):
        frame_q_values[int(start_idx) : int(end_idx)] = float(q_value)

    return TrajQResult(
        q_values=q_values,
        frame_q_values=frame_q_values,
        ee_xyz=ee_np,
        ins_xyz=ins_np,
        rec_xyz=rec_np,
        chunk_starts=chunk_starts,
        chunk_ends_exclusive=chunk_ends_exclusive,
    )


def _overlay_q_values_on_video(
    frames: np.ndarray,
    frame_q_values: np.ndarray,
    status_label: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python (cv2) is required to render Q-value overlays on trajectory videos. "
            "Install with: pip install opencv-python"
        ) from exc

    safe_frames = _ensure_uint8(np.asarray(frames))
    if safe_frames.ndim != 4 or safe_frames.shape[0] == 0:
        raise ValueError(f"Expected frames shaped [T, H, W, C], got {safe_frames.shape}")

    frame_q_values = np.asarray(frame_q_values, dtype=np.float32)
    usable_steps = min(safe_frames.shape[0], frame_q_values.shape[0])
    if usable_steps <= 0:
        raise ValueError("No overlapping timesteps between frames and frame_q_values.")

    safe_frames = safe_frames[:usable_steps].copy()
    frame_q_values = frame_q_values[:usable_steps]
    finite_q_values = frame_q_values[np.isfinite(frame_q_values)]
    if finite_q_values.size == 0:
        raise ValueError("No finite Q values available for video overlay.")

    q_min = float(np.min(finite_q_values)) if vmin is None else float(vmin)
    q_max = float(np.max(finite_q_values)) if vmax is None else float(vmax)
    if np.isclose(q_min, q_max):
        q_max = q_min + 1e-6
    norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
    cmap = plt.get_cmap(cmap_name)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    margin = 12
    panel_h = 78
    bar_h = 16

    output_frames: list[np.ndarray] = []
    for idx, frame in enumerate(safe_frames):
        frame_rgb = _ensure_frame_rgb(frame)
        h, w = frame.shape[:2]
        overlay = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
        overlay[:h] = frame_rgb
        overlay[h:] = 24

        q_value = float(frame_q_values[idx])
        has_q_value = np.isfinite(q_value)
        if has_q_value:
            color_rgb = np.asarray(cmap(norm(q_value))[:3]) * 255.0
            color_bgr = tuple(int(x) for x in color_rgb[::-1])
            label = f"Q(current chunk): {q_value:.4f}"
        else:
            color_bgr = (180, 180, 180)
            label = "Q(current chunk): N/A"
        status_text = f"Status: {status_label}"
        range_text = f"Scale [{q_min:.3f}, {q_max:.3f}]"
        cv2.putText(overlay, label, (margin, h + 28), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(overlay, status_text, (margin, h + 56), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        text_size, _ = cv2.getTextSize(range_text, font, 0.5, 1)
        cv2.putText(
            overlay,
            range_text,
            (w - text_size[0] - margin, h + 24),
            font,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        bar_x0 = margin
        bar_x1 = w - margin
        bar_y0 = h + panel_h - bar_h - 10
        bar_y1 = h + panel_h - 10
        gradient = np.linspace(0.0, 1.0, max(bar_x1 - bar_x0, 1), dtype=np.float32)
        gradient_rgb = (np.asarray(cmap(gradient))[:, :3] * 255.0).astype(np.uint8)
        gradient_bgr = gradient_rgb[:, ::-1][np.newaxis, :, :]
        gradient_bgr = np.repeat(gradient_bgr, bar_h, axis=0)
        overlay[bar_y0:bar_y1, bar_x0:bar_x1] = gradient_bgr
        cv2.rectangle(overlay, (bar_x0, bar_y0), (bar_x1, bar_y1), (255, 255, 255), 1)

        if has_q_value:
            marker_x = int(bar_x0 + norm(q_value) * max(bar_x1 - bar_x0 - 1, 1))
            cv2.line(overlay, (marker_x, bar_y0 - 4), (marker_x, bar_y1 + 4), color_bgr, 3)
        output_frames.append(overlay)

    return np.stack(output_frames, axis=0)


def _save_q_overlay_video(frames: np.ndarray, out_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to write videos. Install it in your env.") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264")
    for frame in _ensure_uint8(frames):
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        writer.append_data(_ensure_frame_rgb(frame))
    writer.close()


def _plot_traj_q_colored_3d(
    traj_id: int,
    status_label: str,
    q_values: np.ndarray,
    ee_xyz: np.ndarray,
    ins_xyz: np.ndarray,
    rec_xyz: np.ndarray,
    chunk_starts: np.ndarray,
    chunk_ends_exclusive: np.ndarray,
    out_path: Path,
    cmap_name: str,
    dpi: int,
    vmin: float | None,
    vmax: float | None,
) -> None:
    if q_values.shape[0] < 1:
        raise ValueError("q_values is empty.")
    if not (q_values.shape[0] == chunk_starts.shape[0] == chunk_ends_exclusive.shape[0]):
        raise ValueError(
            "q_values, chunk_starts, and chunk_ends_exclusive must have matching lengths: "
            f"{q_values.shape[0]}, {chunk_starts.shape[0]}, {chunk_ends_exclusive.shape[0]}"
        )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        ee_xyz[:, 0],
        ee_xyz[:, 1],
        ee_xyz[:, 2],
        color="lightgray",
        alpha=0.35,
        linewidth=1.0,
        label="end_effector_path",
    )

    if np.any(chunk_ends_exclusive - chunk_starts >= 2):
        ee_segments = np.stack([ee_xyz[chunk_starts], ee_xyz[chunk_ends_exclusive - 1]], axis=1)
        cmap = plt.get_cmap(cmap_name)
        q_min = float(np.min(q_values)) if vmin is None else float(vmin)
        q_max = float(np.max(q_values)) if vmax is None else float(vmax)
        if np.isclose(q_min, q_max):
            q_max = q_min + 1e-6
        norm = mcolors.Normalize(vmin=q_min, vmax=q_max)

        ee_collection = Line3DCollection(ee_segments, cmap=cmap, norm=norm, linewidth=2.0, alpha=0.95)
        ee_collection.set_array(q_values)
        ax.add_collection(ee_collection)
        cbar = fig.colorbar(ee_collection, ax=ax, fraction=0.03, pad=0.1)
        cbar.set_label("Q(s_t, a_{t:t+k-1}) sampled every N frames")
    else:
        ax.plot(
            ee_xyz[chunk_starts, 0],
            ee_xyz[chunk_starts, 1],
            ee_xyz[chunk_starts, 2],
            linestyle="",
            marker="o",
            markersize=4,
            color="tab:blue",
            label="end_effector_chunks",
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
    if args.video_fps <= 0:
        raise ValueError(f"video_fps must be > 0, got {args.video_fps}")
    if args.q_stride <= 0:
        raise ValueError(f"q_stride must be > 0, got {args.q_stride}")
    if args.vmin is not None and args.vmax is not None and args.vmin >= args.vmax:
        raise ValueError(f"Expected vmin < vmax, got vmin={args.vmin}, vmax={args.vmax}")

    checkpoint_path = Path(args.q_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Q checkpoint not found: {checkpoint_path}")

    dataset_path = Path(args.trajectory_path)
    traj_files = _resolve_trajectory_files(dataset_path)
    traj_entries: list[tuple[str, dict[str, Any]]] = []
    for traj_file in traj_files:
        traj_entries.extend(_iter_trajectory_entries(traj_file))
    if args.max_trajs > 0:
        traj_entries = traj_entries[: args.max_trajs]
    if not traj_entries:
        raise FileNotFoundError("No trajectory entries selected after applying max_trajs.")
    manifest_path = _resolve_manifest_path(dataset_path)
    status_map = _load_manifest_success_map(manifest_path)

    if args.output_dir is None:
        output_root = dataset_path if dataset_path.is_dir() else dataset_path.parent
        output_dir = output_root / "q_pose_3d_plots"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir = output_dir / "videos"

    device = torch.device(args.device)
    model, checkpoint = _load_q_model(checkpoint_path, device)
    expected_state_dim = int(checkpoint["state_dim"])
    action_chunk_dim = int(checkpoint["action_chunk_dim"])

    num_processed = 0
    num_videos = 0
    skipped: list[tuple[str, str]] = []
    for traj_source, data in traj_entries:
        try:
            q_result = _compute_q_values_for_traj(
                model=model,
                data=data,
                expected_state_dim=expected_state_dim,
                action_chunk_dim=action_chunk_dim,
                device=device,
                q_stride=args.q_stride,
            )
            traj_id = _extract_traj_id_from_source(traj_source)
            status_label = status_map.get(traj_id, "UNKNOWN")
            if traj_id >= 0:
                out_name = f"q_pose_3d_traj_{traj_id:06d}.png"
            else:
                source_stem = Path(traj_source.split("::")[-1]).stem
                out_name = f"q_pose_3d_{source_stem}.png"
            _plot_traj_q_colored_3d(
                traj_id=traj_id if traj_id >= 0 else 0,
                status_label=status_label,
                q_values=q_result.q_values,
                ee_xyz=q_result.ee_xyz,
                ins_xyz=q_result.ins_xyz,
                rec_xyz=q_result.rec_xyz,
                chunk_starts=q_result.chunk_starts,
                chunk_ends_exclusive=q_result.chunk_ends_exclusive,
                out_path=output_dir / out_name,
                cmap_name=args.cmap,
                dpi=args.dpi,
                vmin=args.vmin,
                vmax=args.vmax,
            )

            frames = _get_frames_from_traj(data)
            if frames is not None:
                q_video = _overlay_q_values_on_video(
                    frames=frames,
                    frame_q_values=q_result.frame_q_values,
                    status_label=status_label,
                    cmap_name=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                )
                video_name = f"q_pose_video_traj_{traj_id:06d}.mp4" if traj_id >= 0 else f"q_pose_video_{out_name[:-4]}.mp4"
                _save_q_overlay_video(q_video, video_output_dir / video_name, args.video_fps)
                num_videos += 1

            num_processed += 1
        except Exception as exc:
            skipped.append((traj_source, str(exc)))
            print(f"[WARN] Skipping {traj_source}: {exc}")

    if num_processed == 0:
        reasons = "\n".join(f"- {p}: {r}" for p, r in skipped[:10])
        raise RuntimeError(
            "No trajectories were successfully processed.\n"
            f"Attempted: {len(traj_entries)}\n"
            f"Examples:\n{reasons}"
        )

    print(f"[INFO] Read trajectory files: {len(traj_files)}")
    print(f"[INFO] Expanded trajectory entries: {len(traj_entries)}")
    print(f"[INFO] Processed trajectories: {num_processed}/{len(traj_entries)}")
    print(f"[INFO] Saved Q-overlay videos: {num_videos}")
    print(f"[INFO] Skipped trajectories: {len(skipped)}")
    print(f"[INFO] Saved Q-colored 3D plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
