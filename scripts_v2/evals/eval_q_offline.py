# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline Q evaluation: dataset trajectories, expert action chunks, RGB videos with Q (and optional V) overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import dill
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIFFUSION_POLICY_ROOT = _REPO_ROOT / "diffusion_policy"
if str(_DIFFUSION_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_POLICY_ROOT))

from diffusion_policy.dataset.utils import (  # noqa: E402
    pose_axis_angle_to_pos_quat,
    process_actions,
)
from diffusion_policy.policy.q_image_policy import QImagePolicy  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402


VIDEO_RGB_KEYS = ("front_rgb", "side_rgb", "wrist_rgb")


def _open_zarr_root(dataset_path: str) -> Any:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    name = path.name
    if name.endswith(".zarr.zip"):
        store = zarr.ZipStore(str(path), mode="r")
        return zarr.open_group(store=store, mode="r")
    if path.is_dir() and (path.suffix == ".zarr" or name.endswith(".zarr")):
        return zarr.open_group(str(path), mode="r")
    raise ValueError(f"Expected a .zarr directory or .zarr.zip file, got: {path}")


def _load_q_image_policy(
    ckpt_path: str, device: torch.device, label: str = "checkpoint"
) -> tuple[QImagePolicy, dict[str, Any]]:
    path = Path(ckpt_path).expanduser()
    with path.open("rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    model = cast(Any, workspace).model
    if not isinstance(model, QImagePolicy):
        raise TypeError(f"Expected QImagePolicy in {label}, got {type(model)}")
    shape_meta = OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)
    shape_meta = cast(dict[str, Any], shape_meta)
    model.eval()
    model.to(device)
    return model, shape_meta


def _load_q_model(
    ckpt_path: str, device: torch.device
) -> tuple[QImagePolicy, dict[str, Any]]:
    return _load_q_image_policy(ckpt_path, device, label="Q checkpoint")


def _convert_actions_abs_like_dataset(
    actions: np.ndarray, end_effector_pose: np.ndarray
) -> np.ndarray:
    """Match Sim2RealImageDataset relative -> absolute conversion (global)."""
    action_data = np.asarray(actions, dtype=np.float32)
    ref_pose = torch.from_numpy(np.asarray(end_effector_pose, dtype=np.float32))
    ref_pos, ref_quat = pose_axis_angle_to_pos_quat(ref_pose)
    abs_pos, abs_quat = process_actions(
        torch.from_numpy(action_data[:, :6]).to(dtype=torch.float32),
        ref_pos,
        ref_quat,
    )
    gripper_action = torch.from_numpy(action_data[:, 6:]).to(dtype=torch.float32)
    return torch.cat([abs_pos, abs_quat, gripper_action], dim=-1).cpu().numpy()


def _ensure_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """Return HxWx3 uint8."""
    x = np.asarray(img)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[0] < x.shape[1]:
        x = np.moveaxis(x, 0, -1)
    if x.shape[-1] == 4:
        x = x[..., :3]
    if np.issubdtype(x.dtype, np.floating):
        if float(np.max(x)) <= 1.0:
            x = (x * 255.0).clip(0, 255)
        x = x.astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return x


def _obs_window_1d(
    series: Any,
    ep_start: int,
    ep_end: int,
    t: int,
    n_obs: int,
) -> np.ndarray:
    """Return series[t0:t+1] with length n_obs, pad before episode start by repeating first in-window row."""
    start_idx = t - n_obs + 1
    end_idx = t + 1
    if start_idx < ep_start:
        chunk = np.asarray(series[ep_start:end_idx])
        pad_n = n_obs - len(chunk)
        if pad_n > 0:
            first = np.expand_dims(chunk[0], axis=0)
            chunk = np.concatenate([np.repeat(first, pad_n, axis=0), chunk], axis=0)
    else:
        chunk = np.asarray(series[start_idx:end_idx])
    if chunk.shape[0] != n_obs:
        raise RuntimeError(
            f"Obs window length mismatch: got {chunk.shape[0]}, expected {n_obs}"
        )
    return chunk


def _action_chunk_pad(
    actions: np.ndarray, ep_end: int, t: int, n_action: int
) -> np.ndarray:
    chunk = np.asarray(actions[t : min(t + n_action, ep_end)])
    if len(chunk) < n_action:
        if len(chunk) == 0:
            raise ValueError(f"Empty action chunk at t={t}, ep_end={ep_end}")
        pad = np.repeat(np.expand_dims(chunk[-1], 0), n_action - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def _window_to_obs_tensor_rgb(window: np.ndarray) -> torch.Tensor:
    """(T, H, W, C) or (T, C, H, W) -> float tensor (1, T, C, H, W)."""
    w = np.asarray(window)
    if w.ndim != 4:
        raise ValueError(f"Expected 4D rgb window, got shape {w.shape}")
    if w.shape[-1] in (3, 4):
        x = np.moveaxis(w, -1, 1).astype(np.float32) / 255.0
    elif w.shape[1] in (1, 3, 4):
        x = w.astype(np.float32)
        if float(np.max(x)) > 1.5:
            x = x / 255.0
    else:
        raise ValueError(f"Unrecognized rgb layout, shape {w.shape}")
    return torch.from_numpy(x).unsqueeze(0)


def _window_to_obs_tensor_lowdim(window: np.ndarray) -> torch.Tensor:
    """(T, D) -> (1, T, D)."""
    w = np.asarray(window, dtype=np.float32)
    return torch.from_numpy(w).unsqueeze(0)


def _build_obs_dict(
    obs_group: zarr.Group,
    shape_meta_obs: dict[str, Any],
    ep_start: int,
    ep_end: int,
    t: int,
    n_obs: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, attr in shape_meta_obs.items():
        typ = attr.get("type", "low_dim")
        arr = obs_group[key]
        win = _obs_window_1d(arr, ep_start, ep_end, t, n_obs)
        if typ == "rgb":
            out[key] = _window_to_obs_tensor_rgb(win).to(device, non_blocking=True)
        else:
            out[key] = _window_to_obs_tensor_lowdim(win).to(device, non_blocking=True)
    return out


def _concat_video_frame(
    obs_group: zarr.Group, t: int, target_height: int = 224
) -> np.ndarray:
    tiles = []
    for name in VIDEO_RGB_KEYS:
        if name not in obs_group:
            raise KeyError(
                f"Video key '{name}' missing in zarr data/obs. Present: {list(obs_group.keys())}"
            )
        raw = np.asarray(obs_group[name][t])
        img = _ensure_uint8_hwc(raw)
        h, w = img.shape[0], img.shape[1]
        if h != target_height:
            scale = target_height / h
            img = cv2.resize(img, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)
        tiles.append(img)
    max_h = max(im.shape[0] for im in tiles)
    resized = []
    for im in tiles:
        if im.shape[0] != max_h:
            scale = max_h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), max_h), interpolation=cv2.INTER_AREA)
        resized.append(im)
    return np.concatenate(resized, axis=1)


def _overlay_metrics(
    frame: np.ndarray,
    q_val: float,
    t: int,
    episode: int,
    value_val: float | None = None,
) -> np.ndarray:
    out = frame.copy()
    bar_h = 72 if value_val is not None else 40
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    line1 = f"ep {episode}  t {t}  Q = {q_val:.5f}"
    cv2.putText(
        out,
        line1,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if value_val is not None:
        line2 = f"V = {value_val:.5f}"
        cv2.putText(
            out,
            line2,
            (8, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 255, 200),
            2,
            cv2.LINE_AA,
        )
    return out


def _resolve_checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Return (q_ckpt, value_ckpt_or_none). Requires q_ckpt to exist as a file."""
    q_default: Path | None = None
    v_default: Path | None = None
    if args.checkpoint:
        d = Path(args.checkpoint).expanduser()
        if not d.is_dir():
            raise NotADirectoryError(f"--checkpoint must be a directory: {d}")
        q_default = d / "q_best.ckpt"
        v_default = d / "value_best.ckpt"
        if not q_default.is_file():
            raise FileNotFoundError(f"Expected Q checkpoint at {q_default}")
        if not v_default.is_file():
            raise FileNotFoundError(f"Expected value checkpoint at {v_default}")

    q_path = Path(args.q_checkpoint).expanduser() if args.q_checkpoint else q_default
    if q_path is None:
        raise ValueError(
            "Provide --checkpoint DIR (containing q_best.ckpt and value_best.ckpt) or --q_checkpoint PATH."
        )
    if not q_path.is_file():
        raise FileNotFoundError(f"Q checkpoint not found: {q_path}")

    if args.value_checkpoint:
        v_path = Path(args.value_checkpoint).expanduser()
        if not v_path.is_file():
            raise FileNotFoundError(f"Value checkpoint not found: {v_path}")
    elif v_default is not None:
        v_path = v_default
    else:
        v_path = None

    return q_path, v_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Q eval with RGB videos from zarr.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Directory containing q_best.ckpt and value_best.ckpt (loads both).",
    )
    parser.add_argument(
        "--q_checkpoint",
        type=str,
        default=None,
        help="Q checkpoint path (overrides --checkpoint/q_best.ckpt if both given).",
    )
    parser.add_argument("--dataset", type=str, required=True, help=".zarr dir or .zarr.zip")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fps", type=float, default=10.0, help="Nominal dataset / playback FPS.")
    parser.add_argument(
        "--slowdown",
        type=float,
        default=2.0,
        help="Output fps = fps / slowdown (2 => half fps, 2x slower).",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--abs_action",
        action="store_true",
        help="Convert relative actions to absolute (needs end_effector_pose in obs).",
    )
    parser.add_argument(
        "--value_checkpoint",
        type=str,
        default=None,
        help="Value-head checkpoint (overrides --checkpoint/value_best.ckpt). Q-only if omitted without --checkpoint.",
    )
    args = parser.parse_args()

    try:
        q_ckpt_path, value_ckpt_path = _resolve_checkpoint_paths(args)
    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        parser.error(str(e))

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    q_model, shape_meta = _load_q_model(str(q_ckpt_path), device)
    shape_meta_obs = shape_meta["obs"]
    assert isinstance(shape_meta_obs, dict)
    action_dim_expect = int(shape_meta["action"]["shape"][0])

    value_model: QImagePolicy | None = None
    value_meta_obs: dict[str, Any] | None = None
    value_n_obs: int | None = None
    if value_ckpt_path is not None:
        value_model, value_shape_meta = _load_q_image_policy(
            str(value_ckpt_path), device, label="value checkpoint"
        )
        vo = value_shape_meta["obs"]
        assert isinstance(vo, dict)
        value_meta_obs = vo
        value_n_obs = int(value_model.n_obs_steps)
        if value_model.action_dim != action_dim_expect:
            print(
                f"Warning: value checkpoint action_dim={value_model.action_dim} != Q action_dim={action_dim_expect} "
                "(value head does not use actions; continuing)."
            )

    n_obs = int(q_model.n_obs_steps)
    n_act = int(q_model.n_action_steps)
    if q_model.action_dim != action_dim_expect:
        raise ValueError(
            f"shape_meta action dim {action_dim_expect} vs model.action_dim {q_model.action_dim}"
        )

    root = _open_zarr_root(args.dataset)
    if "data" not in root or "obs" not in root["data"]:
        raise KeyError("Expected zarr layout with group data/obs.")
    obs_grp = root["data"]["obs"]
    if "actions" not in root["data"]:
        raise KeyError("Expected zarr data/actions.")
    actions_z = root["data"]["actions"]
    episode_ends = np.asarray(root["meta"]["episode_ends"])

    for key in shape_meta_obs:
        if key not in obs_grp:
            raise KeyError(
                f"Q obs key '{key}' not found in zarr data/obs. Available: {list(obs_grp.keys())}"
            )

    if value_meta_obs is not None:
        for key in value_meta_obs:
            if key not in obs_grp:
                raise KeyError(
                    f"Value obs key '{key}' not found in zarr data/obs. Available: {list(obs_grp.keys())}"
                )

    actions = np.asarray(actions_z[:])
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D (T, A), got {actions.shape}")

    if args.abs_action:
        if "end_effector_pose" not in obs_grp:
            raise ValueError("--abs_action requires end_effector_pose in data/obs.")
        ee = np.asarray(obs_grp["end_effector_pose"][:])
        actions = _convert_actions_abs_like_dataset(actions, ee)

    if actions.shape[1] != action_dim_expect:
        raise ValueError(
            f"Dataset action dim {actions.shape[1]} does not match Q model ({action_dim_expect}). "
            "Use or omit --abs_action to match how Q was trained."
        )

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fps = float(args.fps) / float(args.slowdown)

    ep_starts = np.insert(episode_ends, 0, 0)[:-1].astype(np.int64)
    n_eps = len(episode_ends)
    cap = args.max_episodes if args.max_episodes is not None else n_eps

    msg = (
        f"Q model: n_obs_steps={n_obs}, n_action_steps={n_act}, action_dim={action_dim_expect}. "
        f"Episodes: {n_eps}, writing up to {cap}. Output FPS: {out_fps:.3f}"
    )
    if value_model is not None and value_n_obs is not None:
        msg += f" | Value model: n_obs_steps={value_n_obs}"
    print(msg)

    for ep_idx in range(min(cap, n_eps)):
        ep_start = int(ep_starts[ep_idx])
        ep_end = int(episode_ends[ep_idx])
        frames: list[np.ndarray] = []
        for t in range(ep_start, ep_end):
            obs_dict = _build_obs_dict(
                obs_grp, shape_meta_obs, ep_start, ep_end, t, n_obs, device
            )
            a_chunk = _action_chunk_pad(actions, ep_end, t, n_act)
            act_t = torch.from_numpy(np.expand_dims(a_chunk, 0)).to(
                device, dtype=torch.float32, non_blocking=True
            )
            v_val: float | None = None
            with torch.inference_mode():
                q_t = q_model.predict_q(obs_dict, act_t).squeeze()
                if value_model is not None and value_meta_obs is not None and value_n_obs is not None:
                    if value_n_obs == n_obs and value_meta_obs.keys() == shape_meta_obs.keys():
                        obs_for_value = obs_dict
                    else:
                        obs_for_value = _build_obs_dict(
                            obs_grp,
                            value_meta_obs,
                            ep_start,
                            ep_end,
                            t,
                            value_n_obs,
                            device,
                        )
                    v_val = float(value_model.predict_value(obs_for_value).squeeze().item())
            q_val = float(q_t.item())
            vid = _concat_video_frame(obs_grp, t)
            frames.append(_overlay_metrics(vid, q_val, t, ep_idx, v_val))

        out_path = out_dir / f"ep_{ep_idx:04d}.mp4"
        imageio.mimsave(str(out_path), frames, fps=out_fps, codec="libx264")  # type: ignore[arg-type]
        print(f"Wrote {out_path} ({len(frames)} frames)")

    print("Done.")


if __name__ == "__main__":
    main()
