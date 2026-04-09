"""Offline Q-noise sweep: dataset action chunks, Gaussian noise in raw space, aggregate Q vs noise."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import dill
import hydra
import matplotlib
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


@dataclass
class QModelBundle:
    model: QImagePolicy
    shape_meta: dict[str, Any]
    abs_action: bool


@dataclass
class RunningStats:
    total: float = 0.0
    total_sq: float = 0.0
    count: int = 0

    def update(self, values: np.ndarray) -> None:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        self.total += float(vals.sum())
        self.total_sq += float(np.square(vals).sum())
        self.count += int(vals.size)

    def to_row(self, noise_std: float) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError(f"No values accumulated for noise std {noise_std}.")
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        stderr = std / float(np.sqrt(self.count))
        return {
            "noise_std": float(noise_std),
            "mean_q": float(mean),
            "std_q": std,
            "stderr_q": float(stderr),
            "count": int(self.count),
        }


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


def _load_payload(ckpt_path: str) -> Any:
    path = Path(ckpt_path).expanduser()
    with path.open("rb") as f:
        return torch.load(f, pickle_module=dill, map_location="cpu")


def _instantiate_workspace(cfg: Any) -> BaseWorkspace:
    cls = hydra.utils.get_class(cfg._target_)
    try:
        workspace = cls(cfg, output_dir=None)
    except TypeError:
        workspace = cls(cfg)
    return cast(BaseWorkspace, workspace)


def _shape_meta_from_cfg(cfg: Any) -> dict[str, Any]:
    shape_meta = OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)
    return cast(dict[str, Any], shape_meta)


def _abs_action_from_cfg(cfg: Any) -> bool:
    return bool(OmegaConf.select(cfg, "task.dataset.abs_action", default=False))


def _load_q_model(ckpt_path: str, device: torch.device) -> QModelBundle:
    payload = _load_payload(ckpt_path)
    cfg = payload["cfg"]
    workspace = _instantiate_workspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    model = getattr(workspace, "model", None)
    if not isinstance(model, QImagePolicy):
        raise TypeError(f"Expected QImagePolicy checkpoint, got {type(model)}")
    model.eval()
    model.to(device)
    return QModelBundle(
        model=model,
        shape_meta=_shape_meta_from_cfg(cfg),
        abs_action=_abs_action_from_cfg(cfg),
    )


def _obs_window_1d(
    series: Any,
    ep_start: int,
    ep_end: int,
    t: int,
    n_obs: int,
) -> np.ndarray:
    """Return series[t0:t+1] with length n_obs, padded before episode start."""
    del ep_end
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
        raise RuntimeError(f"Obs window length mismatch: got {chunk.shape[0]}, expected {n_obs}")
    return chunk


def _action_chunk_pad(actions: np.ndarray, ep_end: int, t: int, n_action: int) -> np.ndarray:
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
    return torch.from_numpy(np.asarray(window, dtype=np.float32)).unsqueeze(0)


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
        arr = obs_group[key]
        window = _obs_window_1d(arr, ep_start, ep_end, t, n_obs)
        if attr.get("type", "low_dim") == "rgb":
            out[key] = _window_to_obs_tensor_rgb(window).to(device, non_blocking=True)
        else:
            out[key] = _window_to_obs_tensor_lowdim(window).to(device, non_blocking=True)
    return out


def _repeat_obs_dict(obs_dict: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    if repeats <= 1:
        return obs_dict
    return {k: v.repeat((repeats,) + (1,) * (v.ndim - 1)) for k, v in obs_dict.items()}


def _convert_actions_abs_like_dataset(actions: np.ndarray, end_effector_pose: np.ndarray) -> np.ndarray:
    """Match Sim2RealImageDataset relative -> absolute conversion (per timestep)."""
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


def _float_list(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("noise_stds must contain at least one value.")
    return [float(v) for v in values]


def _save_stats_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_std", "mean_q", "std_q", "stderr_q", "count"])
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(rows: list[dict[str, float | int]], path: Path) -> None:
    xs = [float(row["noise_std"]) for row in rows]
    ys = [float(row["mean_q"]) for row in rows]
    yerrs = [float(row["stderr_q"]) for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(xs, ys, yerr=yerrs, marker="o", capsize=4)
    plt.xlabel("Gaussian noise std (raw action space)")
    plt.ylabel("Average Q")
    plt.title("Q value vs action noise (dataset chunks)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate how a Q model scores dataset action chunks under raw action noise."
    )
    parser.add_argument("--q_checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help=".zarr dir or .zarr.zip")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--noise_stds",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        help="Raw-action Gaussian noise standard deviations to sweep.",
    )
    parser.add_argument(
        "--samples_per_std",
        type=int,
        default=32,
        help="Number of noisy samples to score per state and per noise std.",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional cap on total dataset states evaluated across all episodes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--abs_action",
        action="store_true",
        help="Convert relative zarr actions to absolute (needs end_effector_pose in obs).",
    )
    args = parser.parse_args()

    if args.samples_per_std <= 0:
        raise ValueError("--samples_per_std must be positive.")
    noise_stds = _float_list(list(args.noise_stds))
    if any(std < 0.0 for std in noise_stds):
        raise ValueError("--noise_stds must all be non-negative.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be positive when provided.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    q_bundle = _load_q_model(args.q_checkpoint, device)
    q_model = q_bundle.model
    print(q_model.normalizer["action"].params_dict["scale"], q_model.normalizer["action"].params_dict["offset"])
    q_shape_meta_obs = cast(dict[str, Any], q_bundle.shape_meta["obs"])

    q_n_obs = int(q_model.n_obs_steps)
    q_n_act = int(q_model.n_action_steps)
    q_action_dim = int(q_bundle.shape_meta["action"]["shape"][0])
    if int(q_model.action_dim) != q_action_dim:
        raise ValueError(
            f"Q checkpoint shape_meta action dim {q_action_dim} does not match model.action_dim {q_model.action_dim}."
        )

    if not args.abs_action and q_bundle.abs_action:
        print(
            "Warning: Q checkpoint was trained with abs_action=True but --abs_action was not set. "
            "Dataset actions are assumed already absolute; use --abs_action if your zarr stores relative actions."
        )
    elif args.abs_action and not q_bundle.abs_action:
        print(
            "Warning: --abs_action set but Q checkpoint cfg has abs_action=False. "
            "Converted actions may not match Q training if Q expected relative actions."
        )

    root = _open_zarr_root(args.dataset)
    if "data" not in root or "obs" not in root["data"]:
        raise KeyError("Expected zarr layout with group data/obs.")
    obs_grp = root["data"]["obs"]
    if "actions" not in root["data"]:
        raise KeyError("Expected zarr data/actions.")
    actions_z = root["data"]["actions"]
    episode_ends = np.asarray(root["meta"]["episode_ends"])

    for key in q_shape_meta_obs:
        if key not in obs_grp:
            raise KeyError(
                f"Q obs key '{key}' not found in zarr data/obs. Available: {list(obs_grp.keys())}"
            )

    actions = np.asarray(actions_z[:])
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D (T, A), got {actions.shape}")

    if args.abs_action:
        if "end_effector_pose" not in obs_grp:
            raise ValueError("--abs_action requires end_effector_pose in data/obs.")
        ee = np.asarray(obs_grp["end_effector_pose"][:])
        actions = _convert_actions_abs_like_dataset(actions, ee)

    if actions.shape[1] != q_action_dim:
        raise ValueError(
            f"Dataset action dim {actions.shape[1]} does not match Q model ({q_action_dim}). "
            "Use or omit --abs_action to match how Q was trained."
        )

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_by_std = {std: RunningStats() for std in noise_stds}
    ep_starts = np.insert(episode_ends, 0, 0)[:-1].astype(np.int64)
    n_eps = len(episode_ends)
    max_episodes = args.max_episodes if args.max_episodes is not None else n_eps
    rng = np.random.default_rng(args.seed)
    total_states = 0

    print(
        f"Q model: n_obs_steps={q_n_obs}, n_action_steps={q_n_act}, action_dim={q_action_dim}. "
        f"Episodes: {n_eps}, evaluating up to {min(max_episodes, n_eps)}. "
        f"Dataset abs_action flag (--abs_action): {args.abs_action}."
    )

    stop = False
    start_idx = np.random.randint(0, n_eps - max_episodes)
    for ep_idx in range(start_idx, start_idx + max_episodes):
        ep_start = int(ep_starts[ep_idx])
        ep_end = int(episode_ends[ep_idx])

        for t in range(ep_start, ep_end):
            if args.max_steps is not None and total_states >= args.max_steps:
                stop = True
                break

            q_obs = _build_obs_dict(
                obs_grp,
                q_shape_meta_obs,
                ep_start,
                ep_end,
                t,
                q_n_obs,
                device,
            )
            base_action_chunk = _action_chunk_pad(actions, ep_end, t, q_n_act)
            if tuple(base_action_chunk.shape) != (q_n_act, q_action_dim):
                raise ValueError(
                    f"Action chunk has shape {tuple(base_action_chunk.shape)}; "
                    f"expected ({q_n_act}, {q_action_dim})."
                )

            repeated_q_obs = _repeat_obs_dict(q_obs, args.samples_per_std)
            action_batch = np.repeat(base_action_chunk[None, ...], args.samples_per_std, axis=0)

            with torch.inference_mode():
                for noise_std in noise_stds:
                    if noise_std == 0.0:
                        noisy_actions = action_batch
                    else:
                        noise = rng.normal(
                            loc=0.0,
                            scale=noise_std,
                            size=action_batch.shape,
                        ).astype(np.float32)
                        noisy_actions = action_batch + noise
                    act_t = torch.from_numpy(noisy_actions).to(
                        device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    q_values = q_model.predict_q(repeated_q_obs, act_t).reshape(-1)
                    stats_by_std[noise_std].update(q_values.detach().cpu().numpy())

            total_states += 1
            if total_states % 100 == 0:
                print(f"Processed {total_states} states...")

        print(f"Finished episode {ep_idx} ({ep_end - ep_start} states available).")
        if stop:
            break

    if total_states == 0:
        raise RuntimeError("No dataset states were evaluated. Check --max_episodes/--max_steps.")

    rows = [stats_by_std[std].to_row(std) for std in noise_stds]
    csv_path = out_dir / "q_noise_stats.csv"
    json_path = out_dir / "q_noise_stats.json"
    plot_path = out_dir / "q_noise_curve.png"

    _save_stats_csv(rows, csv_path)
    with json_path.open("w") as f:
        json.dump(
            {
                "q_checkpoint": str(Path(args.q_checkpoint).expanduser()),
                "dataset": str(Path(args.dataset).expanduser()),
                "seed": int(args.seed),
                "abs_action": bool(args.abs_action),
                "samples_per_std": int(args.samples_per_std),
                "total_states": int(total_states),
                "rows": rows,
            },
            f,
            indent=2,
        )
    _save_plot(rows, plot_path)

    print(f"Processed {total_states} states total.")
    print(f"Wrote plot to {plot_path}")
    print(f"Wrote stats to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
