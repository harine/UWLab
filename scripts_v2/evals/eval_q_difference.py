"""Compare bad-policy action distance from a good policy against Q deltas on random dataset states."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
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

from diffusion_policy.policy.q_image_policy import QImagePolicy  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402


@dataclass
class ModelBundle:
    model: Any
    shape_meta: dict[str, Any]
    abs_action: bool


@dataclass
class ScatterPoint:
    episode_index: int
    global_timestep: int
    episode_timestep: int
    sample_index: int
    l2_distance: float
    q_bad_minus_good: float
    q_good: float
    q_bad: float


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


def _load_policy_model(ckpt_path: str, device: torch.device, label: str) -> ModelBundle:
    payload = _load_payload(ckpt_path)
    cfg = payload["cfg"]
    workspace = _instantiate_workspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace_any = cast(Any, workspace)
    use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False))
    policy = (
        workspace_any.ema_model
        if use_ema and hasattr(workspace_any, "ema_model")
        else workspace_any.model
    )
    if not hasattr(policy, "predict_action"):
        raise TypeError(f"Expected policy checkpoint in {label}, got {type(policy)}")
    policy.eval()
    policy.to(device)
    return ModelBundle(
        model=policy,
        shape_meta=_shape_meta_from_cfg(cfg),
        abs_action=_abs_action_from_cfg(cfg),
    )


def _load_q_model(ckpt_path: str, device: torch.device) -> ModelBundle:
    payload = _load_payload(ckpt_path)
    cfg = payload["cfg"]
    workspace = _instantiate_workspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    model = getattr(workspace, "model", None)
    if not isinstance(model, QImagePolicy):
        raise TypeError(f"Expected QImagePolicy checkpoint, got {type(model)}")
    model.eval()
    model.to(device)
    return ModelBundle(
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
        raise RuntimeError(
            f"Obs window length mismatch: got {chunk.shape[0]}, expected {n_obs}"
        )
    return chunk


def _window_to_obs_tensor_rgb(window: np.ndarray) -> torch.Tensor:
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


def _extract_single_action(result: dict[str, Any]) -> torch.Tensor:
    action = result.get("action")
    if not torch.is_tensor(action):
        raise TypeError("predict_action() must return a tensor under the 'action' key.")
    if action.ndim != 3 or action.shape[0] != 1:
        raise ValueError(
            "Expected predict_action() for one state to return shape (1, Ta, Da), "
            f"got {tuple(action.shape)}."
        )
    return action


def _extract_candidate_actions(action_result: dict[str, Any]) -> torch.Tensor:
    actions = action_result.get("actions")
    if not torch.is_tensor(actions):
        raise TypeError("predict_k_actions() must return a tensor under the 'actions' key.")
    if actions.ndim == 3:
        return actions
    if actions.ndim != 4:
        raise ValueError(f"Expected actions with 3 or 4 dims, got shape {tuple(actions.shape)}")
    if actions.shape[0] == 1:
        return actions[0]
    if actions.shape[1] == 1:
        return actions[:, 0]
    raise ValueError(
        "Expected predict_k_actions() output for a single state to have a singleton batch dimension, "
        f"got shape {tuple(actions.shape)}."
    )


def _sample_one_action(policy: Any, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(policy, "predict_k_actions"):
        result = policy.predict_k_actions(obs_dict, 1)
        actions = _extract_candidate_actions(result)
        if actions.shape[0] != 1:
            raise ValueError(f"Expected one sampled action, got shape {tuple(actions.shape)}")
        return actions
    return _extract_single_action(policy.predict_action(obs_dict))


def _sample_k_actions(policy: Any, obs_dict: dict[str, torch.Tensor], k: int) -> torch.Tensor:
    if hasattr(policy, "predict_k_actions"):
        return _extract_candidate_actions(policy.predict_k_actions(obs_dict, k))
    samples = [_extract_single_action(policy.predict_action(obs_dict)) for _ in range(k)]
    return torch.cat(samples, dim=0)


def _save_rows_csv(rows: list[ScatterPoint], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _save_plot(rows: list[ScatterPoint], path: Path, title: str) -> None:
    unique_eps = sorted({row.episode_index for row in rows})
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    cmap = plt.get_cmap("tab10", len(unique_eps))

    fig, ax = plt.subplots(figsize=(8, 6))
    for color_idx, episode_index in enumerate(unique_eps):
        ep_rows = [row for row in rows if row.episode_index == episode_index]
        xs = [row.l2_distance for row in ep_rows]
        ys = [row.q_bad_minus_good for row in ep_rows]
        ax.scatter(
            xs,
            ys,
            s=28,
            alpha=0.8,
            color=cmap(color_idx),
            marker=markers[color_idx % len(markers)],
            label=f"episode {episode_index}",
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("L2 distance from good action")
    ax.set_ylabel("Q(bad action) - Q(good action)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For one random state from each of several random dataset episodes, compare "
            "bad-policy action distance from a good-policy sample against Q deltas."
        )
    )
    parser.add_argument("--good_policy_checkpoint", type=str, required=True)
    parser.add_argument("--bad_policy_checkpoint", type=str, required=True)
    parser.add_argument("--q_checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help=".zarr dir or .zarr.zip")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of bad-policy actions sampled per selected state.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of random episodes to sample.",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    good_bundle = _load_policy_model(args.good_policy_checkpoint, device, "good policy checkpoint")
    bad_bundle = _load_policy_model(args.bad_policy_checkpoint, device, "bad policy checkpoint")
    q_bundle = _load_q_model(args.q_checkpoint, device)

    good_policy = good_bundle.model
    bad_policy = bad_bundle.model
    q_model = cast(QImagePolicy, q_bundle.model)

    good_shape_meta_obs = cast(dict[str, Any], good_bundle.shape_meta["obs"])
    bad_shape_meta_obs = cast(dict[str, Any], bad_bundle.shape_meta["obs"])
    q_shape_meta_obs = cast(dict[str, Any], q_bundle.shape_meta["obs"])

    good_n_obs = int(getattr(good_policy, "n_obs_steps"))
    bad_n_obs = int(getattr(bad_policy, "n_obs_steps"))
    q_n_obs = int(q_model.n_obs_steps)

    good_n_act = int(getattr(good_policy, "n_action_steps"))
    bad_n_act = int(getattr(bad_policy, "n_action_steps"))
    q_n_act = int(q_model.n_action_steps)

    good_action_dim = int(good_bundle.shape_meta["action"]["shape"][0])
    bad_action_dim = int(bad_bundle.shape_meta["action"]["shape"][0])
    q_action_dim = int(q_bundle.shape_meta["action"]["shape"][0])

    if good_n_act != q_n_act:
        raise ValueError(
            f"Good policy n_action_steps ({good_n_act}) does not match Q n_action_steps ({q_n_act})."
        )
    if bad_n_act != q_n_act:
        raise ValueError(
            f"Bad policy n_action_steps ({bad_n_act}) does not match Q n_action_steps ({q_n_act})."
        )
    if good_action_dim != q_action_dim:
        raise ValueError(
            f"Good policy action dim ({good_action_dim}) does not match Q action dim ({q_action_dim})."
        )
    if bad_action_dim != q_action_dim:
        raise ValueError(
            f"Bad policy action dim ({bad_action_dim}) does not match Q action dim ({q_action_dim})."
        )

    if good_bundle.abs_action != q_bundle.abs_action or bad_bundle.abs_action != q_bundle.abs_action:
        print(
            "Warning: at least one policy checkpoint disagrees with the Q checkpoint on abs_action. "
            "Q comparisons may be hard to interpret if action conventions differ."
        )

    root = _open_zarr_root(args.dataset)
    if "data" not in root or "obs" not in root["data"]:
        raise KeyError("Expected zarr layout with group data/obs.")
    obs_grp = root["data"]["obs"]
    episode_ends = np.asarray(root["meta"]["episode_ends"])

    for key in good_shape_meta_obs:
        if key not in obs_grp:
            raise KeyError(
                f"Good policy obs key '{key}' not found in zarr data/obs. Available: {list(obs_grp.keys())}"
            )
    for key in bad_shape_meta_obs:
        if key not in obs_grp:
            raise KeyError(
                f"Bad policy obs key '{key}' not found in zarr data/obs. Available: {list(obs_grp.keys())}"
            )
    for key in q_shape_meta_obs:
        if key not in obs_grp:
            raise KeyError(
                f"Q obs key '{key}' not found in zarr data/obs. Available: {list(obs_grp.keys())}"
            )

    n_eps = int(len(episode_ends))
    if n_eps < args.num_episodes:
        raise ValueError(
            f"Dataset has only {n_eps} episodes, but --num_episodes={args.num_episodes} was requested."
        )

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_starts = np.insert(episode_ends, 0, 0)[:-1].astype(np.int64)
    rng = np.random.default_rng(args.seed)
    selected_episodes = np.sort(rng.choice(n_eps, size=args.num_episodes, replace=False))

    rows: list[ScatterPoint] = []
    selected_states: list[dict[str, int]] = []

    print(
        f"Good policy: n_obs_steps={good_n_obs}, n_action_steps={good_n_act}, action_dim={good_action_dim}. "
        f"Bad policy: n_obs_steps={bad_n_obs}, n_action_steps={bad_n_act}, action_dim={bad_action_dim}. "
        f"Q: n_obs_steps={q_n_obs}, n_action_steps={q_n_act}, action_dim={q_action_dim}. "
        f"Sampling {args.num_episodes} episodes with k={args.k} bad actions each."
    )

    with torch.inference_mode():
        for episode_index in selected_episodes.tolist():
            ep_start = int(ep_starts[episode_index])
            ep_end = int(episode_ends[episode_index])
            if ep_end <= ep_start:
                raise ValueError(f"Episode {episode_index} is empty.")

            t = int(rng.integers(ep_start, ep_end))
            selected_states.append(
                {
                    "episode_index": episode_index,
                    "global_timestep": t,
                    "episode_timestep": t - ep_start,
                    "episode_length": ep_end - ep_start,
                }
            )

            if hasattr(good_policy, "reset"):
                good_policy.reset()
            if hasattr(bad_policy, "reset"):
                bad_policy.reset()

            good_obs = _build_obs_dict(
                obs_grp,
                good_shape_meta_obs,
                ep_start,
                ep_end,
                t,
                good_n_obs,
                device,
            )
            bad_obs = _build_obs_dict(
                obs_grp,
                bad_shape_meta_obs,
                ep_start,
                ep_end,
                t,
                bad_n_obs,
                device,
            )
            q_obs = _build_obs_dict(
                obs_grp,
                q_shape_meta_obs,
                ep_start,
                ep_end,
                t,
                q_n_obs,
                device,
            )

            good_action = _sample_one_action(good_policy, good_obs).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            bad_actions = _sample_k_actions(bad_policy, bad_obs, args.k).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            if tuple(good_action.shape) != (1, q_n_act, q_action_dim):
                raise ValueError(
                    f"Good policy action shape mismatch: got {tuple(good_action.shape)}, "
                    f"expected (1, {q_n_act}, {q_action_dim})."
                )
            if tuple(bad_actions.shape[1:]) != (q_n_act, q_action_dim):
                raise ValueError(
                    f"Bad policy action shape mismatch: got {tuple(bad_actions.shape)}, "
                    f"expected (K, {q_n_act}, {q_action_dim})."
                )

            q_good = float(q_model.predict_q(q_obs, good_action).reshape(-1)[0].item())
            repeated_q_obs = _repeat_obs_dict(q_obs, int(bad_actions.shape[0]))
            q_bad = q_model.predict_q(repeated_q_obs, bad_actions).reshape(-1)

            good_action_flat = good_action[0].reshape(-1)
            distances = torch.linalg.vector_norm(
                bad_actions.reshape(bad_actions.shape[0], -1) - good_action_flat.unsqueeze(0),
                dim=1,
            )
            q_deltas = q_bad - q_good

            for sample_index in range(int(bad_actions.shape[0])):
                rows.append(
                    ScatterPoint(
                        episode_index=episode_index,
                        global_timestep=t,
                        episode_timestep=t - ep_start,
                        sample_index=sample_index,
                        l2_distance=float(distances[sample_index].item()),
                        q_bad_minus_good=float(q_deltas[sample_index].item()),
                        q_good=q_good,
                        q_bad=float(q_bad[sample_index].item()),
                    )
                )

            print(
                f"Episode {episode_index}: chose timestep {t - ep_start}/{ep_end - ep_start - 1}, "
                f"q_good={q_good:.6f}, sampled {bad_actions.shape[0]} bad actions."
            )

    if not rows:
        raise RuntimeError("No scatter points were produced.")

    plot_path = out_dir / "q_difference_scatter.png"
    csv_path = out_dir / "q_difference_points.csv"
    json_path = out_dir / "q_difference_summary.json"

    title = "Bad-policy distance from good action vs Q delta"
    _save_plot(rows, plot_path, title)
    _save_rows_csv(rows, csv_path)

    summary = {
        "good_policy_checkpoint": str(Path(args.good_policy_checkpoint).expanduser()),
        "bad_policy_checkpoint": str(Path(args.bad_policy_checkpoint).expanduser()),
        "q_checkpoint": str(Path(args.q_checkpoint).expanduser()),
        "dataset": str(Path(args.dataset).expanduser()),
        "seed": int(args.seed),
        "k": int(args.k),
        "num_episodes": int(args.num_episodes),
        "selected_episodes": selected_episodes.tolist(),
        "selected_states": selected_states,
        "num_points": len(rows),
        "mean_l2_distance": float(np.mean([row.l2_distance for row in rows])),
        "mean_q_bad_minus_good": float(np.mean([row.q_bad_minus_good for row in rows])),
    }
    with json_path.open("w") as f:
        json.dump(
            {
                **summary,
                "rows": [asdict(row) for row in rows],
            },
            f,
            indent=2,
        )

    print(f"Wrote plot to {plot_path}")
    print(f"Wrote point data to {csv_path}")
    print(f"Wrote summary to {json_path}")


if __name__ == "__main__":
    main()
