"""Offline policy/Q variance evaluation over dataset states."""

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
class StateMetrics:
    state_index: int
    episode_index: int
    timestep: int
    action_variance: float
    q_variance: float
    q_mean: float
    q_min: float
    q_max: float
    num_candidates: int


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


def _load_policy_model(ckpt_path: str, device: torch.device) -> ModelBundle:
    payload = _load_payload(ckpt_path)
    cfg = payload["cfg"]
    workspace = _instantiate_workspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace_any = cast(Any, workspace)
    use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False))
    policy = workspace_any.ema_model if use_ema and hasattr(workspace_any, "ema_model") else workspace_any.model
    if not hasattr(policy, "predict_k_actions"):
        raise TypeError(
            f"Expected policy checkpoint with predict_k_actions(), got {type(policy)}"
        )
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
        raise RuntimeError(f"Obs window length mismatch: got {chunk.shape[0]}, expected {n_obs}")
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


def _candidate_count(actions: torch.Tensor) -> int:
    if actions.ndim != 3:
        raise ValueError(f"Expected candidate actions shaped (K, Ta, Da), got {tuple(actions.shape)}")
    return int(actions.shape[0])


def _metric_array(values: list[StateMetrics], key: str) -> np.ndarray:
    return np.asarray([getattr(row, key) for row in values], dtype=np.float64)


def _save_rows_csv(rows: list[StateMetrics], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _mask_below_percentile(values: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    cutoff = float(np.percentile(values, percentile))
    return values <= cutoff, cutoff


def _save_plot(rows: list[StateMetrics], path: Path) -> None:
    state_ids = np.arange(len(rows), dtype=np.int64)
    action_vars = _metric_array(rows, "action_variance")
    q_vars = _metric_array(rows, "q_variance")
    action_mask, action_cutoff = _mask_below_percentile(action_vars, 99.0)
    q_mask, q_cutoff = _mask_below_percentile(q_vars, 99.0)
    scatter_mask = action_mask & q_mask

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

    axes[0].plot(state_ids[action_mask], action_vars[action_mask], linewidth=1.2, alpha=0.85)
    axes[0].axhline(action_vars[action_mask].mean(), color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"Action variance across policy samples (<=99th pct, cutoff={action_cutoff:.4g})")
    axes[0].set_ylabel("Mean elementwise variance")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(state_ids[q_mask], q_vars[q_mask], linewidth=1.2, alpha=0.85, color="tab:green")
    axes[1].axhline(q_vars[q_mask].mean(), color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"Q variance across sampled action chunks (<=99th pct, cutoff={q_cutoff:.4g})")
    axes[1].set_ylabel("Variance")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(action_vars[scatter_mask], q_vars[scatter_mask], s=12, alpha=0.7)
    axes[2].set_title("Per-state action variance vs Q variance (<=99th pct)")
    axes[2].set_xlabel("Action variance")
    axes[2].set_ylabel("Q variance")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure policy action variance and induced Q variance over offline dataset states."
    )
    parser.add_argument("--policy_checkpoint", type=str, required=True)
    parser.add_argument("--q_checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help=".zarr dir or .zarr.zip")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of candidate action chunks to sample per state.",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional cap on total states evaluated across all episodes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive.")
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

    policy_bundle = _load_policy_model(args.policy_checkpoint, device)
    q_bundle = _load_q_model(args.q_checkpoint, device)
    policy = policy_bundle.model
    q_model = cast(QImagePolicy, q_bundle.model)

    policy_shape_meta_obs = cast(dict[str, Any], policy_bundle.shape_meta["obs"])
    q_shape_meta_obs = cast(dict[str, Any], q_bundle.shape_meta["obs"])

    policy_n_obs = int(getattr(policy, "n_obs_steps"))
    policy_n_act = int(getattr(policy, "n_action_steps"))
    q_n_obs = int(q_model.n_obs_steps)
    q_n_act = int(q_model.n_action_steps)

    policy_action_dim = int(policy_bundle.shape_meta["action"]["shape"][0])
    q_action_dim = int(q_bundle.shape_meta["action"]["shape"][0])

    if policy_n_act != q_n_act:
        raise ValueError(
            f"Policy n_action_steps ({policy_n_act}) does not match Q n_action_steps ({q_n_act})."
        )
    if policy_action_dim != q_action_dim:
        raise ValueError(
            f"Policy action dim ({policy_action_dim}) does not match Q action dim ({q_action_dim})."
        )
    if policy_bundle.abs_action != q_bundle.abs_action:
        print(
            "Warning: policy and Q checkpoints disagree on abs_action. "
            "Q scores may be hard to interpret if they use different action conventions."
        )

    root = _open_zarr_root(args.dataset)
    if "data" not in root or "obs" not in root["data"]:
        raise KeyError("Expected zarr layout with group data/obs.")
    obs_grp = root["data"]["obs"]
    episode_ends = np.asarray(root["meta"]["episode_ends"])

    for key in policy_shape_meta_obs:
        if key not in obs_grp:
            raise KeyError(
                f"Policy obs key '{key}' not found in zarr data/obs. Available: {list(obs_grp.keys())}"
            )
    for key in q_shape_meta_obs:
        if key not in obs_grp:
            raise KeyError(
                f"Q obs key '{key}' not found in zarr data/obs. Available: {list(obs_grp.keys())}"
            )

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_starts = np.insert(episode_ends, 0, 0)[:-1].astype(np.int64)
    n_eps = len(episode_ends)
    max_episodes = n_eps if args.max_episodes is None else min(args.max_episodes, n_eps)

    rows: list[StateMetrics] = []
    total_states = 0
    stop = False

    print(
        f"Policy: n_obs_steps={policy_n_obs}, n_action_steps={policy_n_act}, action_dim={policy_action_dim}. "
        f"Q: n_obs_steps={q_n_obs}, n_action_steps={q_n_act}, action_dim={q_action_dim}. "
        f"Episodes: {n_eps}, evaluating up to {max_episodes}, k={args.k}."
    )

    with torch.inference_mode():
        for ep_idx in range(max_episodes):
            ep_start = int(ep_starts[ep_idx])
            ep_end = int(episode_ends[ep_idx])

            for t in range(ep_start, ep_end):
                if args.max_steps is not None and total_states >= args.max_steps:
                    stop = True
                    break

                if hasattr(policy, "reset"):
                    policy.reset()

                policy_obs = _build_obs_dict(
                    obs_grp,
                    policy_shape_meta_obs,
                    ep_start,
                    ep_end,
                    t,
                    policy_n_obs,
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

                action_result = policy.predict_k_actions(policy_obs, args.k)
                candidate_actions = _extract_candidate_actions(action_result).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                if tuple(candidate_actions.shape[1:]) != (q_n_act, q_action_dim):
                    raise ValueError(
                        "Policy candidate action shape does not match Q expectation: "
                        f"got {tuple(candidate_actions.shape)}, expected (K, {q_n_act}, {q_action_dim})."
                    )

                num_candidates = _candidate_count(candidate_actions)
                repeated_q_obs = _repeat_obs_dict(q_obs, num_candidates)
                q_values = q_model.predict_q(repeated_q_obs, candidate_actions).reshape(-1)

                action_variance = float(candidate_actions.var(dim=0, unbiased=False).mean().item())
                q_variance = float(q_values.var(unbiased=False).item())
                rows.append(
                    StateMetrics(
                        state_index=total_states,
                        episode_index=ep_idx,
                        timestep=t,
                        action_variance=action_variance,
                        q_variance=q_variance,
                        q_mean=float(q_values.mean().item()),
                        q_min=float(q_values.min().item()),
                        q_max=float(q_values.max().item()),
                        num_candidates=num_candidates,
                    )
                )
                total_states += 1

                if total_states % 100 == 0:
                    print(f"Processed {total_states} states...")

            print(f"Finished episode {ep_idx} ({ep_end - ep_start} states available).")
            if stop:
                break

    if not rows:
        raise RuntimeError("No dataset states were evaluated. Check --max_episodes/--max_steps.")

    action_vars = _metric_array(rows, "action_variance")
    q_vars = _metric_array(rows, "q_variance")
    summary = {
        "policy_checkpoint": str(Path(args.policy_checkpoint).expanduser()),
        "q_checkpoint": str(Path(args.q_checkpoint).expanduser()),
        "dataset": str(Path(args.dataset).expanduser()),
        "seed": int(args.seed),
        "k": int(args.k),
        "total_states": int(total_states),
        "policy_abs_action": bool(policy_bundle.abs_action),
        "q_abs_action": bool(q_bundle.abs_action),
        "mean_action_variance": float(action_vars.mean()),
        "std_action_variance": float(action_vars.std()),
        "mean_q_variance": float(q_vars.mean()),
        "std_q_variance": float(q_vars.std()),
        "corr_action_q_variance": (
            float(np.corrcoef(action_vars, q_vars)[0, 1]) if len(rows) > 1 else None
        ),
    }

    csv_path = out_dir / "policy_q_variance_per_state.csv"
    json_path = out_dir / "policy_q_variance_summary.json"
    plot_path = out_dir / "policy_q_variance.png"

    _save_rows_csv(rows, csv_path)
    with json_path.open("w") as f:
        json.dump(
            {
                **summary,
                "rows": [asdict(row) for row in rows],
            },
            f,
            indent=2,
        )
    _save_plot(rows, plot_path)

    print(f"Processed {total_states} states total.")
    print(f"Average action variance: {summary['mean_action_variance']:.6f}")
    print(f"Average Q variance: {summary['mean_q_variance']:.6f}")
    print(f"Wrote plot to {plot_path}")
    print(f"Wrote per-state stats to {csv_path}")
    print(f"Wrote summary to {json_path}")


if __name__ == "__main__":
    main()
