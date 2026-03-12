from __future__ import annotations

"""Evaluate a checkpointed policy: save positions and camera videos, then plot trajectories and success rate."""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
L2SML_DIR = SCRIPT_DIR.parent  # l2sml/
DEFAULT_CONFIG_PATH = L2SML_DIR / "configs" / "eval_policy.yaml"
Q_FUNCTION_DIR = SCRIPT_DIR / "q_function"
MODELS_PATH = Q_FUNCTION_DIR / "models.py"

_RSL_RL_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_RSL_RL_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_RSL_RL_SCRIPTS_DIR))
import cli_args  # noqa: E402


def _load_gaussian_policy_module():
    import importlib.util
    if not MODELS_PATH.exists():
        raise FileNotFoundError(f"GaussianPolicy models not found at {MODELS_PATH}")
    spec = importlib.util.spec_from_file_location("q_function_models", MODELS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load models module from {MODELS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GaussianPolicy


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        # Try cwd first, then relative to l2sml dir
        if not path.exists():
            alt = L2SML_DIR / path
            if alt.exists():
                path = alt
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at '{path}' must be a YAML mapping.")
    return data


def _build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a policy: rollouts (no noise), save positions and videos, plot trajectories and success.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH.resolve()),
        help="YAML config path.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=config.get("task", "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0"),
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=config.get("agent", "rsl_rl_cfg_entry_point"),
        help="RL agent config entry point.",
    )
    parser.add_argument("--output_dir", type=str, default=config.get("output_dir", "data/eval_policy"))
    parser.add_argument("--num_rollouts", type=int, default=int(config.get("num_rollouts", 50)))
    parser.add_argument("--horizon", type=int, default=int(config.get("horizon", 800)))
    parser.add_argument("--num_envs", type=int, default=int(config.get("num_envs", 1)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 0)))
    parser.add_argument(
        "--capture_rendered_images",
        "--capture-rendered-images",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("capture_rendered_images", True)),
        help="Capture camera frames for video.",
    )
    parser.add_argument("--video_fps", type=int, default=int(config.get("video_fps", 10)))
    parser.add_argument(
        "--exact_success",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("exact_success", False)),
        help="If true, use get_successes(env); otherwise use get_successes_approx(rewards).",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default=config.get("policy_type", "gaussian"),
        choices=("gaussian", "rsl_rl"),
        help="Policy type: 'gaussian' for pi_base-trained GaussianPolicy, 'rsl_rl' for RSL-RL actor-critic.",
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(
        checkpoint=config.get("checkpoint", "l2sml/outputs/pi_base/pi_base_chunked_best.pt"),
        headless=bool(config.get("headless", True)),
    )
    return parser


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH.resolve()))
_pre_args, _ = _pre_parser.parse_known_args()
_config = _load_yaml_config(_pre_args.config)
parser = _build_parser(_config)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.capture_rendered_images:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _tensordict_to_dict(obs: Any) -> dict:
    if hasattr(obs, "to_dict"):
        return obs.to_dict()
    return obs if isinstance(obs, dict) else {}


def _to_cpu_detached(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


def _is_image_tensor(value: torch.Tensor) -> bool:
    if value.ndim < 3:
        return False
    channels = value.shape[-1]
    channels_alt = value.shape[-3] if value.ndim >= 4 else -1
    return channels in (1, 3, 4) or channels_alt in (1, 3, 4)


def _image_tensor_to_uint8_frame(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise RuntimeError(f"Expected image tensor with 3 dims, got shape={tuple(tensor.shape)}.")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[-1] not in (1, 3, 4):
        raise RuntimeError(
            f"Expected image tensor in CHW or HWC format with 1/3/4 channels, got shape={tuple(tensor.shape)}."
        )
    frame = tensor.numpy()
    if np.issubdtype(frame.dtype, np.floating):
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0)
    else:
        frame = np.clip(frame, 0, 255)
    frame = frame.astype(np.uint8)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _extract_xyz_terms_from_positions_obs(
    positions_obs: Any,
) -> dict[str, torch.Tensor]:
    positions_obs = _tensordict_to_dict(positions_obs)
    required_terms = {
        "ee_positions": "end_effector_pose",
        "insertive_positions": "insertive_asset_pose",
        "receptive_positions": "receptive_asset_pose",
    }
    if isinstance(positions_obs, dict):
        xyz_terms: dict[str, torch.Tensor] = {}
        missing_keys: list[str] = []
        for out_key, obs_key in required_terms.items():
            pose = positions_obs.get(obs_key)
            if pose is None:
                missing_keys.append(obs_key)
                continue
            if not isinstance(pose, torch.Tensor):
                raise RuntimeError(f"Expected {obs_key} to be a torch.Tensor.")
            if pose.ndim == 1:
                pose = pose.view(1, -1)
            if pose.shape[-1] < 3:
                raise RuntimeError(
                    f"Expected {obs_key} with >=3 dims, got shape={tuple(pose.shape)}."
                )
            xyz_terms[out_key] = pose[..., :3]
        if len(xyz_terms) == len(required_terms):
            return xyz_terms
    if isinstance(positions_obs, torch.Tensor):
        if positions_obs.ndim == 1:
            positions_obs = positions_obs.view(1, -1)
        if positions_obs.shape[-1] < 9:
            raise RuntimeError(
                f"Expected positions observation with >=9 dims, got shape={tuple(positions_obs.shape)}."
            )
        block_size = positions_obs.shape[-1] // 3
        if block_size < 3:
            raise RuntimeError(f"Expected each positions term to have >=3 dims, got block_size={block_size}.")
        return {
            "ee_positions": positions_obs[..., 0:3],
            "insertive_positions": positions_obs[..., block_size : block_size + 3],
            "receptive_positions": positions_obs[..., 2 * block_size : 2 * block_size + 3],
        }
    if fallback_obs is not None:
        return _extract_xyz_terms_from_positions_obs(fallback_obs, fallback_obs=None)
    raise RuntimeError(f"Unsupported positions observation type: {type(positions_obs)}.")


def _make_env_from_cfg(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    *,
    concatenate_policy_terms: bool,
    concatenate_positions_terms: bool,
    render_mode: str | None,
) -> gym.Env:
    env_cfg_local = copy.deepcopy(env_cfg)
    env_cfg_local.recorders = None
    env_cfg_local.observations.policy.concatenate_terms = concatenate_policy_terms
    if hasattr(env_cfg_local.observations, "positions"):
        env_cfg_local.observations.positions.concatenate_terms = concatenate_positions_terms
    env = gym.make(args_cli.task, cfg=env_cfg_local, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return env


def _build_policy_term_slices(
    env: gym.Env,
) -> tuple[list[str], list[tuple[int, ...]], list[tuple[int, int]]]:
    obs_manager = env.unwrapped.observation_manager
    term_names = list(obs_manager.active_terms["policy"])
    term_dims = list(obs_manager.group_obs_term_dim["policy"])
    slices: list[tuple[int, int]] = []
    start = 0
    for dims in term_dims:
        if len(dims) != 1:
            raise RuntimeError(
                f"Only 1D concatenated policy terms are supported, got dims={dims} for policy group."
            )
        length = int(dims[0])
        slices.append((start, start + length))
        start += length
    return term_names, term_dims, slices


def _split_policy_obs(
    policy_obs_tensor: torch.Tensor, term_names: list[str], slices: list[tuple[int, int]]
) -> dict[str, torch.Tensor]:
    return {name: policy_obs_tensor[:, start:end] for name, (start, end) in zip(term_names, slices)}


def _resolve_checkpoint_path(agent_cfg: RslRlBaseRunnerCfg) -> str:
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _load_gaussian_policy_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Any, int]:
    """Load GaussianPolicy from pi_base checkpoint. Returns (model, action_chunk_dim)."""
    GaussianPolicyClass = _load_gaussian_policy_module()
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        path = Path(retrieve_file_path(str(checkpoint_path)))
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(ckpt)}")
    state_dim = ckpt.get("state_dim")
    action_chunk_dim = ckpt.get("action_chunk_dim")
    hidden_dims = ckpt.get("hidden_dims")
    if state_dim is None or action_chunk_dim is None or hidden_dims is None:
        raise KeyError(
            f"Checkpoint must contain state_dim, action_chunk_dim, hidden_dims; got keys {list(ckpt.keys())}"
        )
    model = GaussianPolicyClass(
        state_dim=int(state_dim),
        action_dim=int(action_chunk_dim),
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, int(action_chunk_dim)


def get_successes(env: RslRlVecEnvWrapper) -> torch.Tensor:
    successes = torch.zeros((env.num_envs,), device=env.device)
    for i in range(env.num_envs):
        successes[i] = env.unwrapped.reward_manager.get_active_iterable_terms(i)[7][1][0]
    return successes


def get_successes_approx(rew: torch.Tensor) -> torch.Tensor:
    return torch.where(rew > 0.05, 1.0, 0.0)


def _eval_trajectory_template() -> dict[str, Any]:
    return {
        "ee_positions": [],
        "insertive_positions": [],
        "receptive_positions": [],
        "camera_frames": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
    }


def _append_eval_step(
    traj: dict[str, Any],
    env_idx: int,
    position_terms_xyz: dict[str, torch.Tensor],
    cameras_obs: dict[str, Any],
    rew: torch.Tensor,
    done: torch.Tensor,
    time_out: torch.Tensor,
    capture_frames: bool,
) -> None:
    traj["ee_positions"].append(_to_cpu_detached(position_terms_xyz["ee_positions"][env_idx : env_idx + 1]))
    traj["insertive_positions"].append(
        _to_cpu_detached(position_terms_xyz["insertive_positions"][env_idx : env_idx + 1])
    )
    traj["receptive_positions"].append(
        _to_cpu_detached(position_terms_xyz["receptive_positions"][env_idx : env_idx + 1])
    )
    traj["rewards"].append(_to_cpu_detached(rew[env_idx : env_idx + 1].view(1, 1)))
    traj["terminated"].append(
        _to_cpu_detached((done[env_idx] & ~time_out[env_idx]).float().view(1, 1))
    )
    traj["truncated"].append(_to_cpu_detached(time_out[env_idx].float().view(1, 1)))
    if capture_frames and cameras_obs:
        first_frame = None
        for cam_key, cam_tensor in cameras_obs.items():
            if isinstance(cam_tensor, torch.Tensor) and _is_image_tensor(cam_tensor):
                first_frame = _image_tensor_to_uint8_frame(cam_tensor[env_idx : env_idx + 1].squeeze(0))
                break
        if first_frame is not None:
            traj["camera_frames"].append(first_frame)


def _finalize_eval_trajectory(traj: dict[str, Any], success: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["ee_positions"] = (
        torch.cat(traj["ee_positions"], dim=0) if traj["ee_positions"] else torch.empty((0, 3))
    )
    out["insertive_positions"] = (
        torch.cat(traj["insertive_positions"], dim=0) if traj["insertive_positions"] else torch.empty((0, 3))
    )
    out["receptive_positions"] = (
        torch.cat(traj["receptive_positions"], dim=0) if traj["receptive_positions"] else torch.empty((0, 3))
    )
    if traj["camera_frames"]:
        out["camera_frames"] = np.stack(traj["camera_frames"], axis=0)
    else:
        out["camera_frames"] = np.empty((0,), dtype=np.uint8)
    out["success"] = float(success)
    out["steps"] = int(out["ee_positions"].shape[0])
    return out


def _save_eval_trajectory(
    traj: dict[str, Any],
    traj_idx: int,
    traj_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    done: bool,
    success: float,
) -> None:
    serialized = _finalize_eval_trajectory(traj, success)
    traj_file = traj_dir / f"traj_{traj_idx:06d}.pt"
    torch.save(serialized, traj_file)
    manifest["files"].append(
        {
            "trajectory_id": traj_idx,
            "file": str(traj_file.relative_to(output_dir)),
            "steps": serialized["steps"],
            "done": done,
            "success": success,
        }
    )


def _save_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _ensure_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    clipped = np.clip(frames, 0, 255)
    if clipped.size and float(np.max(clipped)) <= 1.0:
        clipped = clipped * 255.0
    return clipped.astype(np.uint8)


def _write_videos(traj_dir: Path, output_dir: Path, num_rollouts: int, video_fps: int) -> None:
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to write videos. Install it in your env.") from exc
    for traj_idx in range(num_rollouts):
        traj_file = traj_dir / f"traj_{traj_idx:06d}.pt"
        if not traj_file.exists():
            continue
        data = torch.load(traj_file, map_location="cpu", weights_only=False)
        frames = data.get("camera_frames")
        if frames is None or not isinstance(frames, np.ndarray) or frames.size == 0:
            continue
        if frames.ndim == 3:
            frames = np.expand_dims(frames, axis=0)
        safe_frames = _ensure_uint8(frames)
        out_path = videos_dir / f"traj_{traj_idx:06d}.mp4"
        writer = imageio.get_writer(out_path, fps=video_fps, codec="libx264")
        for frame in safe_frames:
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            writer.append_data(frame)
        writer.close()
    print(f"[INFO] Saved trajectory videos to {videos_dir}")


def _plot_all_trajectories_3d(traj_dir: Path, output_dir: Path, num_rollouts: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for traj_idx in range(num_rollouts):
        traj_file = traj_dir / f"traj_{traj_idx:06d}.pt"
        if not traj_file.exists():
            continue
        data = torch.load(traj_file, map_location="cpu", weights_only=False)
        ee = data["ee_positions"].numpy()
        ins = data["insertive_positions"].numpy()
        rec = data["receptive_positions"].numpy()
        success = data.get("success", 0.0)
        alpha = 0.4 if success else 0.25
        color_ee = "green" if success else "red"
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color_ee, alpha=alpha, linewidth=0.8)
        ax.plot(ins[:, 0], ins[:, 1], ins[:, 2], color="tab:orange", alpha=alpha, linewidth=0.8)
        ax.plot(rec[:, 0], rec[:, 1], rec[:, 2], color="tab:green", alpha=alpha, linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("All trajectories (green=success, red=fail)")
    plt.tight_layout()
    plt.savefig(plots_dir / "all_trajectories_3d.png", dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved 3D trajectory plot to {plots_dir / 'all_trajectories_3d.png'}")


def _plot_success_rate(num_success: int, num_failure: int, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    total = num_success + num_failure
    percent_success = (100.0 * num_success / total) if total else 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0], [num_success], width=0.4, label="Success", color="green", alpha=0.8)
    ax.bar([1], [num_failure], width=0.4, label="Failure", color="red", alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Success", "Failure"])
    ax.set_ylabel("Count")
    ax.set_title(f"Success rate: {percent_success:.1f}% ({num_success}/{total})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "success_rate.png", dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved success rate plot to {plots_dir / 'success_rate.png'}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    output_dir = Path(args_cli.output_dir)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    gym_env = _make_env_from_cfg(
        env_cfg,
        concatenate_policy_terms=True,
        concatenate_positions_terms=False,
        render_mode="rgb_array" if args_cli.capture_rendered_images else None,
    )
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    device = torch.device(env.unwrapped.device if hasattr(env.unwrapped, "device") else "cuda")
    num_actions = env.num_actions

    if args_cli.policy_type == "gaussian":
        checkpoint_path = args_cli.checkpoint or "l2sml/outputs/pi_base/pi_base_chunked_best.pt"
        print(f"[INFO] Loading Gaussian policy from: {checkpoint_path}")
        gaussian_model, action_chunk_dim = _load_gaussian_policy_checkpoint(checkpoint_path, device)

        def policy(obs: Any) -> torch.Tensor:
            obs_flat = obs["policy"]
            if isinstance(obs_flat, torch.Tensor):
                state = obs_flat.to(device)
            else:
                state = torch.as_tensor(obs_flat, device=device, dtype=torch.float32)
            with torch.inference_mode():
                mean, _ = gaussian_model(state)
            return mean[:, :num_actions]

        resume_path = str(Path(checkpoint_path).resolve())
    else:
        resume_path = _resolve_checkpoint_path(agent_cfg)
        print(f"[INFO] Loading RSL-RL checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_term_names, _, policy_term_slices = _build_policy_term_slices(gym_env)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()) if args_cli.checkpoint else resume_path,
        "output_dir": str(output_dir.resolve()),
        "num_rollouts": args_cli.num_rollouts,
        "horizon": args_cli.horizon,
        "num_envs": num_envs,
        "seed": args_cli.seed,
        "files": [],
    }

    target = args_cli.num_rollouts
    collected = 0
    trajectories: list[dict[str, Any]] = [_eval_trajectory_template() for _ in range(num_envs)]
    step_counts = [0] * num_envs

    try:
        obs = _tensordict_to_dict(env.get_observations())
        with tqdm(total=target, desc="Evaluating policy") as pbar:
            while collected < target:
                position_terms_xyz = _extract_xyz_terms_from_positions_obs(
                    obs.get("positions"),
                )
                cameras_obs = _tensordict_to_dict(obs.get("cameras")) if obs.get("cameras") is not None else {}
                if not isinstance(cameras_obs, dict):
                    cameras_obs = {}

                with torch.inference_mode():
                    action = policy(obs)

                obs, rew, dones, extras = env.step(action)
                time_outs = extras.get("time_outs", torch.zeros_like(dones))

                for ei in range(num_envs):
                    if collected >= target:
                        break
                    _append_eval_step(
                        trajectories[ei],
                        ei,
                        position_terms_xyz,
                        cameras_obs,
                        rew,
                        dones,
                        time_outs,
                        args_cli.capture_rendered_images,
                    )
                    step_counts[ei] += 1
                    env_done = bool(dones[ei].item())
                    horizon_hit = step_counts[ei] >= args_cli.horizon

                    if env_done or horizon_hit:
                        if args_cli.exact_success:
                            success_tensor = get_successes(env)
                            success_val = float(success_tensor[ei].item())
                        else:
                            traj_rew = torch.cat(trajectories[ei]["rewards"], dim=0)
                            success_val = float(get_successes_approx(traj_rew).any().item())
                        _save_eval_trajectory(
                            trajectories[ei],
                            collected,
                            traj_dir,
                            output_dir,
                            manifest,
                            env_done,
                            success_val,
                        )
                        collected += 1
                        pbar.update(1)
                        trajectories[ei] = _eval_trajectory_template()
                        step_counts[ei] = 0
    finally:
        env.close()

    num_success = sum(1 for f in manifest["files"] if f.get("success", 0) > 0.5)
    num_failure = len(manifest["files"]) - num_success
    percent_success = (100.0 * num_success / len(manifest["files"])) if manifest["files"] else 0.0
    manifest["num_success"] = num_success
    manifest["num_failure"] = num_failure
    manifest["percent_success"] = percent_success
    _save_run_manifest(output_dir, manifest)

    print(f"[INFO] Saved {len(manifest['files'])} rollouts to {output_dir.resolve()}")
    print(f"[INFO] Success: {num_success}, Failure: {num_failure}, Rate: {percent_success:.1f}%")

    if args_cli.capture_rendered_images:
        _write_videos(traj_dir, output_dir, len(manifest["files"]), args_cli.video_fps)
    else:
        print("[INFO] No camera frames; skipping video creation.")

    _plot_all_trajectories_3d(traj_dir, output_dir, len(manifest["files"]))
    _plot_success_rate(num_success, num_failure, output_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
