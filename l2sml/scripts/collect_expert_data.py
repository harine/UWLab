from __future__ import annotations

"""Collect expert trajectories from a checkpointed policy."""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm

import yaml
from isaaclab.app import AppLauncher

_RSL_RL_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_RSL_RL_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_RSL_RL_SCRIPTS_DIR))
import cli_args  # noqa: E402


def _load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at '{path}' must be a YAML mapping.")
    return data


def _build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect trajectories from an expert policy.")
    parser.add_argument("--config", type=str, default="l2sml/configs/collect_expert_data.yaml", help="YAML config path.")
    parser.add_argument("--task", type=str, default=config.get("task", "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0"))
    parser.add_argument(
        "--agent", type=str, default=config.get("agent", "rsl_rl_cfg_entry_point"), help="RL agent config entry point."
    )
    parser.add_argument("--action_std", type=float, default=float(config.get("action_std", 0.0)), help="Action noise standard deviation.")
    parser.add_argument("--expert_checkpoint", type=str, default=None, help="Alias for --checkpoint.")
    parser.add_argument("--output_dir", type=str, default=config.get("output_dir", "data/peg_expert"))
    parser.add_argument("--num_trajectories", type=int, default=int(config.get("num_trajectories", 1000)))
    parser.add_argument("--horizon", type=int, default=int(config.get("horizon", 800)))
    parser.add_argument("--num_envs", type=int, default=int(config.get("num_envs", 1)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 0)))
    parser.add_argument("--save_every", type=int, default=int(config.get("save_every", 50)))
    parser.add_argument(
        "--proprio_keys",
        nargs="*",
        default=list(config.get("proprio_keys", ["prev_actions", "joint_pos", "end_effector_pose"])),
    )
    parser.add_argument(
        "--asset_keys",
        nargs="*",
        default=list(
            config.get(
                "asset_keys",
                [
                    "insertive_asset_pose",
                    "receptive_asset_pose",
                    "insertive_asset_in_receptive_asset_frame",
                ],
            )
        ),
    )
    parser.add_argument(
        "--image_keys",
        nargs="*",
        default=list(config.get("image_keys", [])),
        help="Observation keys to treat as image observations. Empty means auto-detect.",
    )
    parser.add_argument(
        "--required_obs_keys",
        nargs="*",
        default=list(
            config.get(
                "required_obs_keys",
                [
                    "prev_actions",
                    "joint_pos",
                    "end_effector_pose",
                    "insertive_asset_pose",
                    "receptive_asset_pose",
                    "insertive_asset_in_receptive_asset_frame",
                ],
            )
        ),
        help="Fail fast if any required policy observation keys are missing.",
    )
    parser.add_argument(
        "--capture_rendered_images",
        "--capture-rendered-images",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("capture_rendered_images", False)),
        help="Capture a stitched front/side RGB video frame every step.",
    )
    parser.add_argument(
        "--render_image_obs_key",
        "--render-image-obs-key",
        type=str,
        default=str(config.get("render_image_obs_key", "render_rgb")),
        help="When rendering is enabled, also store stitched frames under obs_images[render_image_obs_key].",
    )
    parser.add_argument(
        "--collect_positions",
        "--collect-positions",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("collect_positions", config.get("collect positions", False))),
        help="Store xyz positions from positions obs group into ee/insertive/receptive trajectory fields.",
    )
    parser.add_argument(
        "--disable_fabric", action="store_true", default=bool(config.get("disable_fabric", False)), help="Disable fabric."
    )
    parser.add_argument(
        "--exact_success",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("exact_success", False)),
        help="If true, use get_successes(env); otherwise use get_successes_approx(rewards).",
    )
    cli_args.add_rsl_rl_args(parser)
    parser.set_defaults(checkpoint=config.get("checkpoint", config.get("expert_checkpoint", "peg_state_rl_expert.pt")))
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=bool(config.get("headless", True)))
    return parser


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--config", type=str, default="l2sml/configs/collect_expert_data.yaml")
_pre_args, _ = _pre_parser.parse_known_args()
_config = _load_yaml_config(_pre_args.config)
parser = _build_parser(_config)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.expert_checkpoint:
    args_cli.checkpoint = args_cli.expert_checkpoint
if args_cli.capture_rendered_images:
    args_cli.enable_cameras = True
# align with rsl_rl/play.py hydra arg handling
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


def _is_image_tensor(value: torch.Tensor) -> bool:
    if value.ndim < 3:
        return False
    channels = value.shape[-1]
    channels_alt = value.shape[-3] if value.ndim >= 4 else -1
    return channels in (1, 3, 4) or channels_alt in (1, 3, 4)


def _to_cpu_detached(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


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





def _normalize_obs_dict(policy_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in policy_obs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 0:
            normalized[key] = value.view(1, 1)
        elif value.ndim == 1:
            normalized[key] = value.view(1, -1)
        elif value.ndim >= 2:
            normalized[key] = value
    return normalized


def _concat_policy_obs(policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = []
    for _, value in policy_obs.items():
        if value.ndim == 1:
            part = value.view(1, -1)
        else:
            part = value.reshape(value.shape[0], -1)
        parts.append(part)
    if not parts:
        raise RuntimeError("Policy observation dict is empty; cannot run expert policy.")
    return torch.cat(parts, dim=-1)


def _classify_obs_keys(
    policy_obs: dict[str, torch.Tensor], proprio_keys: list[str], asset_keys: list[str], image_keys: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    all_keys = [k for k in policy_obs.keys() if isinstance(policy_obs[k], torch.Tensor)]
    image_set = set(image_keys)
    if not image_set:
        image_set = {k for k in all_keys if _is_image_tensor(policy_obs[k])}
    proprio_set = set(proprio_keys)
    asset_set = set(asset_keys)
    proprio_found = [k for k in all_keys if k in proprio_set and k not in image_set]
    asset_found = [k for k in all_keys if k in asset_set and k not in image_set]
    image_found = [k for k in all_keys if k in image_set]
    other_found = [k for k in all_keys if k not in image_set and k not in proprio_set and k not in asset_set]
    return proprio_found, asset_found, image_found, other_found


def _validate_required_obs_keys(policy_obs: dict[str, torch.Tensor], required_keys: list[str]) -> None:
    missing = [k for k in required_keys if k not in policy_obs]
    if missing:
        raise RuntimeError(f"Missing required observation keys: {missing}. Available keys: {sorted(policy_obs.keys())}")


def _resolve_checkpoint_path(agent_cfg: RslRlBaseRunnerCfg) -> str:
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


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
                f"Only 1D concatenated policy terms are supported for slicing, got dims={dims} for policy group."
            )
        length = int(dims[0])
        slices.append((start, start + length))
        start += length
    return term_names, term_dims, slices


def _split_policy_obs(
    policy_obs_tensor: torch.Tensor, term_names: list[str], slices: list[tuple[int, int]]
) -> dict[str, torch.Tensor]:
    return {name: policy_obs_tensor[:, start:end] for name, (start, end) in zip(term_names, slices)}


def _tensordict_to_dict(obs):
    if hasattr(obs, "to_dict"):
        return obs.to_dict()
    return obs


def _extract_xyz_terms_from_positions_obs(positions_obs: Any) -> dict[str, torch.Tensor]:
    positions_obs = _tensordict_to_dict(positions_obs)
    if isinstance(positions_obs, dict):
        required_terms = {
            "ee_positions": "end_effector_pose",
            "insertive_positions": "insertive_asset_pose",
            "receptive_positions": "receptive_asset_pose",
        }
        xyz_terms: dict[str, torch.Tensor] = {}
        for out_key, obs_key in required_terms.items():
            pose = positions_obs.get(obs_key)
            if pose is None:
                raise RuntimeError(
                    f"collect_positions=True requires observations['positions']['{obs_key}'] to be available."
                )
            if not isinstance(pose, torch.Tensor):
                raise RuntimeError(f"collect_positions=True expected positions.{obs_key} to be a torch.Tensor.")
            if pose.ndim == 1:
                pose = pose.view(1, -1)
            if pose.shape[-1] < 3:
                raise RuntimeError(
                    f"collect_positions=True expected positions.{obs_key} with >=3 dims, got shape={tuple(pose.shape)}."
                )
            xyz_terms[out_key] = pose[..., :3]
        return xyz_terms
    if isinstance(positions_obs, torch.Tensor):
        if positions_obs.ndim == 1:
            positions_obs = positions_obs.view(1, -1)
        if positions_obs.shape[-1] < 9:
            raise RuntimeError(
                f"collect_positions=True expected positions observation with >=9 dims, got shape={tuple(positions_obs.shape)}."
            )
        block_size = positions_obs.shape[-1] // 3
        if block_size < 3:
            raise RuntimeError(
                f"collect_positions=True expected each positions term to have >=3 dims, got block_size={block_size}."
            )
        return {
            "ee_positions": positions_obs[..., 0:3],
            "insertive_positions": positions_obs[..., block_size : block_size + 3],
            "receptive_positions": positions_obs[..., 2 * block_size : 2 * block_size + 3],
        }
    raise RuntimeError(
        f"collect_positions=True got unsupported positions observation type: {type(positions_obs)}."
    )


def _save_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _trajectory_template(*, collect_positions: bool = False) -> dict[str, Any]:
    traj = {
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "obs_flat": [],
        "obs_proprio": {},
        "obs_assets": {},
        "obs_images": {},
        "obs_other_state": {},
        "rendered_images": [],
    }
    if collect_positions:
        traj["ee_positions"] = []
        traj["insertive_positions"] = []
        traj["receptive_positions"] = []
    return traj


def _append_obs_to_trajectory(
    traj: dict[str, Any],
    policy_obs: dict[str, torch.Tensor],
    obs_flat: torch.Tensor,
    env_idx: int,
    proprio_keys: list[str],
    asset_keys: list[str],
    image_keys: list[str],
    other_keys: list[str],
    *,
    collect_positions: bool = False,
    position_terms_xyz: dict[str, torch.Tensor] | None = None,
) -> None:
    traj["obs_flat"].append(_to_cpu_detached(obs_flat[env_idx : env_idx + 1]))
    for key in proprio_keys:
        traj["obs_proprio"].setdefault(key, []).append(_to_cpu_detached(policy_obs[key][env_idx : env_idx + 1]))
    for key in asset_keys:
        traj["obs_assets"].setdefault(key, []).append(_to_cpu_detached(policy_obs[key][env_idx : env_idx + 1]))
    for key in image_keys:
        traj["obs_images"].setdefault(key, []).append(_to_cpu_detached(policy_obs[key][env_idx : env_idx + 1]))
    for key in other_keys:
        traj["obs_other_state"].setdefault(key, []).append(_to_cpu_detached(policy_obs[key][env_idx : env_idx + 1]))
    if collect_positions:
        if position_terms_xyz is None:
            raise RuntimeError("collect_positions=True but positions observation was not provided.")
        traj["ee_positions"].append(_to_cpu_detached(position_terms_xyz["ee_positions"][env_idx : env_idx + 1]))
        traj["insertive_positions"].append(
            _to_cpu_detached(position_terms_xyz["insertive_positions"][env_idx : env_idx + 1])
        )
        traj["receptive_positions"].append(
            _to_cpu_detached(position_terms_xyz["receptive_positions"][env_idx : env_idx + 1])
        )


def _finalize_trajectory(traj: dict[str, Any], success: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["actions"] = torch.cat(traj["actions"], dim=0) if traj["actions"] else torch.empty(0)
    out["rewards"] = torch.cat(traj["rewards"], dim=0) if traj["rewards"] else torch.empty(0)
    out["terminated"] = torch.cat(traj["terminated"], dim=0) if traj["terminated"] else torch.empty(0, dtype=torch.bool)
    out["truncated"] = torch.cat(traj["truncated"], dim=0) if traj["truncated"] else torch.empty(0, dtype=torch.bool)
    out["obs_flat"] = torch.cat(traj["obs_flat"], dim=0) if traj["obs_flat"] else torch.empty(0)

    out["obs_proprio"] = {
        key: torch.cat(values, dim=0) if values else torch.empty(0) for key, values in traj["obs_proprio"].items()
    }
    out["obs_assets"] = {
        key: torch.cat(values, dim=0) if values else torch.empty(0) for key, values in traj["obs_assets"].items()
    }
    out["obs_images"] = {
        key: torch.cat(values, dim=0) if values else torch.empty(0) for key, values in traj["obs_images"].items()
    }
    out["obs_other_state"] = {
        key: torch.cat(values, dim=0) if values else torch.empty(0) for key, values in traj["obs_other_state"].items()
    }
    if traj["rendered_images"]:
        out["rendered_images"] = np.stack(traj["rendered_images"], axis=0)
    else:
        out["rendered_images"] = np.empty((0,), dtype=np.uint8)
    if "ee_positions" in traj:
        out["ee_positions"] = torch.cat(traj["ee_positions"], dim=0) if traj["ee_positions"] else torch.empty((0, 3))
        out["insertive_positions"] = (
            torch.cat(traj["insertive_positions"], dim=0) if traj["insertive_positions"] else torch.empty((0, 3))
        )
        out["receptive_positions"] = (
            torch.cat(traj["receptive_positions"], dim=0) if traj["receptive_positions"] else torch.empty((0, 3))
        )
    if success is not None:
        out["success"] = float(success)
    return out


def _save_trajectory(
    traj: dict[str, Any],
    traj_idx: int,
    traj_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    done: bool,
    success: float,
) -> None:
    serialized = _finalize_trajectory(traj, success=success)
    traj_file = traj_dir / f"traj_{traj_idx:06d}.pt"
    torch.save(serialized, traj_file)
    manifest["files"].append(
        {
            "trajectory_id": traj_idx,
            "file": str(traj_file.relative_to(output_dir)),
            "steps": int(serialized["actions"].shape[0]),
            "done": done,
            "success": success,
        }
    )

def get_successes(env: RslRlVecEnvWrapper) -> torch.Tensor:
    successes = torch.zeros((env.num_envs,), device=env.device)
    for i in range(env.num_envs):
        successes[i] = env.unwrapped.reward_manager.get_active_iterable_terms(i)[7][1][0]
    return successes

def get_successes_approx(rew: torch.Tensor) -> torch.Tensor:
    return torch.where(rew > 0.05, 1.0, 0.0)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
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

    resume_path = _resolve_checkpoint_path(agent_cfg)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
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
        "expert_checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_trajectories": args_cli.num_trajectories,
        "horizon": args_cli.horizon,
        "num_envs": num_envs,
        "seed": args_cli.seed,
        "files": [],
    }

    target = args_cli.num_trajectories
    collected = 0
    trajectories: list[dict[str, Any]] = [
        _trajectory_template(collect_positions=args_cli.collect_positions) for _ in range(num_envs)
    ]
    step_counts = [0] * num_envs

    try:
        obs = _tensordict_to_dict(env.get_observations())
        noise = torch.randn(obs["policy"].shape, device=env.device) * args_cli.action_std
        with tqdm(total=target, desc="Collecting trajectories") as pbar:
            while collected < target:
                obs_flat = obs["policy"]
                policy_obs_raw = _split_policy_obs(obs_flat, policy_term_names, policy_term_slices)
                _validate_required_obs_keys(policy_obs_raw, args_cli.required_obs_keys)
                proprio_keys, asset_keys, image_keys, other_keys = _classify_obs_keys(
                    policy_obs_raw, args_cli.proprio_keys, args_cli.asset_keys, args_cli.image_keys
                )
                position_terms_xyz = None
                if args_cli.collect_positions:
                    position_terms_xyz = _extract_xyz_terms_from_positions_obs(obs.get("positions"))

                cameras_obs = _tensordict_to_dict(obs.get("cameras")) if obs.get("cameras") is not None else {}
                if not isinstance(cameras_obs, dict):
                    cameras_obs = {}

                for ei in range(num_envs):
                    if collected >= target:
                        break
                    _append_obs_to_trajectory(
                        trajectories[ei], policy_obs_raw, obs_flat, ei,
                        proprio_keys, asset_keys, image_keys, other_keys,
                        collect_positions=args_cli.collect_positions,
                        position_terms_xyz=position_terms_xyz,
                    )
                    if args_cli.capture_rendered_images and cameras_obs:
                        for cam_key, cam_tensor in cameras_obs.items():
                            if isinstance(cam_tensor, torch.Tensor) and _is_image_tensor(cam_tensor):
                                trajectories[ei]["obs_images"].setdefault(cam_key, []).append(
                                    _to_cpu_detached(cam_tensor[ei : ei + 1])
                                )

                with torch.inference_mode():
                    obs["policy"] = obs["policy"] + noise
                    action = policy(obs)
                    # action = action + torch.randn_like(action) * args_cli.action_std

                obs, rew, dones, extras = env.step(action)
                time_outs = extras.get("time_outs", torch.zeros_like(dones))

                for ei in range(num_envs):
                    if collected >= target:
                        break
                    trajectories[ei]["actions"].append(_to_cpu_detached(action[ei : ei + 1]))
                    trajectories[ei]["rewards"].append(_to_cpu_detached(rew[ei : ei + 1].view(1, 1)))
                    trajectories[ei]["terminated"].append(
                        _to_cpu_detached((dones[ei] & ~time_outs[ei]).float().view(1, 1))
                    )
                    trajectories[ei]["truncated"].append(
                        _to_cpu_detached(time_outs[ei].float().view(1, 1))
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
                        _save_trajectory(
                            trajectories[ei], collected, traj_dir, output_dir, manifest, env_done, success_val,
                        )
                        collected += 1
                        pbar.update(1)
                        trajectories[ei] = _trajectory_template(collect_positions=args_cli.collect_positions)
                        step_counts[ei] = 0

                        if collected % args_cli.save_every == 0 or collected == target:
                            _save_run_manifest(output_dir, manifest)
                            print(f"[INFO] Collected {collected}/{target} trajectories.")

    finally:
        env.close()

    _save_run_manifest(output_dir, manifest)
    print(f"[INFO] Saved dataset to {output_dir.resolve()}")
    print(f"[INFO] Total trajectories: {len(manifest['files'])}")


if __name__ == "__main__":
    main()
    simulation_app.close()
