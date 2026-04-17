# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained diffusion policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
from typing import Any

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using diffusion policy for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to diffusion policy checkpoint.")
parser.add_argument("--q_checkpoint", type=str, default=None, help="Path to Q-function checkpoint for best-of-k.")
parser.add_argument(
    "--k",
    type=int,
    default=1,
    help=(
        "With --q_checkpoint: number of Q-scored candidates per replanning step. "
        "If --action_noise_std is set, one policy action plus k Gaussian-noisy variants; "
        "otherwise k independent policy samples (best-of-k)."
    ),
)
parser.add_argument(
    "--action_noise_std",
    type=float,
    default=None,
    help=(
        "If set with --q_checkpoint, use one predict_action sample and add N(0, std^2) noise to build "
        "k candidates, then select by Q. If unset, use best-of-k (requires predict_k_actions)."
    ),
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel.")
parser.add_argument(
    "--num_trajectories",
    type=int,
    default=100,
    help="Number of trajectories to evaluate. If None, run until simulation is stopped.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video of the policy.")
parser.add_argument(
    "--use_absolute",
    action="store_true",
    default=False,
    help="Use the latest end-effector pose observation to convert absolute policy actions into environment actions.",
)
parser.add_argument(
    "--execute_horizon",
    type=int,
    default=None,
    help="Number of actions to execute from each predicted chunk before replanning. Defaults to the full chunk.",
)
parser.add_argument(
    "--temporal_ensemble",
    action="store_true",
    default=False,
    help="Enable temporal ensembling: query the policy every step and average overlapping action chunk predictions.",
)
parser.add_argument(
    "--temporal_ensemble_decay",
    type=float,
    default=0.5,
    help="Exponential decay rate for temporal ensemble weights (higher = less smoothing).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save videos and stats JSON. Defaults to outputs/eval/<run_name>.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import gymnasium as gym
import numpy as np
import random
import torch
from contextlib import nullcontext
from tqdm import tqdm

import dill
import hydra
import imageio
import isaaclab_tasks  # noqa: F401
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Diffusion policy imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

# Import the Diffusion policy wrapper
from uwlab_rl.wrappers.best_of_k import BestOfKWrapper
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper
from uwlab_rl.wrappers.noise_around_action import NoiseAroundActionWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _reconcile_mlp_cfg(cfg, state_dict: dict):
    """Detect MLP-policy `autoregressive` flag from saved weights when cfg doesn't specify it.

    Training scripts for `MLPImagePolicy` sometimes omit `autoregressive`, so cfg falls back
    to the code default at eval time. If that default has changed since training, the
    re-instantiated model shape won't match the checkpoint. We recover by inspecting
    `mean_head.weight` and `trunk.0.weight`.
    """
    if "mean_head.weight" not in state_dict or "trunk.0.weight" not in state_dict:
        return
    policy_cfg = cfg.policy
    target = str(getattr(policy_cfg, "_target_", ""))
    if "MLPImagePolicy" not in target:
        return
    try:
        from omegaconf import open_dict
        from diffusion_policy.policy.mlp_image_policy import MLPImagePolicy  # noqa
    except Exception:
        open_dict = None  # type: ignore

    shape_meta = cfg.shape_meta
    action_dim = int(shape_meta.action.shape[0])
    n_action_steps = int(policy_cfg.n_action_steps)
    out_dim = state_dict["mean_head.weight"].shape[0]

    if out_dim == action_dim:
        desired = True
    elif out_dim == n_action_steps * action_dim:
        desired = False
    else:
        return

    current = bool(getattr(policy_cfg, "autoregressive", True))
    if current != desired:
        print(
            f"Reconciling MLP cfg: autoregressive {current} -> {desired} "
            f"(mean_head out={out_dim}, action_dim={action_dim}, n_action_steps={n_action_steps})"
        )
        if open_dict is not None:
            with open_dict(policy_cfg):
                policy_cfg.autoregressive = desired
        else:
            policy_cfg.autoregressive = desired


def _load_policy(ckpt_path: str, device: torch.device, use_ema: bool = False) -> BaseImagePolicy:
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    _reconcile_mlp_cfg(cfg, payload["state_dicts"].get("model", {}))
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace_any: Any = workspace
    policy = workspace_any.ema_model if cfg.training.use_ema else workspace_any.model
    policy.abs_action = bool(getattr(getattr(cfg.task, "dataset", {}), "abs_action", False))
    return policy.eval().to(device)


def _load_q_function(ckpt_path: str, device: torch.device):
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace_any: Any = workspace
    q_function = workspace_any.model
    if not hasattr(q_function, "predict_q"):
        raise TypeError(f"Expected Q-function checkpoint with predict_q(), got {type(q_function)}")
    return q_function.eval().to(device)


def _get_policy_chunk_info(policy: Any) -> tuple[int, int]:
    inner_policy: Any = policy
    while hasattr(inner_policy, "policy"):
        inner_policy = inner_policy.policy
    n_obs_steps = int(getattr(inner_policy, "n_obs_steps", 1))
    n_action_steps = int(getattr(inner_policy, "n_action_steps", 1))
    return n_obs_steps, n_action_steps


def _get_policy_obs_keys(policy: Any) -> set[str] | None:
    """Return the set of observation keys the policy's normalizer was trained with."""
    inner = policy
    while hasattr(inner, "policy"):
        inner = inner.policy
    normalizer = getattr(inner, "normalizer", None)
    if normalizer is None:
        return None
    params = getattr(normalizer, "params_dict", None)
    if params is None:
        return None
    keys = {k for k in params.keys() if k not in ("_default", "action")}
    return keys if keys else None


def _prune_env_obs(env_cfg, policy_obs_keys: set[str]):
    """Remove observation terms from env_cfg.observations.policy that the policy doesn't expect."""
    policy_group = env_cfg.observations.policy
    group_meta = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}
    env_terms = {
        name for name in vars(policy_group).keys()
        if not name.startswith("_") and name not in group_meta
    }
    extra = env_terms - policy_obs_keys
    if extra:
        print(f"Pruning env obs terms not in policy normalizer: {sorted(extra)}")
        for name in extra:
            delattr(policy_group, name)


def _discover_cameras(obs_dict, env):
    """Return (cam_keys, scene_cam_names) for video recording."""
    cam_keys = sorted(k for k in obs_dict["policy"] if "rgb" in k)
    if cam_keys:
        return cam_keys, []
    scene_cam_names = sorted(
        name
        for name, sensor in env.unwrapped.scene._sensors.items()
        if hasattr(sensor, "data") and hasattr(sensor.data, "output") and "rgb" in sensor.data.output
    )
    if scene_cam_names:
        print(f"Using scene cameras for video: {scene_cam_names}")
    return cam_keys, scene_cam_names


def _capture_frame(obs_dict, env, env_idx: int, cam_keys: list, scene_cam_names: list) -> np.ndarray | None:
    """Capture and concatenate camera images for one environment."""
    imgs = []
    if cam_keys:
        for cam in cam_keys:
            img = obs_dict["camera"][cam][env_idx].detach().cpu().permute(1, 2, 0).numpy()
            imgs.append((img * 255).clip(0, 255).astype("uint8"))
    elif scene_cam_names:
        for cam_name in scene_cam_names:
            img = env.unwrapped.scene._sensors[cam_name].data.output["rgb"][env_idx].detach().cpu().numpy()
            if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1]:
                img = img.transpose(1, 2, 0)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype("uint8")
            if img.shape[-1] == 4:
                img = img[..., :3]
            imgs.append(img)
    return np.concatenate(imgs, axis=1) if imgs else None


def _count_successes(env, reset_ids: torch.Tensor, term_names: list[str]) -> tuple[int, list[bool]]:
    """Return (success_count, per_env_success_flags) for the given reset_ids."""
    count = 0
    flags: list[bool] = []
    term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]
    for term_row in term_dones:
        active = term_row.nonzero(as_tuple=False).flatten().cpu().tolist()
        is_success = any(term_names[idx] == "success" for idx in active)
        flags.append(is_success)
        if is_success:
            count += 1
    return count, flags


def _collect_metrics(infos: dict, episode_metrics: dict):
    if "log" not in infos:
        return
    for key, value in infos["log"].items():
        if key.startswith("Metrics/") or key.startswith("Episode_Reward/"):
            episode_metrics.setdefault(key, []).append(value)


def _print_results(episodes: int, successful_episodes: int, episode_metrics: dict) -> dict:
    stats: dict = {"total_trajectories": episodes, "successful_trajectories": successful_episodes}
    print("\nFinal Statistics:")
    print(f"Total trajectories evaluated: {episodes}")
    if successful_episodes > 0 or "Episode_Termination/success" in episode_metrics:
        stats["success_rate"] = successful_episodes / episodes * 100 if episodes > 0 else 0.0
        print(f"Successful trajectories: {successful_episodes}")
        print(f"Success rate: {stats['success_rate']:.2f}%")
    else:
        print("Success rate: Not calculable (success metric not found in environment)")
    if episode_metrics:
        print("\nAverage Metrics:")
        avg_metrics: dict = {}
        for metric_name, values in sorted(episode_metrics.items()):
            if values:
                floats = [float(v) if isinstance(v, torch.Tensor) else v for v in values]
                avg_metrics[metric_name] = sum(floats) / len(floats)
                print(f"{metric_name}: {avg_metrics[metric_name]:.4f}")
        stats["metrics"] = avg_metrics
    return stats


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Run a trained diffusion policy with Isaac Lab environment."""
    _set_seeds(args_cli.seed)

    device = torch.device(args_cli.device if args_cli.device else "cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.concatenate_terms = False
    if not hasattr(env_cfg.observations.policy, "concatenate_dim"):
        env_cfg.observations.policy.concatenate_dim = -1

    del env_cfg.observations.data_collection

    if not args_cli.save_video:
        del env_cfg.scene.front_camera
        del env_cfg.scene.side_camera
        del env_cfg.scene.wrist_camera
        del env_cfg.observations.camera

    policy = _load_policy(args_cli.checkpoint, device)
    policy_obs_keys = _get_policy_obs_keys(policy)
    if policy_obs_keys is not None:
        _prune_env_obs(env_cfg, policy_obs_keys)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if args_cli.k < 1:
        raise ValueError("--k must be at least 1.")
    if args_cli.q_checkpoint is None:
        if args_cli.k != 1:
            raise ValueError("--k can only be set above 1 when --q_checkpoint is provided.")
        if args_cli.action_noise_std is not None:
            raise ValueError("--action_noise_std requires --q_checkpoint.")
    else:
        if args_cli.action_noise_std is not None and args_cli.action_noise_std < 0:
            raise ValueError("--action_noise_std must be non-negative.")

    use_q_selection = args_cli.q_checkpoint is not None
    use_noise_around = use_q_selection and args_cli.action_noise_std is not None
    use_best_of_k = use_q_selection and not use_noise_around
    q_function = None
    if use_noise_around:
        q_function = _load_q_function(args_cli.q_checkpoint, device)
        policy = NoiseAroundActionWrapper(policy, q_function, args_cli.k, args_cli.action_noise_std)
        setattr(policy, "abs_action", getattr(policy.policy, "abs_action", False))
    elif use_best_of_k:
        if not hasattr(policy, "predict_k_actions"):
            raise AttributeError(
                f"{policy.__class__.__name__} must implement predict_k_actions(obs_dict, k) for best-of-k evaluation."
            )
        q_function = _load_q_function(args_cli.q_checkpoint, device)
        policy = BestOfKWrapper(policy, q_function, args_cli.k)
        setattr(policy, "abs_action", getattr(policy.policy, "abs_action", False))

    n_obs_steps, n_action_steps = _get_policy_chunk_info(policy)
    if args_cli.execute_horizon is not None and args_cli.execute_horizon < 1:
        raise ValueError("--execute_horizon must be at least 1 when provided.")
    execute_horizon = args_cli.execute_horizon
    print(
        f"Loaded policy `{policy.__class__.__name__}` "
        f"with n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}."
    )
    if use_best_of_k and q_function is not None:
        print(
            f"Best-of-k enabled with `{q_function.__class__.__name__}`: "
            f"sampling {args_cli.k} action chunks and selecting the highest-Q chunk."
        )
    if use_noise_around and q_function is not None:
        print(
            f"Noise-around-action Q selection with `{q_function.__class__.__name__}`: "
            f"one policy action + {args_cli.k} noisy candidates (std={args_cli.action_noise_std}), "
            "selecting the highest-Q chunk."
        )
    if n_action_steps > 1:
        if execute_horizon is None:
            print("Action chunking enabled: full predicted chunks will be executed before replanning.")
        else:
            print(
                "Action chunking enabled: "
                f"executing the first {execute_horizon} actions from each predicted chunk before replanning."
            )
    wrapped_policy = DiffusionPolicyWrapper(
        policy,
        device,
        n_obs_steps=n_obs_steps,
        num_envs=args_cli.num_envs,
        execute_horizon=execute_horizon,
        use_absolute_actions=args_cli.use_absolute,
        temporal_ensemble=args_cli.temporal_ensemble,
        temporal_ensemble_decay=args_cli.temporal_ensemble_decay,
    )

    if args_cli.output_dir is not None:
        output_dir = args_cli.output_dir
    else:
        te_tag = "te" if args_cli.temporal_ensemble else "no_te"
        eh_tag = f"eh{execute_horizon}" if execute_horizon is not None else "eh_full"
        ckpt_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        output_dir = os.path.join("outputs", "eval", f"{ckpt_name}_{te_tag}_{eh_tag}")
    os.makedirs(output_dir, exist_ok=True)

    obs_dict, _ = env.reset()
    dones = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
    wrapped_policy.reset((dones > 0).nonzero(as_tuple=False).reshape(-1))

    term_names = env.unwrapped.termination_manager._term_names  # type: ignore
    assert "success" in term_names, "Success term not found in termination manager"

    episodes, steps, successful_episodes = 0, 0, 0
    episode_metrics: dict = {}

    pbar = None
    if args_cli.num_trajectories is not None:
        pbar = tqdm(total=args_cli.num_trajectories, desc="Evaluating trajectories (Success: 0.00%)")

    # Video recording state -- save one successful and one unsuccessful rollout
    cam_keys, scene_cam_names = [], []
    env_frames: list[list] = []
    saved_success_video = False
    saved_failure_video = False
    if args_cli.save_video:
        cam_keys, scene_cam_names = _discover_cameras(obs_dict, env)
        env_frames = [[] for _ in range(args_cli.num_envs)]

    while simulation_app.is_running():
        if args_cli.num_trajectories is not None and episodes >= args_cli.num_trajectories:
            print(f"\nReached target number of trajectories ({args_cli.num_trajectories}). Stopping evaluation.")
            break

        with torch.inference_mode(), torch.autocast(device_type=device.type) if args_cli.use_amp else nullcontext():
            actions = wrapped_policy.predict_action(obs_dict)

            if args_cli.save_video and not (saved_success_video and saved_failure_video):
                for i in range(args_cli.num_envs):
                    frame = _capture_frame(obs_dict, env, i, cam_keys, scene_cam_names)
                    if frame is not None:
                        env_frames[i].append(frame)

            step_result = env.step(actions)
            if len(step_result) == 4:
                obs_dict, rewards, dones, infos = step_result
            else:
                obs_dict, rewards, terminated, truncated, infos = step_result
                dones = terminated | truncated

            steps += 1

            if isinstance(dones, torch.Tensor):
                new_ids = (dones > 0).nonzero(as_tuple=False)
                episodes += len(new_ids)
            elif dones:
                new_ids = [0]
                episodes += 1
            else:
                new_ids = []

            if pbar is not None:
                pbar.set_postfix(steps=steps)

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                succ_count, succ_flags = _count_successes(env, reset_ids, term_names)
                successful_episodes += succ_count
                wrapped_policy.reset(reset_ids)
                _collect_metrics(infos, episode_metrics)
                steps = 0

                if args_cli.save_video:
                    for idx, env_id in enumerate(reset_ids.tolist()):
                        frames = env_frames[env_id]
                        env_frames[env_id] = []
                        if not frames:
                            continue
                        if succ_flags[idx] and not saved_success_video:
                            path = os.path.join(output_dir, "success.mp4")
                            imageio.mimsave(path, frames, fps=10, codec="libx264")
                            saved_success_video = True
                        elif not succ_flags[idx] and not saved_failure_video:
                            path = os.path.join(output_dir, "failure.mp4")
                            imageio.mimsave(path, frames, fps=10, codec="libx264")
                            saved_failure_video = True
                    if saved_success_video and saved_failure_video:
                        cam_keys, scene_cam_names = [], []

                if pbar is not None:
                    pbar.update(len(new_ids))
                    rate = (successful_episodes / episodes * 100) if episodes > 0 else 0.0
                    pbar.set_description(f"Evaluating trajectories (Success: {rate:.2f}%)")

    stats = _print_results(episodes, successful_episodes, episode_metrics)
    stats["checkpoint"] = args_cli.checkpoint
    stats["task"] = args_cli.task
    stats["temporal_ensemble"] = args_cli.temporal_ensemble
    stats["execute_horizon"] = execute_horizon
    stats["num_envs"] = args_cli.num_envs
    stats["seed"] = args_cli.seed
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {os.path.abspath(stats_path)}")
    print(f"Successful episodes: {successful_episodes}", flush=True)
    if args_cli.save_video:
        print(f"Videos saved to: {os.path.abspath(output_dir)}")
    if pbar is not None:
        pbar.close()
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
