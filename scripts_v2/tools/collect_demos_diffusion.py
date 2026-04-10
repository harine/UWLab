# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations from a trained diffusion policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os
import random
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations from a trained diffusion policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.zarr", help="Output dataset path.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Ignored for diffusion policies (no Gaussian mean/std); kept for CLI parity with collect_demos.",
)
parser.add_argument("--action_std", type=float, default=0.0, help="Threshold for action std.")
parser.add_argument("--collect_failed_demos", action="store_true", default=False, help="Collect failed demos.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to diffusion policy checkpoint.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dill
import gymnasium as gym
import hydra
import isaaclab_tasks  # noqa: F401
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils.datasets import HDF5DatasetFileHandler

from uwlab.utils.datasets import ZarrDatasetFileHandler
from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_policy(ckpt_path: str, device: torch.device, use_ema: bool = False) -> BaseImagePolicy:
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace_any: Any = workspace
    policy = workspace_any.ema_model if cfg.training.use_ema else workspace_any.model
    policy.abs_action = bool(getattr(getattr(cfg.task, "dataset", {}), "abs_action", False))
    return policy.eval().to(device)


def _get_policy_chunk_info(policy: Any) -> tuple[int, int]:
    inner_policy: Any = policy
    while hasattr(inner_policy, "policy"):
        inner_policy = inner_policy.policy
    n_obs_steps = int(getattr(inner_policy, "n_obs_steps", 1))
    n_action_steps = int(getattr(inner_policy, "n_action_steps", 1))
    return n_obs_steps, n_action_steps


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, _agent_cfg):
    """Collect demonstrations from the environment using a diffusion policy."""
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required.")

    _set_seeds(args_cli.seed)

    device = torch.device(args_cli.device if args_cli.device else "cuda" if torch.cuda.is_available() else "cpu")

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # add recordermanager to save data
    use_zarr_format = args_cli.dataset_file.endswith(".zarr")
    if use_zarr_format:
        dataset_handler = ZarrDatasetFileHandler
    else:
        dataset_handler = HDF5DatasetFileHandler

    # Setup recorder for raw actions
    env_cfg.recorders = ActionStateRecorderManagerCfg()

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    if args_cli.collect_failed_demos:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = dataset_handler

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.concatenate_terms = False

    num_envs = env_cfg.scene.num_envs

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    policy = _load_policy(args_cli.checkpoint, device)
    n_obs_steps, n_action_steps = _get_policy_chunk_info(policy)
    if args_cli.execute_horizon is not None and args_cli.execute_horizon < 1:
        raise ValueError("--execute_horizon must be at least 1 when provided.")
    execute_horizon = args_cli.execute_horizon

    print(
        f"Loaded policy `{policy.__class__.__name__}` "
        f"with n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}."
    )
    if n_action_steps > 1:
        if execute_horizon is None:
            print("Action chunking: full predicted chunks before replanning.")
        else:
            print(f"Action chunking: execute_horizon={execute_horizon}.")
    if args_cli.deterministic:
        print("[Policy] --deterministic has no effect for diffusion sampling (CLI parity with collect_demos).")

    wrapped_policy = DiffusionPolicyWrapper(
        policy,
        device,
        n_obs_steps=n_obs_steps,
        num_envs=num_envs,
        execute_horizon=execute_horizon,
        use_absolute_actions=args_cli.use_absolute,
    )

    obs_dict, _ = env.reset()
    dones = torch.ones(num_envs, dtype=torch.bool, device=device)
    wrapped_policy.reset((dones > 0).nonzero(as_tuple=False).reshape(-1))

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    env_step_count = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        pbar = tqdm(total=args_cli.num_demos, desc="Recording Demonstrations", unit="demo")

        while True:
            amp_ctx = torch.autocast(device_type=device.type) if args_cli.use_amp else nullcontext()
            with amp_ctx:
                actions = wrapped_policy.predict_action(obs_dict)

            # Mask actions to zero for environments in their first step after reset since first image may not be valid
            first_step_mask = env.unwrapped.episode_length_buf == 0
            if torch.any(first_step_mask):
                actions[first_step_mask, :-1] = 0.0
                actions[first_step_mask, -1] = -1.0  # close gripper

            if args_cli.action_std > 0.0:
                noise = torch.randn_like(actions) * args_cli.action_std
                actions = actions + noise

            # Record executed action as point-mass "expert" for downstream schemas that expect expert_action_*.
            env.unwrapped.obs_buf["data_collection"]["expert_action_mean"] = actions.clone()
            env.unwrapped.obs_buf["data_collection"]["expert_action_std"] = torch.zeros_like(actions)

            step_result = env.step(actions)
            env_step_count += 1
            if len(step_result) == 4:
                obs_dict, _rewards, dones, _infos = step_result
            else:
                obs_dict, _rewards, terminated, truncated, _infos = step_result
                dones = terminated | truncated

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                wrapped_policy.reset(reset_ids)

            if args_cli.collect_failed_demos:
                new_count = env.unwrapped.recorder_manager.exported_successful_episode_count + env.unwrapped.recorder_manager.exported_failed_episode_count
            else:
                new_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                increment = new_count - current_recorded_demo_count
                current_recorded_demo_count = new_count
                pbar.update(increment)

            pbar.set_postfix(demos=current_recorded_demo_count, steps=env_step_count, successful_demos=env.unwrapped.recorder_manager.exported_successful_episode_count)

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            if env.unwrapped.sim.is_stopped():
                break

        pbar.close()

    env.close()


if __name__ == "__main__":
    main()  # type: ignore
    simulation_app.close()
