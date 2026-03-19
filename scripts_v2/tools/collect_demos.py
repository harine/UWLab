# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Script to collect demonstrations from a trained RL policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os
import gymnasium as gym
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations from trained RL policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.zarr", help="Output dataset path.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument(
    "--deterministic", action="store_true", default=False,
    help="Use the mean of the policy distribution instead of sampling.",
)
parser.add_argument("--store_camera_data", action="store_true", default=False, help="Enable cameras.")
parser.add_argument("--expert_path", type=str, required=True, help="Path to the expert policy.")
parser.add_argument("--action_std", type=float, default=0.5, help="Action noise standard deviation.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

# Import dataset handlers
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab.managers.recorder_manager import DatasetExportMode

from uwlab.utils.datasets import ZarrDatasetFileHandler
from uwlab_tasks.utils.hydra import hydra_task_compose
from uwlab_tasks.manager_based.manipulation.reset_states.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
)

from rsl_rl.runners import OnPolicyRunner, DistillationRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def process_agent_cfg(env_cfg, agent_cfg):
    if hasattr(agent_cfg.algorithm, "behavior_cloning_cfg"):
        if agent_cfg.algorithm.behavior_cloning_cfg is None:
            del agent_cfg.algorithm.behavior_cloning_cfg
        else:
            bc_cfg = agent_cfg.algorithm.behavior_cloning_cfg
            if bc_cfg.experts_observation_group_cfg is not None:
                import importlib

                # resolve path to the module location
                mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                mod = importlib.import_module(mod_name)
                cfg_cls = mod
                for attr in attr_name.split("."):
                    cfg_cls = getattr(cfg_cls, attr)
                cfg = cfg_cls()
                setattr(env_cfg.observations, "expert_obs", cfg)

    if hasattr(agent_cfg.algorithm, "offline_algorithm_cfg"):
        if agent_cfg.algorithm.offline_algorithm_cfg is None:
            del agent_cfg.algorithm.offline_algorithm_cfg
        else:
            if agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg is None:
                del agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
            else:
                bc_cfg = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
                if bc_cfg.experts_observation_group_cfg is not None:
                    import importlib

                    # resolve path to the module location
                    mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                    mod = importlib.import_module(mod_name)
                    cfg_cls = mod
                    for attr in attr_name.split("."):
                        cfg_cls = getattr(cfg_cls, attr)
                    cfg = cfg_cls()
                    setattr(env_cfg.observations, "expert_obs", cfg)
    return agent_cfg


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect demonstrations from the environment using RSL-RL policy."""
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # add recordermanager to save data
    use_zarr_format = args_cli.dataset_file.endswith('.zarr')
    if use_zarr_format:
        dataset_handler = ZarrDatasetFileHandler
    else:
        dataset_handler = HDF5DatasetFileHandler

    # Setup recorder for raw actions
    env_cfg.recorders = ActionStateRecorderManagerCfg()

    if not args_cli.store_camera_data:
        delattr(env_cfg.recorders, "record_pre_step_camera_data")

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = dataset_handler

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    # add expert obs into env_cfg
    agent_cfg = process_agent_cfg(env_cfg, agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load expert
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(args_cli.expert_path)
    expert_policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(f"[Policy] {'Deterministic (mean)' if args_cli.deterministic else 'Stochastic (sampled)'} actions")

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        # Initialize tqdm progress bar if num_demos > 0
        pbar = tqdm(total=args_cli.num_demos, desc="Recording Demonstrations", unit="demo")
        # noise = torch.randn((args_cli.num_envs,env.num_actions), device=env.device) * args_cli.action_std
        while True:
            # agent stepping
            expert_policy_obs = env.get_observations()
            with torch.inference_mode():
                actions = expert_policy(expert_policy_obs) # + noise

            # Mask actions to zero for environments in their first step after reset since first image may not be valid
            first_step_mask = (env.unwrapped.episode_length_buf == 0)
            if torch.any(first_step_mask):
                actions[first_step_mask, :-1] = 0.0
                actions[first_step_mask, -1] = -1.0  # close gripper

            # Inject expert distribution into obs_buf so recorder saves them alongside observations
            env.unwrapped.obs_buf["data_collection"]["expert_action"] = actions.clone()

            # env stepping
            env.step(actions)

            # print out the current demo count if it has changed
            new_count = env.unwrapped.recorder_manager.exported_successful_episode_count + env.unwrapped.recorder_manager.exported_failed_episode_count
            if new_count > current_recorded_demo_count:
                increment = new_count - current_recorded_demo_count
                current_recorded_demo_count = new_count
                pbar.update(increment)

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

        pbar.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()