# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
MDP terminations.
"""


def invalid_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return true if the RigidBody position reads nan"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.isnan(asset.data.body_pos_w).any(dim=-1).any(dim=-1)


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)

def consecutive_success_state_with_min_length(
    env: ManagerBasedRLEnv, num_consecutive_successes: int = 10, min_episode_length: int = 0
):
    """Like consecutive_success_state but rejects episodes shorter than min_episode_length.

    Episodes that start already assembled will reach num_consecutive_successes quickly,
    but won't be marked as success until min_episode_length steps have passed.
    Combined with a separate early termination, these episodes get terminated as failures.
    """
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    continuous_success_counter = getattr(context_term, "continuous_success_counter")
    success = continuous_success_counter >= num_consecutive_successes
    if min_episode_length > 0:
        success = success & (env.episode_length_buf >= min_episode_length)
    return success
