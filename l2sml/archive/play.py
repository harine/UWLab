"""Roll out a trained state-based diffusion policy in a UWLab environment.

Self-contained — does NOT depend on the ``diffusion_policy`` repository.
The ConditionalUnet1D architecture, normalizer logic, and DDPM sampling are
reimplemented inline so that only standard PyPI packages are needed
(torch, einops, diffusers, omegaconf, dill).

Given a single ``.ckpt`` / ``.pt`` checkpoint **or** a folder of them, the
script rolls out the policy for the requested number of episodes, saves every
trajectory to disk, and reports the success rate.

Hydra overrides for the environment or agent config can be appended directly
on the command line (same convention as OctiLab ``scripts_v2``).

Usage (from the UWLab repo root):

    # Single checkpoint
    python l2sml/scripts/imitation_learning/play.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Collect-v0 \\
        --checkpoint path/to/latest.ckpt \\
        --num_rollouts 50 \\
        --output_dir data/eval_results \\
        --headless

    # With Hydra env overrides
    python l2sml/scripts/imitation_learning/play.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Collect-v0 \\
        --checkpoint path/to/latest.ckpt \\
        --num_rollouts 20 --headless \\
        env.scene.insertive_object=peg env.scene.receptive_object=peghole

    # Folder of checkpoints (each evaluated independently)
    python l2sml/scripts/imitation_learning/play.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Collect-v0 \\
        --checkpoint path/to/checkpoints_folder/ \\
        --num_rollouts 20 --headless
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Union

# ---------------------------------------------------------------------------
# AppLauncher must be started before any Isaac Sim / Isaac Lab imports.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Roll out a diffusion policy checkpoint (standalone, no diffusion_policy repo)."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to a .ckpt/.pt file, or a directory containing them.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--output_dir", type=str, default="data/eval_results", help="Where to save trajectories.")
parser.add_argument("--horizon", type=int, default=800, help="Maximum steps per episode.")
parser.add_argument(
    "--n_obs_steps", type=int, default=None,
    help="Override n_obs_steps (default: read from checkpoint config).",
)
parser.add_argument(
    "--n_action_steps", type=int, default=None,
    help="Override n_action_steps (default: read from checkpoint config).",
)
parser.add_argument(
    "--obs_keys", nargs="*",
    default=[
        "prev_actions", "joint_pos", "end_effector_pose",
        "insertive_asset_pose", "receptive_asset_pose",
        "insertive_asset_in_receptive_asset_frame",
    ],
)
parser.add_argument(
    "--proprio_keys", nargs="*",
    default=["prev_actions", "joint_pos", "end_effector_pose"],
)
parser.add_argument(
    "--asset_keys", nargs="*",
    default=[
        "insertive_asset_pose", "receptive_asset_pose",
        "insertive_asset_in_receptive_asset_frame",
    ],
)
parser.add_argument("--use_ema", action="store_true", default=True)
parser.add_argument("--no_ema", dest="use_ema", action="store_false")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", default=False, help="Record videos of rollouts.")
parser.add_argument(
    "--policy_type", type=str, default="auto", choices=["auto", "diffusion", "mlp"],
    help="Policy type: 'auto' detects from checkpoint, 'diffusion' for UNet, 'mlp' for MLPLowdimPolicy.",
)
parser.add_argument(
    "--sim_to_policy_key_map", nargs="*",
    default=[
        "joint_pos=arm_joint_pos:6",
        "prev_actions=last_arm_action:6,last_gripper_action:1",
        "end_effector_pose=end_effector_pose",
        "insertive_asset_pose=insertive_asset_pose",
        "receptive_asset_pose=receptive_asset_pose",
        "insertive_asset_in_receptive_asset_frame=insertive_asset_in_receptive_asset_frame",
    ],
    help="Sim obs term to policy key mapping.  Format: sim_key=policy_key[:dim][,policy_key2[:dim2]]",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=False)

args_cli, remaining_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# All remaining imports (after Isaac Sim has been initialised).
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import dill  # noqa: E402
import einops  # noqa: E402
from einops.layers.torch import Rearrange  # noqa: E402
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


# ===================================================================
# Self-contained model architecture (mirrors diffusion_policy repo)
# ===================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int,
                 kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 3, n_groups: int = 8,
                 cond_predict_scale: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size,
                        n_groups=n_groups),
        ])
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] | None = None,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        if down_dims is None:
            down_dims = [256, 512, 1024]

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(
                    local_cond_dim, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    local_cond_dim, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            ),
        ])

        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device,
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        h_local: list[torch.Tensor] = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder  # type: ignore[misc]
            h_local.append(resnet(local_cond, global_feature))
            h_local.append(resnet2(local_cond, global_feature))

        x = sample
        h: list[torch.Tensor] = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b t h -> b h t")


# ===================================================================
# Normalizer helpers
# ===================================================================

def _apply_normalizer(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    forward: bool = True,
) -> torch.Tensor:
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    return x.reshape(src_shape)


# ===================================================================
# Inline MLP policy (mirrors diffusion_policy MLPLowdimPolicy)
# ===================================================================

class InlineMLP(nn.Module):
    """Gaussian MLP with trunk + mean_head (only mean used at inference)."""

    def __init__(self, input_dim: int, action_dim: int,
                 hidden_dim: int = 512, hidden_depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(hidden_depth):
            layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.trunk(x))


def _parse_key_map(raw_entries: list[str]) -> dict[str, list[tuple[str, int | None]]]:
    """Parse --sim_to_policy_key_map entries.

    Each entry: ``sim_key=policy_key[:dim][,policy_key2[:dim2]]``
    Returns ``{sim_key: [(policy_key, dim_or_None), ...]}``.
    """
    mapping: dict[str, list[tuple[str, int | None]]] = {}
    for entry in raw_entries:
        sim_key, _, rhs = entry.partition("=")
        targets: list[tuple[str, int | None]] = []
        for part in rhs.split(","):
            if ":" in part:
                pkey, dim_str = part.rsplit(":", 1)
                targets.append((pkey, int(dim_str)))
            else:
                targets.append((part, None))
        mapping[sim_key] = targets
    return mapping


# ===================================================================
# Checkpoint loading
# ===================================================================

def _resolve_cfg(cfg) -> dict[str, Any]:
    """Resolve an OmegaConf config into a plain dict, handling eval resolvers."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _get(d: dict, *key_paths, default=None):
    """Try dotted key paths against a nested dict and return the first hit."""
    for path in key_paths:
        cur = d
        try:
            for part in path.split("."):
                cur = cur[part]
            if cur is not None:
                return cur
        except (KeyError, TypeError):
            continue
    return default


def load_checkpoint(ckpt_path: str, device: torch.device, use_ema: bool = True):
    """Load a TrainDiffusionUnetLowdimWorkspace checkpoint.

    Returns ``(unet, scheduler, norm_params, pcfg)`` where:
    - *unet* is the ``ConditionalUnet1D`` in eval mode on *device*.
    - *scheduler* is a ``DDPMScheduler``.
    - *norm_params* is ``{'obs': {'scale', 'offset'}, 'action': {'scale', 'offset'}}``.
    - *pcfg* is a plain dict with policy hyper-parameters.
    """
    payload = torch.load(
        open(ckpt_path, "rb"), pickle_module=dill, weights_only=False,
    )
    cfg = _resolve_cfg(payload["cfg"])

    obs_dim = int(_get(cfg, "obs_dim", "policy.obs_dim", "task.obs_dim"))
    action_dim = int(_get(cfg, "action_dim", "policy.action_dim", "task.action_dim"))
    horizon = int(_get(cfg, "horizon", "policy.horizon"))
    n_obs_steps = int(_get(cfg, "n_obs_steps", "policy.n_obs_steps"))
    n_action_steps = int(_get(cfg, "n_action_steps", "policy.n_action_steps"))
    obs_as_global = bool(_get(cfg, "obs_as_global_cond", "policy.obs_as_global_cond", default=True))
    obs_as_local = bool(_get(cfg, "obs_as_local_cond", "policy.obs_as_local_cond", default=False))
    pred_act_only = bool(_get(cfg, "pred_action_steps_only", "policy.pred_action_steps_only", default=False))
    oa_step_conv = bool(_get(cfg, "policy.oa_step_convention", default=False))
    num_inf_steps = int(_get(cfg, "policy.num_inference_steps", default=100))

    mcfg = _get(cfg, "policy.model", default={})
    input_dim = action_dim if (obs_as_local or obs_as_global) else (obs_dim + action_dim)
    global_cond_dim = obs_dim * n_obs_steps if obs_as_global else None
    local_cond_dim = obs_dim if obs_as_local else None

    unet = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=local_cond_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=int(mcfg.get("diffusion_step_embed_dim", 256)),
        down_dims=list(mcfg.get("down_dims", [256, 512, 1024])),
        kernel_size=int(mcfg.get("kernel_size", 5)),
        n_groups=int(mcfg.get("n_groups", 8)),
        cond_predict_scale=bool(mcfg.get("cond_predict_scale", False)),
    )

    scfg = _get(cfg, "policy.noise_scheduler", default={})
    scheduler = DDPMScheduler(
        num_train_timesteps=int(scfg.get("num_train_timesteps", 100)),
        beta_start=float(scfg.get("beta_start", 0.0001)),
        beta_end=float(scfg.get("beta_end", 0.02)),
        beta_schedule=str(scfg.get("beta_schedule", "squaredcos_cap_v2")),
        variance_type=str(scfg.get("variance_type", "fixed_small")),
        clip_sample=bool(scfg.get("clip_sample", True)),
        prediction_type=str(scfg.get("prediction_type", "epsilon")),
    )

    sd_key = "ema_model" if (use_ema and "ema_model" in payload["state_dicts"]) else "model"
    full_sd = payload["state_dicts"][sd_key]
    print(f"[INFO] Loading weights from '{sd_key}' key.")

    unet_sd = {
        k[len("model."):]: v
        for k, v in full_sd.items()
        if k.startswith("model.")
    }
    unet.load_state_dict(unet_sd)
    unet.eval().to(device)

    norm_params: dict[str, dict[str, torch.Tensor]] = {}
    for field in ("obs", "action"):
        norm_params[field] = {
            "scale": full_sd[f"normalizer.params_dict.{field}.scale"].to(device),
            "offset": full_sd[f"normalizer.params_dict.{field}.offset"].to(device),
        }

    pcfg = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "obs_as_global_cond": obs_as_global,
        "obs_as_local_cond": obs_as_local,
        "pred_action_steps_only": pred_act_only,
        "oa_step_convention": oa_step_conv,
        "num_inference_steps": num_inf_steps,
    }
    return unet, scheduler, norm_params, pcfg


def detect_policy_type(ckpt_path: str) -> str:
    """Return 'mlp' or 'diffusion' based on checkpoint contents."""
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = _resolve_cfg(payload["cfg"])
    name = cfg.get("name", "")
    target = _get(cfg, "_target_", default="")
    if "mlp" in name.lower() or "mlp" in target.lower():
        return "mlp"
    return "diffusion"


def load_mlp_checkpoint(
    ckpt_path: str, device: torch.device, use_ema: bool = False,
):
    """Load a TrainMLPLowdimWorkspace checkpoint.

    Returns ``(mlp, per_key_norm, action_norm, pcfg)`` where:
    - *mlp* is the ``InlineMLP`` in eval mode on *device*.
    - *per_key_norm* is ``{policy_key: {'scale', 'offset'}}``.
    - *action_norm* is ``{'scale', 'offset'}``.
    - *pcfg* is a plain dict with policy hyper-parameters.
    """
    payload = torch.load(
        open(ckpt_path, "rb"), pickle_module=dill, weights_only=False,
    )
    cfg = _resolve_cfg(payload["cfg"])

    shape_meta = _get(cfg, "task.shape_meta", "shape_meta")
    obs_meta = shape_meta["obs"]
    lowdim_keys = sorted([
        k for k, v in obs_meta.items()
        if v.get("type", "low_dim") == "low_dim"
    ])
    obs_key_dims = {k: obs_meta[k]["shape"][0] for k in lowdim_keys}
    obs_feature_dim = sum(obs_key_dims.values())
    action_dim = shape_meta["action"]["shape"][0]

    n_obs_steps = int(_get(cfg, "n_obs_steps", "policy.n_obs_steps"))
    n_action_steps = int(_get(cfg, "n_action_steps", "policy.n_action_steps"))
    horizon = int(_get(cfg, "horizon", "policy.horizon"))
    hidden_dim = int(_get(cfg, "policy.hidden_dim", default=512))
    hidden_depth = int(_get(cfg, "policy.hidden_depth", default=4))

    input_dim = obs_feature_dim * n_obs_steps
    mlp = InlineMLP(input_dim, action_dim, hidden_dim, hidden_depth)

    sd_key = "ema_model" if (use_ema and "ema_model" in payload["state_dicts"]) else "model"
    full_sd = payload["state_dicts"][sd_key]
    print(f"[INFO] Loading MLP weights from '{sd_key}' key.")

    mlp_sd = {}
    for k, v in full_sd.items():
        for prefix in ("trunk.", "mean_head."):
            if k.startswith(prefix):
                mlp_sd[k] = v
    mlp.load_state_dict(mlp_sd)
    mlp.eval().to(device)

    per_key_norm: dict[str, dict[str, torch.Tensor]] = {}
    for key in lowdim_keys:
        per_key_norm[key] = {
            "scale": full_sd[f"normalizer.params_dict.{key}.scale"].to(device),
            "offset": full_sd[f"normalizer.params_dict.{key}.offset"].to(device),
        }
    action_norm = {
        "scale": full_sd["normalizer.params_dict.action.scale"].to(device),
        "offset": full_sd["normalizer.params_dict.action.offset"].to(device),
    }

    pcfg = {
        "obs_dim": obs_feature_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "lowdim_keys": lowdim_keys,
        "obs_key_dims": obs_key_dims,
        "policy_type": "mlp",
    }
    return mlp, per_key_norm, action_norm, pcfg


# ===================================================================
# Diffusion inference
# ===================================================================

@torch.no_grad()
def predict_action(
    unet: ConditionalUnet1D,
    scheduler: DDPMScheduler,
    norm_params: dict[str, dict[str, torch.Tensor]],
    pcfg: dict[str, Any],
    obs_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run one diffusion inference step.

    Args:
        obs_tensor: ``(B, n_obs_steps, obs_dim)`` on *device*.

    Returns:
        ``(B, n_action_steps, action_dim)`` un-normalised actions.
    """
    nobs = _apply_normalizer(
        obs_tensor, norm_params["obs"]["scale"], norm_params["obs"]["offset"],
        forward=True,
    )
    B = nobs.shape[0]
    To = pcfg["n_obs_steps"]
    T = pcfg["horizon"]
    Da = pcfg["action_dim"]
    Do = pcfg["obs_dim"]
    dtype = nobs.dtype

    local_cond = None
    global_cond = None

    if pcfg["obs_as_local_cond"]:
        local_cond = torch.zeros((B, T, Do), device=device, dtype=dtype)
        local_cond[:, :To] = nobs[:, :To]
        shape = (B, T, Da)
        cond_data = torch.zeros(shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    elif pcfg["obs_as_global_cond"]:
        global_cond = nobs[:, :To].reshape(B, -1)
        shape = (B, pcfg["n_action_steps"], Da) if pcfg["pred_action_steps_only"] else (B, T, Da)
        cond_data = torch.zeros(shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    else:
        shape = (B, T, Da + Do)
        cond_data = torch.zeros(shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        cond_data[:, :To, Da:] = nobs[:, :To]
        cond_mask[:, :To, Da:] = True

    trajectory = torch.randn(cond_data.shape, dtype=dtype, device=device)
    scheduler.set_timesteps(pcfg["num_inference_steps"])
    for t in scheduler.timesteps:
        trajectory[cond_mask] = cond_data[cond_mask]
        model_output = unet(
            trajectory, t, local_cond=local_cond, global_cond=global_cond,
        )
        trajectory = scheduler.step(model_output, t, trajectory).prev_sample
    trajectory[cond_mask] = cond_data[cond_mask]

    naction_pred = trajectory[..., :Da]
    action_pred = _apply_normalizer(
        naction_pred,
        norm_params["action"]["scale"],
        norm_params["action"]["offset"],
        forward=False,
    )

    if pcfg["pred_action_steps_only"]:
        return action_pred

    start = To - 1 if pcfg["oa_step_convention"] else To
    end = start + pcfg["n_action_steps"]
    return action_pred[:, start:end]


# ===================================================================
# MLP inference
# ===================================================================

@torch.no_grad()
def predict_action_mlp(
    mlp: InlineMLP,
    per_key_norm: dict[str, dict[str, torch.Tensor]],
    action_norm: dict[str, torch.Tensor],
    pcfg: dict[str, Any],
    obs_dict_seq: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Run one MLP inference step.

    Args:
        obs_dict_seq: ``{policy_key: (B, n_obs_steps, key_dim)}`` on *device*.

    Returns:
        ``(B, 1, action_dim)`` un-normalised actions.
    """
    lowdim_keys = pcfg["lowdim_keys"]
    To = pcfg["n_obs_steps"]
    B = next(iter(obs_dict_seq.values())).shape[0]

    parts = []
    for key in lowdim_keys:
        x = obs_dict_seq[key][:, :To]
        nx = _apply_normalizer(
            x, per_key_norm[key]["scale"], per_key_norm[key]["offset"],
            forward=True,
        )
        parts.append(nx.reshape(B, To, -1))

    per_step = torch.cat(parts, dim=-1)  # (B, To, obs_feature_dim)
    flat_input = per_step.reshape(B, -1)  # (B, To * obs_feature_dim)

    action_mean = mlp(flat_input)  # (B, action_dim)
    action = _apply_normalizer(
        action_mean, action_norm["scale"], action_norm["offset"],
        forward=False,
    )
    return action.unsqueeze(1)  # (B, 1, action_dim)


def _sim_obs_to_policy_dict(
    obs_dict: dict[str, torch.Tensor],
    key_map: dict[str, list[tuple[str, int | None]]],
) -> dict[str, torch.Tensor]:
    """Convert sim obs_dict to policy obs_dict using key_map."""
    policy_dict: dict[str, torch.Tensor] = {}
    for sim_key, targets in key_map.items():
        if sim_key not in obs_dict:
            continue
        val = obs_dict[sim_key]
        if len(targets) == 1 and targets[0][1] is None:
            policy_dict[targets[0][0]] = val
        else:
            offset = 0
            for pkey, dim in targets:
                if dim is None:
                    policy_dict[pkey] = val[..., offset:]
                    break
                policy_dict[pkey] = val[..., offset:offset + dim]
                offset += dim
    return policy_dict


# ===================================================================
# Environment helpers
# ===================================================================

def _build_policy_term_slices(env: gym.Env):
    obs_manager = env.unwrapped.observation_manager
    term_names = list(obs_manager.active_terms["policy"])
    term_dims = list(obs_manager.group_obs_term_dim["policy"])
    slices: list[tuple[int, int]] = []
    start = 0
    for dims in term_dims:
        if len(dims) != 1:
            raise RuntimeError(f"Only 1-D policy terms supported; got dims={dims}")
        end = start + int(dims[0])
        slices.append((start, end))
        start = end
    return term_names, slices


def _get_policy_obs(obs_raw):
    """Extract the ``"policy"`` value from obs, handling dict / TensorDict / tensor."""
    if isinstance(obs_raw, torch.Tensor):
        return obs_raw
    try:
        return obs_raw["policy"]
    except (KeyError, TypeError, IndexError):
        return obs_raw


def _check_success_from_reward_manager(env: gym.Env, env_idx: int = 0, verbose: bool = False) -> bool:
    """Fallback success check via the reward manager's ``progress_context`` term.

    The peg insertion env tracks ``success`` (position + orientation aligned)
    inside the ``progress_context`` reward term rather than as a termination.
    """
    try:
        ctx = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
        success_val = bool(getattr(ctx, "success")[env_idx])
        if verbose:
            pos_aligned = bool(getattr(ctx, "position_aligned")[env_idx])
            ori_aligned = bool(getattr(ctx, "orientation_aligned")[env_idx])
            xyz_dist = float(getattr(ctx, "xyz_distance")[env_idx])
            euler_dist = float(getattr(ctx, "euler_xy_distance")[env_idx])
            print(f"  [DEBUG] success={success_val}, pos_aligned={pos_aligned}, "
                  f"ori_aligned={ori_aligned}, xyz_dist={xyz_dist:.5f}, euler_dist={euler_dist:.5f}")
        return success_val
    except Exception as e:
        print(f"  [WARN] _check_success_from_reward_manager failed: {e}")
        return False


def _extract_obs_dict(obs_policy, term_names, slices):
    if isinstance(obs_policy, torch.Tensor):
        return {n: obs_policy[:, s:e] for n, (s, e) in zip(term_names, slices)}
    return {n: obs_policy[n] for n in term_names}


def _obs_dict_to_vector(obs_dict, obs_keys):
    parts = []
    for key in obs_keys:
        t = obs_dict[key]
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        parts.append(t.squeeze(0).cpu().numpy())
    return np.concatenate(parts, axis=-1).astype(np.float32)


# ===================================================================
# Trajectory I/O
# ===================================================================

def _trajectory_template() -> dict[str, Any]:
    return {
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "obs_flat": [],
        "obs_proprio": {},
        "obs_assets": {},
        "obs_images": {},
        "obs_other_state": {},
    }


def _append_obs(
    traj: dict[str, Any],
    obs_dict: dict[str, torch.Tensor],
    obs_flat_vec: np.ndarray,
    proprio_keys: list[str],
    asset_keys: list[str],
):
    traj["obs_flat"].append(torch.from_numpy(obs_flat_vec).unsqueeze(0))
    for key in proprio_keys:
        if key in obs_dict:
            traj["obs_proprio"].setdefault(key, []).append(
                obs_dict[key].detach().cpu(),
            )
    for key in asset_keys:
        if key in obs_dict:
            traj["obs_assets"].setdefault(key, []).append(
                obs_dict[key].detach().cpu(),
            )


def _finalize_trajectory(traj: dict[str, Any], success: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"success": success}
    out["actions"] = torch.cat(traj["actions"]) if traj["actions"] else torch.empty(0)
    out["rewards"] = torch.cat(traj["rewards"]) if traj["rewards"] else torch.empty(0)
    out["terminated"] = (
        torch.cat(traj["terminated"]) if traj["terminated"]
        else torch.empty(0, dtype=torch.bool)
    )
    out["truncated"] = (
        torch.cat(traj["truncated"]) if traj["truncated"]
        else torch.empty(0, dtype=torch.bool)
    )
    out["obs_flat"] = torch.cat(traj["obs_flat"]) if traj["obs_flat"] else torch.empty(0)
    for group in ("obs_proprio", "obs_assets", "obs_images", "obs_other_state"):
        out[group] = {
            k: torch.cat(v) if v else torch.empty(0)
            for k, v in traj[group].items()
        }
    return out


def _save_trajectory(
    traj: dict[str, Any],
    success: bool,
    traj_idx: int,
    traj_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
):
    serialized = _finalize_trajectory(traj, success)
    traj_file = traj_dir / f"traj_{traj_idx:06d}.pt"
    torch.save(serialized, traj_file)
    manifest["files"].append({
        "trajectory_id": traj_idx,
        "file": str(traj_file.relative_to(output_dir)),
        "steps": int(serialized["actions"].shape[0]),
        "success": success,
    })


# ===================================================================
# Single-episode rollout
# ===================================================================

def rollout_episode(
    unet: ConditionalUnet1D,
    scheduler: DDPMScheduler,
    norm_params: dict,
    pcfg: dict,
    env: gym.Env,
    rsl_env,
    term_names: list[str],
    slices: list[tuple[int, int]],
    obs_keys: list[str],
    proprio_keys: list[str],
    asset_keys: list[str],
    horizon: int,
    device: torch.device,
    success_term=None,
    verbose: bool = False,
) -> tuple[bool, int, dict[str, Any]]:
    """Run one episode. Returns ``(success, steps, trajectory_dict)``."""
    n_obs_steps = pcfg["n_obs_steps"]
    n_action_steps = pcfg["n_action_steps"]
    obs_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=n_obs_steps)
    traj = _trajectory_template()

    obs_raw = rsl_env.get_observations()
    obs_policy = _get_policy_obs(obs_raw)
    obs_dict = _extract_obs_dict(obs_policy, term_names, slices)
    first_obs = _obs_dict_to_vector(obs_dict, obs_keys)
    for _ in range(n_obs_steps):
        obs_buffer.append(first_obs.copy())
    _append_obs(traj, obs_dict, first_obs, proprio_keys, asset_keys)

    if verbose:
        print(f"  [DEBUG] Initial obs range: min={first_obs.min():.4f}, max={first_obs.max():.4f}, "
              f"mean={first_obs.mean():.4f}, has_nan={np.isnan(first_obs).any()}")

    step = 0
    while step < horizon:
        obs_seq = np.stack(list(obs_buffer), axis=0)
        obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(device)

        actions = predict_action(
            unet, scheduler, norm_params, pcfg, obs_tensor, device,
        )
        actions_np = actions.squeeze(0).cpu().numpy()

        if verbose and step == 0:
            print(f"  [DEBUG] First action chunk (8 actions):\n{actions_np}")
            nobs = _apply_normalizer(
                obs_tensor, norm_params["obs"]["scale"], norm_params["obs"]["offset"],
                forward=True,
            )
            print(f"  [DEBUG] Normalized obs range: min={nobs.min().item():.4f}, "
                  f"max={nobs.max().item():.4f}")

        for a_idx in range(min(n_action_steps, horizon - step)):
            action = torch.from_numpy(actions_np[a_idx]).unsqueeze(0).to(device)
            obs_raw, rew, dones, extras = rsl_env.step(action)

            obs_policy = _get_policy_obs(obs_raw)
            obs_dict = _extract_obs_dict(obs_policy, term_names, slices)
            obs_vec = _obs_dict_to_vector(obs_dict, obs_keys)
            obs_buffer.append(obs_vec)
            step += 1

            traj["actions"].append(action.detach().cpu())
            traj["rewards"].append(rew[:1].detach().cpu().view(1, 1))
            time_outs = extras.get("time_outs", torch.zeros_like(dones))
            traj["terminated"].append(
                (dones[:1] & ~time_outs[:1]).float().detach().cpu().view(1, 1),
            )
            traj["truncated"].append(
                time_outs[:1].float().detach().cpu().view(1, 1),
            )
            _append_obs(traj, obs_dict, obs_vec, proprio_keys, asset_keys)

            is_success = False
            if success_term is not None:
                try:
                    success_val = success_term.func(
                        env.unwrapped, **success_term.params,
                    )
                    is_success = bool(success_val[0])
                except Exception:
                    pass
            else:
                is_success = _check_success_from_reward_manager(env, verbose=(verbose and step % 20 == 0))

            if is_success:
                return True, step, traj

            if bool(dones[0].item()):
                is_timeout = bool(time_outs[0].item())
                is_terminated = bool((dones[0] & ~time_outs[0]).item())
                if verbose:
                    print(f"  [DEBUG] Episode ended at step {step}: "
                          f"timeout={is_timeout}, terminated(abnormal)={is_terminated}")
                    if is_terminated:
                        try:
                            robot = env.unwrapped.scene["robot"]
                            jvel = robot.data.joint_vel[0].abs()
                            jlim = robot.data.joint_vel_limits[0, :, 1].abs() * 2
                            violated = jvel > jlim
                            if violated.any():
                                idx = violated.nonzero(as_tuple=True)[0]
                                print(f"  [DEBUG] Joint vel limit violated at joints: {idx.tolist()}")
                                print(f"  [DEBUG]   vel={jvel[idx].tolist()}, limit(2x)={jlim[idx].tolist()}")
                        except Exception as e:
                            print(f"  [DEBUG] Could not inspect joint vels: {e}")
                    _check_success_from_reward_manager(env, verbose=True)
                return False, step, traj

    if verbose:
        print(f"  [DEBUG] Episode ended at horizon ({horizon} steps)")
        _check_success_from_reward_manager(env, verbose=True)
    return False, step, traj


def rollout_episode_mlp(
    mlp: InlineMLP,
    per_key_norm: dict,
    action_norm: dict,
    pcfg: dict,
    env: gym.Env,
    rsl_env,
    term_names: list[str],
    slices: list[tuple[int, int]],
    key_map: dict[str, list[tuple[str, int | None]]],
    obs_keys: list[str],
    proprio_keys: list[str],
    asset_keys: list[str],
    horizon: int,
    device: torch.device,
    success_term=None,
    verbose: bool = False,
) -> tuple[bool, int, dict[str, Any]]:
    """Run one episode with MLP policy. Returns ``(success, steps, trajectory_dict)``."""
    n_obs_steps = pcfg["n_obs_steps"]
    n_action_steps = pcfg["n_action_steps"]
    lowdim_keys = pcfg["lowdim_keys"]

    obs_dict_buffer: collections.deque[dict[str, np.ndarray]] = collections.deque(maxlen=n_obs_steps)
    traj = _trajectory_template()

    obs_raw = rsl_env.get_observations()
    obs_policy = _get_policy_obs(obs_raw)
    sim_obs_dict = _extract_obs_dict(obs_policy, term_names, slices)
    first_obs_flat = _obs_dict_to_vector(sim_obs_dict, obs_keys)

    policy_obs = _sim_obs_to_policy_dict(
        {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 1 else v
         for k, v in sim_obs_dict.items()},
        key_map,
    )
    policy_obs_np = {k: v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v
                     for k, v in policy_obs.items()}

    for _ in range(n_obs_steps):
        obs_dict_buffer.append({k: v.copy() for k, v in policy_obs_np.items()})
    _append_obs(traj, sim_obs_dict, first_obs_flat, proprio_keys, asset_keys)

    if verbose:
        print(f"  [DEBUG] MLP policy keys: {lowdim_keys}")
        for k, v in policy_obs_np.items():
            print(f"  [DEBUG]   {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

    step = 0
    while step < horizon:
        obs_seq = {
            k: torch.from_numpy(np.stack([buf[k] for buf in obs_dict_buffer], axis=0)).unsqueeze(0).to(device)
            for k in lowdim_keys
        }
        actions = predict_action_mlp(mlp, per_key_norm, action_norm, pcfg, obs_seq, device)
        actions_np = actions.squeeze(0).cpu().numpy()

        if verbose and step == 0:
            print(f"  [DEBUG] First MLP action: {actions_np[0]}")

        for a_idx in range(min(n_action_steps, horizon - step)):
            action = torch.from_numpy(actions_np[a_idx]).unsqueeze(0).to(device)
            obs_raw, rew, dones, extras = rsl_env.step(action)

            obs_policy_raw = _get_policy_obs(obs_raw)
            sim_obs_dict = _extract_obs_dict(obs_policy_raw, term_names, slices)
            obs_flat = _obs_dict_to_vector(sim_obs_dict, obs_keys)

            policy_obs = _sim_obs_to_policy_dict(
                {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 1 else v
                 for k, v in sim_obs_dict.items()},
                key_map,
            )
            policy_obs_np = {k: v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v
                             for k, v in policy_obs.items()}
            obs_dict_buffer.append(policy_obs_np)
            step += 1

            traj["actions"].append(action.detach().cpu())
            traj["rewards"].append(rew[:1].detach().cpu().view(1, 1))
            time_outs = extras.get("time_outs", torch.zeros_like(dones))
            traj["terminated"].append(
                (dones[:1] & ~time_outs[:1]).float().detach().cpu().view(1, 1),
            )
            traj["truncated"].append(
                time_outs[:1].float().detach().cpu().view(1, 1),
            )
            _append_obs(traj, sim_obs_dict, obs_flat, proprio_keys, asset_keys)

            is_success = False
            if success_term is not None:
                try:
                    success_val = success_term.func(
                        env.unwrapped, **success_term.params,
                    )
                    is_success = bool(success_val[0])
                except Exception:
                    pass
            else:
                is_success = _check_success_from_reward_manager(env, verbose=(verbose and step % 20 == 0))

            if is_success:
                return True, step, traj

            if bool(dones[0].item()):
                is_timeout = bool(time_outs[0].item())
                is_terminated = bool((dones[0] & ~time_outs[0]).item())
                if verbose:
                    print(f"  [DEBUG] Episode ended at step {step}: "
                          f"timeout={is_timeout}, terminated(abnormal)={is_terminated}")
                    _check_success_from_reward_manager(env, verbose=True)
                return False, step, traj

    if verbose:
        print(f"  [DEBUG] Episode ended at horizon ({horizon} steps)")
        _check_success_from_reward_manager(env, verbose=True)
    return False, step, traj


# ===================================================================
# Checkpoint discovery
# ===================================================================

def discover_checkpoints(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        ckpts = sorted(p.glob("*.ckpt")) + sorted(p.glob("*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt or .pt files found in {p}")
        return ckpts
    raise FileNotFoundError(f"Checkpoint path does not exist: {p}")


class _FrameRecorder(gym.Wrapper):
    """Captures ``render()`` output after every step for video export."""

    def __init__(self, env):
        super().__init__(env)
        self.frames: list[np.ndarray] = []

    def step(self, action):
        result = self.env.step(action)
        frame = self.env.render()
        if frame is not None:
            self.frames.append(frame)
        return result

    def save_video(self, path, fps: int = 30):
        if not self.frames:
            return
        import imageio
        imageio.mimwrite(str(path), self.frames, fps=fps)
        self.frames.clear()


# ===================================================================
# Main
# ===================================================================

@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg) -> None:
    """Roll out a trained diffusion policy and report success rate."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.recorders = None
    env_cfg.observations.policy.concatenate_terms = True

    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is not None:
        env_cfg.terminations.success = None

    render_mode = "rgb_array" if args_cli.video else None
    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)

    frame_recorder = None
    if args_cli.video:
        frame_recorder = _FrameRecorder(gym_env)
        gym_env = frame_recorder
        print("[INFO] Video recording enabled.")

    rsl_env = RslRlVecEnvWrapper(gym_env)

    device = torch.device(env_cfg.sim.device if env_cfg.sim.device else "cuda:0")
    term_names, slices = _build_policy_term_slices(gym_env)
    print(f"[INFO] Policy obs terms: {term_names}")

    ckpt_paths = discover_checkpoints(args_cli.checkpoint)
    print(f"[INFO] Found {len(ckpt_paths)} checkpoint(s) to evaluate.")

    output_root = Path(args_cli.output_dir)
    all_results: dict[str, dict[str, Any]] = {}

    key_map = _parse_key_map(args_cli.sim_to_policy_key_map)

    for ckpt_path in ckpt_paths:
        ckpt_name = ckpt_path.stem
        print(f"\n{'='*60}")
        print(f"[INFO] Evaluating: {ckpt_path}")
        print(f"{'='*60}")

        policy_type = args_cli.policy_type
        if policy_type == "auto":
            policy_type = detect_policy_type(str(ckpt_path))
        print(f"[INFO] Detected policy type: {policy_type}")

        unet = scheduler = norm_params = None
        mlp = per_key_norm = action_norm = None

        if policy_type == "mlp":
            mlp, per_key_norm, action_norm, pcfg = load_mlp_checkpoint(
                str(ckpt_path), device, use_ema=args_cli.use_ema,
            )
        else:
            unet, scheduler, norm_params, pcfg = load_checkpoint(
                str(ckpt_path), device, use_ema=args_cli.use_ema,
            )

        if args_cli.n_obs_steps is not None:
            pcfg["n_obs_steps"] = args_cli.n_obs_steps
        if args_cli.n_action_steps is not None:
            pcfg["n_action_steps"] = args_cli.n_action_steps

        print(f"[INFO] Policy config: horizon={pcfg['horizon']}, "
              f"n_obs_steps={pcfg['n_obs_steps']}, "
              f"n_action_steps={pcfg['n_action_steps']}, "
              f"obs_dim={pcfg['obs_dim']}, action_dim={pcfg['action_dim']}")

        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

        ckpt_output = output_root / ckpt_name
        traj_dir = ckpt_output / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "checkpoint": str(ckpt_path.resolve()),
            "task": args_cli.task,
            "num_rollouts": args_cli.num_rollouts,
            "horizon": args_cli.horizon,
            "seed": args_cli.seed,
            "policy_config": {k: v for k, v in pcfg.items() if not isinstance(v, (dict, list)) or k in ("lowdim_keys",)},
            "files": [],
        }

        results: list[bool] = []
        for trial in range(args_cli.num_rollouts):
            print(f"\n[INFO] Trial {trial + 1}/{args_cli.num_rollouts}")

            if policy_type == "mlp":
                success, steps, traj = rollout_episode_mlp(
                    mlp=mlp,
                    per_key_norm=per_key_norm,
                    action_norm=action_norm,
                    pcfg=pcfg,
                    env=gym_env,
                    rsl_env=rsl_env,
                    term_names=term_names,
                    slices=slices,
                    key_map=key_map,
                    obs_keys=args_cli.obs_keys,
                    proprio_keys=args_cli.proprio_keys,
                    asset_keys=args_cli.asset_keys,
                    horizon=args_cli.horizon,
                    device=device,
                    success_term=success_term,
                    verbose=args_cli.verbose,
                )
            else:
                success, steps, traj = rollout_episode(
                    unet=unet,
                    scheduler=scheduler,
                    norm_params=norm_params,
                    pcfg=pcfg,
                    env=gym_env,
                    rsl_env=rsl_env,
                    term_names=term_names,
                    slices=slices,
                    obs_keys=args_cli.obs_keys,
                    proprio_keys=args_cli.proprio_keys,
                    asset_keys=args_cli.asset_keys,
                    horizon=args_cli.horizon,
                    device=device,
                    success_term=success_term,
                    verbose=args_cli.verbose,
                )
            results.append(success)
            print(f"[INFO]   -> {'SUCCESS' if success else 'FAILURE'} in {steps} steps")

            _save_trajectory(
                traj, success, trial, traj_dir, ckpt_output, manifest,
            )

            if frame_recorder is not None and frame_recorder.frames:
                vid_dir = ckpt_output / "videos"
                vid_dir.mkdir(parents=True, exist_ok=True)
                vid_path = vid_dir / f"rollout_{trial:03d}.mp4"
                frame_recorder.save_video(vid_path)
                print(f"[INFO]   Video saved to {vid_path}")

        n_success = sum(results)
        rate = 100.0 * n_success / args_cli.num_rollouts
        manifest["success_count"] = n_success
        manifest["success_rate"] = rate

        with open(ckpt_output / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Checkpoint : {ckpt_name}")
        print(f"  Success    : {n_success}/{args_cli.num_rollouts} ({rate:.1f}%)")
        print(f"  Saved to   : {ckpt_output.resolve()}")
        print(f"{'='*60}")

        all_results[ckpt_name] = {
            "success_count": n_success,
            "total": args_cli.num_rollouts,
            "success_rate": rate,
        }

    if len(all_results) > 1:
        print(f"\n\n{'='*60}")
        print("  SUMMARY ACROSS ALL CHECKPOINTS")
        print(f"{'='*60}")
        for name, res in all_results.items():
            print(f"  {name:40s}  {res['success_count']}/{res['total']} "
                  f"({res['success_rate']:.1f}%)")
        print(f"{'='*60}")

        with open(output_root / "summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    rsl_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
