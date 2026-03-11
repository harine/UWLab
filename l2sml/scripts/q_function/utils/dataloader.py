from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class ActionChunkZarrDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        chunk_horizon: int = 16,
        image_keys=("front_rgb", "side_rgb", "wrist_rgb"),
        use_images=False,
        use_lowdim=True,
        normalize_actions: bool = False,
        gamma: float = 0.97,
    ):
        open_group = cast(Any, getattr(zarr, "open_group"))
        self.root = open_group(zarr_path, mode="a")
        self.H = chunk_horizon
        self.use_images = use_images
        self.use_lowdim = use_lowdim
        self.gamma = float(gamma)

        data = cast(Any, self.root["data"])
        obs = cast(Any, data["obs"])
        cam = cast(Any, data["camera"]) if "camera" in data else None
        meta = cast(Any, self.root["meta"])
        self.data = data
        self.meta = meta

        self.actions = cast(Any, data["actions"])
        self.rewards = cast(Any, data["rewards"])
        self.episode_ends = np.asarray(cast(Any, meta["episode_ends"])[:], dtype=np.int64)

        if self.H <= 0:
            raise ValueError(f"chunk_horizon must be positive, got {self.H}")
        if not self.episode_ends.size:
            raise ValueError(f"No episodes found in Zarr dataset: {zarr_path}")

        # Low-dimensional observations
        self.lowdim_arrays = {}
        lowdim_keys = [
            "joint_pos",
            "end_effector_pose",
            "insertive_asset_in_receptive_asset_frame",
            "insertive_asset_pose",
            "receptive_asset_pose",
            "prev_actions",
        ]
        for k in lowdim_keys:
            if k in obs:
                self.lowdim_arrays[k] = obs[k]

        # Cameras
        self.image_arrays = {}
        if cam is not None:
            for k in image_keys:
                if k in cam:
                    self.image_arrays[k] = cam[k]

        if self.use_lowdim and not self.lowdim_arrays:
            raise ValueError("No low-dimensional observation arrays found in Zarr dataset.")

        self.return_to_go = self._get_or_compute_cached_return_to_go()

        # Build valid sample start indices
        self.valid_starts = []
        start = 0
        for end in self.episode_ends:
            # episode is [start, end)
            max_start = end - self.H
            for t in range(start, max_start + 1):
                self.valid_starts.append(t)
            start = end
        self.valid_starts = np.asarray(self.valid_starts, dtype=np.int64)
        if len(self.valid_starts) == 0:
            raise ValueError(
                f"No valid samples found for chunk_horizon={self.H}. "
                "All episodes are shorter than the requested chunk length."
            )

        # Action normalization stats
        self.normalize_actions = normalize_actions
        actions_np = np.asarray(self.actions[:], dtype=np.float32)
        self.action_mean = actions_np.mean(axis=0)
        self.action_std = actions_np.std(axis=0) + 1e-6
        self.action_dim = int(actions_np.shape[1])

        if self.use_lowdim:
            state_dim = 0
            for arr in self.lowdim_arrays.values():
                state_dim += int(np.asarray(arr[0]).reshape(-1).shape[0])
            self.state_dim = state_dim
        else:
            self.state_dim = 0
        self.action_chunk_dim = self.H * self.action_dim

    def __len__(self):
        return len(self.valid_starts)

    def _compute_return_to_go(self, rewards: np.ndarray) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if rewards.size == 0:
            return rewards.astype(np.float32)
        if self.gamma == 0.0:
            return rewards.astype(np.float32)

        discounts = np.power(self.gamma, np.arange(rewards.shape[0], dtype=np.float64))
        if np.any(discounts == 0.0):
            returns = np.empty_like(rewards, dtype=np.float64)
            running = 0.0
            for t in range(rewards.shape[0] - 1, -1, -1):
                running = float(rewards[t]) + self.gamma * running
                returns[t] = running
            return returns.astype(np.float32)

        discounted_rewards = rewards * discounts
        returns = np.cumsum(discounted_rewards[::-1])[::-1] / discounts
        return returns.astype(np.float32)

    def _should_recompute_return_to_go(self, cached_gamma: Any, cached_returns: Any) -> bool:
        if cached_returns is None:
            return True
        if cached_gamma is None:
            return True

        cached_shape = tuple(getattr(cached_returns, "shape", ()))
        expected_shape = (int(self.actions.shape[0]),)
        if cached_shape != expected_shape:
            return True

        try:
            cached_gamma_float = float(cached_gamma)
        except (TypeError, ValueError):
            return True
        return not np.isclose(cached_gamma_float, self.gamma)

    def _write_return_to_go_cache(self, return_to_go: np.ndarray) -> None:
        if "returns_to_go" in self.data:
            cached_returns = cast(Any, self.data["returns_to_go"])
            if tuple(cached_returns.shape) != tuple(return_to_go.shape):
                cached_returns.resize(return_to_go.shape)
            cached_returns[:] = return_to_go
        else:
            chunk_len = max(1, min(int(return_to_go.shape[0]), 5000))
            self.data.create_dataset(
                "returns_to_go",
                data=return_to_go,
                chunks=(chunk_len,),
                dtype=np.float32,
            )

        self.meta.attrs["return_to_go_gamma"] = self.gamma

    def _get_or_compute_cached_return_to_go(self) -> np.ndarray:
        cached_returns = cast(Any, self.data["returns_to_go"]) if "returns_to_go" in self.data else None
        cached_gamma = self.meta.attrs.get("return_to_go_gamma")
        if not self._should_recompute_return_to_go(cached_gamma=cached_gamma, cached_returns=cached_returns):
            assert cached_returns is not None
            return np.asarray(cached_returns[:], dtype=np.float32)

        return_to_go = np.zeros(int(self.actions.shape[0]), dtype=np.float32)
        start = 0
        for end in self.episode_ends:
            rewards_np = np.asarray(self.rewards[start:end], dtype=np.float32).reshape(-1)
            return_to_go[start:end] = self._compute_return_to_go(rewards_np)
            start = int(end)

        self._write_return_to_go_cache(return_to_go)
        return return_to_go

    def _process_image(self, x):
        x = np.asarray(x)

        # Convert uint8 image to float in [0,1]
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        else:
            x = x.astype(np.float32)

        # Convert NHWC -> CHW if needed
        if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
            x = np.transpose(x, (2, 0, 1))

        return x

    def __getitem__(self, idx):
        t = int(self.valid_starts[idx])

        # Current observation at time t
        lowdim_obs = None
        if self.use_lowdim:
            lowdim_parts = []
            for k, arr in self.lowdim_arrays.items():
                x = np.asarray(arr[t], dtype=np.float32).reshape(-1)
                lowdim_parts.append(x)
            if not lowdim_parts:
                raise ValueError("No low-dimensional features available for Q-function training.")
            lowdim_obs = torch.from_numpy(np.concatenate(lowdim_parts, axis=0))

        if self.use_images:
            for k, arr in self.image_arrays.items():
                img = self._process_image(arr[t])
                _ = torch.from_numpy(img)

        # Target chunk of future actions [t, t+H)
        action_chunk = np.asarray(self.actions[t : t + self.H], dtype=np.float32)
        if self.normalize_actions:
            action_chunk = (action_chunk - self.action_mean) / self.action_std

        action_chunk = torch.from_numpy(action_chunk.reshape(-1))
        returns = torch.tensor([self.return_to_go[t]], dtype=torch.float32)

        if lowdim_obs is None:
            raise ValueError("Q-function training requires low-dimensional observations.")

        return lowdim_obs, action_chunk, returns