from __future__ import annotations

from typing import Any

import torch


class NoiseAroundActionWrapper:
    """Sample one action from the policy, add i.i.d. Gaussian noise to build k candidates, pick best by Q."""

    def __init__(self, policy, q_function, k: int, noise_std: float):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}.")
        self.policy = policy
        self.k = k
        self.noise_std = float(noise_std)
        self.q_function = q_function

    def _extract_policy_obs(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        nested_obs = obs_dict.get("obs")
        if isinstance(nested_obs, dict):
            return nested_obs
        return obs_dict

    def _repeat_obs_batch(self, obs_dict: dict[str, Any], repeats: int) -> dict[str, Any]:
        repeated_obs = {}
        for key, value in obs_dict.items():
            if isinstance(value, dict):
                repeated_obs[key] = self._repeat_obs_batch(value, repeats)
            elif isinstance(value, torch.Tensor):
                repeated_obs[key] = value.repeat_interleave(repeats, dim=0)
            else:
                repeated_obs[key] = value
        return repeated_obs

    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        if not hasattr(self.policy, "predict_action"):
            raise AttributeError(
                f"{self.policy.__class__.__name__} must implement predict_action(obs_dict) "
                "to be used with NoiseAroundActionWrapper."
            )
        if not hasattr(self.q_function, "predict_q"):
            raise AttributeError(
                f"{self.q_function.__class__.__name__} must implement predict_q(obs_dict, action) "
                "to be used with NoiseAroundActionWrapper."
            )

        policy_obs = self._extract_policy_obs(obs_dict)
        base = self.policy.predict_action(policy_obs)
        if not isinstance(base, dict) or "action" not in base:
            raise ValueError("predict_action() must return a dict containing an 'action' tensor.")

        actions = base["action"]
        if actions.ndim < 2:
            raise ValueError(f"Expected action with shape (batch, ...), got {tuple(actions.shape)}.")

        batch_size = actions.shape[0]
        noise = torch.randn(
            self.k,
            batch_size,
            *actions.shape[1:],
            device=actions.device,
            dtype=actions.dtype,
        )
        candidates = actions.unsqueeze(0) + self.noise_std * noise

        batched_actions = candidates.movedim(0, 1).reshape(batch_size * self.k, *actions.shape[1:])
        repeated_obs = self._repeat_obs_batch(policy_obs, self.k)

        q_values = self.q_function.predict_q(repeated_obs, batched_actions)
        q_values = q_values.reshape(batch_size, self.k, -1)
        if q_values.shape[-1] != 1:
            raise ValueError(
                f"Expected scalar Q-values per candidate, got trailing shape {tuple(q_values.shape[2:])}."
            )

        best_indices = q_values.squeeze(-1).argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=actions.device)
        best_actions = candidates.movedim(0, 1)[batch_indices, best_indices]

        out: dict[str, torch.Tensor] = {"action": best_actions}
        if "action_pred" in base:
            out["action_pred"] = base["action_pred"]
        else:
            out["action_pred"] = best_actions
        return out

    def reset(self):
        self.policy.reset()
        if hasattr(self.q_function, "reset"):
            self.q_function.reset()
