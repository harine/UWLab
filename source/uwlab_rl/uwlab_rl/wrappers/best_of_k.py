from __future__ import annotations

from typing import Any

import torch


class BestOfKWrapper:
    def __init__(self, policy, q_function, k: int):
        self.policy = policy
        self.k = k
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
        if not hasattr(self.policy, "predict_k_actions"):
            raise AttributeError(
                f"{self.policy.__class__.__name__} must implement predict_k_actions(obs_dict, k) "
                "to be used with BestOfKWrapper."
            )
        if not hasattr(self.q_function, "predict_q"):
            raise AttributeError(
                f"{self.q_function.__class__.__name__} must implement predict_q(obs_dict, action) "
                "to be used with BestOfKWrapper."
            )

        policy_obs = self._extract_policy_obs(obs_dict)
        result = self.policy.predict_k_actions(policy_obs, self.k)

        if not isinstance(result, dict) or "actions" not in result:
            raise ValueError(
                "predict_k_actions() must return a dict containing an 'actions' tensor "
                "with shape (k, batch_size, action_horizon, action_dim)."
            )

        actions = result["actions"]
        if actions.ndim < 3:
            raise ValueError(
                f"Expected candidate actions with shape (k, batch_size, ...), got {tuple(actions.shape)}."
            )
        if actions.shape[0] != self.k:
            raise ValueError(
                f"Expected {self.k} candidate action chunks, but policy returned {actions.shape[0]}."
            )

        batch_size = actions.shape[1]
        batched_actions = actions.movedim(0, 1).reshape(batch_size * self.k, *actions.shape[2:])
        repeated_obs = self._repeat_obs_batch(policy_obs, self.k)

        q_values = self.q_function.predict_q(repeated_obs, batched_actions)
        q_values = q_values.reshape(batch_size, self.k, -1)
        if q_values.shape[-1] != 1:
            raise ValueError(
                f"Expected scalar Q-values per candidate, got trailing shape {tuple(q_values.shape[2:])}."
            )

        best_indices = q_values.squeeze(-1).argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=actions.device)

        best_actions = actions.movedim(0, 1)[batch_indices, best_indices]
        action_preds = result.get("action_preds")
        if action_preds is not None:
            best_action_preds = action_preds.movedim(0, 1)[batch_indices, best_indices]
        else:
            best_action_preds = best_actions

        return {
            "action": best_actions,
            "action_pred": best_action_preds,
        }

    def reset(self):
        self.policy.reset()
        if hasattr(self.q_function, "reset"):
            self.q_function.reset()