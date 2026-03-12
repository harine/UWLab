from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Q-function training data with "
            "(state, action_chunk, next_state, return_to_go) "
            "from saved expert trajectories."
        )
    )
    parser.add_argument(
        "--trajectory_path",
        type=str,
        required=True,
        help="Path to a single trajectory .pt file, or a directory containing traj_*.pt files.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Number of consecutive actions in each action chunk.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output .pt path for processed Q-data. Default is <trajectory_path>/q_data_chunk_<chunk_size>.pt.",
    )
    parser.add_argument(
        "--max_trajs",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 means all.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.97,
        help="Discount factor for return-to-go calculations.",
    )
    return parser.parse_args()


def _resolve_trajectory_files(trajectory_path: Path) -> list[Path]:
    if trajectory_path.is_file():
        if trajectory_path.suffix != ".pt":
            raise ValueError(f"Expected a .pt trajectory file, got: {trajectory_path}")
        return [trajectory_path]

    if trajectory_path.is_dir():
        traj_files = sorted(trajectory_path.glob("traj_*.pt"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files matching traj_*.pt found under {trajectory_path}")
        return traj_files

    raise FileNotFoundError(f"Trajectory path does not exist: {trajectory_path}")


def _load_traj(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Trajectory at {path} is not a dict.")
    for key in ("obs_flat", "actions", "rewards"):
        if key not in data:
            raise KeyError(f"Trajectory at {path} is missing required key: {key}")
    return data


def _ensure_2d_time_major(tensor: torch.Tensor, name: str, traj_path: Path) -> torch.Tensor:
    if tensor.ndim == 0:
        raise ValueError(f"{name} in {traj_path} must have a time dimension, got scalar.")
    if tensor.ndim == 1:
        return tensor.view(-1, 1).to(dtype=torch.float32)
    return tensor.reshape(tensor.shape[0], -1).to(dtype=torch.float32)


def _ensure_1d_time(tensor: torch.Tensor, name: str, traj_path: Path) -> torch.Tensor:
    if tensor.ndim == 0:
        raise ValueError(f"{name} in {traj_path} must have a time dimension, got scalar.")
    return tensor.reshape(tensor.shape[0], -1)[:, 0].to(dtype=torch.float32)


def _compute_return_to_go(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.empty_like(rewards, dtype=torch.float32)
    running = 0.0
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


def _build_samples_for_traj(
    obs_flat: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    chunk_size: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = rewards.shape[0]
    # Observations are recorded before each action is executed, so the state after
    # an action chunk starting at time t is the observation at t + chunk_size.
    valid_steps = steps - chunk_size
    if valid_steps <= 0:
        empty_state = obs_flat.new_empty((0, obs_flat.shape[1]))
        empty_action_chunks = actions.new_empty((0, chunk_size * actions.shape[1]))
        empty_next_states = obs_flat.new_empty((0, obs_flat.shape[1]))
        empty_returns = rewards.new_empty((0,))
        return empty_state, empty_action_chunks, empty_next_states, empty_returns

    returns = _compute_return_to_go(rewards, gamma)
    states = obs_flat[:valid_steps]
    action_chunks = actions.unfold(0, chunk_size, 1)[:valid_steps].reshape(valid_steps, -1)
    next_states = obs_flat[chunk_size : chunk_size + valid_steps]
    returns_at_states = returns[:valid_steps]
    return states, action_chunks, next_states, returns_at_states


def main() -> None:
    args = _parse_args()
    if args.chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {args.chunk_size}")

    gamma = args.gamma
    input_path = Path(args.trajectory_path)
    traj_files = _resolve_trajectory_files(input_path)
    if args.max_trajs > 0:
        traj_files = traj_files[: args.max_trajs]
    if not traj_files:
        raise FileNotFoundError("No trajectory files selected after applying max_trajs.")

    if args.output_path is None:
        if input_path.is_dir():
            output_path = input_path / f"q_data_chunk_{args.chunk_size}_gamma_{args.gamma}.pt"
        else:
            output_path = input_path.with_name(f"{input_path.stem}_q_data_chunk_{args.chunk_size}_gamma_{args.gamma}.pt")
    else:
        output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    states_all: list[torch.Tensor] = []
    action_chunks_all: list[torch.Tensor] = []
    next_states_all: list[torch.Tensor] = []
    returns_all: list[torch.Tensor] = []
    used_files: list[str] = []
    skipped_too_short: list[str] = []
    skipped_empty: list[str] = []

    for traj_path in traj_files:
        data = _load_traj(traj_path)
        obs_flat_raw = data["obs_flat"]
        actions_raw = data["actions"]
        rewards_raw = data["rewards"]

        if not isinstance(obs_flat_raw, torch.Tensor):
            obs_flat_raw = torch.as_tensor(obs_flat_raw)
        if not isinstance(actions_raw, torch.Tensor):
            actions_raw = torch.as_tensor(actions_raw)
        if not isinstance(rewards_raw, torch.Tensor):
            rewards_raw = torch.as_tensor(rewards_raw)

        obs_flat = _ensure_2d_time_major(obs_flat_raw, "obs_flat", traj_path)
        actions = _ensure_2d_time_major(actions_raw, "actions", traj_path)
        rewards = _ensure_1d_time(rewards_raw, "rewards", traj_path)

        if obs_flat.shape[0] == 0 or actions.shape[0] == 0 or rewards.shape[0] == 0:
            skipped_empty.append(str(traj_path))
            continue

        if not (obs_flat.shape[0] == actions.shape[0] == rewards.shape[0]):
            raise ValueError(
                f"Time dimension mismatch in {traj_path}: "
                f"obs_flat={obs_flat.shape[0]}, actions={actions.shape[0]}, rewards={rewards.shape[0]}"
            )

        states, action_chunks, next_states, returns = _build_samples_for_traj(
            obs_flat=obs_flat,
            actions=actions,
            rewards=rewards,
            chunk_size=args.chunk_size,
            gamma=gamma,
        )
        if states.shape[0] == 0:
            skipped_too_short.append(str(traj_path))
            continue

        states_all.append(states)
        action_chunks_all.append(action_chunks)
        next_states_all.append(next_states)
        returns_all.append(returns)
        used_files.append(str(traj_path))

    if not states_all:
        raise RuntimeError(
            "No valid samples generated. All trajectories were empty, too short, or filtered out."
        )

    states_out = torch.cat(states_all, dim=0)
    action_chunks_out = torch.cat(action_chunks_all, dim=0)
    next_states_out = torch.cat(next_states_all, dim=0)
    returns_out = torch.cat(returns_all, dim=0)

    output: dict[str, Any] = {
        "states": states_out,
        "action_chunks": action_chunks_out,
        "next_states": next_states_out,
        "returns": returns_out,
        "meta": {
            "chunk_size": int(args.chunk_size),
            "gamma": float(gamma),
            "num_source_files": len(used_files),
            "num_samples": int(states_out.shape[0]),
            "state_dim": int(states_out.shape[1]),
            "next_state_dim": int(next_states_out.shape[1]),
            "action_chunk_dim": int(action_chunks_out.shape[1]),
            "source_files": used_files,
            "skipped_too_short": skipped_too_short,
            "skipped_empty": skipped_empty,
        },
    }
    torch.save(output, output_path)

    print(f"[INFO] Read trajectory files: {len(traj_files)}")
    print(f"[INFO] Used trajectory files: {len(used_files)}")
    print(f"[INFO] Skipped too short: {len(skipped_too_short)}")
    print(f"[INFO] Skipped empty: {len(skipped_empty)}")
    print(
        "[INFO] Output shapes: "
        f"states={tuple(states_out.shape)}, "
        f"action_chunks={tuple(action_chunks_out.shape)}, "
        f"next_states={tuple(next_states_out.shape)}, "
        f"returns={tuple(returns_out.shape)}"
    )
    print(f"[INFO] Saved Q-data to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
