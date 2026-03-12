from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from models import QFunction, ValueFunction

try:
    import wandb
except ImportError:
    wandb = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a value function, then a Q-function from processed trajectory data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="l2sml/configs/train_q.yaml",
        help="Path to YAML config containing training hyperparameters.",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")
    return cfg


def _required(cfg: dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]


def _parse_hidden_dims(value: Any, *, key_name: str) -> list[int]:
    if isinstance(value, int):
        dims = [value]
    elif isinstance(value, (list, tuple)):
        dims = [int(dim) for dim in value]
    else:
        raise ValueError(f"{key_name} must be an int or a list of ints, got {type(value)}")

    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"{key_name} must contain positive hidden dimensions, got {dims}")
    return dims


def _load_value_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[ValueFunction, dict[str, Any]]:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Value checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected dict checkpoint at {checkpoint_path}, got {type(checkpoint)}")

    state_dim = checkpoint.get("state_dim")
    hidden_dims = checkpoint.get("hidden_dims")
    model_state_dict = checkpoint.get("model_state_dict")
    if state_dim is None or hidden_dims is None or model_state_dict is None:
        raise KeyError(
            f"Value checkpoint at {checkpoint_path} must contain 'state_dim', 'hidden_dims', and 'model_state_dict'."
        )

    model = ValueFunction(
        state_dim=int(state_dim),
        hidden_dims=_parse_hidden_dims(hidden_dims, key_name="checkpoint.hidden_dims"),
    ).to(device)
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    return model, checkpoint


def _maybe_init_wandb(cfg: dict[str, Any]) -> Any:
    use_wandb = bool(cfg.get("use_wandb", False))
    if not use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed, but use_wandb=true was requested.")

    wandb_run = wandb.init(
        project=str(cfg.get("wandb_project", "uwlab-q-function")),
        entity=cfg.get("wandb_entity"),
        group=cfg.get("wandb_group"),
        name=cfg.get("wandb_name"),
        tags=cfg.get("wandb_tags"),
        config=cfg,
    )
    return wandb_run


def _prepare_tensors(
    data: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for key in ("states", "action_chunks", "next_states", "returns"):
        if key not in data:
            raise KeyError(f"Processed data is missing key: {key}")

    states = data["states"]
    action_chunks = data["action_chunks"]
    next_states = data["next_states"]
    returns = data["returns"]

    if not isinstance(states, torch.Tensor):
        states = torch.as_tensor(states)
    if not isinstance(action_chunks, torch.Tensor):
        action_chunks = torch.as_tensor(action_chunks)
    if not isinstance(next_states, torch.Tensor):
        next_states = torch.as_tensor(next_states)
    if not isinstance(returns, torch.Tensor):
        returns = torch.as_tensor(returns)

    states = states.reshape(states.shape[0], -1).to(dtype=torch.float32)
    action_chunks = action_chunks.reshape(action_chunks.shape[0], -1).to(dtype=torch.float32)
    next_states = next_states.reshape(next_states.shape[0], -1).to(dtype=torch.float32)
    returns = returns.reshape(-1, 1).to(dtype=torch.float32)

    if not (states.shape[0] == action_chunks.shape[0] == next_states.shape[0] == returns.shape[0]):
        raise ValueError(
            "states/action_chunks/next_states/returns sample count mismatch: "
            f"{states.shape[0]}, {action_chunks.shape[0]}, {next_states.shape[0]}, {returns.shape[0]}"
        )
    if states.shape[0] == 0:
        raise ValueError("Processed data contains zero samples.")
    if states.shape[1] != next_states.shape[1]:
        raise ValueError(
            f"state and next_state dimensions must match, got {states.shape[1]} and {next_states.shape[1]}"
        )

    return states, action_chunks, next_states, returns


def _evaluate_value(
    model: ValueFunction,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for states, _action_chunks, _next_states, target_returns in data_loader:
            states = states.to(device)
            target_returns = target_returns.to(device)
            pred_values = model(states)
            loss = criterion(pred_values, target_returns)
            batch_size = states.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def _evaluate_q(
    model: QFunction,
    value_model: ValueFunction,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for states, action_chunks, next_states in data_loader:
            states = states.to(device)
            action_chunks = action_chunks.to(device)
            next_states = next_states.to(device)
            advantages = value_model(next_states) - value_model(states)
            pred_advantages = model(states, action_chunks)
            loss = criterion(pred_advantages, advantages)
            batch_size = states.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def _predict_values(
    model: ValueFunction,
    states: torch.Tensor,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> torch.Tensor:
    pred_loader = DataLoader(
        TensorDataset(states),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    preds: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for (batch_states,) in pred_loader:
            batch_states = batch_states.to(device)
            preds.append(model(batch_states).cpu())
    return torch.cat(preds, dim=0)


def main() -> None:
    args = _parse_args()
    cfg = _load_config(Path(args.config))
    wandb_run = _maybe_init_wandb(cfg)

    data_path = Path(str(_required(cfg, "data_path")))
    output_dir = Path(str(_required(cfg, "output_dir")))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(cfg.get("run_name", "q_function"))
    seed = int(cfg.get("seed", 0))
    train_split = float(cfg.get("train_split", 0.9))
    batch_size = int(cfg.get("batch_size", 256))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    shared_hidden_dims = _parse_hidden_dims(cfg.get("hidden_dims", cfg.get("hidden_dim", [256])), key_name="hidden_dims")
    value_hidden_dims = _parse_hidden_dims(cfg.get("value_hidden_dims", shared_hidden_dims), key_name="value_hidden_dims")
    q_hidden_dims = _parse_hidden_dims(cfg.get("q_hidden_dims", shared_hidden_dims), key_name="q_hidden_dims")
    value_checkpoint_cfg = cfg.get("value_checkpoint")
    epochs = int(cfg.get("epochs", 100))
    value_epochs = int(cfg.get("value_epochs", epochs))
    q_epochs = int(cfg.get("q_epochs", epochs))
    value_learning_rate = float(cfg.get("value_learning_rate", learning_rate))
    q_learning_rate = float(cfg.get("q_learning_rate", learning_rate))
    value_weight_decay = float(cfg.get("value_weight_decay", weight_decay))
    q_weight_decay = float(cfg.get("q_weight_decay", weight_decay))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 50))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
    device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    if not (0.0 < train_split < 1.0):
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if batch_size <= 0 or value_epochs <= 0 or q_epochs <= 0:
        raise ValueError(
            f"batch_size, value_epochs, and q_epochs must be > 0, got {batch_size}, {value_epochs}, {q_epochs}"
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in processed data file: {data_path}")
    states, action_chunks, next_states, returns = _prepare_tensors(data)

    dataset = TensorDataset(states, action_chunks, next_states, returns)
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train
    if n_train == 0 or n_val == 0:
        raise ValueError(
            f"Dataset split invalid for {n_total} samples and train_split={train_split}: "
            f"train={n_train}, val={n_val}"
        )

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    value_model = ValueFunction(
        state_dim=states.shape[1],
        hidden_dims=value_hidden_dims,
    ).to(device)
    value_optimizer = torch.optim.Adam(
        value_model.parameters(),
        lr=value_learning_rate,
        weight_decay=value_weight_decay,
    )
    value_criterion = nn.MSELoss()

    history: dict[str, list[float]] = {
        "value_train_loss": [],
        "value_val_loss": [],
        "q_train_loss": [],
        "q_val_loss": [],
    }
    wandb_step_offset = 0

    value_best_val = float("inf")
    value_best_model_path = output_dir / f"{run_name}_value_best.pt"
    value_last_model_path = output_dir / f"{run_name}_value_last.pt"

    loaded_value_checkpoint_path: str | None = None
    if value_checkpoint_cfg:
        value_checkpoint_path = Path(str(value_checkpoint_cfg))
        value_model, value_checkpoint = _load_value_checkpoint(value_checkpoint_path, device=device)
        if int(value_checkpoint["state_dim"]) != states.shape[1]:
            raise ValueError(
                "Value checkpoint state_dim does not match dataset state_dim: "
                f"{value_checkpoint['state_dim']} != {states.shape[1]}"
            )
        value_hidden_dims = _parse_hidden_dims(value_checkpoint["hidden_dims"], key_name="checkpoint.hidden_dims")
        value_best_val = float(value_checkpoint.get("best_val_loss", float("nan")))
        loaded_value_checkpoint_path = str(value_checkpoint_path)
        print(f"[INFO] Stage 1/2: loaded value checkpoint from {value_checkpoint_path}. Skipping value training.")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "value/loaded_checkpoint": 1,
                    "value/best_val_loss": value_best_val,
                },
                step=0,
            )
        wandb_step_offset = 0
    else:
        print("[INFO] Stage 1/2: training value function on state -> return targets.")
        for epoch in range(1, value_epochs + 1):
            value_model.train()
            total_loss = 0.0
            total_samples = 0

            for batch_states, _batch_action_chunks, _batch_next_states, batch_returns in train_loader:
                batch_states = batch_states.to(device)
                batch_returns = batch_returns.to(device)

                pred_values = value_model(batch_states)
                loss = value_criterion(pred_values, batch_returns)

                value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=grad_clip_norm)
                value_optimizer.step()

                batch_size_actual = batch_states.shape[0]
                total_loss += float(loss.item()) * batch_size_actual
                total_samples += batch_size_actual

            train_loss = total_loss / total_samples
            val_loss = _evaluate_value(value_model, val_loader, value_criterion, device)
            history["value_train_loss"].append(train_loss)
            history["value_val_loss"].append(val_loss)

            if val_loss < value_best_val:
                value_best_val = val_loss
                torch.save(
                    {
                        "model_state_dict": value_model.state_dict(),
                        "state_dim": states.shape[1],
                        "hidden_dims": value_hidden_dims,
                        "best_val_loss": value_best_val,
                        "epoch": epoch,
                        "target_type": "return_to_go",
                        "config": cfg,
                    },
                    value_best_model_path,
                )

            if epoch % save_every == 0:
                ckpt_path = output_dir / f"{run_name}_value_epoch_{epoch:04d}.pt"
                torch.save(
                    {
                        "model_state_dict": value_model.state_dict(),
                        "state_dim": states.shape[1],
                        "hidden_dims": value_hidden_dims,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "target_type": "return_to_go",
                        "config": cfg,
                    },
                    ckpt_path,
                )

            if epoch % log_every == 0 or epoch == 1 or epoch == value_epochs:
                print(
                    f"[VALUE] Epoch {epoch:04d}/{value_epochs} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "value/epoch": epoch,
                        "value/train_loss": train_loss,
                        "value/val_loss": val_loss,
                        "value/best_val_loss": value_best_val,
                    },
                    step=epoch,
                )
        wandb_step_offset = value_epochs

        torch.save(
            {
                "model_state_dict": value_model.state_dict(),
                "state_dim": states.shape[1],
                "hidden_dims": value_hidden_dims,
                "epoch": value_epochs,
                "best_val_loss": value_best_val,
                "target_type": "return_to_go",
                "config": cfg,
            },
            value_last_model_path,
        )

        value_model, _ = _load_value_checkpoint(value_best_model_path, device=device)
        loaded_value_checkpoint_path = str(value_best_model_path)
        print(f"[INFO] Reloaded best value checkpoint for Q training: {value_best_model_path}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "value/reloaded_best_checkpoint": 1,
                    "value/best_val_loss": value_best_val,
                },
                step=value_epochs,
            )

    q_dataset = TensorDataset(states, action_chunks, next_states)
    q_train_dataset = Subset(q_dataset, train_dataset.indices)
    q_val_dataset = Subset(q_dataset, val_dataset.indices)

    q_train_loader = DataLoader(
        q_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    q_val_loader = DataLoader(
        q_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    q_model = QFunction(
        state_dim=states.shape[1],
        action_chunk_dim=action_chunks.shape[1],
        hidden_dims=q_hidden_dims,
    ).to(device)
    q_optimizer = torch.optim.Adam(
        q_model.parameters(),
        lr=q_learning_rate,
        weight_decay=q_weight_decay,
    )
    q_criterion = nn.MSELoss()

    q_best_val = float("inf")
    q_best_model_path = output_dir / f"{run_name}_best.pt"
    q_last_model_path = output_dir / f"{run_name}_last.pt"

    print("[INFO] Stage 2/2: training Q-function on (state, action_chunk) -> V(next_state) - V(state) targets.")
    for epoch in range(1, q_epochs + 1):
        q_model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_states, batch_action_chunks, batch_next_states in q_train_loader:
            batch_states = batch_states.to(device)
            batch_action_chunks = batch_action_chunks.to(device)
            batch_next_states = batch_next_states.to(device)
            batch_target_values = value_model(batch_next_states) - value_model(batch_states)

            pred_values = q_model(batch_states, batch_action_chunks)
            loss = q_criterion(pred_values, batch_target_values)

            q_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(q_model.parameters(), max_norm=grad_clip_norm)
            q_optimizer.step()

            batch_size_actual = batch_states.shape[0]
            total_loss += float(loss.item()) * batch_size_actual
            total_samples += batch_size_actual

        train_loss = total_loss / total_samples
        val_loss = _evaluate_q(q_model, value_model, q_val_loader, q_criterion, device)
        history["q_train_loss"].append(train_loss)
        history["q_val_loss"].append(val_loss)

        if val_loss < q_best_val:
            q_best_val = val_loss
            torch.save(
                {
                    "model_state_dict": q_model.state_dict(),
                    "state_dim": states.shape[1],
                    "action_chunk_dim": action_chunks.shape[1],
                    "hidden_dims": q_hidden_dims,
                    "best_val_loss": q_best_val,
                    "epoch": epoch,
                    "target_type": "advantage_of_action_chunk",
                    "value_model_path": loaded_value_checkpoint_path or str(value_best_model_path),
                    "config": cfg,
                },
                q_best_model_path,
            )

        if epoch % save_every == 0:
            ckpt_path = output_dir / f"{run_name}_q_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "model_state_dict": q_model.state_dict(),
                    "state_dim": states.shape[1],
                    "action_chunk_dim": action_chunks.shape[1],
                    "hidden_dims": q_hidden_dims,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "target_type": "advantage_of_action_chunk",
                    "value_model_path": loaded_value_checkpoint_path or str(value_best_model_path),
                    "config": cfg,
                },
                ckpt_path,
            )

        if epoch % log_every == 0 or epoch == 1 or epoch == q_epochs:
            print(
                f"[Q] Epoch {epoch:04d}/{q_epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )
        if wandb_run is not None:
            q_wandb_step = wandb_step_offset + epoch
            wandb_run.log(
                {
                    "q/epoch": epoch,
                    "q/global_epoch": q_wandb_step,
                    "q/train_loss": train_loss,
                    "q/val_loss": val_loss,
                    "q/best_val_loss": q_best_val,
                },
                step=q_wandb_step,
            )

    torch.save(
        {
            "model_state_dict": q_model.state_dict(),
            "state_dim": states.shape[1],
            "action_chunk_dim": action_chunks.shape[1],
            "hidden_dims": q_hidden_dims,
            "epoch": q_epochs,
            "best_val_loss": q_best_val,
            "target_type": "advantage_of_action_chunk",
            "value_model_path": loaded_value_checkpoint_path or str(value_best_model_path),
            "config": cfg,
        },
        q_last_model_path,
    )

    metrics_path = output_dir / f"{run_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "value_train_loss": history["value_train_loss"],
                "value_val_loss": history["value_val_loss"],
                "q_train_loss": history["q_train_loss"],
                "q_val_loss": history["q_val_loss"],
                "best_value_val_loss": value_best_val,
                "best_q_val_loss": q_best_val,
                "value_epochs": value_epochs,
                "q_epochs": q_epochs,
                "value_checkpoint": loaded_value_checkpoint_path,
                "num_samples": n_total,
                "num_train": n_train,
                "num_val": n_val,
            },
            f,
            indent=2,
        )

    if loaded_value_checkpoint_path is None:
        print(f"[INFO] Value training complete. Best val loss: {value_best_val:.6f}")
        print(f"[INFO] Saved best value model: {value_best_model_path}")
        print(f"[INFO] Saved last value model: {value_last_model_path}")
    else:
        print(f"[INFO] Reused value checkpoint: {loaded_value_checkpoint_path}")
    print(f"[INFO] Q training complete. Best val loss: {q_best_val:.6f}")
    print(f"[INFO] Saved best Q model: {q_best_model_path}")
    print(f"[INFO] Saved last Q model: {q_last_model_path}")
    print(f"[INFO] Saved metrics: {metrics_path}")
    if wandb_run is not None:
        wandb_run.summary["best_value_val_loss"] = value_best_val
        wandb_run.summary["best_q_val_loss"] = q_best_val
        wandb_run.summary["value_checkpoint"] = loaded_value_checkpoint_path
        wandb_run.summary["best_q_model_path"] = str(q_best_model_path)
        wandb_run.summary["metrics_path"] = str(metrics_path)
        wandb_run.finish()


if __name__ == "__main__":
    main()