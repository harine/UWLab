from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import QFunction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Q-function from processed trajectory data.")
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


def _prepare_tensors(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for key in ("states", "action_chunks", "returns"):
        if key not in data:
            raise KeyError(f"Processed data is missing key: {key}")

    states = data["states"]
    action_chunks = data["action_chunks"]
    returns = data["returns"]
    if not isinstance(states, torch.Tensor):
        states = torch.as_tensor(states)
    if not isinstance(action_chunks, torch.Tensor):
        action_chunks = torch.as_tensor(action_chunks)
    if not isinstance(returns, torch.Tensor):
        returns = torch.as_tensor(returns)

    states = states.reshape(states.shape[0], -1).to(dtype=torch.float32)
    action_chunks = action_chunks.reshape(action_chunks.shape[0], -1).to(dtype=torch.float32)
    returns = returns.reshape(-1, 1).to(dtype=torch.float32)

    if not (states.shape[0] == action_chunks.shape[0] == returns.shape[0]):
        raise ValueError(
            "states/action_chunks/returns sample count mismatch: "
            f"{states.shape[0]}, {action_chunks.shape[0]}, {returns.shape[0]}"
        )
    if states.shape[0] == 0:
        raise ValueError("Processed data contains zero samples.")
    return states, action_chunks, returns


def _evaluate(model: QFunction, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for states, action_chunks, target_q in data_loader:
            states = states.to(device)
            action_chunks = action_chunks.to(device)
            target_q = target_q.to(device)
            pred_q = model(states, action_chunks)
            loss = criterion(pred_q, target_q)
            batch_size = states.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def main() -> None:
    args = _parse_args()
    cfg = _load_config(Path(args.config))

    data_path = Path(str(_required(cfg, "data_path")))
    output_dir = Path(str(_required(cfg, "output_dir")))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(cfg.get("run_name", "q_function"))
    seed = int(cfg.get("seed", 0))
    train_split = float(cfg.get("train_split", 0.9))
    batch_size = int(cfg.get("batch_size", 256))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    epochs = int(cfg.get("epochs", 100))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 50))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
    device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    if not (0.0 < train_split < 1.0):
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if batch_size <= 0 or epochs <= 0:
        raise ValueError(f"batch_size and epochs must be > 0, got {batch_size}, {epochs}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in processed data file: {data_path}")
    states, action_chunks, returns = _prepare_tensors(data)

    dataset = TensorDataset(states, action_chunks, returns)
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

    model = QFunction(
        state_dim=states.shape[1],
        action_chunk_dim=action_chunks.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_model_path = output_dir / f"{run_name}_best.pt"
    last_model_path = output_dir / f"{run_name}_last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_states, batch_action_chunks, batch_returns in train_loader:
            batch_states = batch_states.to(device)
            batch_action_chunks = batch_action_chunks.to(device)
            batch_returns = batch_returns.to(device)

            pred_q = model(batch_states, batch_action_chunks)
            loss = criterion(pred_q, batch_returns)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            batch_size_actual = batch_states.shape[0]
            total_loss += float(loss.item()) * batch_size_actual
            total_samples += batch_size_actual

        train_loss = total_loss / total_samples
        val_loss = _evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": states.shape[1],
                    "action_chunk_dim": action_chunks.shape[1],
                    "hidden_dim": hidden_dim,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "config": cfg,
                },
                best_model_path,
            )

        if epoch % save_every == 0:
            ckpt_path = output_dir / f"{run_name}_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": states.shape[1],
                    "action_chunk_dim": action_chunks.shape[1],
                    "hidden_dim": hidden_dim,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                ckpt_path,
            )

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[INFO] Epoch {epoch:04d}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": states.shape[1],
            "action_chunk_dim": action_chunks.shape[1],
            "hidden_dim": hidden_dim,
            "epoch": epochs,
            "best_val_loss": best_val,
            "config": cfg,
        },
        last_model_path,
    )

    metrics_path = output_dir / f"{run_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "best_val_loss": best_val,
                "epochs": epochs,
                "num_samples": n_total,
                "num_train": n_train,
                "num_val": n_val,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Training complete. Best val loss: {best_val:.6f}")
    print(f"[INFO] Saved best model: {best_model_path}")
    print(f"[INFO] Saved last model: {last_model_path}")
    print(f"[INFO] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()