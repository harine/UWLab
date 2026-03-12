from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
Q_FUNCTION_DIR = SCRIPT_DIR.parent / "q_function"
MODELS_PATH = Q_FUNCTION_DIR / "models.py"
MODELS_SPEC = importlib.util.spec_from_file_location("q_function_models", MODELS_PATH)
if MODELS_SPEC is None or MODELS_SPEC.loader is None:
    raise ImportError(f"Unable to load models module from {MODELS_PATH}")
MODELS_MODULE = importlib.util.module_from_spec(MODELS_SPEC)
MODELS_SPEC.loader.exec_module(MODELS_MODULE)
GaussianPolicy = MODELS_MODULE.GaussianPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gaussian policy from processed action-chunk data.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
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


def _resolve_data_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        if data_path.suffix != ".pt":
            raise ValueError(f"Expected a .pt file, got: {data_path}")
        return [data_path]

    if data_path.is_dir():
        processed_files = sorted(data_path.glob("q_data*.pt"))
        if processed_files:
            return processed_files

        pt_files = sorted(data_path.glob("*.pt"))
        if pt_files:
            return pt_files

        raise FileNotFoundError(f"No .pt processed-data files found under {data_path}")

    raise FileNotFoundError(f"Data path does not exist: {data_path}")


def _ensure_2d_samples(tensor: torch.Tensor, name: str, path: Path) -> torch.Tensor:
    if tensor.ndim == 0:
        raise ValueError(f"{name} in {path} must have a sample dimension, got scalar.")
    if tensor.ndim == 1:
        return tensor.view(-1, 1).to(dtype=torch.float32)
    return tensor.reshape(tensor.shape[0], -1).to(dtype=torch.float32)


def _load_data_file(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in data file {path}, got {type(data)}")

    if "states" in data and "action_chunks" in data:
        states_raw = data["states"]
        action_chunks_raw = data["action_chunks"]
    else:
        raise KeyError(
            f"Data file {path} must contain ('states', 'action_chunks'), as produced by process_data_q.py."
        )

    if not isinstance(states_raw, torch.Tensor):
        states_raw = torch.as_tensor(states_raw)
    if not isinstance(action_chunks_raw, torch.Tensor):
        action_chunks_raw = torch.as_tensor(action_chunks_raw)

    states = _ensure_2d_samples(states_raw, "states", path)
    action_chunks = _ensure_2d_samples(action_chunks_raw, "action_chunks", path)

    if states.shape[0] != action_chunks.shape[0]:
        raise ValueError(
            f"Sample count mismatch in {path}: states={states.shape[0]}, action_chunks={action_chunks.shape[0]}"
        )
    if states.shape[0] == 0:
        raise ValueError(f"Data file {path} contains zero samples.")

    return states, action_chunks


def _prepare_dataset(data_path: Path, max_files: int = 0) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    data_files = _resolve_data_files(data_path)
    if max_files > 0:
        data_files = data_files[:max_files]
    if not data_files:
        raise FileNotFoundError("No data files selected after applying max_files.")

    states_all: list[torch.Tensor] = []
    action_chunks_all: list[torch.Tensor] = []
    used_files: list[str] = []

    for file_path in data_files:
        states, action_chunks = _load_data_file(file_path)
        states_all.append(states)
        action_chunks_all.append(action_chunks)
        used_files.append(str(file_path))

    states_out = torch.cat(states_all, dim=0)
    action_chunks_out = torch.cat(action_chunks_all, dim=0)
    return states_out, action_chunks_out, used_files


def _mean_nll_loss(model: GaussianPolicy, states: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
    dist = model.distribution(states)
    log_prob = dist.log_prob(action_chunks).sum(dim=-1)
    return -log_prob.mean()


def _mean_action_chunk_mse(model: GaussianPolicy, states: torch.Tensor, action_chunks: torch.Tensor) -> float:
    with torch.inference_mode():
        mean, _ = model(states)
        return float(torch.mean((mean - action_chunks) ** 2).item())


def _evaluate(model: GaussianPolicy, data_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.inference_mode():
        for states, action_chunks in data_loader:
            states = states.to(device)
            action_chunks = action_chunks.to(device)

            batch_nll = _mean_nll_loss(model, states, action_chunks)
            batch_mse = _mean_action_chunk_mse(model, states, action_chunks)
            batch_size = states.shape[0]

            total_nll += float(batch_nll.item()) * batch_size
            total_mse += batch_mse * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_nll / total_samples, total_mse / total_samples


def main() -> None:
    args = _parse_args()
    cfg = _load_config(Path(args.config))

    data_path = Path(str(_required(cfg, "data_path")))
    output_dir = Path(str(_required(cfg, "output_dir")))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(cfg.get("run_name", "pi_base"))
    seed = int(cfg.get("seed", 0))
    train_split = float(cfg.get("train_split", 0.9))
    batch_size = int(cfg.get("batch_size", 256))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    hidden_dims = cfg.get("hidden_dims")
    epochs = int(cfg.get("epochs", 100))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 50))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
    max_files = int(cfg.get("max_files", 0))
    device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    if not (0.0 < train_split < 1.0):
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if batch_size <= 0 or epochs <= 0:
        raise ValueError(f"batch_size and epochs must be > 0, got {batch_size}, {epochs}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    states, action_chunks, used_files = _prepare_dataset(data_path=data_path, max_files=max_files)
    dataset = TensorDataset(states, action_chunks)
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

    model = GaussianPolicy(
        state_dim=states.shape[1],
        action_dim=action_chunks.shape[1],
        hidden_dims=hidden_dims,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: dict[str, list[float]] = {
        "train_nll": [],
        "val_nll": [],
        "train_action_chunk_mse": [],
        "val_action_chunk_mse": [],
    }
    best_val_nll = float("inf")
    best_model_path = output_dir / f"{run_name}_best.pt"
    last_model_path = output_dir / f"{run_name}_last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_nll = 0.0
        total_mse = 0.0
        total_samples = 0

        for batch_states, batch_action_chunks in train_loader:
            batch_states = batch_states.to(device)
            batch_action_chunks = batch_action_chunks.to(device)

            loss = _mean_nll_loss(model, batch_states, batch_action_chunks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            batch_size_actual = batch_states.shape[0]
            total_nll += float(loss.item()) * batch_size_actual
            total_mse += _mean_action_chunk_mse(model, batch_states, batch_action_chunks) * batch_size_actual
            total_samples += batch_size_actual

        train_nll = total_nll / total_samples
        train_mse = total_mse / total_samples
        val_nll, val_mse = _evaluate(model, val_loader, device)

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["train_action_chunk_mse"].append(train_mse)
        history["val_action_chunk_mse"].append(val_mse)

        checkpoint_common = {
            "model_state_dict": model.state_dict(),
            "state_dim": states.shape[1],
            "action_chunk_dim": action_chunks.shape[1],
            "hidden_dims": list(model.hidden_dims),
            "epoch": epoch,
            "config": cfg,
        }

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save(
                {
                    **checkpoint_common,
                    "best_val_nll": best_val_nll,
                    "val_action_chunk_mse": val_mse,
                },
                best_model_path,
            )

        if epoch % save_every == 0:
            ckpt_path = output_dir / f"{run_name}_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    **checkpoint_common,
                    "train_nll": train_nll,
                    "val_nll": val_nll,
                    "train_action_chunk_mse": train_mse,
                    "val_action_chunk_mse": val_mse,
                },
                ckpt_path,
            )

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[INFO] Epoch {epoch:04d}/{epochs} "
                f"train_nll={train_nll:.6f} val_nll={val_nll:.6f} "
                f"train_action_chunk_mse={train_mse:.6f} val_action_chunk_mse={val_mse:.6f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": states.shape[1],
            "action_chunk_dim": action_chunks.shape[1],
            "hidden_dims": list(model.hidden_dims),
            "epoch": epochs,
            "best_val_nll": best_val_nll,
            "config": cfg,
        },
        last_model_path,
    )

    metrics_path = output_dir / f"{run_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_nll": history["train_nll"],
                "val_nll": history["val_nll"],
                "train_action_chunk_mse": history["train_action_chunk_mse"],
                "val_action_chunk_mse": history["val_action_chunk_mse"],
                "best_val_nll": best_val_nll,
                "epochs": epochs,
                "num_samples": n_total,
                "num_train": n_train,
                "num_val": n_val,
                "num_source_files": len(used_files),
                "source_files": used_files,
            },
            f,
            indent=2,
        )

    print(f"[INFO] Training complete. Best val NLL: {best_val_nll:.6f}")
    print(f"[INFO] Saved best model: {best_model_path}")
    print(f"[INFO] Saved last model: {last_model_path}")
    print(f"[INFO] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
