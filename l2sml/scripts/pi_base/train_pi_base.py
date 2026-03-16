import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from models import DeterministicPolicy
import torch.nn as nn
import wandb
import argparse
import yaml
from pathlib import Path
from typing import Any

def load_data(data_path: str):
    data = torch.load(data_path)
    states = data["states"]
    action_chunks = data["action_chunks"]
    states = states.reshape(states.shape[0], -1).to(dtype=torch.float32)
    action_chunks = action_chunks.reshape(action_chunks.shape[0], -1).to(dtype=torch.float32)
    return states, action_chunks

def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")
    return cfg

def evaluate(model: DeterministicPolicy, loader: DataLoader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in loader:
            states, action_chunks = batch
            states = states.to(device)
            action_chunks = action_chunks.to(device)
            outputs = model(states)
            loss = criterion(outputs, action_chunks)
            total_loss += loss.item() * action_chunks.shape[0]
            total_samples += action_chunks.shape[0]
    return total_loss / total_samples

def train_pi_base(model, states: torch.Tensor, action_chunks: torch.Tensor, train_split: float, seed: int, epochs: int, device: str):
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
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            states, action_chunks = batch
            states = states.to(device)
            action_chunks = action_chunks.to(device)
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, action_chunks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * action_chunks.shape[0]
            total_samples += action_chunks.shape[0]
        val_loss = evaluate(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / f"best_model.pt")
        print(f"Epoch {epoch}, Train Loss: {train_loss/total_samples}, Val Loss: {val_loss}")
        wandb.log({
            "train_loss": train_loss/total_samples,
            "val_loss": val_loss,
        })
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    device = torch.device(config["device"])
    states, action_chunks = load_data(config["data_path"])
    states = states.to(dtype=torch.float32).to(device)
    action_chunks = action_chunks.to(dtype=torch.float32).to(device)
    action_mean = action_chunks.mean(dim=0)
    action_std = action_chunks.std(dim=0)
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0)
    states = (states - state_mean) / state_std
    action_chunks = (action_chunks - action_mean) / action_std
    
    model = DeterministicPolicy(states.shape[1], action_chunks.shape[1], config["hidden_dims"]).to(device)
    
    wandb.init(
        project="pi_base",
        name=config["run_name"],
        config=config,
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = train_pi_base(model, states, action_chunks, config["train_split"], config["seed"], config["epochs"], str(device))

    
    torch.save({
        "model_state_dict": model.state_dict(),
        "action_mean": action_mean,
        "action_std": action_std,
        "state_mean": state_mean,
        "state_std": state_std,
    }, output_dir / f"final_model.pt")
