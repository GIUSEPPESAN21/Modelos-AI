"""Attention LSTM training pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import AttentionLSTM
from utils import collate_fn, load_and_split
from scripts.shared_utils import set_seed, save_metrics


def train(config_path: str | Path | None = None) -> dict:
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split(config)

    train_loader = DataLoader(
        data["train"], batch_size=config["training"]["batch_size"],
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        data["val"], batch_size=config["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn,
    )

    model = AttentionLSTM(
        input_dim=data["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()
    best_val = float("inf")
    save_dir = MODEL_DIR / config["paths"]["model_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "attention_lstm_best.pt"

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            pred, _ = model(x, lengths)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(data["train"])

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                pred, _ = model(x, lengths)
                val_loss += criterion(pred.squeeze(), y).item() * x.size(0)
        val_loss /= len(data["val"])

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": config}, best_path)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Save last checkpoint if none saved (e.g. empty val set)
    if not best_path.exists() and len(data["train"]) > 0:
        torch.save({"model_state": model.state_dict(), "config": config}, best_path)

    metrics = {"best_val_mse": best_val, "epochs": config["training"]["epochs"]}
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "train_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
