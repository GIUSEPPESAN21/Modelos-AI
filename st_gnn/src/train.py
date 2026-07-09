"""ST-GNN training pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import STGNN
from utils import load_graph_data
from scripts.shared_utils import set_seed, save_metrics


def train(config_path: str | Path | None = None) -> dict:
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph_data(config)

    model = STGNN(
        in_features=data["in_features"],
        hidden_dim=config["model"]["hidden_dim"],
        gcn_layers=config["model"]["gcn_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    x_seq = data["x_seq"].to(device)
    edge_index = data["edge_index"].to(device)
    targets = data["targets"].to(device)
    train_mask = data["train_mask"].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()

    save_dir = MODEL_DIR / config["paths"]["model_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "st_gnn_best.pt"
    best_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        pred = model(x_seq, edge_index)
        loss = criterion(pred[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "countries": data["countries"],
            }, best_path)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    metrics = {"best_train_loss": best_loss, "graph_stats": data["graph_stats"]}
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "train_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
