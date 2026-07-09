"""Evaluate ST-GNN on held-out countries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import STGNN
from utils import load_graph_data
from scripts.shared_utils import save_metrics


def evaluate(config_path: str | Path | None = None) -> dict:
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph_data(config)

    ckpt_path = MODEL_DIR / config["paths"]["model_dir"] / "st_gnn_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = STGNN(
        in_features=data["in_features"],
        hidden_dim=config["model"]["hidden_dim"],
        gcn_layers=config["model"]["gcn_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred = model(data["x_seq"].to(device), data["edge_index"].to(device))
    test_mask = data["test_mask"]
    y_true = data["targets"][test_mask].numpy()
    y_pred = pred[test_mask].cpu().numpy()

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test_nodes": int(test_mask.sum()),
    }
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "eval_metrics.json")
    print(f"ST-GNN Eval — RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
