"""Evaluate Attention LSTM on test countries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import AttentionLSTM
from utils import collate_fn, load_and_split
from scripts.shared_utils import save_metrics


def evaluate(config_path: str | Path | None = None) -> dict:
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split(config)
    loader = DataLoader(data["test"], batch_size=16, collate_fn=collate_fn)

    ckpt_path = MODEL_DIR / config["paths"]["model_dir"] / "attention_lstm_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AttentionLSTM(
        input_dim=data["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, lengths = x.to(device), lengths.to(device)
            pred, _ = model(x, lengths)
            preds.extend(pred.squeeze().cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds)),
    }
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "eval_metrics.json")
    print(f"LSTM Eval — RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
