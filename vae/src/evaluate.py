"""
VAE evaluation: imputation RMSE/MAE on artificially masked test entries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import PublicHealthVAE
from utils import load_and_split, make_dataloader
from scripts.shared_utils import save_metrics


@torch.no_grad()
def evaluate_imputation(model, loader, device) -> dict:
    """Evaluate reconstruction on observed entries (forward pass, not impute)."""
    model.eval()
    all_true, all_pred, all_mask = [], [], []
    for x, target, mask in loader:
        x, target, mask = x.to(device), target.to(device), mask.to(device)
        recon, _, _ = model(x)
        all_true.append(target.cpu().numpy())
        all_pred.append(recon.cpu().numpy())
        all_mask.append(mask.cpu().numpy())

    true = np.vstack(all_true)
    pred = np.vstack(all_pred)
    mask = np.vstack(all_mask)

    # Evaluate only on observed entries
    obs = mask.astype(bool)
    y_true = true[obs]
    y_pred = pred[obs]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "n_evaluated": int(obs.sum())}


def evaluate(config_path: str | Path | None = None) -> dict:
    """Load trained VAE and evaluate on test set."""
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split(config)
    test_loader = make_dataloader(*data["test"], batch_size=64, shuffle=False)

    ckpt_path = MODEL_DIR / config["paths"]["model_dir"] / "vae_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PublicHealthVAE(
        input_dim=len(data["feature_cols"]),
        hidden_dims=config["model"]["hidden_dims"],
        latent_dim=config["model"]["latent_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate_imputation(model, test_loader, device)
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "eval_metrics.json")
    print(f"VAE Eval — RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VAE")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
