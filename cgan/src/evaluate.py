"""Evaluate cGAN: MMD proxy and counterfactual generation quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import ConditionalGAN
from utils import load_and_split
from scripts.shared_utils import save_metrics


def evaluate(config_path: str | Path | None = None) -> dict:
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split(config)
    loader = DataLoader(data["test"], batch_size=64, shuffle=False)

    ckpt = torch.load(
        MODEL_DIR / config["paths"]["model_dir"] / "cgan_best.pt",
        map_location=device, weights_only=False,
    )
    cgan = ConditionalGAN(
        feature_dim=data["feature_dim"],
        cond_dim=data["cond_dim"],
        noise_dim=config["model"]["noise_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)
    cgan.generator.load_state_dict(ckpt["generator"])
    cgan.eval()

    real_all, fake_all = [], []
    with torch.no_grad():
        for real_x, cond in loader:
            real_x, cond = real_x.to(device), cond.to(device)
            fake_x = cgan.generate(cond)
            real_all.append(real_x.cpu().numpy())
            fake_all.append(fake_x.cpu().numpy())

    real = np.vstack(real_all)
    fake = np.vstack(fake_all)
    # Mean/std matching as simple quality proxy
    mean_diff = float(np.mean(np.abs(real.mean(0) - fake.mean(0))))
    std_diff = float(np.mean(np.abs(real.std(0) - fake.std(0))))

    metrics = {
        "mean_feature_diff": mean_diff,
        "std_feature_diff": std_diff,
        "n_generated": len(fake),
    }
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "eval_metrics.json")
    print(f"cGAN Eval — mean diff: {mean_diff:.4f}, std diff: {std_diff:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
