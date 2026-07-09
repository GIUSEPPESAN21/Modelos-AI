"""cGAN training with label smoothing and gradient penalty (optional)."""

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

from model import ConditionalGAN
from utils import load_and_split
from scripts.shared_utils import set_seed, save_metrics


def train(config_path: str | Path | None = None) -> dict:
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split(config)

    loader = DataLoader(data["train"], batch_size=config["training"]["batch_size"], shuffle=True)
    cgan = ConditionalGAN(
        feature_dim=data["feature_dim"],
        cond_dim=data["cond_dim"],
        noise_dim=config["model"]["noise_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    ).to(device)

    opt_g = torch.optim.Adam(
        cgan.generator.parameters(), lr=config["training"]["lr_g"], betas=(0.5, 0.999)
    )
    opt_d = torch.optim.Adam(
        cgan.discriminator.parameters(), lr=config["training"]["lr_d"], betas=(0.5, 0.999)
    )
    bce = nn.BCEWithLogitsLoss()
    save_dir = MODEL_DIR / config["paths"]["model_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    d_losses, g_losses = [], []
    for epoch in range(config["training"]["epochs"]):
        d_epoch, g_epoch = 0.0, 0.0
        for real_x, cond in loader:
            real_x, cond = real_x.to(device), cond.to(device)
            bs = real_x.size(0)
            real_lab = torch.ones(bs, 1, device=device) * 0.9
            fake_lab = torch.zeros(bs, 1, device=device) + 0.1

            # Train discriminator
            opt_d.zero_grad()
            z = torch.randn(bs, config["model"]["noise_dim"], device=device)
            fake_x = cgan.generator(z, cond)
            d_real = cgan.discriminator(real_x, cond)
            d_fake = cgan.discriminator(fake_x.detach(), cond)
            loss_d = bce(d_real, real_lab) + bce(d_fake, fake_lab)
            loss_d.backward()
            opt_d.step()

            # Train generator
            opt_g.zero_grad()
            z = torch.randn(bs, config["model"]["noise_dim"], device=device)
            fake_x = cgan.generator(z, cond)
            d_fake = cgan.discriminator(fake_x, cond)
            loss_g = bce(d_fake, real_lab)
            loss_g.backward()
            opt_g.step()

            d_epoch += loss_d.item()
            g_epoch += loss_g.item()

        d_losses.append(d_epoch / len(loader))
        g_losses.append(g_epoch / len(loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: D={d_losses[-1]:.4f}, G={g_losses[-1]:.4f}")

    torch.save({
        "generator": cgan.generator.state_dict(),
        "discriminator": cgan.discriminator.state_dict(),
        "config": config,
        "feature_cols": data["feature_cols"],
        "scaler_x_mean": data["scaler_x"].mean_.tolist(),
        "scaler_x_scale": data["scaler_x"].scale_.tolist(),
        "scaler_c_mean": data["scaler_c"].mean_.tolist(),
        "scaler_c_scale": data["scaler_c"].scale_.tolist(),
    }, save_dir / "cgan_best.pt")

    metrics = {"final_d_loss": d_losses[-1], "final_g_loss": g_losses[-1]}
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "train_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
