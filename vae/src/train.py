"""
VAE training pipeline with early stopping and checkpointing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import PublicHealthVAE
from utils import load_and_split, make_dataloader
from scripts.shared_utils import set_seed, save_metrics


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_epoch(model, loader, optimizer, device, beta: float) -> float:
    model.train()
    total = 0.0
    for x, target, mask in loader:
        x, target, mask = x.to(device), target.to(device), mask.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, _, _ = model.loss_function(recon, target, mask, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device, beta: float) -> float:
    model.eval()
    total = 0.0
    for x, target, mask in loader:
        x, target, mask = x.to(device), target.to(device), mask.to(device)
        recon, mu, logvar = model(x)
        loss, _, _ = model.loss_function(recon, target, mask, mu, logvar, beta=beta)
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def train(config_path: str | Path | None = None) -> dict:
    """Run full VAE training and save checkpoint."""
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split(config)

    train_loader = make_dataloader(
        *data["train"], batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = make_dataloader(
        *data["val"], batch_size=config["training"]["batch_size"], shuffle=False
    )

    input_dim = len(data["feature_cols"])
    model = PublicHealthVAE(
        input_dim=input_dim,
        hidden_dims=config["model"]["hidden_dims"],
        latent_dim=config["model"]["latent_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-5),
    )
    early_stop = EarlyStopping(patience=config["training"]["patience"])

    save_dir = MODEL_DIR / config["paths"]["model_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "vae_best.pt"

    history = {"train_loss": [], "val_loss": []}
    beta = config["training"].get("beta", 1.0)

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, device, beta)
        val_loss = eval_epoch(model, val_loader, device, beta)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = early_stop.step(val_loss)
        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "feature_cols": data["feature_cols"],
                    "scaler_mean": data["scaler"].mean_.tolist(),
                    "scaler_scale": data["scaler"].scale_.tolist(),
                },
                best_path,
            )
        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    metrics = {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": early_stop.best_loss,
        "epochs_run": len(history["train_loss"]),
    }
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "train_metrics.json")
    print(f"VAE training complete. Best val loss: {early_stop.best_loss:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE for health data imputation")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
