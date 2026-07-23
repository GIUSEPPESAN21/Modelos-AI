"""DML training / estimation pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import LinearDML, fit_doubleml, fit_econml_dml
from utils import load_and_split
from scripts.shared_utils import set_seed, save_metrics


def train(config_path: str | Path | None = None) -> dict:
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    data = load_and_split(config)
    Y, D, X = data["train"]

    backend = config["model"].get("backend", "auto")
    if backend == "econml":
        result = fit_econml_dml(Y, D, X, seed=config.get("seed", 42))
    elif backend == "doubleml":
        result = fit_doubleml(Y, D, X, seed=config.get("seed", 42))
        if result is None:
            result = LinearDML(n_folds=config["model"]["n_folds"]).fit(Y, D, X)
    else:
        # Try EconML, then DoubleML, then manual
        result = fit_econml_dml(Y, D, X, seed=config.get("seed", 42))
        if result.method == "LinearDML_crossfit":
            dml2 = fit_doubleml(Y, D, X, seed=config.get("seed", 42))
            if dml2 is not None:
                result = dml2

    save_dir = MODEL_DIR / config["paths"]["model_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(asdict(result), save_dir / "dml_result.joblib")

    metrics = asdict(result)
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "train_metrics.json")
    print(f"DML ATE: {result.ate:.4f} (95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}])")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
