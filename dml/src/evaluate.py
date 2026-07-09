"""Evaluate DML: placebo test and overlap diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR / "src"))

from model import LinearDML
from utils import load_and_split
from scripts.shared_utils import save_metrics


def evaluate(config_path: str | Path | None = None) -> dict:
    config_path = config_path or MODEL_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_and_split(config)
    Y_te, D_te, X_te = data["test"]

    result_path = MODEL_DIR / config["paths"]["model_dir"] / "dml_result.joblib"
    train_result = joblib.load(result_path)

    # Placebo: shuffle treatment on test set and re-estimate
    rng = np.random.default_rng(config.get("seed", 42))
    D_placebo = D_te.copy()
    rng.shuffle(D_placebo)
    placebo = LinearDML(n_folds=3, random_state=config.get("seed", 42)).fit(Y_te, D_placebo, X_te)

    metrics = {
        "train_ate": train_result["ate"],
        "train_ci": [train_result["ci_lower"], train_result["ci_upper"]],
        "placebo_ate": placebo.ate,
        "placebo_ci": [placebo.ci_lower, placebo.ci_upper],
        "treatment_rate_test": float(D_te.mean()),
        "method": train_result["method"],
    }
    save_metrics(metrics, MODEL_DIR / config["paths"]["metrics_dir"] / "eval_metrics.json")
    print(f"Placebo ATE: {placebo.ate:.4f} (should be ~0)")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
