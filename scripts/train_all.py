"""Train and evaluate all 5 models sequentially (demo mode)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ["vae", "attention_lstm", "st_gnn", "dml", "cgan"]


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    # Generate data first
    print("=== Generating synthetic data ===")
    from scripts.generate_data import generate_dataset
    generate_dataset(ROOT / "data")

    for model in MODELS:
        print(f"\n=== Training {model.upper()} ===")
        rc = run([sys.executable, f"{model}/src/train.py"])
        if rc != 0:
            print(f"WARNING: {model} training returned code {rc}")
        print(f"=== Evaluating {model.upper()} ===")
        run([sys.executable, f"{model}/src/evaluate.py"])

    print("\n=== All models complete ===")


if __name__ == "__main__":
    main()
