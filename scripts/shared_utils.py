"""
Shared utilities for all model pipelines: data loading, seeding, paths.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_panel_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load main panel dataset from Parquet."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / "la_public_health.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python scripts/generate_data.py"
        )
    return pd.read_parquet(path)


def load_adjacency(data_dir: Path | None = None) -> pd.DataFrame:
    """Load country adjacency matrix."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / "country_adjacency.parquet"
    return pd.read_parquet(path)


def load_metadata(data_dir: Path | None = None) -> dict[str, Any]:
    """Load dataset metadata JSON."""
    data_dir = data_dir or DATA_DIR
    with open(data_dir / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_columns() -> list[str]:
    """Return standard feature column names from metadata or defaults."""
    meta = load_metadata()
    return meta["features"]


def train_val_test_split_panel(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split panel data by country to avoid leakage across time for same country.

    Returns train, validation, and test DataFrames.
    """
    rng = np.random.default_rng(seed)
    countries = df["country"].unique()
    rng.shuffle(countries)
    n = len(countries)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test_c = set(countries[:n_test])
    val_c = set(countries[n_test : n_test + n_val])
    train_c = set(countries[n_test + n_val :])

    train = df[df["country"].isin(train_c)].copy()
    val = df[df["country"].isin(val_c)].copy()
    test = df[df["country"].isin(test_c)].copy()
    return train, val, test


def impute_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Simple median imputation for baseline comparisons."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].fillna(out[col].median())
    return out


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Save evaluation metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
