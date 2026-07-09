"""cGAN data utilities: condition vectors from country/year metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared_utils import get_feature_columns, load_panel_data, set_seed, train_val_test_split_panel


class CGANDataset(Dataset):
    """Real feature vectors with condition (year_norm, gdp, urbanization)."""

    def __init__(self, features: np.ndarray, conditions: np.ndarray) -> None:
        self.features = torch.FloatTensor(features)
        self.conditions = torch.FloatTensor(conditions)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.conditions[idx]


def build_cgan_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler_x: StandardScaler | None = None,
    scaler_c: StandardScaler | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Build feature and condition arrays; drop rows with too many missing features."""
    sub = df.copy()
    cond_cols = ["year", "gdp_per_capita_usd", "urbanization_pct"]
    sub = sub.dropna(subset=cond_cols, how="any")

    feats = sub[feature_cols].values.astype(np.float32)
    valid_rows = (~np.isnan(feats)).sum(axis=1) >= len(feature_cols) * 0.5
    sub = sub.iloc[np.where(valid_rows)[0]]
    feats = sub[feature_cols].values.astype(np.float32)
    cond = sub[cond_cols].values.astype(np.float32)

    col_med = np.nanmedian(feats, axis=0)
    inds = np.where(np.isnan(feats))
    feats_copy = feats.copy()
    feats_copy[inds] = np.take(col_med, inds[1])

    cond[:, 0] = (cond[:, 0] - 2000) / 23.0  # normalize year

    if fit:
        scaler_x = scaler_x or StandardScaler()
        scaler_c = scaler_c or StandardScaler()
        scaler_x.fit(feats_copy)
        scaler_c.fit(cond)
    assert scaler_x is not None and scaler_c is not None

    return (
        scaler_x.transform(feats_copy).astype(np.float32),
        scaler_c.transform(cond).astype(np.float32),
        scaler_x,
        scaler_c,
    )


def load_and_split(config: dict) -> dict:
    set_seed(config.get("seed", 42))
    df = load_panel_data()
    feature_cols = get_feature_columns()
    train_df, val_df, test_df = train_val_test_split_panel(
        df, val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"], seed=config.get("seed", 42),
    )
    tr_x, tr_c, sx, sc = build_cgan_arrays(train_df, feature_cols, fit=True)
    va_x, va_c, _, _ = build_cgan_arrays(val_df, feature_cols, sx, sc)
    te_x, te_c, _, _ = build_cgan_arrays(test_df, feature_cols, sx, sc)
    return {
        "train": CGANDataset(tr_x, tr_c),
        "val": CGANDataset(va_x, va_c),
        "test": CGANDataset(te_x, te_c),
        "feature_dim": len(feature_cols),
        "cond_dim": 3,
        "scaler_x": sx,
        "scaler_c": sc,
        "feature_cols": feature_cols,
    }
