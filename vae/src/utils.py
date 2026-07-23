"""
VAE training utilities: data preparation, masking, normalization.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared_utils import get_feature_columns, load_panel_data, set_seed


class MaskedHealthDataset(Dataset):
    """Dataset returning (filled_values, original, mask) tensors."""

    def __init__(self, array: np.ndarray, mask: np.ndarray) -> None:
        self.data = torch.FloatTensor(array)
        self.mask = torch.FloatTensor(mask)
        self.original = self.data.clone()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.data[idx].clone()
        m = self.mask[idx]
        x = torch.where(m.bool(), x, torch.zeros_like(x))
        return x, self.original[idx], m


def prepare_vae_data(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Prepare normalized feature matrix and observation mask.

    Returns (values, mask, scaler, feature_cols).
    """
    feature_cols = feature_cols or get_feature_columns()
    values = df[feature_cols].values.astype(np.float32)
    mask = (~np.isnan(values)).astype(np.float32)

    # Fill NaN with column median for scaling only
    col_medians = np.nanmedian(values, axis=0)
    inds = np.where(np.isnan(values))
    values_filled = values.copy()
    values_filled[inds] = np.take(col_medians, inds[1])

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        scaled = scaler.fit_transform(values_filled)
    else:
        scaled = scaler.transform(values_filled)

    return scaled.astype(np.float32), mask, scaler, feature_cols


def make_dataloader(
    values: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for VAE training."""
    ds = MaskedHealthDataset(values, mask)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_and_split(config: dict) -> dict:
    """Load panel data and split for VAE pipeline."""
    set_seed(config.get("seed", 42))
    df = load_panel_data()
    feature_cols = get_feature_columns()

    from scripts.shared_utils import train_val_test_split_panel

    train_df, val_df, test_df = train_val_test_split_panel(
        df,
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config.get("seed", 42),
    )

    train_x, train_m, scaler, cols = prepare_vae_data(train_df, feature_cols, fit_scaler=True)
    val_x, val_m, _, _ = prepare_vae_data(val_df, cols, scaler=scaler, fit_scaler=False)
    test_x, test_m, _, _ = prepare_vae_data(test_df, cols, scaler=scaler, fit_scaler=False)

    return {
        "train": (train_x, train_m),
        "val": (val_x, val_m),
        "test": (test_x, test_m),
        "scaler": scaler,
        "feature_cols": cols,
        "raw_test_df": test_df,
    }
