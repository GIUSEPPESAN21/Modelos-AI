"""
Attention LSTM data utilities: country time series, dynamic padding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared_utils import get_feature_columns, load_panel_data, set_seed, train_val_test_split_panel


class CountrySequenceDataset(Dataset):
    """One sample per country: padded time series with variable valid length."""

    def __init__(
        self,
        sequences: list[np.ndarray],
        targets: list[float],
        lengths: list[int],
    ) -> None:
        self.sequences = [torch.FloatTensor(s) for s in sequences]
        self.targets = torch.FloatTensor(targets)
        self.lengths = torch.LongTensor(lengths)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx], self.lengths[idx]


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic padding collate for variable-length sequences."""
    seqs, targets, lengths = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return padded, torch.stack(targets), torch.stack(lengths)


def build_country_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[list[np.ndarray], list[float], list[int], StandardScaler]:
    """
    Build per-country sequences sorted by year.
    
    Excludes target_col from feature_cols to prevent data leakage.
    Trims sequences to years 2000-2020 for input; target uses 2021-2023.
    """
    # Exclude target column from features
    feature_cols = [c for c in feature_cols if c != target_col]
    
    sequences, targets, lengths = [], [], []
    grouped = df.groupby("country")

    all_rows = []
    for _, gdf in grouped:
        gdf = gdf.sort_values("year")
        # Use only years 2000-2020 for input features
        gdf_input = gdf[gdf["year"] <= 2020]
        feats = gdf_input[feature_cols].values.astype(np.float32)
        medians = np.nanmedian(feats, axis=0)
        inds = np.where(np.isnan(feats))
        feats_f = feats.copy()
        feats_f[inds] = np.take(medians, inds[1])
        all_rows.append(feats_f)

    if fit_scaler:
        scaler = scaler or StandardScaler()
        flat = np.vstack(all_rows)
        scaler.fit(flat)

    assert scaler is not None
    for country, gdf in grouped:
        gdf = gdf.sort_values("year")
        # Use only years 2000-2020 for input features
        gdf_input = gdf[gdf["year"] <= 2020]
        feats = gdf_input[feature_cols].values.astype(np.float32)
        medians = np.nanmedian(feats, axis=0)
        inds = np.where(np.isnan(feats))
        feats_f = feats.copy()
        feats_f[inds] = np.take(medians, inds[1])
        scaled = scaler.transform(feats_f)
        # Target uses 2021-2023 (years after input)
        gdf_target = gdf[gdf["year"] >= 2021]
        tgt = gdf_target[target_col].values.astype(np.float32)
        valid_tgt = tgt[~np.isnan(tgt)]
        if len(valid_tgt) < 1:
            continue
        sequences.append(scaled.astype(np.float32))
        targets.append(float(np.mean(valid_tgt)))
        lengths.append(len(scaled))

    return sequences, targets, lengths, scaler


def load_and_split(config: dict) -> dict:
    """Prepare train/val/test sequence datasets."""
    set_seed(config.get("seed", 42))
    df = load_panel_data()
    feature_cols = get_feature_columns()
    target_col = config["data"]["target_col"]

    train_df, val_df, test_df = train_val_test_split_panel(
        df,
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config.get("seed", 42),
    )

    tr_seq, tr_tgt, tr_len, scaler = build_country_sequences(
        train_df, feature_cols, target_col, fit_scaler=True
    )
    va_seq, va_tgt, va_len, _ = build_country_sequences(
        val_df, feature_cols, target_col, scaler=scaler
    )
    te_seq, te_tgt, te_len, _ = build_country_sequences(
        test_df, feature_cols, target_col, scaler=scaler
    )

    return {
        "train": CountrySequenceDataset(tr_seq, tr_tgt, tr_len),
        "val": CountrySequenceDataset(va_seq, va_tgt, va_len),
        "test": CountrySequenceDataset(te_seq, te_tgt, te_len),
        "input_dim": len(feature_cols),
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
