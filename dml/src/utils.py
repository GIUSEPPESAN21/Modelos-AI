"""DML data preparation utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared_utils import get_feature_columns, load_panel_data, set_seed, train_val_test_split_panel


def prepare_dml_data(
    df: pd.DataFrame,
    treatment_col: str = "health_policy_treatment",
    outcome_col: str = "outcome_life_expectancy",
    covariate_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare Y, D, X arrays for DML estimation.

    Drops rows with missing outcome; median-imputes covariates.
    """
    covariate_cols = covariate_cols or get_feature_columns()
    sub = df[[treatment_col, outcome_col] + covariate_cols].copy()
    sub = sub.dropna(subset=[outcome_col, treatment_col])

    for col in covariate_cols:
        sub[col] = sub[col].fillna(sub[col].median())

    Y = sub[outcome_col].values.astype(np.float64)
    D = sub[treatment_col].values.astype(np.float64)
    X = sub[covariate_cols].values.astype(np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return Y, D, X, sub


def load_and_split(config: dict) -> dict:
    """Load and split data for DML."""
    set_seed(config.get("seed", 42))
    df = load_panel_data()
    train_df, val_df, test_df = train_val_test_split_panel(
        df,
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config.get("seed", 42),
    )
    # DML uses train for estimation, test for validation
    Y_tr, D_tr, X_tr, _ = prepare_dml_data(train_df)
    Y_te, D_te, X_te, _ = prepare_dml_data(test_df)
    return {"train": (Y_tr, D_tr, X_tr), "test": (Y_te, D_te, X_te)}
