"""
Double/Debiased Machine Learning (DML) for causal effect estimation.

Estimates Average Treatment Effect (ATE) of health policy intervention on
life expectancy using cross-fitted nuisance models and orthogonalization.
Supports EconML and DoubleML backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class DMLResult:
    """Container for DML estimation results."""

    ate: float
    ate_se: float
    ci_lower: float
    ci_upper: float
    method: str
    n_samples: int


class LinearDML:
    """
    Manual DML implementation with cross-fitting (Chernozhukov et al.).

    Partially linear model: Y = θ·D + g(X) + ε, D = m(X) + η
    Nuisance models: gradient boosting for g and m.
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        self.ate_: float | None = None
        self.ate_se_: float | None = None

    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
    ) -> DMLResult:
        """
        Estimate ATE via cross-fitted residual-on-residual regression.

        Parameters
        ----------
        Y : outcome (n,)
        D : treatment (n,) binary or continuous
        X : covariates (n, p)
        """
        n = len(Y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        Y_res = np.zeros(n)
        D_res = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            Y_tr, D_tr = Y[train_idx], D[train_idx]

            # Outcome nuisance model g(X)
            g_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=self.random_state
            )
            g_model.fit(X_tr, Y_tr)
            Y_res[test_idx] = Y[test_idx] - g_model.predict(X_te)

            # Treatment nuisance model m(X)
            if len(np.unique(D)) <= 2:
                m_model = LogisticRegression(max_iter=500, random_state=self.random_state)
                m_model.fit(X_tr, D_tr)
                D_res[test_idx] = D[test_idx] - m_model.predict_proba(X_te)[:, 1]
            else:
                m_model = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, random_state=self.random_state
                )
                m_model.fit(X_tr, D_tr)
                D_res[test_idx] = D[test_idx] - m_model.predict(X_te)

        # Final stage: OLS of Y_res on D_res
        theta = np.sum(D_res * Y_res) / np.sum(D_res ** 2)
        residuals = Y_res - theta * D_res
        se = np.sqrt(np.mean(residuals ** 2) / (np.sum(D_res ** 2) / n))

        self.ate_ = float(theta)
        self.ate_se_ = float(se)
        z = 1.96
        return DMLResult(
            ate=self.ate_,
            ate_se=self.ate_se_,
            ci_lower=self.ate_ - z * self.ate_se_,
            ci_upper=self.ate_ + z * self.ate_se_,
            method="LinearDML_crossfit",
            n_samples=n,
        )


def fit_econml_dml(Y: np.ndarray, D: np.ndarray, X: np.ndarray, seed: int = 42) -> DMLResult:
    """Estimate ATE using EconML LinearDML if available."""
    try:
        from econml.dml import LinearDML as EconLinearDML

        model = EconLinearDML(
            model_y=RandomForestRegressor(n_estimators=50, random_state=seed),
            model_t=RandomForestRegressor(n_estimators=50, random_state=seed),
            cv=3,
            random_state=seed,
        )
        model.fit(Y, D, X=X)
        ate = float(model.ate(X))
        inf = model.ate_inference(X)
        se = float(inf.stderr_mean)
        z = 1.96
        return DMLResult(
            ate=ate, ate_se=se, ci_lower=ate - z * se, ci_upper=ate + z * se,
            method="EconML_LinearDML", n_samples=len(Y),
        )
    except ImportError:
        return LinearDML(random_state=seed).fit(Y, D, X)


def fit_doubleml(Y: np.ndarray, D: np.ndarray, X: np.ndarray, seed: int = 42) -> DMLResult | None:
    """Estimate ATE using DoubleML package if available."""
    try:
        import pandas as pd
        from doubleml import DoubleMLData, DoubleMLPLR
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler as SS

        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        df["y"] = Y
        df["d"] = D
        dml_data = DoubleMLData(df, "y", "d")

        ml_l = Pipeline([("scaler", SS()), ("reg", LassoCV(cv=3, random_state=seed))])
        ml_m = Pipeline([("scaler", SS()), ("reg", LassoCV(cv=3, random_state=seed))])
        dml = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=3)
        dml.fit()
        ate = float(dml.coef[0])
        se = float(dml.se[0])
        z = 1.96
        return DMLResult(
            ate=ate, ate_se=se, ci_lower=ate - z * se, ci_upper=ate + z * se,
            method="DoubleML_PLR", n_samples=len(Y),
        )
    except Exception:
        return None
