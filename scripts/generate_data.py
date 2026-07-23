"""
Generate synthetic Latin American public health dataset (2000-2023, 42 countries).

Mimics real-world characteristics: ~1000 panel observations, severe missingness
(up to 83%), correlated health/economic indicators, and spatial structure.
Outputs Parquet and DuckDB formats compatible with all model pipelines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Reproducibility
RANDOM_SEED = 42

# 42 Latin American and Caribbean countries (ISO3 codes)
COUNTRIES = [
    "ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM", "ECU", "SLV",
    "GTM", "GUY", "HTI", "HND", "JAM", "MEX", "NIC", "PAN", "PRY", "PER",
    "PRI", "URY", "VEN", "BLZ", "SUR", "TTO", "BRB", "BHS", "GRD", "LCA",
    "VCT", "ATG", "DMA", "KNA", "ABW", "CUW", "SXM", "GUF", "GLP", "MTQ",
    "MSR", "TCA",
]

YEARS = list(range(2000, 2024))

FEATURES = [
    "life_expectancy",
    "infant_mortality",
    "health_expenditure_pct_gdp",
    "physicians_per_1000",
    "hospital_beds_per_1000",
    "maternal_mortality",
    "tb_incidence",
    "hiv_prevalence",
    "under5_mortality",
    "gdp_per_capita_usd",
    "gini_index",
    "poverty_rate",
    "education_index",
    "urbanization_pct",
    "sanitation_access_pct",
    "clean_water_access_pct",
    "vaccination_coverage",
    "obesity_rate",
    "diabetes_prevalence",
    "air_pollution_pm25",
]

# Base values per country (simplified latent health index)
COUNTRY_BASE = {
    c: 0.3 + 0.7 * (i / (len(COUNTRIES) - 1)) for i, c in enumerate(COUNTRIES)
}


def _country_trend(country: str, year: int) -> float:
    """Slow improvement trend over time with country heterogeneity."""
    base = COUNTRY_BASE[country]
    trend = (year - 2000) * 0.008 * base
    noise = np.sin(year * 0.3 + hash(country) % 100) * 0.02
    return base + trend + noise


def _generate_feature(
    rng: np.random.Generator,
    country: str,
    year: int,
    feat: str,
    health_idx: float,
) -> float:
    """Generate a single feature value from latent health index."""
    scales = {
        "life_expectancy": (65, 82, 1.0),
        "infant_mortality": (5, 45, -1.0),
        "health_expenditure_pct_gdp": (3, 12, 0.8),
        "physicians_per_1000": (0.5, 4.5, 0.9),
        "hospital_beds_per_1000": (0.8, 5.0, 0.7),
        "maternal_mortality": (20, 180, -0.9),
        "tb_incidence": (5, 120, -0.8),
        "hiv_prevalence": (0.1, 2.5, -0.5),
        "under5_mortality": (8, 55, -1.0),
        "gdp_per_capita_usd": (2000, 18000, 0.95),
        "gini_index": (0.35, 0.62, -0.3),
        "poverty_rate": (5, 55, -0.85),
        "education_index": (0.4, 0.9, 0.85),
        "urbanization_pct": (45, 92, 0.4),
        "sanitation_access_pct": (50, 98, 0.9),
        "clean_water_access_pct": (55, 99, 0.85),
        "vaccination_coverage": (60, 98, 0.8),
        "obesity_rate": (8, 35, 0.5),
        "diabetes_prevalence": (3, 14, 0.3),
        "air_pollution_pm25": (8, 45, -0.4),
    }
    lo, hi, w = scales[feat]
    val = lo + (hi - lo) * (0.5 + 0.5 * health_idx * w)
    val += rng.normal(0, (hi - lo) * 0.04)
    return float(np.clip(val, lo * 0.85, hi * 1.15))


def apply_missingness(
    df: pd.DataFrame,
    rng: np.random.Generator,
    max_missing_rate: float = 0.83,
) -> pd.DataFrame:
    """
    Apply structured missingness: MCAR, MAR (by GDP), and MNAR (extreme values).

    Different features have different missing rates up to max_missing_rate.
    """
    out = df.copy()
    n = len(out)
    feature_cols = [c for c in out.columns if c in FEATURES]

    for i, col in enumerate(feature_cols):
        # Vary missing rate by feature (45% to 83%)
        rate = 0.55 + (i / max(len(feature_cols) - 1, 1)) * (max_missing_rate - 0.55)
        rate = min(rate, max_missing_rate)

        mask_mcar = rng.random(n) < rate * 0.50
        gdp_rank = out["gdp_per_capita_usd"].rank(pct=True, na_option="keep").fillna(0.5)
        mask_mar = (rng.random(n) < rate * 0.40) & (gdp_rank < 0.50)
        col_vals = out[col].copy()
        q95, q05 = col_vals.quantile(0.95), col_vals.quantile(0.05)
        mask_mnar = ((col_vals > q95) | (col_vals < q05)) & (rng.random(n) < rate * 0.35)

        missing = mask_mcar | mask_mar | mask_mnar
        out.loc[missing, col] = np.nan

    return out


def generate_adjacency(countries: list[str]) -> pd.DataFrame:
    """Build simplified geographic adjacency matrix for ST-GNN."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(countries)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Spatial proximity proxy via country index + random edges
            dist = abs(i - j)
            prob = max(0.05, 0.5 - dist * 0.02)
            if rng.random() < prob:
                adj[i, j] = adj[j, i] = 1.0
    # Ensure connectivity via k-nearest
    for i in range(n):
        if adj[i].sum() == 0:
            j = (i + 1) % n
            adj[i, j] = adj[j, i] = 1.0
    return pd.DataFrame(adj, index=countries, columns=countries)


def generate_dataset(output_dir: Path | str | None = None) -> pd.DataFrame:
    """Generate full panel dataset and save to Parquet/DuckDB."""
    rng = np.random.default_rng(RANDOM_SEED)
    root = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "data"
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for country in COUNTRIES:
        for year in YEARS:
            health_idx = _country_trend(country, year)
            row = {"country": country, "year": year, "health_index_latent": health_idx}
            for feat in FEATURES:
                row[feat] = _generate_feature(rng, country, year, feat, health_idx)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = apply_missingness(df, rng)

    # Treatment variable for DML: health policy intervention (synthetic)
    df["health_policy_treatment"] = (
        (df["health_expenditure_pct_gdp"].fillna(df["health_expenditure_pct_gdp"].median()) > 6.5)
        & (df["year"] >= 2010)
    ).astype(int)
    df["outcome_life_expectancy"] = df["life_expectancy"]

    # Save formats
    parquet_path = root / "la_public_health.parquet"
    df.to_parquet(parquet_path, index=False)

    adj = generate_adjacency(COUNTRIES)
    adj.to_parquet(root / "country_adjacency.parquet")

    metadata = {
        "n_countries": len(COUNTRIES),
        "years": YEARS,
        "features": FEATURES,
        "n_observations": len(df),
        "missing_rate_overall": float(df[FEATURES].isna().mean().mean()),
        "random_seed": RANDOM_SEED,
    }
    with open(root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    con = duckdb.connect(str(root / "la_public_health.duckdb"))
    con.execute("CREATE OR REPLACE TABLE panel AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
    con.execute(
        "CREATE OR REPLACE TABLE adjacency AS SELECT * FROM read_parquet(?)",
        [str(root / "country_adjacency.parquet")],
    )
    con.close()

    print(f"Generated {len(df)} observations -> {parquet_path}")
    print(f"Overall missing rate: {metadata['missing_rate_overall']:.1%}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LA public health synthetic data")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    generate_dataset(args.output_dir)


if __name__ == "__main__":
    main()
