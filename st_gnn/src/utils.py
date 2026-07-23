"""ST-GNN data utilities: graph construction, spatio-temporal tensors."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared_utils import get_feature_columns, load_adjacency, load_panel_data, set_seed


def adjacency_to_edge_index(adj: pd.DataFrame) -> torch.Tensor:
    """Convert adjacency DataFrame to PyG edge_index."""
    countries = list(adj.index)
    edges = []
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if i != j and adj.loc[c1, c2] > 0:
                edges.append([i, j])
    if not edges:
        edges = [[i, (i + 1) % len(countries)] for i in range(len(countries))]
    return torch.LongTensor(edges).t().contiguous()


def build_spatiotemporal_tensor(
    df: pd.DataFrame,
    countries: list[str],
    feature_cols: list[str],
    years: list[int],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Build (T, N, F) tensor and (N,) target vector.
    
    Excludes life_expectancy from feature_cols to prevent data leakage.
    Uses years 2000-2020 for input tensor; 2023 for targets.
    """
    # Exclude life_expectancy from features to prevent data leakage
    feature_cols = [c for c in feature_cols if c != "life_expectancy"]
    
    # Split years: input uses 2000-2020, target uses 2023
    input_years = [y for y in years if y <= 2020]
    target_year = 2023 if 2023 in years else years[-1]
    
    n = len(countries)
    f = len(feature_cols)
    t = len(input_years)
    arr = np.zeros((t, n, f), dtype=np.float32)
    targets = np.full(n, np.nan, dtype=np.float32)

    for yi, year in enumerate(input_years):
        ydf = df[df["year"] == year].set_index("country")
        for ci, country in enumerate(countries):
            if country in ydf.index:
                vals = ydf.loc[country, feature_cols].values.astype(np.float32)
                med = np.nanmedian(vals)
                vals = np.where(np.isnan(vals), med, vals)
                arr[yi, ci] = vals
    
    # Target from the latest year (2023 if available)
    tgt_df = df[df["year"] == target_year].set_index("country")
    for ci, country in enumerate(countries):
        if country in tgt_df.index and "life_expectancy" in tgt_df.columns:
            targets[ci] = tgt_df.loc[country, "life_expectancy"]

    flat = arr.reshape(-1, f)
    if fit_scaler:
        scaler = scaler or StandardScaler()
        scaler.fit(flat)
    assert scaler is not None
    scaled = scaler.transform(flat).reshape(t, n, f)
    tgt_med = np.nanmedian(targets)
    targets = np.where(np.isnan(targets), tgt_med, targets)
    return torch.FloatTensor(scaled), torch.FloatTensor(targets), scaler


def load_graph_data(config: dict) -> dict:
    """Load panel, adjacency, and build tensors."""
    set_seed(config.get("seed", 42))
    df = load_panel_data()
    adj = load_adjacency()
    countries = list(adj.index)
    feature_cols = get_feature_columns()
    years = sorted(df["year"].unique().tolist())

    # Use all data for demo; split countries
    rng = np.random.default_rng(config.get("seed", 42))
    idx = np.arange(len(countries))
    rng.shuffle(idx)
    n_test = max(2, int(len(countries) * config["data"]["test_ratio"]))
    test_idx = set(idx[:n_test].tolist())

    x_seq, targets, scaler = build_spatiotemporal_tensor(
        df, countries, feature_cols, years, fit_scaler=True
    )
    edge_index = adjacency_to_edge_index(adj)

    # NetworkX graph stats for metadata
    G = nx.from_pandas_adjacency(adj)
    graph_stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
    }

    train_mask = torch.tensor([i not in test_idx for i in range(len(countries))])
    test_mask = ~train_mask

    return {
        "x_seq": x_seq,
        "targets": targets,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "countries": countries,
        "in_features": len(feature_cols),
        "graph_stats": graph_stats,
        "scaler": scaler,
    }
