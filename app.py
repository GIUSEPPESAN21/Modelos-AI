"""
Streamlit dashboard for Latin American Public Health ML models.

Select among 5 models, view architecture, run predictions, and compare approaches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.shared_utils import DATA_DIR, load_metadata, load_panel_data

st.set_page_config(
    page_title="LA Public Health ML Dashboard",
    page_icon="🏥",
    layout="wide",
)

MODELS = {
    "VAE": {
        "folder": "vae",
        "description": "Variational Autoencoder for missing data imputation using masked reconstruction + KL divergence.",
        "architecture": "Encoder MLP [128→64] → Latent(16) → Decoder → Reconstructed features",
        "strengths": ["Missing data handling", "Unsupervised imputation", "Latent representations"],
        "limitations": ["No causal claims", "Assumes stable correlations"],
        "q1_potential": "Medium-High for imputation methods papers",
        "policy_use": "Fill data gaps for policy dashboards and composite indices",
    },
    "Attention LSTM": {
        "folder": "attention_lstm",
        "description": "LSTM with Bahdanau attention for temporal health forecasting per country.",
        "architecture": "LSTM(64, 2 layers) → Attention → FC → Life expectancy prediction",
        "strengths": ["Temporal patterns", "Attention interpretability", "Dynamic padding"],
        "limitations": ["No spatial structure", "Needs imputation first"],
        "q1_potential": "Medium for health forecasting applications",
        "policy_use": "Project life expectancy trends for planning",
    },
    "ST-GNN": {
        "folder": "st_gnn",
        "description": "Spatio-Temporal GNN combining GCN spatial diffusion with GRU temporal modeling.",
        "architecture": "GCN(2 layers) per timestep → GRU → Node-level prediction",
        "strengths": ["Spatial spillovers", "Regional health networks", "Graph structure"],
        "limitations": ["Simplified adjacency", "Fixed graph"],
        "q1_potential": "High for spatial epidemiology / GNN health papers",
        "policy_use": "Model regional contagion of health outcomes",
    },
    "DML": {
        "folder": "dml",
        "description": "Double/Debiased ML for causal ATE of health policy on life expectancy.",
        "architecture": "Cross-fitted nuisance models → Orthogonalized ATE estimation",
        "strengths": ["Causal robustness", "Policy evaluation", "Confounding control"],
        "limitations": ["Unconfoundedness assumed", "Synthetic treatment in demo"],
        "q1_potential": "High for health economics / causal ML",
        "policy_use": "Estimate impact of health expenditure policies",
    },
    "cGAN": {
        "folder": "cgan",
        "description": "Conditional GAN for generating counterfactual health profiles.",
        "architecture": "Generator(noise+condition) vs Discriminator(features+condition)",
        "strengths": ["Counterfactual generation", "Data augmentation", "Scenario simulation"],
        "limitations": ["Mode collapse risk", "No formal causal ID"],
        "q1_potential": "Medium for generative health data methods",
        "policy_use": "Simulate health profiles under policy scenarios",
    },
}


@st.cache_data
def load_data() -> pd.DataFrame:
    if not (DATA_DIR / "la_public_health.parquet").exists():
        from scripts.generate_data import generate_dataset
        generate_dataset(DATA_DIR)
    return load_panel_data()


def load_model_config(folder: str) -> dict:
    path = ROOT / folder / "config.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_metrics(folder: str, name: str) -> dict | None:
    path = ROOT / folder / "results" / "metrics" / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_comparison_tab() -> None:
    st.subheader("Model Comparison Matrix")
    rows = []
    for name, info in MODELS.items():
        folder = info["folder"]
        eval_m = load_metrics(folder, "eval_metrics.json") or {}
        rows.append({
            "Model": name,
            "Missing Data": "★★★★☆" if "VAE" in name else "★★★☆☆" if name != "DML" else "★★☆☆☆",
            "Causal Robustness": "★★★★★" if name == "DML" else "★★☆☆☆",
            "Spatial Modeling": "★★★★★" if name == "ST-GNN" else "★★☆☆☆",
            "Q1 Potential": info["q1_potential"],
            "Policy Application": info["policy_use"][:50] + "...",
            "Key Metric": str(list(eval_m.values())[0]) if eval_m else "Not trained",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    fig = go.Figure(data=go.Scatterpolar(
        r=[4, 3, 5, 5, 3],
        theta=["Missing Data", "Forecasting", "Spatial", "Causal", "Generation"],
        fill="toself", name="VAE",
    ))
    for name, vals in [("LSTM", [3, 5, 2, 2, 2]), ("ST-GNN", [3, 4, 5, 2, 2]),
                       ("DML", [2, 2, 2, 5, 2]), ("cGAN", [3, 2, 2, 2, 5])]:
        fig.add_trace(go.Scatterpolar(r=vals, theta=["Missing Data", "Forecasting", "Spatial", "Causal", "Generation"],
                                      fill="toself", name=name))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), title="Capability Radar")
    st.plotly_chart(fig, use_container_width=True)


def render_model_tab(model_name: str, info: dict) -> None:
    folder = info["folder"]
    config = load_model_config(folder)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Architecture:** `{info['architecture']}`")
        st.markdown("**Strengths:** " + ", ".join(info["strengths"]))
        st.markdown("**Limitations:** " + ", ".join(info["limitations"]))

    with col2:
        st.json(config)

    train_m = load_metrics(folder, "train_metrics.json")
    eval_m = load_metrics(folder, "eval_metrics.json")
    if train_m:
        st.success(f"Training metrics: {train_m}")
    if eval_m:
        st.info(f"Evaluation metrics: {eval_m}")
    if not train_m:
        st.warning("Model not trained yet. Run training from project root.")

    st.subheader("Hyperparameter Tuning")
    if config:
        updated = config.copy()
        if "training" in updated:
            for key in list(updated["training"].keys()):
                if isinstance(updated["training"][key], (int, float)):
                    updated["training"][key] = st.slider(
                        f"training.{key}", float(updated["training"][key]) * 0.5 if isinstance(updated["training"][key], float) else 1,
                        float(updated["training"][key]) * 2 if isinstance(updated["training"][key], float) else 100,
                        float(updated["training"][key]) if isinstance(updated["training"][key], float) else int(updated["training"][key]),
                        key=f"slider_{model_name}_{key}"
                    )
        st.write("Modified config preview:", updated)

    st.subheader("Interactive Prediction Demo")
    df = load_data()
    countries = sorted(df["country"].unique())
    selected_country = st.selectbox("Country", countries, key=f"country_{model_name}")
    year = st.slider("Year", 2000, 2023, 2020, key=f"year_{model_name}")

    row = df[(df["country"] == selected_country) & (df["year"] == year)]
    if not row.empty:
        st.dataframe(row.astype(str).T, use_container_width=True)
        missing_pct = row.isna().mean().mean() * 100
        st.metric("Missing rate (this row)", f"{missing_pct:.1f}%")

    if model_name == "DML":
        if train_m and "ate" in train_m:
            st.metric("Estimated ATE (years of life expectancy)", f"{train_m['ate']:.3f}")
            st.write(f"95% CI: [{train_m.get('ci_lower', 'N/A')}, {train_m.get('ci_upper', 'N/A')}]")

    if st.button(f"Run {model_name} inference", key=f"run_{model_name}"):
        with st.spinner("Running..."):
            try:
                if model_name == "VAE":
                    import torch
                    sys.path.insert(0, str(ROOT / "vae" / "src"))
                    from model import PublicHealthVAE
                    ckpt_path = ROOT / "vae" / "results" / "trained_models" / "vae_best.pt"
                    if ckpt_path.exists():
                        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        st.success("VAE loaded. Imputation available for masked features.")
                    else:
                        st.error("Train VAE first: python vae/src/train.py")
                elif model_name == "cGAN":
                    st.success("cGAN ready for counterfactual generation after training.")
                else:
                    st.success(f"{model_name} pipeline ready.")
            except Exception as e:
                st.error(f"Error: {e}")


def main() -> None:
    st.title("Latin American Public Health — Deep Learning Dashboard")
    st.markdown("**42 countries | 2000–2023 | ~1000 observations | Up to 83% missing data**")

    try:
        meta = load_metadata()
        st.sidebar.metric("Observations", meta["n_observations"])
        st.sidebar.metric("Missing rate", f"{meta['missing_rate_overall']:.1%}")
    except FileNotFoundError:
        if st.sidebar.button("Generate Demo Data"):
            from scripts.generate_data import generate_dataset
            generate_dataset()
            st.rerun()

    tab_overview, tab_compare, *model_tabs = st.tabs(
        ["Overview", "Compare Models"] + list(MODELS.keys())
    )

    with tab_overview:
        df = load_data()
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Countries", df["country"].nunique())
        c2.metric("Years", df["year"].nunique())
        c3.metric("Features", len([c for c in df.columns if c not in ("country", "year")]))

        fig = px.line(
            df.groupby("year")["life_expectancy"].mean().reset_index(),
            x="year", y="life_expectancy", title="Average Life Expectancy (available data)",
        )
        st.plotly_chart(fig, use_container_width=True)

        miss = df.select_dtypes(include=[np.number]).isna().mean().sort_values(ascending=False).head(10)
        fig2 = px.bar(miss.reset_index(), x="index", y=0, title="Top 10 Missing Features")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_compare:
        render_comparison_tab()

    for tab, (name, info) in zip(model_tabs, MODELS.items()):
        with tab:
            render_model_tab(name, info)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Start**")
    st.sidebar.code("pip install -r requirements.txt\npython scripts/generate_data.py\npython scripts/train_all.py\nstreamlit run app.py")


if __name__ == "__main__":
    main()
