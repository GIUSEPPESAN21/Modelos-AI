# Latin American Public Health — Deep Learning Models

A modular Python project implementing **5 deep learning / causal ML models** for Latin American public health panel data (2000–2023, 42 countries, ~1000 observations, up to 83% missing data).

## Models

| # | Model | Folder | Purpose |
|---|-------|--------|---------|
| 1 | **VAE** | `vae/` | Missing data imputation |
| 2 | **Attention LSTM** | `attention_lstm/` | Temporal health forecasting |
| 3 | **ST-GNN** | `st_gnn/` | Spatio-temporal graph modeling |
| 4 | **DML** | `dml/` | Causal policy effect (ATE) |
| 5 | **cGAN** | `cgan/` | Counterfactual profile generation |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic demo dataset
python scripts/generate_data.py

# 3. Train all models (demo epochs)
python scripts/train_all.py

# 4. Launch Streamlit dashboard
streamlit run app.py
```

## Individual Model Training

```bash
python vae/src/train.py && python vae/src/evaluate.py
python attention_lstm/src/train.py && python attention_lstm/src/evaluate.py
python st_gnn/src/train.py && python st_gnn/src/evaluate.py
python dml/src/train.py && python dml/src/evaluate.py
python cgan/src/train.py && python cgan/src/evaluate.py
```

## Project Structure

```
├── data/                    # Shared Parquet + DuckDB dataset
├── scripts/
│   ├── generate_data.py     # Synthetic LA health data generator
│   ├── shared_utils.py      # Common utilities
│   ├── train_all.py         # Train all models
│   └── create_notebooks.py  # Regenerate notebooks
├── vae/                     # Model 1
├── attention_lstm/          # Model 2
├── st_gnn/                  # Model 3
├── dml/                     # Model 4
├── cgan/                    # Model 5
├── app.py                   # Streamlit dashboard
└── requirements.txt
```

Each model folder contains:
- `src/` — `model.py`, `train.py`, `evaluate.py`, `utils.py`
- `notebooks/` — 3 progressive levels (medio, maestría, doctorado)
- `config.yaml` — hyperparameters
- `results/` — trained models, metrics, visualizations

## Data

Synthetic dataset mimicking LA public health indicators:
- **42 countries**, **2000–2023**, **20 health/economic features**
- Structured missingness (MCAR, MAR, MNAR) up to **83%**
- Formats: `data/la_public_health.parquet`, `data/la_public_health.duckdb`

## Reproducibility

- Fixed random seed: `42` (configurable in each `config.yaml`)
- Pinned dependency versions in `requirements.txt`

## Colab Compatibility

Upload the project folder and run:

```python
!pip install -r requirements.txt
!python scripts/generate_data.py
!python scripts/train_all.py
```

## Limitations

- **Synthetic data only** — no real PAHO/WHO/BM data included
- **Demo training** — reduced epochs for quick execution; increase in `config.yaml` for production
- **Small sample** (~1000 obs) limits model complexity and causal identification
- **Simplified geography** — adjacency matrix is approximate, not real borders
- **Implementacion Modelos.docx** was not found in workspace; requirements inferred from specification

## Streamlit App

Interactive dashboard with:
- Model selection and architecture descriptions
- Hyperparameter inspection
- Metrics from trained models
- Model comparison (missing data, causal robustness, Q1 potential, policy use)
- Dataset exploration visualizations

## License

Research / educational use — UNIANDES Estancia UNINUÑEZ project.
