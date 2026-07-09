# Variational Autoencoder (VAE) — Missing Data Imputation

## Architecture

```
Input (20 features, masked) → Encoder MLP [128→64] → μ, log σ²
    → Reparameterization → Latent z (16-dim)
    → Decoder MLP [64→128] → Reconstructed features
Loss: Masked MSE (observed only) + β·KL(q||p)
```

## Purpose

Imputes missing values in the Latin American public health panel (2000–2023, 42 countries) by learning a low-dimensional latent representation of correlated health indicators.

## Libraries

- **PyTorch**: model and training
- **Missingno concepts**: structured MCAR/MAR/MNAR missingness handling via masks
- **Pyro** (optional extension): probabilistic VAE variant in doctorado notebook

## Pipeline

```bash
# From project root
python scripts/generate_data.py
python vae/src/train.py
python vae/src/evaluate.py
```

## Hyperparameters

See `config.yaml` — key params: `latent_dim=16`, `hidden_dims=[128,64]`, `beta=1.0`.

## Limitations

- Assumes feature correlations are stable across countries/years
- Median-filled inputs for scaling may bias extreme missingness patterns
- Not designed for causal inference (use DML for treatment effects)
- Small sample (~1000 obs) limits latent dimensionality
- Demo training uses 30 epochs; production needs cross-validation

## Notebooks

| Notebook | Level | Content |
|----------|-------|---------|
| `01_nivel_medio.ipynb` | Undergraduate | VAE basics, missing data viz, simple training |
| `02_nivel_maestria.ipynb` | Master's | β-VAE, missingness mechanisms, imputation metrics |
| `03_nivel_doctorado.ipynb` | PhD | Pyro probabilistic VAE, sensitivity analysis |
