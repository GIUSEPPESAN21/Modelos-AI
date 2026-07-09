# Attention-based LSTM — Temporal Health Forecasting

## Architecture

```
Per-country sequence (T×F) → LSTM (2 layers, 64 hidden)
    → Bahdanau Attention → Context vector → FC → Target prediction
Dynamic padding via pack_padded_sequence
```

## Purpose

Forecasts health outcomes (e.g., life expectancy) using attention-weighted temporal patterns across 2000–2023 for each country.

## Pipeline

```bash
python attention_lstm/src/train.py
python attention_lstm/src/evaluate.py
```

## Limitations

- Requires median imputation before sequence construction
- Country-level split reduces effective sample size (~30 countries per split)
- Attention interpretability limited with high missingness
- No explicit spatial dependencies (see ST-GNN)
