# Spatio-Temporal Graph Neural Network (ST-GNN)

## Architecture

```
Yearly snapshots: (N countries × F features)
    → GCN layers (spatial diffusion via adjacency)
    → GRU (temporal aggregation)
    → Node-level prediction (life expectancy)
```

## Libraries

- **PyTorch Geometric**: GCNConv (with pure-PyTorch fallback)
- **NetworkX**: graph statistics and validation
- **DGL**: referenced in doctorado notebook for alternative implementation

## Limitations

- Simplified geographic adjacency (not real borders)
- Full graph training; limited generalization to new countries
- Missing values median-imputed before graph construction
