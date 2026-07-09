"""Generate progressive notebooks for all 5 models."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_SPECS = {
    "vae": {
        "01_nivel_medio": ("VAE — Nivel Medio", [
            "# VAE para Imputación de Datos de Salud\n",
            "Introducción a autoencoders variacionales y datos faltantes en salud pública latinoamericana.\n",
            "```python\nimport sys\nsys.path.insert(0, '../..')\nfrom scripts.shared_utils import load_panel_data\nimport missingno as msno\n\ndf = load_panel_data()\nmsno.matrix(df.iloc[:, :10])\n```\n",
            "## Entrenamiento básico\n",
            "```python\n!python ../src/train.py\n!python ../src/evaluate.py\n```\n",
        ]),
        "02_nivel_maestria": ("VAE — Nivel Maestría", [
            "# β-VAE y Mecanismos de Missingness\n",
            "Análisis MCAR/MAR/MNAR y ajuste del parámetro β en config.yaml.\n",
            "```python\nimport yaml\nwith open('../config.yaml') as f:\n    cfg = yaml.safe_load(f)\ncfg['training']['beta'] = 0.5\n```\n",
            "## Métricas de imputación\n",
            "```python\nimport json\nwith open('../results/metrics/eval_metrics.json') as f:\n    print(json.load(f))\n```\n",
        ]),
        "03_nivel_doctorado": ("VAE — Nivel Doctorado", [
            "# VAE Probabilístico con Pyro (opcional)\n",
            "Extensión bayesiana para incertidumbre en imputación.\n",
            "```python\n# pip install pyro-ppl\n# import pyro\n# Modelo probabilístico avanzado — ver documentación Pyro\n```\n",
            "## Análisis de sensibilidad\n",
            "Comparar imputaciones bajo diferentes tasas de missingness simuladas.\n",
        ]),
    },
    "attention_lstm": {
        "01_nivel_medio": ("Attention LSTM — Nivel Medio", [
            "# LSTM con Atención para Series Temporales de Salud\n",
            "```python\nimport sys\nsys.path.insert(0, '../..')\nfrom scripts.shared_utils import load_panel_data\ndf = load_panel_data()\nprint(df.groupby('country')['life_expectancy'].mean().head())\n```\n",
            "## Entrenar modelo\n```python\n!python ../src/train.py\n```\n",
        ]),
        "02_nivel_maestria": ("Attention LSTM — Maestría", [
            "# Padding Dinámico y Visualización de Atención\n",
            "```python\nimport torch\nckpt = torch.load('../results/trained_models/attention_lstm_best.pt', weights_only=False)\nprint(ckpt.keys())\n```\n",
            "Interpretación de pesos de atención por año.\n",
        ]),
        "03_nivel_doctorado": ("Attention LSTM — Doctorado", [
            "# Multi-horizon Forecasting y Transfer Learning\n",
            "Extensión a predicción multi-paso y comparación con Transformers.\n",
        ]),
    },
    "st_gnn": {
        "01_nivel_medio": ("ST-GNN — Nivel Medio", [
            "# Grafos Espacio-Temporales en Salud Pública\n",
            "```python\nimport networkx as nx\nimport pandas as pd\nadj = pd.read_parquet('../../data/country_adjacency.parquet')\nG = nx.from_pandas_adjacency(adj)\nprint('Nodos:', G.number_of_nodes(), 'Aristas:', G.number_of_edges())\n```\n",
        ]),
        "02_nivel_maestria": ("ST-GNN — Maestría", [
            "# PyTorch Geometric y Diffusion Spatial\n",
            "```python\n!python ../src/train.py\n!python ../src/evaluate.py\n```\n",
        ]),
        "03_nivel_doctorado": ("ST-GNN — Doctorado", [
            "# Implementación alternativa con DGL\n",
            "Comparación GCN vs GraphSAGE vs GAT en redes de salud regional.\n",
        ]),
    },
    "dml": {
        "01_nivel_medio": ("DML — Nivel Medio", [
            "# Inferencia Causal con DML\n",
            "Efecto promedio del tratamiento (ATE) de políticas de salud.\n",
            "```python\n!python ../src/train.py\n```\n",
        ]),
        "02_nivel_maestria": ("DML — Maestría", [
            "# EconML y DoubleML\n",
            "```python\nimport json\nwith open('../results/metrics/train_metrics.json') as f:\n    r = json.load(f)\nprint(f\"ATE={r['ate']:.3f}, CI=[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]\")\n```\n",
        ]),
        "03_nivel_doctorado": ("DML — Doctorado", [
            "# Diagnósticos de overlap y sensibilidad a confounders\n",
            "Placebo tests y análisis de variables instrumentales.\n",
        ]),
    },
    "cgan": {
        "01_nivel_medio": ("cGAN — Nivel Medio", [
            "# GANs Condicionales para Perfiles de Salud\n",
            "```python\n!python ../src/train.py\n!python ../src/evaluate.py\n```\n",
        ]),
        "02_nivel_maestria": ("cGAN — Maestría", [
            "# Inferencia Contrafactual\n",
            "Generar perfiles sintéticos bajo escenarios de política.\n",
        ]),
        "03_nivel_doctorado": ("cGAN — Doctorado", [
            "# Wasserstein GAN y restricciones físicas\n",
            "Extensión WGAN-GP con penalización por violación de rangos clínicos.\n",
        ]),
    },
}


def make_notebook(title: str, cells_md: list[str]) -> dict:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n"]},
    ]
    for block in cells_md:
        if block.strip().startswith("```python"):
            code = block.replace("```python\n", "").replace("```\n", "").rstrip("`")
            cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.split("\n") if line or code.split("\n").index(line) == 0]})
        else:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [block]})
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }


def main() -> None:
    for model, specs in NOTEBOOK_SPECS.items():
        nb_dir = ROOT / model / "notebooks"
        nb_dir.mkdir(parents=True, exist_ok=True)
        for fname, (title, blocks) in specs.items():
            nb = make_notebook(title, blocks)
            path = nb_dir / f"{fname}.ipynb"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            print(f"Created {path}")


if __name__ == "__main__":
    main()
