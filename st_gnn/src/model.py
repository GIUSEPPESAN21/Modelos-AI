"""
Spatio-Temporal Graph Neural Network for regional health modeling.

Combines GCN spatial message passing with temporal GRU over yearly snapshots.
Uses PyTorch Geometric when available; falls back to pure PyTorch implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class SimpleGCNLayer(nn.Module):
    """Fallback GCN layer when PyG is unavailable."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: (N, F), edge_index: (2, E)."""
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg = deg.clamp(min=1)
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        agg = agg / deg.unsqueeze(1)
        return F.relu(self.linear(agg + x))


class STGNN(nn.Module):
    """
    Spatio-Temporal GNN: GCN per timestep + GRU temporal aggregation.

    Input: node features (N countries × F features) over T timesteps.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        gcn_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gcn_layers = nn.ModuleList()
        prev = in_features
        for _ in range(gcn_layers):
            if HAS_PYG:
                self.gcn_layers.append(GCNConv(prev, hidden_dim))
            else:
                self.gcn_layers.append(SimpleGCNLayer(prev, hidden_dim))
            prev = hidden_dim

        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x_seq: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        x_seq: (T, N, F) — T timesteps, N nodes, F features
        edge_index: (2, E)
        Returns: (N,) predictions per node at final timestep
        """
        t_steps, n_nodes, _ = x_seq.shape
        spatial_embeds = []
        for t in range(t_steps):
            h = x_seq[t]
            for gcn in self.gcn_layers:
                if HAS_PYG:
                    h = F.relu(gcn(h, edge_index))
                else:
                    h = gcn(h, edge_index)
            spatial_embeds.append(h)
        # (T, N, H) -> GRU per node
        seq = torch.stack(spatial_embeds, dim=1)  # (N, T, H)
        out, _ = self.temporal(seq)
        final = out[:, -1, :]  # (N, H)
        return self.head(final).squeeze(-1)
