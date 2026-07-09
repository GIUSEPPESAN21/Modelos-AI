"""
Attention-based LSTM for spatio-temporal health indicator forecasting.

Processes per-country time series (2000-2023) with dynamic padding and
Bahdanau-style attention over LSTM hidden states.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Bahdanau additive attention over LSTM outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, lstm_out: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_out : (batch, seq_len, hidden_dim)
        mask : (batch, seq_len) — 1 for valid timesteps

        Returns context vector and attention weights.
        """
        scores = self.v(torch.tanh(self.W(lstm_out))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class AttentionLSTM(nn.Module):
    """
    LSTM encoder with attention for multi-step health forecasting.

    Predicts target feature(s) at final observed timestep + horizon.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with packed padded sequences.

        x: (batch, max_seq, features)
        lengths: (batch,) actual sequence lengths
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        context, attn_weights = self.attention(lstm_out, mask.float())
        out = self.fc(context)
        return out, attn_weights
