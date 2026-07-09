"""
Variational Autoencoder (VAE) for missing data imputation in LA public health panel.

Encoder maps observed features to latent distribution; decoder reconstructs
all features. Trained with masked MSE on observed entries only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PublicHealthVAE(nn.Module):
    """
    VAE for multivariate health indicator imputation.

    Parameters
    ----------
    input_dim : int
        Number of feature columns.
    hidden_dims : list[int]
        Hidden layer sizes for encoder/decoder MLPs.
    latent_dim : int
        Dimension of latent space.
    dropout : float
        Dropout rate in hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and log-variance."""
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from N(mu, exp(logvar)) via reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent vector to reconstructed features."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> sample -> decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute masked reconstruction loss + KL divergence.

        mask: 1 where observed, 0 where missing (loss computed on observed only).
        """
        mse = F.mse_loss(recon * mask, target * mask, reduction="sum")
        n_obs = mask.sum().clamp(min=1.0)
        recon_loss = mse / n_obs
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
        total = recon_loss + beta * kl
        return total, recon_loss, kl

    def impute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Impute missing values using decoder reconstruction."""
        self.eval()
        with torch.no_grad():
            x_filled = torch.where(mask.bool(), x, torch.zeros_like(x))
            recon, _, _ = self.forward(x_filled)
            return torch.where(mask.bool(), x, recon)
