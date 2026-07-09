"""
Conditional GAN (cGAN) for counterfactual health profile generation.

Generator produces synthetic health feature vectors conditioned on
country characteristics; discriminator distinguishes real vs generated.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional generator: noise + condition → feature vector."""

    def __init__(
        self,
        noise_dim: int,
        cond_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond], dim=1))


class Discriminator(nn.Module):
    """Conditional discriminator: features + condition → real/fake logit."""

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, cond], dim=1))


class ConditionalGAN(nn.Module):
    """Wrapper combining generator and discriminator."""

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        noise_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.generator = Generator(noise_dim, cond_dim, feature_dim, hidden_dim)
        self.discriminator = Discriminator(feature_dim, cond_dim, hidden_dim)

    def generate(self, cond: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        """Generate counterfactual profiles for given conditions."""
        n = n_samples or cond.size(0)
        if cond.size(0) == 1 and n > 1:
            cond = cond.expand(n, -1)
        z = torch.randn(n, self.noise_dim, device=cond.device)
        return self.generator(z, cond)
