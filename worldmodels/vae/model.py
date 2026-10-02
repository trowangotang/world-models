"""Konvolusjons-VAE som komprimerer 64x64x3-observasjoner til en latent vektor z.

Arkitekturen er en vanlig "halver oppløsningen, doble kanalene"-pyramide:
    64 -> 32 -> 16 -> 8 -> 4 piksler, 32 -> 64 -> 128 -> 256 kanaler
Dekoderen speiler den med transponerte konvolusjoner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VAEConfig:
    latent_dim: int = 32
    image_size: int = 64
    base_channels: int = 32


class ConvVAE(nn.Module):
    def __init__(self, config: VAEConfig | None = None):
        super().__init__()
        self.config = config or VAEConfig()
        c = self.config
        if c.image_size != 64:
            raise ValueError("Arkitekturen er laget for 64x64-bilder")
        ch = [c.base_channels * m for m in (1, 2, 4, 8)]          # 32, 64, 128, 256

        self.encoder = nn.Sequential(
            nn.Conv2d(3, ch[0], 4, stride=2, padding=1), nn.ReLU(),       # 32x32
            nn.Conv2d(ch[0], ch[1], 4, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.Conv2d(ch[1], ch[2], 4, stride=2, padding=1), nn.ReLU(),   # 8x8
            nn.Conv2d(ch[2], ch[3], 4, stride=2, padding=1), nn.ReLU(),   # 4x4
            nn.Flatten(),
        )
        flat = ch[3] * 4 * 4
        self.fc_mu = nn.Linear(flat, c.latent_dim)
        self.fc_logvar = nn.Linear(flat, c.latent_dim)

        self.fc_decode = nn.Linear(c.latent_dim, flat)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (ch[3], 4, 4)),
            nn.ConvTranspose2d(ch[3], ch[2], 4, stride=2, padding=1), nn.ReLU(),  # 8x8
            nn.ConvTranspose2d(ch[2], ch[1], 4, stride=2, padding=1), nn.ReLU(),  # 16x16
            nn.ConvTranspose2d(ch[1], ch[0], 4, stride=2, padding=1), nn.ReLU(),  # 32x32
            nn.ConvTranspose2d(ch[0], 3, 4, stride=2, padding=1),                 # 64x64
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, 3, 64, 64) i [0, 1]. Returnerer (mu, logvar), hver (N, latent_dim)."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim). Returnerer bilde (N, 3, 64, 64) i [0, 1]."""
        return torch.sigmoid(self.decoder(F.relu(self.fc_decode(z))))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar

    # ------------------------------------------------------------ lagring
    def save(self, path, **extra) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> tuple["ConvVAE", dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(VAEConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt
