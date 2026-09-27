"""Tapsfunksjon for VAE-en, med ekstra vekt på pikslene som faktisk betyr noe.

Rundt 85-90 % av hvert bilde er ensfarget bakgrunn. Med vanlig MSE kan en VAE
få lavt tap ved å tegne bakgrunn overalt og "glemme" den lille agenten. Vi
løser det med et vektkart: piksler som ikke er bakgrunn i originalbildet får
vekt object_weight (f.eks. 10), bakgrunnen får 1.
"""

from __future__ import annotations

import torch

from worldmodels.env.gridworld import COLOR_BACKGROUND

_BACKGROUND = torch.tensor(COLOR_BACKGROUND, dtype=torch.float32) / 255.0


def object_weight_map(x: torch.Tensor, object_weight: float) -> torch.Tensor:
    """Vekt per piksel, (N, 1, H, W): object_weight der x ikke er bakgrunn, ellers 1.

    x: (N, 3, H, W) i [0, 1].
    """
    bg = _BACKGROUND.to(x.device).view(1, 3, 1, 1)
    is_object = ((x - bg).abs().amax(dim=1, keepdim=True) > 0.05).float()
    return 1.0 + (object_weight - 1.0) * is_object


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) per eksempel, (N,)."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    object_weight: float = 10.0,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Returnerer {"loss", "recon", "kl"}, alle snittet over batchen.

    recon er vektet kvadratfeil summert over piksler og kanaler, slik at den står
    i et fornuftig forhold til KL-leddet (som også er en sum over dimensjoner).
    """
    w = object_weight_map(x, object_weight)
    recon_loss = (w * (recon - x).pow(2)).sum(dim=(1, 2, 3)).mean()
    kl = kl_divergence(mu, logvar).mean()
    return {"loss": recon_loss + beta * kl, "recon": recon_loss, "kl": kl}
