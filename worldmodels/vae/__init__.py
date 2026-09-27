"""VAE (V i World Models): komprimerer observasjoner til en latent vektor z."""

from worldmodels.vae.model import ConvVAE, VAEConfig

__all__ = ["ConvVAE", "VAEConfig"]
