"""MDN-RNN: en LSTM som predikerer neste latente tilstand og hva som skjer.

Gitt z_t og handling a_t gir modellen:
  * en blanding av K gaussiske fordelinger over z_{t+1} (MDN-hodet),
  * sannsynligheter for hendelsen i skrittet: vanlig flytt, mål eller hindring.

To bevisste avvik fra artikkelen (se decisions.md):
  * Hver komponent er en gaussisk fordeling over *hele* z-vektoren, ikke én blanding per
    dimensjon. I et rutenett er utfallene diskrete (agenten havner i én av noen få celler), og
    alle dimensjonene må endre seg sammen. Én komponent per utfall passer bedre.
  * Middelverdiene er residualer: mu_k = z_t + delta_k. Det meste av scenen står stille, så
    modellen trenger bare lære endringen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from worldmodels.env import NUM_ACTIONS
from worldmodels.mdnrnn.encode import NUM_EVENTS

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -7.0, 2.0


@dataclass(frozen=True)
class MDNRNNConfig:
    latent_dim: int = 32
    num_actions: int = NUM_ACTIONS
    hidden_dim: int = 256
    num_mixtures: int = 5


@dataclass
class MDNOutput:
    logit_pi: torch.Tensor   # (B, T, K)
    mu: torch.Tensor         # (B, T, K, D)
    log_sigma: torch.Tensor  # (B, T, K, D)
    event_logits: torch.Tensor  # (B, T, NUM_EVENTS)
    hidden: tuple[torch.Tensor, torch.Tensor]
    h: torch.Tensor          # (B, T, H) skjult tilstand etter hvert skritt


class MDNRNN(nn.Module):
    def __init__(self, config: MDNRNNConfig | None = None):
        super().__init__()
        self.config = c = config or MDNRNNConfig()
        self.lstm = nn.LSTM(c.latent_dim + c.num_actions, c.hidden_dim, batch_first=True)
        self.pi_head = nn.Linear(c.hidden_dim, c.num_mixtures)
        self.mu_head = nn.Linear(c.hidden_dim, c.num_mixtures * c.latent_dim)
        self.sigma_head = nn.Linear(c.hidden_dim, c.num_mixtures * c.latent_dim)
        self.event_head = nn.Linear(c.hidden_dim, NUM_EVENTS)

    def forward(
        self, z: torch.Tensor, actions: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> MDNOutput:
        """z: (B, T, D) float, actions: (B, T) long."""
        c = self.config
        a = F.one_hot(actions, c.num_actions).float()
        h, hidden = self.lstm(torch.cat([z, a], dim=-1), hidden)
        B, T, _ = h.shape
        delta = self.mu_head(h).view(B, T, c.num_mixtures, c.latent_dim)
        log_sigma = self.sigma_head(h).view(B, T, c.num_mixtures, c.latent_dim)
        return MDNOutput(
            logit_pi=self.pi_head(h),
            mu=z.unsqueeze(2) + delta,
            log_sigma=log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX),
            event_logits=self.event_head(h),
            hidden=hidden,
            h=h,
        )

    # ------------------------------------------------------------ lagring
    def save(self, path, **extra) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> tuple["MDNRNN", dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(MDNRNNConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt


def mdn_nll(out: MDNOutput, target: torch.Tensor) -> torch.Tensor:
    """Negativ log-sannsynlighet for target (B, T, D) under blandingen, per skritt (B, T).

    Delt på D, slik at tallet er sammenlignbart mellom ulike størrelser på z.
    """
    t = target.unsqueeze(2)                                           # (B, T, 1, D)
    log_prob = -0.5 * (((t - out.mu) / out.log_sigma.exp()) ** 2) - out.log_sigma - 0.5 * torch.log(
        torch.tensor(2 * torch.pi)
    )
    log_prob = log_prob.sum(-1)                                       # (B, T, K)
    log_mix = torch.logsumexp(F.log_softmax(out.logit_pi, dim=-1) + log_prob, dim=-1)
    return -log_mix / target.shape[-1]


def most_likely_mean(out: MDNOutput) -> torch.Tensor:
    """Middelverdien til komponenten med høyest vekt, (B, T, D). Deterministisk prediksjon."""
    k = out.logit_pi.argmax(-1)                                        # (B, T)
    idx = k[..., None, None].expand(*k.shape, 1, out.mu.shape[-1])
    return out.mu.gather(2, idx).squeeze(2)


def sample_next(out: MDNOutput, temperature: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Trekk z_{t+1} fra blandingen, (B, T, D).

    temperature skalerer både valg av komponent og spredningen. 0 gir most_likely_mean.
    Høyere temperatur gir en mer uforutsigbar drøm, som i artikkelen brukes for å gjøre
    det vanskeligere for controlleren å utnytte feil i modellen.
    """
    if temperature <= 0:
        return most_likely_mean(out)
    probs = F.softmax(out.logit_pi / temperature, dim=-1)
    k = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator).view(probs.shape[:-1])
    idx = k[..., None, None].expand(*k.shape, 1, out.mu.shape[-1])
    mu = out.mu.gather(2, idx).squeeze(2)
    sigma = out.log_sigma.exp().gather(2, idx).squeeze(2)
    noise = torch.randn(mu.shape, generator=generator)
    return mu + sigma * noise * (temperature ** 0.5)
