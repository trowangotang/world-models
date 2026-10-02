"""Controller (C i World Models): en lineær avbildning fra [z, h] til handling.

Med 32 + 256 inndata og 4 handlinger har controlleren bare 1156 parametre. Det er med vilje:
all "forståelse" skal ligge i verdensmodellen (V og M), og C skal bare velge handling.
Proben i steg 3 viste at agentens posisjon kan leses lineært fra (z, h) i 84 % av tilfellene,
så en lineær controller har det den trenger.

For å kunne evaluere en hel populasjon av kandidater på én gang (se es.py) støtter
`batched_logits` at hver kandidat har sine egne vekter.
"""

from __future__ import annotations

import numpy as np
import torch


class LinearController:
    """extra_dim > 0 gir controlleren ekstra inndata etter [z, h] (se features.py). beliefs, sight og
    lookahead sier hvilke, slik at agenten regner ut de samme inndataene som under trening."""

    def __init__(
        self, z_dim: int, h_dim: int, num_actions: int = 4, extra_dim: int = 0,
        beliefs: bool | None = None, lookahead: bool = False, sight: bool = False,
    ):
        self.z_dim, self.h_dim, self.num_actions, self.extra_dim = z_dim, h_dim, num_actions, extra_dim
        self.beliefs = (extra_dim > 0 and not lookahead and not sight) if beliefs is None else beliefs
        self.lookahead = lookahead
        self.sight = sight
        self.params = np.zeros(self.num_params, dtype=np.float32)

    @property
    def in_dim(self) -> int:
        return self.z_dim + self.h_dim + self.extra_dim

    @property
    def num_params(self) -> int:
        return (self.in_dim + 1) * self.num_actions

    def unflatten(self, params: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat vektor(er) -> (W, b). Støtter (num_params,) og (P, num_params)."""
        p = torch.as_tensor(params, dtype=torch.float32)
        lead = p.shape[:-1]
        n_w = self.in_dim * self.num_actions
        W = p[..., :n_w].reshape(*lead, self.in_dim, self.num_actions)
        b = p[..., n_w:].reshape(*lead, self.num_actions)
        return W, b

    def _inputs(self, z, h, extra):
        parts = [z, h] if extra is None or self.extra_dim == 0 else [z, h, extra]
        return torch.cat(parts, dim=-1)

    def logits(self, z: torch.Tensor, h: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """z: (B, z_dim), h: (B, h_dim), extra: (B, extra_dim) -> (B, num_actions)."""
        W, b = self.unflatten(self.params)
        return self._inputs(z, h, extra) @ W + b

    def act(self, z: torch.Tensor, h: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        return self.logits(z, h, extra).argmax(-1)

    def batched_logits(
        self, population: np.ndarray, z: torch.Tensor, h: torch.Tensor, extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        """population: (P, num_params), z: (P, B, z_dim), h: (P, B, h_dim) -> (P, B, num_actions)."""
        W, b = self.unflatten(population)
        return torch.einsum("pbi,pia->pba", self._inputs(z, h, extra), W) + b[:, None, :]

    def save(self, path, **extra) -> None:
        np.savez(
            path, params=self.params, z_dim=self.z_dim, h_dim=self.h_dim, num_actions=self.num_actions,
            extra_dim=self.extra_dim, beliefs=self.beliefs, lookahead=self.lookahead,
            sight=self.sight, **extra,
        )

    @classmethod
    def load(cls, path) -> "LinearController":
        with np.load(path, allow_pickle=False) as d:
            extra_dim = int(d["extra_dim"]) if "extra_dim" in d.files else 0
            beliefs = bool(d["beliefs"]) if "beliefs" in d.files else None
            lookahead = bool(d["lookahead"]) if "lookahead" in d.files else False
            sight = bool(d["sight"]) if "sight" in d.files else False
            c = cls(int(d["z_dim"]), int(d["h_dim"]), int(d["num_actions"]), extra_dim, beliefs, lookahead, sight)
            c.params = d["params"].astype(np.float32)
        return c
