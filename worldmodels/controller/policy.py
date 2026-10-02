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
    def __init__(self, z_dim: int, h_dim: int, num_actions: int = 4):
        self.z_dim, self.h_dim, self.num_actions = z_dim, h_dim, num_actions
        self.params = np.zeros(self.num_params, dtype=np.float32)

    @property
    def in_dim(self) -> int:
        return self.z_dim + self.h_dim

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

    def logits(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """z: (B, z_dim), h: (B, h_dim) -> (B, num_actions)."""
        W, b = self.unflatten(self.params)
        return torch.cat([z, h], dim=-1) @ W + b

    def act(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.logits(z, h).argmax(-1)

    def batched_logits(self, population: np.ndarray, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """population: (P, num_params), z: (P, B, z_dim), h: (P, B, h_dim) -> (P, B, num_actions)."""
        W, b = self.unflatten(population)
        return torch.einsum("pbi,pia->pba", torch.cat([z, h], dim=-1), W) + b[:, None, :]

    def save(self, path, **extra) -> None:
        np.savez(path, params=self.params, z_dim=self.z_dim, h_dim=self.h_dim, num_actions=self.num_actions, **extra)

    @classmethod
    def load(cls, path) -> "LinearController":
        with np.load(path, allow_pickle=False) as d:
            c = cls(int(d["z_dim"]), int(d["h_dim"]), int(d["num_actions"]))
            c.params = d["params"].astype(np.float32)
        return c
