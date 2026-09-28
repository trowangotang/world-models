"""Kod rollouts om til sekvenser av latente vektorer med en trent VAE.

MDN-RNN-en ser aldri bilder, bare z. Vi koder derfor alle episodene én gang og
lagrer resultatet i én .npz-fil, slik at RNN-treningen blir rask.

    python -m worldmodels.mdnrnn.encode --vae checkpoints/vae_z32_w10.pt \
        --data data/rollouts_20k --out data/zseq_20k.npz

Filen inneholder episodene etter hverandre ("flatet ut"):
    mu, logvar     (sum(T_i + 1), D)  z-fordelingen for hver observasjon
    agent_cell     (sum(T_i + 1),)    agentens celle i originalbildet (for evaluering)
    actions        (sum(T_i),)
    events         (sum(T_i),)        EVENT_MOVE / EVENT_GOAL / EVENT_OBSTACLE
    lengths        (N,)               T_i, antall skritt i episode i
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from worldmodels.data import Episode, episode_paths, load_episode
from worldmodels.env.parse import AGENT, find_cell, parse_cells
from worldmodels.vae.dataset import iterate_minibatches
from worldmodels.vae.model import ConvVAE

EVENT_MOVE, EVENT_GOAL, EVENT_OBSTACLE = 0, 1, 2
EVENT_NAMES = ("move", "goal", "obstacle")
NUM_EVENTS = 3


def episode_events(ep: Episode) -> np.ndarray:
    """Hendelse per skritt. Avkorting etter max_steps regnes som et vanlig skritt,
    fordi det ikke er noe som skjer i verden, bare en tidsgrense."""
    events = np.full(len(ep), EVENT_MOVE, dtype=np.int64)
    ended = ep.terminated
    events[ended & (ep.rewards > 0)] = EVENT_GOAL
    events[ended & (ep.rewards <= 0)] = EVENT_OBSTACLE
    return events


@dataclass
class ZSequences:
    mu: np.ndarray
    logvar: np.ndarray
    agent_cell: np.ndarray
    actions: np.ndarray
    events: np.ndarray
    lengths: np.ndarray

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def obs_offsets(self) -> np.ndarray:
        """Startindeks i mu/logvar/agent_cell for hver episode."""
        return np.concatenate([[0], np.cumsum(self.lengths + 1)[:-1]])

    @property
    def step_offsets(self) -> np.ndarray:
        """Startindeks i actions/events for hver episode."""
        return np.concatenate([[0], np.cumsum(self.lengths)[:-1]])

    def episode(self, i: int) -> dict[str, np.ndarray]:
        o, s, T = self.obs_offsets[i], self.step_offsets[i], self.lengths[i]
        return {
            "mu": self.mu[o:o + T + 1],
            "logvar": self.logvar[o:o + T + 1],
            "agent_cell": self.agent_cell[o:o + T + 1],
            "actions": self.actions[s:s + T],
            "events": self.events[s:s + T],
        }

    def subset(self, indices) -> "ZSequences":
        eps = [self.episode(i) for i in indices]
        return ZSequences(
            **{k: np.concatenate([e[k] for e in eps]) for k in ("mu", "logvar", "agent_cell", "actions", "events")},
            lengths=self.lengths[list(indices)].copy(),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: str | Path) -> "ZSequences":
        with np.load(path) as d:
            return cls(**{k: d[k] for k in ("mu", "logvar", "agent_cell", "actions", "events", "lengths")})

    @classmethod
    def concat(cls, parts: list["ZSequences"]) -> "ZSequences":
        return cls(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in cls.__dataclass_fields__})

    @classmethod
    def load_many(cls, paths) -> "ZSequences":
        """Last én eller flere filer og slå dem sammen (brukes i iterativ trening)."""
        if isinstance(paths, (str, Path)):
            return cls.load(paths)
        return cls.concat([cls.load(p) for p in paths])


@torch.no_grad()
def encode_episodes(vae: ConvVAE, paths: list[Path], batch_size: int = 512) -> ZSequences:
    vae.eval()
    parts = {k: [] for k in ("mu", "logvar", "agent_cell", "actions", "events")}
    lengths = []
    for p in paths:
        ep = load_episode(p)
        mus, logvars = [], []
        for x in iterate_minibatches(ep.obs, batch_size, shuffle=False):
            mu, logvar = vae.encode(x)
            mus.append(mu.numpy())
            logvars.append(logvar.numpy())
        parts["mu"].append(np.concatenate(mus))
        parts["logvar"].append(np.concatenate(logvars))
        parts["agent_cell"].append(find_cell(parse_cells(ep.obs), AGENT))
        parts["actions"].append(ep.actions)
        parts["events"].append(episode_events(ep))
        lengths.append(len(ep))
    return ZSequences(
        **{k: np.concatenate(v) for k, v in parts.items()},
        lengths=np.asarray(lengths, dtype=np.int64),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Kod rollouts til z-sekvenser")
    p.add_argument("--vae", default="checkpoints/vae_z32_w10.pt")
    p.add_argument("--data", default="data/rollouts_20k")
    p.add_argument("--out", default="data/zseq_20k.npz")
    args = p.parse_args(argv)

    vae, _ = ConvVAE.load(args.vae)
    seqs = encode_episodes(vae, episode_paths(args.data))
    seqs.save(args.out)
    print(f"Kodet {len(seqs)} episoder / {int(seqs.lengths.sum())} skritt til {args.out}")


if __name__ == "__main__":
    main()
