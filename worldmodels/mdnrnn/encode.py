"""Kod rollouts om til sekvenser av latente vektorer med en trent VAE.

MDN-RNN-en ser aldri bilder, bare z. Vi koder derfor alle episodene én gang og
lagrer resultatet i én .npz-fil, slik at RNN-treningen blir rask.

    python -m worldmodels.mdnrnn.encode --vae checkpoints/vae_z32_w10.pt \
        --data data/rollouts_20k --out data/zseq_20k.npz

Filen inneholder episodene etter hverandre ("flatet ut"):
    mu, logvar     (sum(T_i + 1), D)  z-fordelingen for hver observasjon
    agent_cell     (sum(T_i + 1),)    agentens celle i originalbildet (for evaluering)
    goal_cell      (sum(T_i + 1),)    målets celle (fast gjennom episoden, -1 i eldre filer)
    near_obstacle  (sum(T_i + 1), 4)  hindring i nabocellen over/under/venstre/høyre for agenten
                                      (1 = ja, 0 = nei eller kant, -1 = ukjent/eldre filer)
    danger         (sum(T_i + 1), 4)  krasj om agenten går den veien nå: hindring i nabocellen i dette
                                      bildet eller i neste (bevegelige hindringer, steg 10). I en stille
                                      verden er det det samme som near_obstacle.
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
from worldmodels.env.gridworld import ACTIONS
from worldmodels.env.parse import AGENT, GOAL, OBSTACLE, find_cell, parse_cells
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


PER_STEP_FIELDS = ("mu", "logvar", "agent_cell", "goal_cell", "near_obstacle", "danger", "actions", "events")


@dataclass
class ZSequences:
    mu: np.ndarray
    logvar: np.ndarray
    agent_cell: np.ndarray
    actions: np.ndarray
    events: np.ndarray
    lengths: np.ndarray
    goal_cell: np.ndarray
    near_obstacle: np.ndarray
    danger: np.ndarray

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
            "goal_cell": self.goal_cell[o:o + T + 1],
            "near_obstacle": self.near_obstacle[o:o + T + 1],
            "danger": self.danger[o:o + T + 1],
            "actions": self.actions[s:s + T],
            "events": self.events[s:s + T],
        }

    def subset(self, indices) -> "ZSequences":
        eps = [self.episode(i) for i in indices]
        return ZSequences(
            **{k: np.concatenate([e[k] for e in eps]) for k in PER_STEP_FIELDS},
            lengths=self.lengths[list(indices)].copy(),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: str | Path) -> "ZSequences":
        with np.load(path) as d:
            fields = {k: d[k] for k in d.files}
        # Filer kodet før målets celle ble lagret: ukjent overalt
        fields.setdefault("goal_cell", np.full(len(fields["agent_cell"]), -1, dtype=np.int64))
        fields.setdefault("near_obstacle", np.full((len(fields["agent_cell"]), len(ACTIONS)), -1, dtype=np.int8))
        # Filer fra den stille verdenen: hindringene flytter seg ikke, så faren er det som står ved siden av
        fields.setdefault("danger", fields["near_obstacle"].copy())
        return cls(**fields)

    @classmethod
    def concat(cls, parts: list["ZSequences"]) -> "ZSequences":
        return cls(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in cls.__dataclass_fields__})

    @classmethod
    def load_many(cls, paths) -> "ZSequences":
        """Last én eller flere filer og slå dem sammen (brukes i iterativ trening)."""
        if isinstance(paths, (str, Path)):
            return cls.load(paths)
        return cls.concat([cls.load(p) for p in paths])


def near_obstacles(obstacle_map: np.ndarray, agent_cells: np.ndarray) -> np.ndarray:
    """(G, G) bool-kart og (N,) agentceller -> (N, 4) int8: hindring i nabocellen i hver retning.

    Kanten teller ikke som hindring (agenten blir bare stående). Ukjent agentcelle gir -1.
    """
    G = obstacle_map.shape[0]
    out = np.full((len(agent_cells), len(ACTIONS)), -1, dtype=np.int8)
    known = agent_cells >= 0
    r, c = agent_cells[known] // G, agent_cells[known] % G
    for k, (dr, dc) in enumerate(ACTIONS):
        nr, nc = r + dr, c + dc
        inside = (nr >= 0) & (nr < G) & (nc >= 0) & (nc < G)
        hit = np.zeros(len(r), dtype=bool)
        hit[inside] = obstacle_map[nr[inside], nc[inside]]
        out[known, k] = hit
    return out


@torch.no_grad()
def encode_episodes(vae: ConvVAE, paths: list[Path], batch_size: int = 512) -> ZSequences:
    vae.eval()
    parts = {k: [] for k in PER_STEP_FIELDS}
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
        cells = parse_cells(ep.obs)
        parts["agent_cell"].append(find_cell(cells, AGENT))
        # Målet flytter seg aldri, men skjules når agenten står på det. Bruk første bilde.
        parts["goal_cell"].append(np.full(len(ep.obs), find_cell(cells[:1], GOAL)[0], dtype=np.int64))
        # Hindringene leses fra hvert bilde, siden noen kan bevege seg (steg 10). Agenten tegnes oppå
        # det den står på, så i en stille verden er første bilde tryggest: der står ingen oppå noe.
        agent = parts["agent_cell"][-1]
        maps = cells == OBSTACLE
        if (maps == maps[:1]).all() or len(maps) == 1:
            near = near_obstacles(maps[0], agent)
            danger = near
        else:
            near = np.stack([near_obstacles(maps[t], agent[t:t + 1])[0] for t in range(len(maps))])
            after = np.stack([near_obstacles(maps[min(t + 1, len(maps) - 1)], agent[t:t + 1])[0] for t in range(len(maps))])
            danger = np.where((near < 0) | (after < 0), -1, near | after).astype(np.int8)
            # Krasjer agenten, står den oppå hindringen i neste bilde og skjuler den. Hendelsen sier fra.
            crash = np.flatnonzero(episode_events(ep) == EVENT_OBSTACLE)
            danger[crash, ep.actions[crash]] = 1
        parts["near_obstacle"].append(near)
        parts["danger"].append(danger)
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
