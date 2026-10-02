"""Kjør hele verdensmodell-agenten (V + M + C) i det ekte miljøet.

I hvert skritt: kod bildet med VAE-en til z, la controlleren velge handling fra [z, h], og
oppdater RNN-ens minne med (z, handling). Mange miljøer kjøres side om side i én batch.
"""

from __future__ import annotations

import numpy as np
import torch

from worldmodels.controller.policy import LinearController
from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.dataset import to_tensor
from worldmodels.vae.model import ConvVAE


class WorldModelAgent:
    def __init__(self, vae: ConvVAE, rnn: MDNRNN, controller: LinearController):
        self.vae, self.rnn, self.controller = vae.eval(), rnn.eval(), controller
        self.hidden = None

    def reset(self, batch_size: int) -> None:
        H = self.rnn.config.hidden_dim
        self.hidden = (torch.zeros(1, batch_size, H), torch.zeros(1, batch_size, H))

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (B, 64, 64, 3) uint8 -> handlinger (B,). Oppdaterer minnet."""
        z, _ = self.vae.encode(to_tensor(obs))
        a = self.controller.act(z, self.hidden[0][-1])
        self.hidden = self.rnn(z.unsqueeze(1), a.unsqueeze(1), self.hidden).hidden
        return a.numpy()


def run_real_episodes(agent: WorldModelAgent, num_episodes: int, seed: int = 100_000, config: GridConfig | None = None) -> dict:
    """Spill num_episodes episoder parallelt. Seeds starter på `seed`, langt unna treningsdataene."""
    envs = [GridDodgeEnv(config) for _ in range(num_episodes)]
    obs = np.stack([env.reset(seed=seed + i) for i, env in enumerate(envs)])
    agent.reset(num_episodes)
    active = np.ones(num_episodes, bool)
    returns = np.zeros(num_episodes)
    lengths = np.zeros(num_episodes, int)
    outcome = np.array(["truncated"] * num_episodes, dtype=object)
    while active.any():
        actions = agent.act(obs)  # alle får en handling; ferdige episoder ignoreres
        for i in np.flatnonzero(active):
            o, r, terminated, truncated, info = envs[i].step(int(actions[i]))
            obs[i] = o
            returns[i] += r
            lengths[i] += 1
            if terminated or truncated:
                active[i] = False
                if terminated:
                    outcome[i] = info["event"]
    return {
        "episodes": num_episodes,
        "goal_rate": float((outcome == "goal").mean()),
        "obstacle_rate": float((outcome == "obstacle").mean()),
        "truncated_rate": float((outcome == "truncated").mean()),
        "mean_return": float(returns.mean()),
        "mean_length": float(lengths.mean()),
    }
