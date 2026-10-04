"""Kjør hele verdensmodell-agenten (V + M + C) i det ekte miljøet.

I hvert skritt: kod bildet med VAE-en til z, la controlleren velge handling fra [z, h], og
oppdater RNN-ens minne med (z, handling). Mange miljøer kjøres side om side i én batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from worldmodels.controller.features import world_features
from worldmodels.controller.policy import LinearController
from worldmodels.data import Episode, save_episode
from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.dataset import to_tensor
from worldmodels.vae.model import ConvVAE


class WorldModelAgent:
    def __init__(self, vae: ConvVAE, rnn: MDNRNN, controller: LinearController):
        self.vae, self.rnn, self.controller = vae.eval(), rnn.eval(), controller
        self.hidden = None
        self.memory = None   # øyets hukommelse (mdnrnn/tracker.py)

    def reset(self, batch_size: int) -> None:
        H = self.rnn.config.hidden_dim
        self.hidden = (torch.zeros(1, batch_size, H), torch.zeros(1, batch_size, H))
        self.memory = None

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        """obs: (B, 64, 64, 3) uint8 -> handlinger (B,). Oppdaterer minnet.

        Med epsilon > 0 byttes hver handling ut med en tilfeldig en med den sannsynligheten.
        Minnet oppdateres alltid med handlingen som faktisk ble tatt."""
        z, _ = self.vae.encode(to_tensor(obs))
        h = self.hidden[0][-1]
        c = self.controller
        extra = None
        if c.extra_dim:
            extra, self.memory = world_features(self.rnn, z, self.hidden, c.beliefs, c.lookahead, c.sight,
                                                self.memory, track=c.track)
        a = self.controller.act(z, h, extra)
        if epsilon > 0:
            rng = rng if rng is not None else np.random.default_rng()
            explore = rng.random(len(a)) < epsilon
            random_actions = rng.integers(self.rnn.config.num_actions, size=len(a))
            a = torch.from_numpy(np.where(explore, random_actions, a.numpy()))
        self.hidden = self.rnn(z.unsqueeze(1), a.unsqueeze(1), self.hidden).hidden
        if self.memory is not None:
            self.memory = self.memory.moved(a)
        return a.numpy()


def play_episodes(
    agent: WorldModelAgent,
    num_episodes: int,
    seed: int = 100_000,
    epsilon: float = 0.0,
    rng_seed: int = 0,
    record: bool = False,
    config: GridConfig | None = None,
) -> tuple[dict, list[Episode]]:
    """Spill num_episodes episoder parallelt med env-seeds seed, seed + 1, ...

    Med record=True lagres hele episodene i samme format som de tilfeldige rolloutene,
    slik at de kan kodes og brukes til å trene MDN-RNN-en videre."""
    rng = np.random.default_rng(rng_seed)
    envs = [GridDodgeEnv(config) for _ in range(num_episodes)]
    obs = np.stack([env.reset(seed=seed + i) for i, env in enumerate(envs)])
    frames = [[o] for o in obs] if record else None
    steps = [[] for _ in range(num_episodes)]  # (handling, belønning, terminated, truncated)
    agent.reset(num_episodes)
    active = np.ones(num_episodes, bool)
    outcome = np.array(["truncated"] * num_episodes, dtype=object)
    while active.any():
        actions = agent.act(obs, epsilon=epsilon, rng=rng)  # alle får en handling; ferdige episoder ignoreres
        for i in np.flatnonzero(active):
            o, r, terminated, truncated, info = envs[i].step(int(actions[i]))
            obs[i] = o
            steps[i].append((int(actions[i]), r, terminated, truncated))
            if record:
                frames[i].append(o.copy())
            if terminated or truncated:
                active[i] = False
                if terminated:
                    outcome[i] = info["event"]
    returns = np.array([sum(s[1] for s in ep) for ep in steps])
    lengths = np.array([len(ep) for ep in steps])
    stats = {
        "episodes": num_episodes,
        "goal_rate": float((outcome == "goal").mean()),
        "obstacle_rate": float((outcome == "obstacle").mean()),
        "truncated_rate": float((outcome == "truncated").mean()),
        "mean_return": float(returns.mean()),
        "mean_length": float(lengths.mean()),
    }
    episodes = []
    if record:
        for f, ep in zip(frames, steps):
            a, r, te, tr = zip(*ep)
            episodes.append(Episode(
                obs=np.stack(f).astype(np.uint8),
                actions=np.asarray(a, dtype=np.int64),
                rewards=np.asarray(r, dtype=np.float32),
                terminated=np.asarray(te, dtype=bool),
                truncated=np.asarray(tr, dtype=bool),
            ))
    return stats, episodes


def run_real_episodes(agent: WorldModelAgent, num_episodes: int, seed: int = 100_000, config: GridConfig | None = None) -> dict:
    """Mål agenten uten utforsking. Seeds starter på `seed`, langt unna treningsdataene."""
    return play_episodes(agent, num_episodes, seed=seed, config=config)[0]


def collect_agent_rollouts(
    agent: WorldModelAgent,
    out_dir: str | Path,
    num_episodes: int,
    seed: int,
    epsilon: float = 0.3,
    batch_size: int = 500,
    config: GridConfig | None = None,
) -> dict:
    """Samle episoder med agenten (og litt utforsking) til out_dir, i biter for å spare minne."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"goal": 0.0, "obstacle": 0.0, "truncated": 0.0}
    for start in range(0, num_episodes, batch_size):
        n = min(batch_size, num_episodes - start)
        stats, episodes = play_episodes(agent, n, seed=seed + start, epsilon=epsilon, rng_seed=seed + start, record=True,
                                         config=config)
        for j, ep in enumerate(episodes):
            save_episode(ep, out / f"episode_{start + j:05d}.npz")
        for k in counts:
            counts[k] += stats[f"{k}_rate"] * n
    return {"episodes": num_episodes, **{f"{k}_rate": v / num_episodes for k, v in counts.items()}}
