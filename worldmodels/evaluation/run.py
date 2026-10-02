"""Kjør en policy på en liste med seeds og lagre én post per episode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.evaluation.policies import bfs_distances

# En avkortet episode regnes som pendling når agenten var innom høyst så mange ulike celler i de
# siste OSCILLATION_WINDOW skrittene (decisions.md D42 fant mønsteret i steg 4e).
OSCILLATION_WINDOW = 20
OSCILLATION_CELLS = 3


@dataclass
class EpisodeRecord:
    seed: int
    outcome: str                 # "goal", "obstacle" eller "truncated"
    episode_return: float
    length: int
    shortest: int                # korteste vei fra start til mål, rundt hindringene
    manhattan: int               # avstand uten hindringer
    obstacles_between: int       # hindringer i rektangelet mellom start og mål (på en mulig rett vei)
    unique_cells: int
    oscillating: bool
    path: list = field(repr=False)   # [(rad, kolonne), ...] fra start til slutt
    actions: list = field(repr=False)

    def to_dict(self) -> dict:
        return {**asdict(self), "path": [list(p) for p in self.path]}


def is_oscillating(path: list, window: int = OSCILLATION_WINDOW, cells: int = OSCILLATION_CELLS) -> bool:
    return len(path) > window and len(set(map(tuple, path[-window:]))) <= cells


def obstacles_between(env: GridDodgeEnv, a: tuple[int, int], g: tuple[int, int]) -> int:
    """Hindringer agenten kan møte om den går rett mot målet (i rektangelet mellom dem)."""
    r0, r1 = sorted((a[0], g[0]))
    c0, c1 = sorted((a[1], g[1]))
    return sum(r0 <= r <= r1 and c0 <= c <= c1 for r, c in env.obstacles)


def evaluate_policy(
    policy, seeds, config: GridConfig | None = None, keep_frames=(), batch_size: int = 1000,
) -> tuple[list[EpisodeRecord], dict[int, np.ndarray]]:
    """Spill én episode per seed. Returnerer poster og bildene for seedene i keep_frames."""
    seeds = [int(s) for s in seeds]
    keep = set(int(s) for s in keep_frames)
    records, frames = [], {}
    for start in range(0, len(seeds), batch_size):
        r, f = _run_batch(policy, seeds[start:start + batch_size], config, keep)
        records += r
        frames.update(f)
    return records, frames


def _run_batch(policy, seeds, config, keep):
    n = len(seeds)
    envs = [GridDodgeEnv(config) for _ in range(n)]
    obs = np.stack([env.reset(seed=s) for env, s in zip(envs, seeds)])
    starts = [(env.agent_pos, env.goal_pos) for env in envs]
    between = [obstacles_between(env, a, g) for env, (a, g) in zip(envs, starts)]
    shortest = [bfs_distances(env, a)[g] for env, (a, g) in zip(envs, starts)]
    paths = [[a] for a, _ in starts]
    actions = [[] for _ in range(n)]
    returns = np.zeros(n)
    outcome = ["truncated"] * n
    frames = {s: [obs[i].copy()] for i, s in enumerate(seeds) if s in keep}
    policy.reset(envs)
    active = np.ones(n, bool)
    while active.any():
        acts = policy.act(obs, envs)  # alle får en handling; ferdige episoder ignoreres
        for i in np.flatnonzero(active):
            o, r, terminated, truncated, info = envs[i].step(int(acts[i]))
            obs[i] = o
            returns[i] += r
            paths[i].append(envs[i].agent_pos)
            actions[i].append(int(acts[i]))
            if seeds[i] in frames:
                frames[seeds[i]].append(o.copy())
            if terminated or truncated:
                active[i] = False
                if terminated:
                    outcome[i] = info["event"]
    records = []
    for i, s in enumerate(seeds):
        (ar, ac), (gr, gc) = starts[i]
        records.append(EpisodeRecord(
            seed=s, outcome=outcome[i], episode_return=float(returns[i]), length=len(actions[i]),
            shortest=shortest[i], manhattan=abs(ar - gr) + abs(ac - gc), obstacles_between=between[i], unique_cells=len(set(paths[i])),
            oscillating=outcome[i] == "truncated" and is_oscillating(paths[i]),
            path=paths[i], actions=actions[i],
        ))
    return records, {s: np.stack(f) for s, f in frames.items()}
