"""Hvor godt vet M hvor agenten og målet er, mens agenten spiller?

Controllerne fra steg 4d og 4e styrer nesten bare etter M sin tro om posisjonene (D32, D39).
Her sammenligner vi troen med fasiten i hvert ekte skritt. Troen controlleren bruker i skritt t
kommer fra minnet før bildet i skritt t er lest, slik agenten faktisk bruker den.
"""

from __future__ import annotations

import numpy as np
import torch

from worldmodels.env import GridConfig, GridDodgeEnv

REPORT_STEPS = (0, 1, 2, 5, 10)


@torch.no_grad()
def belief_accuracy(agent, seeds, config: GridConfig | None = None) -> dict | None:
    """Andel skritt der troen peker på riktig celle, og snittfeil i celler, per skritt."""
    rnn = agent.rnn
    if rnn.num_position_features == 0:
        return None
    side = int(round(rnn.config.grid_cells ** 0.5))
    seeds = [int(s) for s in seeds]
    n = len(seeds)
    envs = [GridDodgeEnv(config) for _ in range(n)]
    obs = np.stack([env.reset(seed=s) for env, s in zip(envs, seeds)])
    agent.reset(n)
    active = np.ones(n, bool)
    per_step = []  # (agent riktig, agent feil i celler, mål riktig, mål feil) for aktive episoder
    while active.any():
        f = rnn.position_features(agent.hidden[0][-1]).numpy()
        cells = np.rint((f + 1) / 2 * (side - 1)).astype(int)
        idx = np.flatnonzero(active)
        true_agent = np.array([envs[i].agent_pos for i in idx])
        true_goal = np.array([envs[i].goal_pos for i in idx])
        err_a = np.abs(cells[idx, :2] - true_agent).sum(1)
        err_g = np.abs(cells[idx, 2:] - true_goal).sum(1)
        per_step.append((err_a, err_g))
        acts = agent.act(obs)
        for i in idx:
            o, _, terminated, truncated, _ = envs[i].step(int(acts[i]))
            obs[i] = o
            if terminated or truncated:
                active[i] = False

    def row(err_a, err_g):
        return {"n": int(len(err_a)), "agent_exact": float((err_a == 0).mean()), "agent_error": float(err_a.mean()),
                "goal_exact": float((err_g == 0).mean()), "goal_error": float(err_g.mean())}

    out = {"by_step": {t: row(*per_step[t]) for t in REPORT_STEPS if t < len(per_step)}}
    out["all"] = row(np.concatenate([a for a, _ in per_step]), np.concatenate([g for _, g in per_step]))
    return out
