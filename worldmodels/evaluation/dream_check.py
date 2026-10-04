"""Drøm mot virkelighet: stemmer det M spår, med det som faktisk skjer?

Controlleren ble trent på drømmens løfter. Her tester vi løftene på ekte brett:

    1. Spill `start_step` ekte skritt med agenten, så minnet vet hvor den er.
    2. Fra akkurat den tilstanden drømmer vi `horizon` skritt (temperatur 0, som i treningen):
       hvor sannsynlig tror M at agenten når målet, eller krasjer, i løpet av drømmen?
    3. Så spiller vi de samme `horizon` skrittene i det ekte miljøet og ser hva som skjedde.

Er drømmen ærlig, er snittet av M sine sannsynligheter nær den ekte raten.
"""

from __future__ import annotations

import numpy as np
import torch

from worldmodels.controller.dream import DreamConfig, WarmStarts, dream_fitness
from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.evaluation.stats import rate
from worldmodels.vae.dataset import to_tensor

CALIBRATION_BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)


@torch.no_grad()
def dream_vs_real(agent, seeds, start_step: int = 5, horizon: int = 10, config: GridConfig | None = None) -> dict:
    seeds = [int(s) for s in seeds]
    n = len(seeds)
    envs = [GridDodgeEnv(config) for _ in range(n)]
    obs = np.stack([env.reset(seed=s) for env, s in zip(envs, seeds)])
    agent.reset(n)
    active = np.ones(n, bool)

    def play(steps: int, events: np.ndarray | None = None):
        for _ in range(steps):
            if not active.any():
                return
            acts = agent.act(obs)
            for i in np.flatnonzero(active):
                o, _, terminated, truncated, info = envs[i].step(int(acts[i]))
                obs[i] = o
                if terminated or truncated:
                    active[i] = False
                    if events is not None and terminated:
                        events[i] = info["event"]

    play(start_step)
    alive = np.flatnonzero(active)
    if len(alive) == 0:
        return {"starts": 0}
    z, _ = agent.vae.encode(to_tensor(obs[alive]))
    h, c = agent.hidden
    memory = None if agent.memory is None else agent.memory.subset(alive).clone()
    starts = WarmStarts(z, h[:, alive].clone(), c[:, alive].clone(), alive, memory)
    ctrl = agent.controller
    d = dream_fitness(ctrl, ctrl.params[None], agent.rnn, starts, DreamConfig(horizon=horizon, temperature=0.0),
                      return_details=True)
    dream_goal, dream_obstacle = d["p_goal_per_start"][0], d["p_obstacle_per_start"][0]

    events = np.array(["none"] * n, dtype=object)
    play(horizon, events)
    real_goal = events[alive] == "goal"
    real_obstacle = events[alive] == "obstacle"
    return {
        "start_step": start_step, "horizon": horizon, "starts": int(len(alive)),
        "goal": _compare(dream_goal, real_goal),
        "obstacle": _compare(dream_obstacle, real_obstacle),
    }


def _compare(predicted: np.ndarray, actual: np.ndarray) -> dict:
    bins = []
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        m = (predicted >= lo) & (predicted < hi)
        if m.any():
            bins.append({"bin": [lo, min(hi, 1.0)], "count": int(m.sum()), "predicted": float(predicted[m].mean()),
                         **{k: v for k, v in rate(actual[m]).items() if k != "n"}})
    return {
        "dream": float(predicted.mean()),
        "real": rate(actual),
        "brier": float(((predicted - actual) ** 2).mean()),
        "auc": auc(predicted, actual),
        "calibration": bins,
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Sannsynligheten for at en tilfeldig positiv får høyere skår enn en tilfeldig negativ."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # Snittrang for like verdier
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
