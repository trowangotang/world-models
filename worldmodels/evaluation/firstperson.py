"""Evaluering av førstepersonsagenten (steg 14) på 2000 nye brett.

    python -m worldmodels.evaluation.firstperson --out docs/evaluation_fp

Sammenligner agenten med tilfeldige handlinger, en håndlaget agent som bruker det samme kompasset og
fremsynet som controlleren får, og to jukseagenter som ser fasiten: samme regel som den håndlagde, og
korteste vei. Skriver results.json, report.md og en bildestripe av noen episoder (strips.png).
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from worldmodels.controller.features import world_features
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.env.gridworld import ACTIONS, CLOCKWISE, FORWARD, TURN_LEFT, TURN_RIGHT
from worldmodels.env.preview import save_png
from worldmodels.evaluation.stats import rate
from worldmodels.mdnrnn.model import COMPASS_SCALE, MDNRNN
from worldmodels.vae.dataset import to_tensor
from worldmodels.vae.model import ConvVAE


def toward_goal(forward: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Regelen begge de håndlagde agentene bruker: fram så lenge målet er foran, ellers snu mot det."""
    return np.where(forward >= 0.5, FORWARD, np.where(right > 0, TURN_RIGHT, TURN_LEFT))


class Policy:
    """Batch av handlinger fra bilder og miljøene (miljøene brukes bare av jukseagentene)."""

    def reset(self, n: int) -> None:
        pass

    def act(self, obs: np.ndarray, envs: list[GridDodgeEnv]) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, num_actions: int, seed: int = 0):
        self.num_actions, self.rng = num_actions, np.random.default_rng(seed)

    def act(self, obs, envs):
        return self.rng.integers(self.num_actions, size=len(envs))


class TruthRule(Policy):
    """Juks: kjenner målets plass og hindringen foran, og bruker regelen toward_goal. Uten målet i sikte
    går den fram og snur tilfeldig, som den håndlagde."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, envs):
        ego = np.array([e.egocentric(e.goal_pos) for e in envs], dtype=float)
        a = toward_goal(ego[:, 0], ego[:, 1])
        for i, e in enumerate(envs):
            dr, dc = ACTIONS[e.heading]
            if a[i] == FORWARD and (e.agent_pos[0] + dr, e.agent_pos[1] + dc) in e.danger():
                a[i] = self.rng.choice([TURN_LEFT, TURN_RIGHT])
        return a


def shortest_first_action(env: GridDodgeEnv) -> tuple[int, int]:
    """Bredde-først-søk over (celle, retning) med fram og snu. Gir (første handling, antall skritt)."""
    g = env.config.grid_size
    start = (env.agent_pos, env.heading)
    seen = {start: None}
    queue = deque([(start, None, 0)])
    while queue:
        (pos, h), first, n = queue.popleft()
        if pos == env.goal_pos:
            return (first if first is not None else FORWARD), n
        i = CLOCKWISE.index(h)
        nexts = [(FORWARD, (pos[0] + ACTIONS[h][0], pos[1] + ACTIONS[h][1]), h),
                 (TURN_LEFT, pos, CLOCKWISE[(i - 1) % 4]), (TURN_RIGHT, pos, CLOCKWISE[(i + 1) % 4])]
        for a, p, nh in nexts:
            if not (0 <= p[0] < g and 0 <= p[1] < g) or p in env.obstacles:
                continue
            if (p, nh) not in seen:
                seen[(p, nh)] = True
                queue.append(((p, nh), a if first is None else first, n + 1))
    return FORWARD, -1


class ShortestPath(Policy):
    """Juks: korteste vei med fram og snu, i den stille verdenen."""

    def act(self, obs, envs):
        return np.array([shortest_first_action(e)[0] for e in envs])


class CompassRule(Policy):
    """Håndlaget agent på verdensmodellens kompass og fremsyn, uten fasit. Viser hva en controller kan
    få ut av de samme inndataene. Har den ikke sett målet, går den fram og snur av og til."""

    def __init__(self, vae: ConvVAE, rnn: MDNRNN, seed: int = 0):
        self.vae, self.rnn, self.rng = vae.eval(), rnn.eval(), np.random.default_rng(seed)

    def reset(self, n):
        H = self.rnn.config.hidden_dim
        self.hidden = (torch.zeros(1, n, H), torch.zeros(1, n, H))

    @torch.no_grad()
    def act(self, obs, envs):
        n = len(obs)
        z, _ = self.vae.encode(to_tensor(obs))
        comp = self.rnn.compass_features(z, self.hidden[0][-1]).numpy()
        danger = self.rnn.lookahead_obstacle(z, self.hidden).numpy()
        explore = np.where(self.rng.random(n) < 0.3, self.rng.choice([TURN_LEFT, TURN_RIGHT], n), FORWARD)
        a = np.where(comp[:, 2] > 0.5, toward_goal(comp[:, 0] * COMPASS_SCALE, comp[:, 1] * COMPASS_SCALE), explore)
        blocked = (a == FORWARD) & (danger[:, FORWARD] > 0.5)
        a[blocked] = np.where(self.rng.random(blocked.sum()) < 0.5, TURN_LEFT, TURN_RIGHT)
        self.hidden = self.rnn(z.unsqueeze(1), torch.from_numpy(a).unsqueeze(1), self.hidden).hidden
        return a


class Learned(Policy):
    """Verdensmodell-agenten: V + M + C, ingen fasit."""

    def __init__(self, vae: ConvVAE, rnn: MDNRNN, controller: LinearController):
        from worldmodels.controller.agent import WorldModelAgent

        self.agent = WorldModelAgent(vae, rnn, controller)

    def reset(self, n):
        self.agent.reset(n)

    def act(self, obs, envs):
        return self.agent.act(obs)


def run(policy: Policy, n: int, seed: int, config: GridConfig, record: int = 0):
    envs = [GridDodgeEnv(config) for _ in range(n)]
    obs = np.stack([e.reset(seed=seed + i) for i, e in enumerate(envs)])
    shortest = np.array([shortest_first_action(e)[1] for e in envs])
    policy.reset(n)
    active = np.ones(n, bool)
    outcome = np.array(["truncated"] * n, dtype=object)
    returns, lengths = np.zeros(n), np.zeros(n, int)
    frames = [[(e.render_topdown(), o.copy())] for e, o in zip(envs[:record], obs[:record])]
    while active.any():
        a = policy.act(obs, envs)
        for i in np.flatnonzero(active):
            o, r, te, tr, info = envs[i].step(int(a[i]))
            obs[i] = o
            returns[i] += r
            lengths[i] += 1
            if i < record:
                frames[i].append((envs[i].render_topdown(), o.copy()))
            if te or tr:
                active[i] = False
                if te:
                    outcome[i] = info["event"]
    goal = outcome == "goal"
    summary = {
        "episodes": n,
        **{k: rate(outcome == k) for k in ("goal", "obstacle", "truncated")},
        "mean_return": float(returns.mean()),
        "mean_length": float(lengths.mean()),
        # Skritt på korteste vei (med snu) delt på skritt brukt, når målet ble nådd
        "path_efficiency": float((shortest[goal] / lengths[goal]).mean()) if goal.any() else None,
    }
    return summary, outcome, frames


def strips(frames, every: int = 2, max_frames: int = 10) -> np.ndarray:
    """Én rad per episode: ovenfra over, førsteperson under, for hvert annet skritt."""
    rows = []
    for ep in frames:
        pick = ep[::every][:max_frames]
        if ep[-1] is not pick[-1]:
            pick = pick[:-1] + [ep[-1]]
        pad = ((1, 1), (1, 1), (0, 0))
        cols = [np.concatenate([np.pad(t, pad, constant_values=255), np.pad(f, pad, constant_values=255)], 0)
                for t, f in pick]
        while len(cols) < max_frames:
            cols.append(np.full_like(cols[0], 255))
        rows.append(np.pad(np.concatenate(cols, 1), ((0, 6), (0, 0), (0, 0)), constant_values=255))
    return np.concatenate(rows, 0)


def report_md(results: dict, hparams: dict) -> str:
    lines = [
        "# Evaluering: førsteperson (steg 14)",
        "",
        f"{results['episodes']} nye brett (seed {hparams['seed']} og oppover), stille hindringer, maks 50 skritt. "
        "Å snu bruker et skritt. 95 %-intervaller i parentes.",
        "",
        "| Policy | Mål | Krasj | Avkortet | Avkastning | Effektivitet |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in results["policies"].items():
        g, o, t = r["goal"], r["obstacle"], r["truncated"]
        eff = "–" if r["path_efficiency"] is None else f"{r['path_efficiency']:.0%}"
        lines.append(
            f"| {name} | {g['rate']:.1%} ({g['lo']:.0%}–{g['hi']:.0%}) | {o['rate']:.1%} | {t['rate']:.1%} "
            f"| {r['mean_return']:+.2f} | {eff} |"
        )
    lines += ["", "Effektivitet: skritt på korteste vei delt på skritt brukt, for episodene som nådde målet.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluer førstepersonsagenten")
    p.add_argument("--vae", default="checkpoints/vae_fp.pt")
    p.add_argument("--rnn", default="checkpoints/mdnrnn_fp20k.pt")
    p.add_argument("--controller", default="checkpoints/controller_fp.npz")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=200_000)
    p.add_argument("--moving", type=int, default=0)
    p.add_argument("--out", default="docs/evaluation_fp")
    a = p.parse_args(argv)
    torch.set_num_threads(4)
    config = GridConfig(first_person=True, moving_obstacles=a.moving)
    vae, _ = ConvVAE.load(a.vae)
    rnn, _ = MDNRNN.load(a.rnn)
    controller = LinearController.load(a.controller)
    policies = {
        "Tilfeldig": RandomPolicy(4),
        "Tilfeldig uten rygging": RandomPolicy(3),
        "**Steg 14: verdensmodell-agenten**": Learned(vae, rnn, controller),
        "Håndlaget regel på kompass + fremsyn": CompassRule(vae, rnn),
        "*Juks: samme regel med fasit*": TruthRule(),
        "*Juks: korteste vei*": ShortestPath(),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"episodes": a.episodes, "policies": {}, "outcomes": {}}
    for name, pol in policies.items():
        record = 6 if pol.__class__ is Learned else 0
        summary, outcome, frames = run(pol, a.episodes, a.seed, config, record=record)
        results["policies"][name] = summary
        results["outcomes"][name] = outcome.tolist()
        print(f"{name:40s} mål {summary['goal']['rate']:.1%}  krasj {summary['obstacle']['rate']:.1%}  "
              f"avkortet {summary['truncated']['rate']:.1%}", flush=True)
        if record:
            save_png(np.kron(strips(frames), np.ones((2, 2, 1), np.uint8)), out / "strips.png")
    hparams = {k: v for k, v in vars(a).items()}
    (out / "results.json").write_text(json.dumps({"hparams": hparams, **results}, indent=1))
    (out / "report.md").write_text(report_md(results, hparams))
    print(f"Skrev {out}/report.md")


if __name__ == "__main__":
    main()
