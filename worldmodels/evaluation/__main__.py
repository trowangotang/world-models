"""Kjør hele evalueringen og skriv rapport, tall og figurer.

    python -m worldmodels.evaluation --out docs/evaluation

Policyer med sjekkpunkter som mangler hoppes over, så kommandoen virker også med bare noen av dem.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from worldmodels.controller.agent import WorldModelAgent
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.env.preview import episode_grid, save_png
from worldmodels.evaluation import figures
from worldmodels.evaluation.beliefs import belief_accuracy
from worldmodels.evaluation.dream_check import dream_vs_real
from worldmodels.evaluation.policies import (
    GreedyGoalBaseline, GreedySafeBaseline, RandomBaseline, SafeRandomBaseline, ShortestPathBaseline, WorldModelPolicy,
)
from worldmodels.evaluation.report import write_report
from worldmodels.evaluation.run import evaluate_policy
from worldmodels.evaluation.stats import paired_difference, summarize
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.model import ConvVAE

# Nye seeds som ingen tidligere steg har brukt (trening 0–19 999, gammel evaluering 100 000+,
# innsamling 400 000–699 999, holdt-ute sett 900 000+).
EVAL_SEED = 200_000

# (navn, M, controller, diagnostikk). Diagnostikk er controllere vi har satt sammen for hånd for å
# finne flaskehalser; de er ikke lært og regnes ikke som et steg. Den siste lærte er sluttagenten.
AGENTS = (
    ("Steg 4: lært i drømmen", "mdnrnn_direct_k5.pt", "controller_s256.npz", False),
    ("Steg 4b: iterativ trening", "iter/mdnrnn_r3.pt", "iter/controller_r3.npz", False),
    ("Steg 4c: mål-bevisst drøm", "mdnrnn_pos.pt", "controller_curriculum_feat.npz", False),
    ("Steg 4d: bedre søk", "mdnrnn_pos.pt", "controller_cma_zh0.npz", False),
    ("Steg 4e: unngå hindringer", "mdnrnn_pos.pt", "controller_look_zh0.npz", False),
    ("Håndlaget på øyet (diagnostikk)", "mdnrnn_eye.pt", "controller_eye_handcrafted.npz", True),
    ("Steg 6: syn", "mdnrnn_eye.pt", "controller_eye_look.npz", False),
    ("Steg 7: drømmen ser", "mdnrnn_sense.pt", "controller_sense.npz", False),
    ("Steg 7 + sporing, håndlagt besøksvekt (diagnostikk)", "mdnrnn_sense.pt", "controller_track_handcrafted.npz", True),
    ("Steg 8: sporing", "mdnrnn_sense.pt", "controller_track.npz", False),
)

# Verdenen med bevegelige hindringer (steg 10, --moving): steg 8 slik den er, og agenten som er
# trent i en drøm der hindringene beveger seg.
MOVING_AGENTS = (
    ("Steg 8: sporing", "mdnrnn_sense.pt", "controller_track.npz", False),
    ("Steg 10: bevegelige hindringer", "mdnrnn_moving.pt", "controller_moving.npz", False),
    # Samme controller, men M leser hindringene og regner ut bevegelsen fra to bilder (D70)
    ("Steg 11: øyet leser hindringene", "mdnrnn_obstacles.pt", "controller_moving.npz", False),
)


def load_agents(checkpoints: Path, vae_name: str, log, agents=AGENTS) -> list[WorldModelPolicy]:
    vae, _ = ConvVAE.load(checkpoints / vae_name)
    out = []
    for name, rnn_name, ctrl_name, diagnostic in agents:
        rnn_path, ctrl_path = checkpoints / rnn_name, checkpoints / ctrl_name
        if not (rnn_path.exists() and ctrl_path.exists()):
            log(f"hopper over {name}: mangler {rnn_path} eller {ctrl_path}")
            continue
        rnn, _ = MDNRNN.load(rnn_path)
        policy = WorldModelPolicy(name, WorldModelAgent(vae, rnn, LinearController.load(ctrl_path)))
        policy.files = {"rnn": str(rnn_path), "controller": str(ctrl_path)}
        policy.diagnostic = diagnostic
        out.append(policy)
    return out


def final_name(agents) -> str | None:
    learned = [p.name for p in agents if not p.diagnostic]
    return learned[-1] if learned else None


def pick(records, outcome: str, k: int) -> list[int]:
    return [r.seed for r in records if r.outcome == outcome][:k]


def first_frame(seed: int, config: GridConfig | None = None) -> np.ndarray:
    return GridDodgeEnv(config).reset(seed=seed)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluer alle controllerne i det ekte miljøet (steg 5)")
    p.add_argument("--out", default="docs/evaluation")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--vae", default="vae_z32_w10.pt")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--dream-start", type=int, default=5, help="ekte skritt før drøm-mot-virkelighet-testen")
    p.add_argument("--moving", type=int, default=0, help="så mange hindringer beveger seg (steg 10)")
    a = p.parse_args(argv)
    config = GridConfig(moving_obstacles=a.moving)
    log = lambda msg: print(msg, flush=True)  # noqa: E731

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seed, a.seed + a.episodes))
    agents = load_agents(Path(a.checkpoints), a.vae, log, MOVING_AGENTS if a.moving else AGENTS)
    policies = [RandomBaseline(seed=a.seed), SafeRandomBaseline(seed=a.seed), GreedyGoalBaseline(seed=a.seed),
                GreedySafeBaseline(seed=a.seed), ShortestPathBaseline(), *agents]

    records, summaries = {}, {}
    for policy in policies:
        t0 = time.time()
        records[policy.name], _ = evaluate_policy(policy, seeds, config)
        summaries[policy.name] = summarize(records[policy.name])
        s = summaries[policy.name]
        log(f"{policy.name}: mål {s['goal']['rate']:.3f} hindring {s['obstacle']['rate']:.3f} "
            f"avkortet {s['truncated']['rate']:.3f} avkastning {s['return']['mean']:.3f}  [{time.time() - t0:.0f}s]")

    # Parvise sammenligninger på de samme brettene: hvert steg mot det forrige, og sluttagenten
    # mot grunnlinjene den bør måles mot.
    learned = [p.name for p in agents if not p.diagnostic]
    pairs = list(zip(learned[1:], learned[:-1]))
    if learned:
        pairs += [(learned[-1], RandomBaseline.name), (learned[-1], GreedySafeBaseline.name)]
        pairs += [(learned[-1], p.name) for p in agents if p.diagnostic]
    paired = [{"a": x, "b": y, **{o: paired_difference(records[x], records[y], o) for o in ("goal", "obstacle")}}
              for x, y in pairs]

    dream = {}
    for policy in agents:
        t0 = time.time()
        dream[policy.name] = dream_vs_real(policy.agent, seeds, start_step=a.dream_start, config=config)
        log(f"drøm mot virkelighet, {policy.name}: [{time.time() - t0:.0f}s]")

    beliefs = {p.name: b for p in agents if (b := belief_accuracy(p.agent, seeds, config)) is not None}

    figure_files = make_figures(out, policies, agents, records, summaries, dream, config)
    results = {
        "seeds": [seeds[0], seeds[-1]],
        "moving_obstacles": a.moving,
        "policies": [{"name": p.name, "cheat": p.cheat, "diagnostic": getattr(p, "diagnostic", False),
                      **getattr(p, "files", {})} for p in policies],
        "final": final_name(agents),
        "summaries": summaries,
        "paired": paired,
        "dream_vs_real": dream,
        "beliefs": beliefs,
        "episodes": {n: [[r.seed, r.outcome, r.length, round(r.episode_return, 2), r.shortest, r.oscillating]
                         for r in rs] for n, rs in records.items()},
    }
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False))
    write_report(out / "report.md", results, figure_files)
    log(f"Skrev {out}/report.md, results.json og {len(figure_files)} figurer")


def make_figures(out: Path, policies, agents, records, summaries, dream, config: GridConfig | None = None) -> dict:
    files = {}
    rows = [(p.name, summaries[p.name]) for p in policies]
    (out / "outcomes.svg").write_text(figures.outcome_bars_svg(rows))
    files["outcomes"] = "outcomes.svg"
    line_names = [RandomBaseline.name, GreedyGoalBaseline.name, GreedySafeBaseline.name] + [p.name for p in agents]
    (out / "goal_by_distance.svg").write_text(
        figures.goal_by_distance_svg([(n, summaries[n]) for n in line_names if n in summaries]))
    files["goal_by_distance"] = "goal_by_distance.svg"
    if not agents:
        return files

    for event in ("goal", "obstacle"):
        (out / f"dream_{event}.svg").write_text(figures.calibration_svg([(n, d) for n, d in dream.items()], event))
        files[f"dream_{event}"] = f"dream_{event}.svg"

    # Sluttagenten: fire episoder av hvert utfall, som vei oppå startbildet og som bildestripe.
    learned = [p for p in agents if not p.diagnostic]
    final = learned[-1]
    rs = records[final.name]
    chosen = {o: pick(rs, o, 4) for o in ("goal", "obstacle", "truncated")}
    by_seed = {r.seed: r for r in rs}
    overlays = [figures.path_overlay(first_frame(s, config), by_seed[s].path) for o in chosen for s in chosen[o]]
    save_png(figures.tile(overlays, columns=4), out / "final_paths.png")
    files["final_paths"] = "final_paths.png"
    strip_seeds = [s for o in chosen for s in chosen[o][:1]]
    _, frames = evaluate_policy(final, strip_seeds, config, keep_frames=strip_seeds)
    save_png(episode_grid([frames[s] for s in strip_seeds], frames=16), out / "final_strips.png")
    files["final_strips"] = "final_strips.png"
    files["chosen"] = chosen

    # Samme brett, to agenter: der forrige agent feilet og sluttagenten kom frem. Vi viser den
    # feilen sluttagenten oftest retter (krasj, eller avkortet som oftest betyr pendling).
    if len(learned) >= 2:
        prev = learned[-2]
        prev_by = {r.seed: r for r in records[prev.name]}
        fixed = {o: [s for s, r in by_seed.items() if r.outcome == "goal" and prev_by[s].outcome == o]
                 for o in ("obstacle", "truncated")}
        failure = max(fixed, key=lambda o: len(fixed[o]))
        flips = fixed[failure][:4]
        if flips:
            imgs = [figures.path_overlay(first_frame(s, config), prev_by[s].path) for s in flips]
            imgs += [figures.path_overlay(first_frame(s, config), by_seed[s].path) for s in flips]
            save_png(figures.tile(imgs, columns=len(flips)), out / "compare_paths.png")
            files["compare_paths"] = "compare_paths.png"
            files["compare"] = {"a": prev.name, "b": final.name, "seeds": flips, "failure": failure}
    return files


if __name__ == "__main__":
    main()
