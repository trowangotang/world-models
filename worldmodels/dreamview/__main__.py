"""Lag siden som viser drømmen (steg 9).

    python -m worldmodels.dreamview --out docs/drom/index.html

Velger noen brett fra evalueringsbrettene, tar opp hva sluttagenten ser, tror og drømmer i hvert
skritt (record.py) og skriver én selvstendig HTML-fil med dataene inni (page.html er malen).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from worldmodels.controller.agent import WorldModelAgent
from worldmodels.controller.policy import LinearController
from worldmodels.dreamview.record import EpisodeRecord, record_episode
from worldmodels.env.preview import png_bytes
from worldmodels.evaluation.policies import WorldModelPolicy
from worldmodels.evaluation.run import evaluate_policy
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.model import ConvVAE

TEMPLATE = Path(__file__).with_name("page.html")
QUANT = 16   # bildene rundes til 16 nivåer per farge; de er uskarpe uansett, og filen blir 6x mindre

# (nøkkel, tittel, hvor mange) i den rekkefølgen de vises
CATEGORIES = (
    ("detour", "Omvei rundt hindringer", 4),
    ("fixed", "Steg 7 pendlet her", 3),
    ("straight", "Rett fram", 2),
    ("crash", "Krasj", 2),
)


def pick_boards(vae, rnn, step7, step8, seeds) -> list[tuple[int, str]]:
    """Velg brett per kategori, de første i seed-rekkefølge, så valget ikke er håndplukket."""
    new, _ = evaluate_policy(WorldModelPolicy("steg 8", WorldModelAgent(vae, rnn, step8)), seeds)
    old, _ = evaluate_policy(WorldModelPolicy("steg 7", WorldModelAgent(vae, rnn, step7)), seeds)
    old_by = {r.seed: r for r in old}

    def category(r) -> str | None:
        if r.outcome == "obstacle":
            return "crash"
        if r.outcome != "goal":
            return None
        if old_by[r.seed].oscillating:
            return "fixed"
        if r.obstacles_between and r.shortest > r.manhattan and r.length >= 6:
            return "detour"
        if not r.obstacles_between and r.length >= 4:
            return "straight"
        return None

    chosen = []
    for key, _, n in CATEGORIES:
        chosen += [(r.seed, key) for r in new if category(r) == key][:n]
    return chosen


def percent(x: np.ndarray) -> list[int]:
    return [int(v) for v in np.rint(np.asarray(x) * 100)]


def sheet(frames: list[np.ndarray]) -> str:
    """Alle bildene fra modellene for ett brett, stablet loddrett i én PNG."""
    img = np.concatenate(frames, axis=0)
    img = (img // QUANT * QUANT + QUANT // 2).astype(np.uint8)
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode()


def board_json(rec: EpisodeRecord, key: str) -> dict:
    frames: list[np.ndarray] = []

    def frame(img) -> int:
        frames.append(img)
        return len(frames) - 1

    steps = []
    for s in rec.steps:
        steps.append({
            "pos": list(s.pos), "recon": frame(s.recon), "eye": percent(s.eye_agent),
            "belief": percent(s.agent_belief), "goal": percent(s.goal_belief),
            "look": percent(s.lookahead), "visits": percent(s.visits),
            "logits": [round(float(x), 3) for x in s.logits], "action": s.action, "event": s.event,
            "dream": [{
                "a": d.action, "pg": round(d.p_goal, 3), "po": round(d.p_obstacle, 3), "alive": round(d.alive, 3),
                "f": frame(d.frame), "belief": percent(d.agent_belief),
                "rp": list(d.real_pos) if d.real_pos else None, "re": d.real_event,
            } for d in s.dream],
        })
    return {"seed": rec.seed, "category": key, "goal": list(rec.goal), "obstacles": [list(o) for o in rec.obstacles],
            "outcome": rec.outcome, "final": list(rec.final_pos), "sheet": sheet(frames), "steps": steps}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Lag siden som viser drømmen")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--rnn", default="mdnrnn_sense.pt")
    p.add_argument("--controller", default="controller_track.npz")
    p.add_argument("--previous", default="controller_sense.npz", help="steg 7, for å finne brettene den pendlet på")
    p.add_argument("--first-seed", type=int, default=200_000)
    p.add_argument("--candidates", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--out", default="docs/drom/index.html")
    p.add_argument("--bare", action="store_true", help="uten <html>/<head>, for verktøy som legger dem til selv")
    a = p.parse_args(argv)
    ck = Path(a.checkpoints)
    vae, _ = ConvVAE.load(ck / "vae_z32_w10.pt")
    rnn, _ = MDNRNN.load(ck / a.rnn)
    step8, step7 = LinearController.load(ck / a.controller), LinearController.load(ck / a.previous)
    boards = pick_boards(vae, rnn, step7, step8, range(a.first_seed, a.first_seed + a.candidates))
    data = {
        "categories": [{"key": k, "title": t} for k, t, _ in CATEGORIES],
        "horizon": a.horizon,
        "boards": [board_json(record_episode(vae, rnn, step8, seed, a.horizon), key) for seed, key in boards],
    }
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    if not a.bare:
        html = ('<!doctype html>\n<html lang="no">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                "</head>\n<body>\n" + html + "\n</body>\n</html>\n")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Skrev {out} ({len(html) / 1e6:.1f} MB, {len(data['boards'])} brett)")


if __name__ == "__main__":
    main()
