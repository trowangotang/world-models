"""Oppsummering med usikkerhet.

Rater får Wilson-intervall (fungerer også nær 0 % og 100 %). Snittavkastning og parvise
forskjeller får bootstrap-intervall. Alle intervaller er 95 %.
"""

from __future__ import annotations

import numpy as np

OUTCOMES = ("goal", "obstacle", "truncated")
# Korteste vei til målet, i skritt. Grensene gir omtrent like store grupper.
DISTANCE_BINS = ((1, 3), (4, 5), (6, 7), (8, 30))
Z95 = 1.959964


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def bootstrap_mean(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(len(v), size=(n_boot, len(v)))].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def rate(flags) -> dict:
    flags = np.asarray(flags, bool)
    lo, hi = wilson(int(flags.sum()), len(flags))
    return {"rate": float(flags.mean()) if len(flags) else float("nan"), "lo": lo, "hi": hi, "n": int(len(flags))}


def summarize(records) -> dict:
    outcome = np.array([r.outcome for r in records])
    returns = np.array([r.episode_return for r in records])
    shortest = np.array([r.shortest for r in records])
    between = np.array([r.obstacles_between for r in records])
    length = np.array([r.length for r in records])
    goal = outcome == "goal"
    truncated = outcome == "truncated"
    lo, hi = bootstrap_mean(returns)
    out = {
        "episodes": len(records),
        **{o: rate(outcome == o) for o in OUTCOMES},
        "return": {"mean": float(returns.mean()), "lo": lo, "hi": hi},
        "mean_length": float(length.mean()),
        # Hvor mange skritt agenten brukte per skritt på korteste vei, når den nådde målet.
        "path_efficiency": float((shortest[goal] / length[goal]).mean()) if goal.any() else None,
        "oscillating_of_truncated": rate([r.oscillating for r in records if r.outcome == "truncated"]),
        "by_distance": [],
        "by_blocking": {},
    }
    for a, b in DISTANCE_BINS:
        m = (shortest >= a) & (shortest <= b)
        out["by_distance"].append({"bin": [a, b], **{o: rate(outcome[m] == o) for o in OUTCOMES}})
    for name, m in (("free", between == 0), ("blocked", between > 0)):
        out["by_blocking"][name] = {o: rate(outcome[m] == o) for o in OUTCOMES}
    out["obstacle_step"] = _histogram([r.length for r in records if r.outcome == "obstacle"])
    out["truncated_with_oscillation"] = int(sum(r.oscillating for r in records))
    out["truncated_count"] = int(truncated.sum())
    return out


def _histogram(lengths, edges=(1, 2, 3, 5, 10, 20, 51)) -> list:
    counts, _ = np.histogram(lengths, bins=edges)
    return [{"steps": [int(a), int(b) - 1], "count": int(c)} for a, b, c in zip(edges[:-1], edges[1:], counts)]


def paired_difference(records_a, records_b, outcome: str = "goal", n_boot: int = 2000, seed: int = 0) -> dict:
    """Forskjell i rate (a - b) på de samme seedene, med bootstrap over seeds.

    Fordi begge spilte de samme brettene, er forskjellen mye skarpere enn to separate intervaller."""
    a = {r.seed: r.outcome == outcome for r in records_a}
    b = {r.seed: r.outcome == outcome for r in records_b}
    seeds = sorted(set(a) & set(b))
    d = np.array([float(a[s]) - float(b[s]) for s in seeds])
    lo, hi = bootstrap_mean(d, n_boot, seed)
    return {
        "diff": float(d.mean()), "lo": lo, "hi": hi, "n": len(seeds),
        "only_a": int((d > 0).sum()), "only_b": int((d < 0).sum()),
    }
