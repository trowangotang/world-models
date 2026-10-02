"""Figurer uten matplotlib: SVG-diagrammer skrevet for hånd, og PNG-bilder av episoder.

Prosjektet avhenger bare av numpy og torch, så vi tegner selv. SVG er tekst, vises direkte på
GitHub og er lett å teste.
"""

from __future__ import annotations

from html import escape

import numpy as np

OUTCOME_COLORS = {"goal": "#3cb45a", "obstacle": "#dc3c3c", "truncated": "#9aa0a6"}
OUTCOME_LABELS = {"goal": "Mål", "obstacle": "Hindring", "truncated": "Avkortet"}
LINE_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf")
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def _svg(width: int, height: int, body: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        *body, "</svg>", ""])


def outcome_bars_svg(rows: list[tuple[str, dict]]) -> str:
    """Én liggende stabel per policy: andel mål, hindring og avkortet. rows: [(navn, summary)]."""
    label_w, bar_w, row_h, top = 300, 420, 30, 40
    height = top + len(rows) * row_h + 40
    body = [f'<text x="10" y="24" {FONT} font-size="15" font-weight="bold">Hvordan episodene endte ({rows[0][1]["episodes"]} like brett)</text>']
    for i, (name, s) in enumerate(rows):
        y = top + i * row_h
        body.append(f'<text x="{label_w - 8}" y="{y + 18}" {FONT} font-size="12" text-anchor="end">{escape(name)}</text>')
        x = label_w
        for o in ("goal", "obstacle", "truncated"):
            w = s[o]["rate"] * bar_w
            body.append(f'<rect x="{x:.1f}" y="{y + 4}" width="{w:.1f}" height="{row_h - 8}" fill="{OUTCOME_COLORS[o]}"/>')
            if w > 28:
                body.append(f'<text x="{x + w / 2:.1f}" y="{y + 19}" {FONT} font-size="11" fill="white" '
                            f'text-anchor="middle">{s[o]["rate"] * 100:.0f}</text>')
            x += w
    y = top + len(rows) * row_h + 12
    for j, o in enumerate(("goal", "obstacle", "truncated")):
        x = label_w + j * 110
        body.append(f'<rect x="{x}" y="{y}" width="12" height="12" fill="{OUTCOME_COLORS[o]}"/>')
        body.append(f'<text x="{x + 16}" y="{y + 11}" {FONT} font-size="12">{OUTCOME_LABELS[o]} (%)</text>')
    return _svg(label_w + bar_w + 20, height, body)


def goal_by_distance_svg(rows: list[tuple[str, dict]]) -> str:
    """Målrate (med 95 %-intervall) mot korteste vei til målet, én linje per policy."""
    left, top, w, h = 60, 40, 420, 260
    legend_x = left + w + 20
    bins = rows[0][1]["by_distance"]
    k = len(bins)
    xs = [left + (i + 0.5) * w / k for i in range(k)]
    body = [f'<text x="10" y="24" {FONT} font-size="15" font-weight="bold">Andel som når målet, etter avstand</text>']
    for t in range(0, 101, 20):
        y = top + h - t / 100 * h
        body.append(f'<line x1="{left}" y1="{y}" x2="{left + w}" y2="{y}" stroke="#e5e5e5"/>')
        body.append(f'<text x="{left - 6}" y="{y + 4}" {FONT} font-size="11" text-anchor="end">{t} %</text>')
    for x, b in zip(xs, bins):
        a, c = b["bin"]
        label = f"{a}–{c}" if c < 30 else f"{a}+"
        body.append(f'<text x="{x}" y="{top + h + 18}" {FONT} font-size="11" text-anchor="middle">{label}</text>')
    body.append(f'<text x="{left + w / 2}" y="{top + h + 36}" {FONT} font-size="12" text-anchor="middle">'
                f'Korteste vei til målet (skritt)</text>')
    for j, (name, s) in enumerate(rows):
        color = LINE_COLORS[j % len(LINE_COLORS)]
        pts = [(x, top + h - b["goal"]["rate"] * h) for x, b in zip(xs, s["by_distance"])]
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
                    + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
        for (x, y), b in zip(pts, s["by_distance"]):
            lo, hi = (top + h - b["goal"][q] * h for q in ("lo", "hi"))
            body.append(f'<line x1="{x:.1f}" y1="{lo:.1f}" x2="{x:.1f}" y2="{hi:.1f}" stroke="{color}" stroke-opacity="0.5"/>')
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        ly = top + j * 20
        body.append(f'<line x1="{legend_x}" y1="{ly + 6}" x2="{legend_x + 20}" y2="{ly + 6}" stroke="{color}" stroke-width="2"/>')
        body.append(f'<text x="{legend_x + 26}" y="{ly + 10}" {FONT} font-size="12">{escape(name)}</text>')
    return _svg(legend_x + 320, top + h + 50, body)


def calibration_svg(rows: list[tuple[str, dict]], event: str = "goal") -> str:
    """Drømmens sannsynlighet mot ekte rate. Punkter på diagonalen betyr en ærlig drøm."""
    left, top, size = 60, 40, 260
    title = {"goal": "nå målet", "obstacle": "treffe en hindring"}[event]
    body = [f'<text x="10" y="24" {FONT} font-size="15" font-weight="bold">Drøm mot virkelighet: {title} innen 10 skritt</text>']
    for t in range(0, 101, 20):
        y, x = top + size - t / 100 * size, left + t / 100 * size
        body.append(f'<line x1="{left}" y1="{y}" x2="{left + size}" y2="{y}" stroke="#e5e5e5"/>')
        body.append(f'<text x="{left - 6}" y="{y + 4}" {FONT} font-size="11" text-anchor="end">{t} %</text>')
        body.append(f'<text x="{x}" y="{top + size + 16}" {FONT} font-size="11" text-anchor="middle">{t} %</text>')
    body.append(f'<line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" stroke="#999" stroke-dasharray="4 3"/>')
    body.append(f'<text x="{left + size / 2}" y="{top + size + 34}" {FONT} font-size="12" text-anchor="middle">Drømmen sa</text>')
    body.append(f'<text x="16" y="{top + size / 2}" {FONT} font-size="12" text-anchor="middle" '
                f'transform="rotate(-90 16 {top + size / 2})">Det skjedde</text>')
    legend_x = left + size + 20
    for j, (name, d) in enumerate(rows):
        color = LINE_COLORS[j % len(LINE_COLORS)]
        pts = [(left + b["predicted"] * size, top + size - b["rate"] * size, b["count"]) for b in d[event]["calibration"]]
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="'
                    + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts) + '"/>')
        for x, y, cnt in pts:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2 + min(5, np.sqrt(cnt) / 6):.1f}" fill="{color}" fill-opacity="0.7"/>')
        ly = top + j * 20
        body.append(f'<circle cx="{legend_x + 8}" cy="{ly + 6}" r="4" fill="{color}"/>')
        body.append(f'<text x="{legend_x + 18}" y="{ly + 10}" {FONT} font-size="12">{escape(name)}</text>')
    return _svg(legend_x + 300, top + size + 46, body)


def path_overlay(first_frame: np.ndarray, path, cell_px: int = 8, scale: int = 4) -> np.ndarray:
    """Tegn agentens vei oppå startbildet. Streken går fra gul (start) til hvit (slutt)."""
    img = first_frame.repeat(scale, 0).repeat(scale, 1).astype(np.float32)
    s = cell_px * scale
    centre = [(r * s + s // 2, c * s + s // 2) for r, c in path]
    n = max(1, len(centre) - 1)
    for k, ((y0, x0), (y1, x1)) in enumerate(zip(centre[:-1], centre[1:])):
        t = k / n
        color = np.array([255, 220 + 35 * t, 40 + 215 * t])
        ya, yb, xa, xb = min(y0, y1) - 2, max(y0, y1) + 2, min(x0, x1) - 2, max(x0, x1) + 2
        img[ya:yb, xa:xb] = 0.25 * img[ya:yb, xa:xb] + 0.75 * color
    y, x = centre[-1]
    img[y - 5:y + 5, x - 5:x + 5] = (255, 255, 255)
    return img.clip(0, 255).astype(np.uint8)


def tile(images: list[np.ndarray], columns: int, pad: int = 4) -> np.ndarray:
    h, w, _ = images[0].shape
    rows = -(-len(images) // columns)
    out = np.full((rows * (h + pad) + pad, columns * (w + pad) + pad, 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, columns)
        out[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = im
    return out
