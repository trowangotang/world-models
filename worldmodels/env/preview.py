"""Lag et forhåndsvisningsbilde (PNG) av noen episoder, uten eksterne bildepakker.

    python -m worldmodels.env.preview --out preview.png --episodes 4 --frames 10

Hver rad er én episode med tilfeldig policy, hver kolonne ett tidssteg.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np


def save_png(img: np.ndarray, path: str | Path) -> None:
    """Skriv et uint8 RGB-bilde (H, W, 3) som PNG med bare zlib."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Forventer uint8-bilde med form (H, W, 3)")
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))  # filtertype 0 per linje

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def episode_grid(episodes: list[np.ndarray], frames: int, scale: int = 2, pad: int = 2) -> np.ndarray:
    """Sett sammen obs-sekvenser til ett bilde. Korte episoder fylles med grått."""
    h, w, _ = episodes[0].shape[1:]
    H, W = h * scale, w * scale
    grid = np.full(
        (len(episodes) * (H + pad) + pad, frames * (W + pad) + pad, 3), 255, dtype=np.uint8
    )
    for i, obs in enumerate(episodes):
        for t in range(frames):
            y, x = pad + i * (H + pad), pad + t * (W + pad)
            if t < len(obs):
                grid[y:y + H, x:x + W] = obs[t].repeat(scale, 0).repeat(scale, 1)
            else:
                grid[y:y + H, x:x + W] = 200
    return grid


def main(argv: list[str] | None = None) -> None:
    from worldmodels.data.rollouts import RandomPolicy, run_episode
    from worldmodels.env import GridDodgeEnv

    p = argparse.ArgumentParser(description="Forhåndsvis GridDodge-episoder som PNG")
    p.add_argument("--out", default="preview.png")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    env = GridDodgeEnv()
    policy = RandomPolicy(seed=args.seed)
    eps = [run_episode(env, policy, seed=args.seed + i).obs for i in range(args.episodes)]
    save_png(episode_grid(eps, args.frames), args.out)
    print(f"Skrev {args.out}")


if __name__ == "__main__":
    main()
