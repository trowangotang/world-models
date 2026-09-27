"""Bildedata for VAE-en, hentet fra rollouts fra steg 1.

Vi deler i trening/validering per *episode*, ikke per bilde. Bilder fra samme
episode er nesten like (samme layout, agenten har flyttet seg én rute), så en
tilfeldig bildedeling ville lekket layouter inn i valideringssettet og gitt
for optimistiske tall.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from worldmodels.data import episode_paths, load_episode


@dataclass
class FrameSplit:
    train: np.ndarray   # (N, 64, 64, 3) uint8
    val: np.ndarray


def load_frames(paths: list[Path]) -> np.ndarray:
    """Alle observasjoner (inkludert siste i hver episode) stablet, uint8."""
    return np.concatenate([load_episode(p).obs for p in paths], axis=0)


def split_episodes(paths: list[Path], val_fraction: float = 0.1, seed: int = 0) -> tuple[list[Path], list[Path]]:
    if not paths:
        raise ValueError("Fant ingen episoder")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n_val = max(1, int(round(len(paths) * val_fraction)))
    val = sorted(paths[i] for i in order[:n_val])
    train = sorted(paths[i] for i in order[n_val:])
    return train, val


def load_split(data_dir: str | Path, val_fraction: float = 0.1, seed: int = 0) -> FrameSplit:
    train_paths, val_paths = split_episodes(episode_paths(data_dir), val_fraction, seed)
    return FrameSplit(train=load_frames(train_paths), val=load_frames(val_paths))


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 i [0, 1]."""
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)


def iterate_minibatches(frames: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator | None = None):
    """Gi tensor-batcher uten å konvertere hele datasettet til float på en gang."""
    idx = np.arange(len(frames))
    if shuffle:
        (rng or np.random.default_rng()).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield to_tensor(frames[idx[start:start + batch_size]])
