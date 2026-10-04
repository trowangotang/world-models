"""Hindringsøyet: leser hindringene fra z og regner ut hvor de er på vei fra to bilder (steg 11).

    python -m worldmodels.mdnrnn.obstacles --rnn checkpoints/mdnrnn_moving.pt \\
        --data data/moving/zseq_agent.npz --out checkpoints/mdnrnn_obstacles.pt

Nærsynet med minne (steg 10, D66) må lese hvor hindringene er på vei fra M sitt minne h, og ser bare
28 % av dem som er på vei inn. Her gjør vi det i to enkle trinn i stedet (D70):

1. **ObstacleEye** leser et kart over hindringene fra z, formet som øyet fra steg 6.
2. **Motion** spår neste kart fra kartet i forrige og dette bildet. En hindring som har flyttet seg,
   fortsetter, og snur bare ved noe den kan se like ved. Tre lag 3x3 ser langt nok.

Fare i en retning er hindring i nabocellen nå eller i det spådde neste kartet. Den erstatter
nærsynets hindringsdel; nærsynet spår fortsatt målet. Begge trenes alene med resten av M fryst.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from worldmodels.env.gridworld import ACTIONS


class ObstacleEye(nn.Module):
    """z (B, D) -> logits (B, side*side): står det en hindring i cellen?"""

    def __init__(self, latent_dim: int = 32, channels: int = 32, side: int = 8):
        super().__init__()
        self.channels, self.side = channels, side
        self.fc = nn.Linear(latent_dim, channels * (side // 2) ** 2)
        self.net = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Conv2d(channels, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        half = self.side // 2
        return self.net(self.fc(z).view(-1, self.channels, half, half)).flatten(1)


class Motion(nn.Module):
    """(forrige kart, dette kartet) (B, 2, side*side) i [0, 1] -> logits for neste kart (B, side*side)."""

    def __init__(self, channels: int = 32, side: int = 8):
        super().__init__()
        self.side = side
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.net(maps.view(-1, 2, self.side, self.side)).flatten(1)


def danger_by_direction(now: torch.Tensor, after: torch.Tensor, agent_map: torch.Tensor, side: int = 8) -> torch.Tensor:
    """Kart nå og etter neste skritt (B, celler) og troen om agenten (B, celler) -> (B, 4)
    sannsynlighet for krasj om agenten går hver vei. Kanten er ingen fare (agenten blir stående)."""
    danger = torch.maximum(now, after).view(-1, side, side)
    padded = F.pad(danger, (1, 1, 1, 1))
    agent = agent_map.view(-1, side, side)
    out = []
    for dr, dc in ACTIONS:
        neighbour = padded[:, 1 + dr:1 + dr + side, 1 + dc:1 + dc + side]   # fare i cellen ved siden av
        out.append((agent * neighbour).sum((-2, -1)))
    return torch.stack(out, -1).clamp(0, 1)


def previous_maps(maps: torch.Tensor, offsets: np.ndarray) -> torch.Tensor:
    """Kartet i forrige bilde. I første bilde i en episode: det samme bildet (ingen bevegelse sett)."""
    prev = maps.clone()
    prev[1:] = maps[:-1]
    prev[offsets] = maps[offsets]
    return prev


def fit(net: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epochs: int, lr: float = 1e-3,
        batch: int = 512, seed: int = 0, log=print) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr)
    rng = np.random.default_rng(seed)
    losses = []
    net.train()
    for epoch in range(epochs):
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(len(targets)))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.binary_cross_entropy_with_logits(net(inputs[b]), targets[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        losses.append(total / len(perm))
        log(f"  epoke {epoch}: tap {losses[-1]:.4f}  [{time.time() - t0:.0f}s]")
    net.eval()
    return losses


@torch.no_grad()
def probabilities(net: nn.Module, x: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    return torch.cat([torch.sigmoid(net(x[i:i + batch])) for i in range(0, len(x), batch)])


@torch.no_grad()
def danger_recall(model, seqs) -> dict:
    """Treff på hindringer som står i nabocellen nå og som er på vei inn, og falske alarmer."""
    last = np.zeros(len(seqs.mu), bool)
    last[seqs.obs_offsets + seqs.lengths] = True
    keep = ~last & (seqs.danger >= 0).all(1) & (seqs.agent_cell >= 0)
    z = torch.from_numpy(seqs.mu).float()
    now = probabilities(model.obstacle_eye, z)
    after = probabilities(model.motion, torch.stack([previous_maps(now, seqs.obs_offsets), now], 1))
    agent = F.one_hot(torch.from_numpy(np.maximum(seqs.agent_cell, 0)), model.config.grid_cells).float()
    hit = danger_by_direction(now, after, agent)[torch.from_numpy(keep)] > 0.5
    near = torch.from_numpy(seqs.near_obstacle[keep] == 1)
    danger = torch.from_numpy(seqs.danger[keep] == 1)
    arriving = danger & ~near
    rate = lambda a, b: float((a & b).sum() / max(1, b.sum()))  # noqa: E731
    return {"present": rate(hit, near), "arriving": rate(hit, arriving), "false_alarm": rate(hit, ~danger)}


def train_obstacles(model, seqs, val_idx, eye_epochs: int = 8, motion_epochs: int = 20, log=print) -> dict:
    """Tren øyet på kartene i dataene, så bevegelsen på øyets egne kart (så den lærer å leve med feilene)."""
    known = (seqs.obstacles >= 0).all(1)
    episode = np.repeat(np.arange(len(seqs)), seqs.lengths + 1)
    is_val = np.isin(episode, val_idx)
    z = torch.from_numpy(seqs.mu).float()
    maps = torch.from_numpy(np.maximum(seqs.obstacles, 0)).float()

    log("Hindringsøyet:")
    train = torch.from_numpy(known & ~is_val)
    eye_loss = fit(model.obstacle_eye, z[train], maps[train], eye_epochs, log=log)
    seen = probabilities(model.obstacle_eye, z)

    log("Bevegelsen:")
    last = np.zeros(len(maps), bool)
    last[seqs.obs_offsets + seqs.lengths] = True
    nxt = torch.zeros_like(maps)
    nxt[:-1] = maps[1:]
    next_known = np.zeros(len(maps), bool)
    next_known[:-1] = known[1:]
    use = torch.from_numpy(~last & known & next_known & ~is_val)
    pairs = torch.stack([previous_maps(seen, seqs.obs_offsets), seen], 1)
    motion_loss = fit(model.motion, pairs[use], nxt[use], motion_epochs, log=log)

    val_maps = torch.from_numpy(known & is_val)
    hit, true = seen[val_maps] > 0.5, maps[val_maps].bool()
    metrics = {"eye_found": float((hit & true).sum() / max(1, true.sum())),
               "eye_cells": float((hit == true).float().mean()),
               "danger": danger_recall(model, seqs.subset(val_idx)),
               "eye_loss": eye_loss, "motion_loss": motion_loss}
    log(f"Validering: øyet finner {metrics['eye_found']:.1%} av hindringene, fare: står der nå "
        f"{metrics['danger']['present']:.1%}, på vei inn {metrics['danger']['arriving']:.1%}, "
        f"falsk alarm {metrics['danger']['false_alarm']:.2%}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    from worldmodels.mdnrnn.encode import ZSequences
    from worldmodels.mdnrnn.model import MDNRNN
    from worldmodels.mdnrnn.train import split_indices

    p = argparse.ArgumentParser(description="Tren hindringsøyet og bevegelsen (steg 11)")
    p.add_argument("--rnn", default="checkpoints/mdnrnn_moving.pt")
    p.add_argument("--data", nargs="+", default=["data/moving/zseq_agent.npz"],
                   help="z-sekvenser med hindringskart (kodet med steg 11 eller nyere)")
    p.add_argument("--out", default="checkpoints/mdnrnn_obstacles.pt")
    p.add_argument("--eye-epochs", type=int, default=8)
    p.add_argument("--motion-epochs", type=int, default=20)
    a = p.parse_args(argv)

    rnn, ckpt = MDNRNN.load(a.rnn)
    seqs = ZSequences.load_many(a.data)
    if (seqs.obstacles < 0).all():
        raise ValueError("dataene mangler hindringskart: kod dem på nytt med python -m worldmodels.mdnrnn.encode")
    _, val_idx = split_indices(len(seqs), seed=0)
    model = MDNRNN(replace(rnn.config, obstacle_eye=True))
    model.load_state_dict(rnn.state_dict(), strict=False)
    model.eval()
    torch.manual_seed(0)
    metrics = train_obstacles(model, seqs, val_idx, a.eye_epochs, a.motion_epochs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    extra = {k: v for k, v in ckpt.items() if k not in ("config", "state_dict")}
    model.save(a.out, **extra, obstacle_metrics=metrics, obstacles_from=a.rnn, obstacle_data=a.data)
    print(f"Skrev {a.out}")


if __name__ == "__main__":
    main()
