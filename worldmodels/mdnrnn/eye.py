"""Øyet: leser agentens og målets celle fra z, altså fra bildet agenten ser akkurat nå.

    python -m worldmodels.mdnrnn.eye --rnn checkpoints/mdnrnn_pos.pt --data data/v3/zseq_*.npz \\
        --out checkpoints/mdnrnn_eye.pt

Steg 5 viste at M bare vet hvor målet er i 26 % av skrittene, og at troen fra minnet h er tom i
første skritt (decisions.md D46). Men VAE-dekoderen tegner målet riktig i 81 % av bildene, så
informasjonen finnes i z. Den er bare viklet inn: en lineær leser finner målet i 2 %, et MLP i 66 %.

Øyet er formet som starten på en dekoder: z blir et lite 4x4-kart, skaleres opp til 8x8 (én
verdi per celle i rutenettet), og hver celle får en logit for "agenten er her" og "målet er her".
Romlig struktur gjør det mye lettere å lære (agent 97 %, mål 90 %, D48).

Øyet trenes alene, med M fryst. Det blir en del av M-sjekkpunktet, slik at controlleren, drømmen
og agenten finner det der de allerede finner M.
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


# Hvert bilde gir målet log(p + dette) per celle. Gulvet hindrer at ett bilde med p ~ 0 for den
# riktige cellen veier mer enn mange bilder som peker riktig (decisions.md D49).
GOAL_EVIDENCE_FLOOR = 1e-3


class SpatialEye(nn.Module):
    def __init__(self, latent_dim: int = 32, channels: int = 32, side: int = 8):
        super().__init__()
        self.channels, self.side = channels, side
        self.fc = nn.Linear(latent_dim, channels * (side // 2) ** 2)
        self.net = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Conv2d(channels, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., D) -> logits (..., 2, side*side) for [agent, mål]."""
        lead = z.shape[:-1]
        half = self.side // 2
        x = self.fc(z.reshape(-1, z.shape[-1])).view(-1, self.channels, half, half)
        return self.net(x).flatten(2).view(*lead, 2, self.side * self.side)


def expected_positions(logits: torch.Tensor, side: int) -> torch.Tensor:
    """(..., 2, side*side) logits -> (..., 4) forventet [agent rad, agent kol, mål rad, mål kol] i [-1, 1]."""
    p = F.softmax(logits, dim=-1).view(*logits.shape[:-1], side, side)
    coords = torch.linspace(-1, 1, side, device=logits.device)
    rows = (p.sum(-1) * coords).sum(-1)
    cols = (p.sum(-2) * coords).sum(-1)
    return torch.stack([rows[..., 0], cols[..., 0], rows[..., 1], cols[..., 1]], dim=-1)


def eye_data(seqs, train_fraction: float = 0.9):
    """(z, celler) for alle bilder med kjent agent og mål, delt på episoder i trening og validering."""
    episode = np.repeat(np.arange(len(seqs)), seqs.lengths + 1)
    cells = np.stack([seqs.agent_cell, seqs.goal_cell], 1)
    known = (cells >= 0).all(1)
    is_train = episode < int(train_fraction * len(seqs))
    z = torch.from_numpy(seqs.mu).float()
    y = torch.from_numpy(cells).long()
    return (z[known & is_train], y[known & is_train]), (z[known & ~is_train], y[known & ~is_train])


@torch.no_grad()
def eye_accuracy(eye: SpatialEye, z: torch.Tensor, y: torch.Tensor, batch: int = 8192) -> tuple[float, float]:
    pred = torch.cat([eye(z[i:i + batch]).argmax(-1) for i in range(0, len(z), batch)])
    acc = (pred == y).float().mean(0)
    return float(acc[0]), float(acc[1])


def train_eye(eye: SpatialEye, train, val, epochs: int = 8, lr: float = 1e-3, batch: int = 512, seed: int = 0,
              log=lambda msg: print(msg, flush=True)) -> list[dict]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(eye.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    z, y = train
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        eye.train()
        perm = torch.from_numpy(rng.permutation(len(z)))
        total = 0.0
        for i in range(0, len(z), batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(eye(z[b]).reshape(-1, eye.side ** 2), y[b].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        eye.eval()
        agent, goal = eye_accuracy(eye, *val)
        history.append({"epoch": epoch, "loss": total / len(z), "val_agent": agent, "val_goal": goal})
        log(f"epoke {epoch}: tap {total / len(z):.4f}, validering agent {agent:.3f} mål {goal:.3f}  [{time.time() - t0:.0f}s]")
    return history


def main(argv: list[str] | None = None) -> None:
    from worldmodels.mdnrnn.encode import ZSequences
    from worldmodels.mdnrnn.model import MDNRNN

    p = argparse.ArgumentParser(description="Tren øyet som leser posisjonene fra z, og legg det til M")
    p.add_argument("--rnn", default="checkpoints/mdnrnn_pos.pt")
    p.add_argument("--data", nargs="+", default=["data/v3/zseq_20k.npz"])
    p.add_argument("--out", default="checkpoints/mdnrnn_eye.pt")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args(argv)

    rnn, ckpt = MDNRNN.load(a.rnn)
    seqs = ZSequences.load_many(a.data)
    train, val = eye_data(seqs)
    print(f"{len(train[0])} bilder til trening, {len(val[0])} til validering", flush=True)
    model = MDNRNN(replace(rnn.config, eye=True, eye_channels=a.channels))
    model.load_state_dict(rnn.state_dict(), strict=False)
    history = train_eye(model.eye, train, val, epochs=a.epochs, lr=a.lr)
    model.eval()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out, eye_history=history, eye_from=a.rnn, eye_data=a.data)
    print(f"Skrev {a.out}")


if __name__ == "__main__":
    main()
