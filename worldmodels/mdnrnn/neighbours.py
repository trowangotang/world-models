"""Nærsyn: hva står i nabocellene? Øyet spår hendelsene i drømmen ut fra bildet.

    python -m worldmodels.mdnrnn.neighbours --rnn checkpoints/mdnrnn_eye.pt --data data/v3/zseq_*.npz \\
        --out checkpoints/mdnrnn_sense.pt

Steg 6 viste at drømmen straffer den strategien som virker: M spår mål og krasj fra minnet h, og
h vet ikke hvor målet er (decisions.md D51). Øyet vet det. Nærsynet ser på bildet (z) og øyets minne
om målet, og sier for hver av de fire retningene:

    * er målet i nabocellen den veien?
    * står det en hindring i nabocellen den veien?

Det er nøyaktig det som avgjør hva som skjer når agenten tar en handling, så drømmen kan spå
hendelsene herfra i stedet for fra h (D52). Hvert bilde gir merkelapper for alle fire retningene,
ikke bare den agenten tok, så det lærer av mye mer enn hendelseshodet.

Nettet starter som øyet (z → 8x8-kart), får målkartet fra minnet som en ekstra kanal, og ender i
ett kart per retning og spørsmål: "om agenten står her, hva er ved siden av?". Svaret er kartet
avlest der øyet ser agenten (vektet med øyets sannsynlighet for agenten i hver celle). Et første
forsøk som selv måtte finne agenten (maksimum over kartet), lærte ingenting (D52).
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
from worldmodels.mdnrnn.encode import EVENT_GOAL, EVENT_MOVE, EVENT_OBSTACLE

NUM_DIRECTIONS = len(ACTIONS)


class NeighbourEye(nn.Module):
    """context_dim > 0: nærsynet får også M sitt minne h (steg 10). Ett bilde viser hvor hindringene
    står, men ikke hvor de er på vei; det vet minnet, som har sett de forrige bildene (D66)."""

    def __init__(self, latent_dim: int = 32, channels: int = 32, side: int = 8, context_dim: int = 0):
        super().__init__()
        self.channels, self.side = channels, side
        self.fc = nn.Linear(latent_dim, channels * (side // 2) ** 2)
        self.context = nn.Linear(context_dim, channels * (side // 2) ** 2) if context_dim else None
        self.up = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.ReLU())
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 2 * NUM_DIRECTIONS, 1),
        )

    def forward(self, z: torch.Tensor, goal_map: torch.Tensor, agent_map: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        """z: (B, D), goal_map og agent_map: (B, side*side) sannsynligheter fra øyet, context: (B, H)
        M sitt minne før skrittet (bare med context_dim > 0) -> logits (B, 2, 4): [mål, fare] x retning."""
        half = self.side // 2
        x = self.fc(z)
        if self.context is not None:
            if context is None:
                raise ValueError("dette nærsynet trenger M sitt minne h")
            x = x + self.context(context)
        x = self.up(x.view(-1, self.channels, half, half))
        x = torch.cat([x, goal_map.view(-1, 1, self.side, self.side)], dim=1)
        maps = self.net(x).flatten(2)                       # (B, 8, side*side)
        return (maps * agent_map.unsqueeze(1)).sum(-1).view(-1, 2, NUM_DIRECTIONS)


def event_probs_from_logits(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Nærsynets logits (B, 2, 4) og handlinger (B,) -> hendelsessannsynligheter (B, 3).

    Mål vinner over hindring (de kan ikke stå i samme celle). Ellers er det en vanlig flytting."""
    idx = torch.arange(len(actions))
    p_goal = torch.sigmoid(logits[idx, 0, actions])
    p_obstacle = (1 - p_goal) * torch.sigmoid(logits[idx, 1, actions])
    out = torch.zeros(len(actions), 3)
    out[:, EVENT_GOAL], out[:, EVENT_OBSTACLE] = p_goal, p_obstacle
    out[:, EVENT_MOVE] = 1 - p_goal - p_obstacle
    return out


def goal_direction_labels(agent_cells: np.ndarray, goal_cells: np.ndarray, side: int = 8) -> np.ndarray:
    """(N,) celler -> (N, 4) int8: ligger målet i nabocellen i hver retning? -1 der noe er ukjent."""
    out = np.full((len(agent_cells), NUM_DIRECTIONS), -1, dtype=np.int8)
    known = (agent_cells >= 0) & (goal_cells >= 0)
    r, c = agent_cells[known] // side, agent_cells[known] % side
    for k, (dr, dc) in enumerate(ACTIONS):
        nr, nc = np.clip(r + dr, 0, side - 1), np.clip(c + dc, 0, side - 1)
        out[known, k] = (nr * side + nc) == goal_cells[known]
    return out


@torch.no_grad()
def eye_maps(model, seqs, batch: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Øyets kart for hvert bilde i z-sekvensene: agenten fra bildet, målet fra minnet (N, celler).

    Målminnet følger samme regel som MDNRNN.see: summen av log(p + gulv) over bildene så langt."""
    from worldmodels.mdnrnn.eye import GOAL_EVIDENCE_FLOOR

    z = torch.from_numpy(seqs.mu).float()
    logits = torch.cat([model.eye(z[i:i + batch]) for i in range(0, len(z), batch)])
    evidence = torch.log(F.softmax(logits[:, 1], dim=-1) + GOAL_EVIDENCE_FLOOR)
    memory = torch.empty_like(evidence)
    for s, n in zip(seqs.obs_offsets, seqs.lengths + 1):
        memory[s:s + n] = evidence[s:s + n].cumsum(0)
    return F.softmax(logits[:, 0], dim=-1), F.softmax(memory, dim=-1)


@torch.no_grad()
def memory_states(model, seqs, batch: int = 512) -> torch.Tensor:
    """M sitt minne h før hvert bilde (N, H): null før første bilde, ellers h etter forrige skritt."""
    from worldmodels.mdnrnn.train import make_batch

    H = model.config.hidden_dim
    out = torch.zeros(len(seqs.mu), H)
    offsets = seqs.obs_offsets
    for start in range(0, len(seqs), batch):
        idx = np.arange(start, min(start + batch, len(seqs)))
        b = make_batch(seqs, idx)
        h = model(b.z_mu[:, :-1], b.actions).h                     # (B, T, H): etter skritt t
        for j, i in enumerate(idx):
            T = int(seqs.lengths[i])
            out[offsets[i] + 1:offsets[i] + T + 1] = h[j, :T]
    return out


def neighbour_data(model, seqs, val_episodes: np.ndarray, memory: bool = False):
    """(z, målkart, agentkart, merkelapper (N, 2, 4), minne eller None) for trening og validering.
    Siste bilde i hver episode, der agenten allerede står i målet eller en hindring, er tatt ut."""
    # "Fare" er en hindring i nabocellen nå eller etter neste skritt; i en stille verden det samme som før
    labels = np.stack([goal_direction_labels(seqs.agent_cell, seqs.goal_cell), seqs.danger], 1)
    episode = np.repeat(np.arange(len(seqs)), seqs.lengths + 1)
    last = np.zeros(len(episode), bool)
    last[seqs.obs_offsets + seqs.lengths] = True
    keep = ~last & (labels >= 0).all((1, 2))
    is_val = np.isin(episode, val_episodes)
    z = torch.from_numpy(seqs.mu).float()
    agent_maps, goal_maps = eye_maps(model, seqs)
    y = torch.from_numpy(labels).float()
    h = memory_states(model, seqs) if memory else None
    pick = lambda m: (z[m], goal_maps[m], agent_maps[m], y[m], None if h is None else h[m])  # noqa: E731
    return pick(keep & ~is_val), pick(keep & is_val)


def _logits(net, data, idx):
    z, goal_maps, agent_maps, _, h = data
    return net(z[idx], goal_maps[idx], agent_maps[idx], None if h is None else h[idx])


@torch.no_grad()
def neighbour_metrics(net: NeighbourEye, data, batch: int = 8192) -> dict:
    y = data[3]
    logits = torch.cat([_logits(net, data, slice(i, i + batch)) for i in range(0, len(y), batch)])
    out = {}
    for k, name in enumerate(("goal", "obstacle")):
        pred, true = torch.sigmoid(logits[:, k]).flatten(), y[:, k].flatten().bool()
        hit = pred > 0.5
        out[name] = {
            "recall": float((hit & true).sum() / max(1, true.sum())),
            "precision": float((hit & true).sum() / max(1, hit.sum())),
            "mean_predicted": float(pred.mean()), "rate": float(true.float().mean()),
        }
    return out


def train_neighbours(net: NeighbourEye, train, val, epochs: int = 6, lr: float = 1e-3, batch: int = 512, seed: int = 0,
                     log=lambda msg: print(msg, flush=True)) -> list[dict]:
    """Vanlig binær kryssentropi uten vekting, slik at sannsynlighetene blir kalibrerte. Drømmen
    regner med dem som sannsynligheter, og vektede klasser gjorde M for redd for krasj (D52)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    y = train[3]
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        net.train()
        perm = torch.from_numpy(rng.permutation(len(y)))
        total = 0.0
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            loss = F.binary_cross_entropy_with_logits(_logits(net, train, b), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        net.eval()
        m = neighbour_metrics(net, val)
        history.append({"epoch": epoch, "loss": total / len(y), **m})
        log(f"epoke {epoch}: tap {total / len(y):.4f}, mål treff {m['goal']['recall']:.3f} presisjon "
            f"{m['goal']['precision']:.3f}, hindring treff {m['obstacle']['recall']:.3f} presisjon "
            f"{m['obstacle']['precision']:.3f}  [{time.time() - t0:.0f}s]")
    return history


def main(argv: list[str] | None = None) -> None:
    from worldmodels.mdnrnn.encode import ZSequences
    from worldmodels.mdnrnn.model import MDNRNN
    from worldmodels.mdnrnn.train import split_indices

    p = argparse.ArgumentParser(description="Tren nærsynet og la drømmen spå hendelser fra det")
    p.add_argument("--rnn", default="checkpoints/mdnrnn_eye.pt")
    p.add_argument("--data", nargs="+", default=["data/v3/zseq_20k.npz"])
    p.add_argument("--out", default="checkpoints/mdnrnn_sense.pt")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--memory", action="store_true", help="gi nærsynet M sitt minne h, for bevegelige hindringer (D66)")
    a = p.parse_args(argv)

    rnn, ckpt = MDNRNN.load(a.rnn)
    if rnn.eye is None:
        raise ValueError("nærsynet trenger øyets minne om målet: bruk en M med øye (python -m worldmodels.mdnrnn.eye)")
    seqs = ZSequences.load_many(a.data)
    _, val_idx = split_indices(len(seqs), seed=ckpt.get("hparams", {}).get("seed", 0))
    train, val = neighbour_data(rnn, seqs, val_idx, memory=a.memory)
    print(f"{len(train[0])} bilder til trening, {len(val[0])} til validering", flush=True)
    model = MDNRNN(replace(rnn.config, neighbours=True, neighbour_channels=a.channels, neighbour_memory=a.memory))
    model.load_state_dict(rnn.state_dict(), strict=False)
    history = train_neighbours(model.neighbours, train, val, epochs=a.epochs, lr=a.lr)
    model.eval()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    extra = {k: v for k, v in ckpt.items() if k not in ("config", "state_dict")}
    model.save(a.out, **extra, neighbour_history=history, neighbours_from=a.rnn, neighbour_data=a.data)
    print(f"Skrev {a.out}")


if __name__ == "__main__":
    main()
