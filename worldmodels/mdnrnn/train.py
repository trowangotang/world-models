"""Tren MDN-RNN-en på z-sekvenser.

    python -m worldmodels.mdnrnn.train --data data/zseq_20k.npz --out checkpoints/mdnrnn.pt

Inndata i skritt t er en *trukket* z_t ~ q(z | x_t), ikke bare middelverdien. Da lærer
modellen å tåle litt støy i sin egen inndata, noe den møter når den senere drømmer og
mater sine egne prediksjoner tilbake. Målet er middelverdien mu_{t+1}.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN, MDNRNNConfig, mdn_nll


@dataclass
class Batch:
    z_mu: torch.Tensor       # (B, T+1, D) middelverdier for alle observasjoner
    z_logvar: torch.Tensor   # (B, T+1, D)
    actions: torch.Tensor    # (B, T)
    events: torch.Tensor     # (B, T)
    agent_cell: torch.Tensor # (B, T+1)
    mask: torch.Tensor       # (B, T) True for ekte skritt, False for utfylling

    def inputs(self, sample: bool, generator: torch.Generator | None = None) -> torch.Tensor:
        mu = self.z_mu[:, :-1]
        if not sample:
            return mu
        std = (0.5 * self.z_logvar[:, :-1]).exp()
        return mu + std * torch.randn(mu.shape, generator=generator)

    @property
    def targets(self) -> torch.Tensor:
        return self.z_mu[:, 1:]


def make_batch(seqs: ZSequences, indices) -> Batch:
    """Fyll opp episodene til samme lengde og lag en maske for de ekte skrittene."""
    eps = [seqs.episode(i) for i in indices]
    T = max(len(e["actions"]) for e in eps)
    B, D = len(eps), seqs.mu.shape[1]
    z_mu = np.zeros((B, T + 1, D), np.float32)
    z_logvar = np.zeros((B, T + 1, D), np.float32)
    actions = np.zeros((B, T), np.int64)
    events = np.zeros((B, T), np.int64)
    agent = np.full((B, T + 1), -1, np.int64)
    mask = np.zeros((B, T), bool)
    for b, e in enumerate(eps):
        n = len(e["actions"])
        z_mu[b, :n + 1] = e["mu"]
        z_logvar[b, :n + 1] = e["logvar"]
        actions[b, :n] = e["actions"]
        events[b, :n] = e["events"]
        agent[b, :n + 1] = e["agent_cell"]
        mask[b, :n] = True
    t = torch.from_numpy
    return Batch(t(z_mu), t(z_logvar), t(actions), t(events), t(agent), t(mask))


def split_indices(n: int, val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def batch_loss(model: MDNRNN, batch: Batch, sample_inputs: bool, event_weight: float) -> dict[str, torch.Tensor]:
    out = model(batch.inputs(sample_inputs), batch.actions)
    m = batch.mask
    nll = mdn_nll(out, batch.targets)[m].mean()
    ce = F.cross_entropy(out.event_logits[m], batch.events[m])
    return {"loss": nll + event_weight * ce, "nll": nll, "event_ce": ce}


def run_epoch(model, seqs, indices, opt, batch_size, event_weight, rng=None) -> dict[str, float]:
    training = opt is not None
    model.train(training)
    order = rng.permutation(indices) if training else indices
    totals, n = {"loss": 0.0, "nll": 0.0, "event_ce": 0.0}, 0
    with torch.set_grad_enabled(training):
        for start in range(0, len(order), batch_size):
            batch = make_batch(seqs, order[start:start + batch_size])
            parts = batch_loss(model, batch, sample_inputs=training, event_weight=event_weight)
            if training:
                opt.zero_grad()
                parts["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            k = int(batch.mask.sum())
            for key in totals:
                totals[key] += parts[key].item() * k
            n += k
    return {k: v / n for k, v in totals.items()}


def train(
    data: str | Path,
    out: str | Path,
    hidden_dim: int = 256,
    num_mixtures: int = 5,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    event_weight: float = 1.0,
    seed: int = 0,
    log=lambda msg: print(msg, flush=True),
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seqs = ZSequences.load(data)
    train_idx, val_idx = split_indices(len(seqs), seed=seed)
    log(f"Trening: {len(train_idx)} episoder, validering: {len(val_idx)} episoder")

    model = MDNRNN(MDNRNNConfig(latent_dim=seqs.mu.shape[1], hidden_dim=hidden_dim, num_mixtures=num_mixtures))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, seqs, train_idx, opt, batch_size, event_weight, rng)
        va = run_epoch(model, seqs, val_idx, None, batch_size, event_weight)
        history.append({"epoch": epoch, "train": tr, "val": va})
        log(
            f"epoke {epoch:2d}  tren nll {tr['nll']:7.3f} hendelse {tr['event_ce']:.3f}  "
            f"val nll {va['nll']:7.3f} hendelse {va['event_ce']:.3f}  [{time.time() - t0:.0f}s]"
        )

    hparams = {
        "hidden_dim": hidden_dim, "num_mixtures": num_mixtures, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "event_weight": event_weight, "seed": seed, "data": str(data),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out, hparams=hparams, history=history)
    log(f"Lagret {out}")
    return {"hparams": hparams, "history": history}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tren MDN-RNN på z-sekvenser")
    p.add_argument("--data", default="data/zseq_20k.npz")
    p.add_argument("--out", default="checkpoints/mdnrnn.pt")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--mixtures", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--event-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    result = train(
        args.data, args.out, hidden_dim=args.hidden_dim, num_mixtures=args.mixtures, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, event_weight=args.event_weight, seed=args.seed,
    )
    print(json.dumps(result["history"][-1]))


if __name__ == "__main__":
    main()
