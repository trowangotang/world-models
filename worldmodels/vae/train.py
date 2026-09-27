"""Tren VAE-en på bilder fra rollouts.

    python -m worldmodels.vae.train --data data/rollouts --out checkpoints/vae.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from worldmodels.vae.dataset import iterate_minibatches, load_split
from worldmodels.vae.loss import vae_loss
from worldmodels.vae.model import ConvVAE, VAEConfig


def run_epoch(model, frames, opt, batch_size, object_weight, beta, rng=None) -> dict[str, float]:
    """Én gjennomgang av frames. Trener hvis opt er gitt, ellers bare evaluerer."""
    training = opt is not None
    model.train(training)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for x in iterate_minibatches(frames, batch_size, shuffle=training, rng=rng):
            recon, mu, logvar = model(x)
            parts = vae_loss(recon, x, mu, logvar, object_weight=object_weight, beta=beta)
            if training:
                opt.zero_grad()
                parts["loss"].backward()
                opt.step()
            for k in totals:
                totals[k] += parts[k].item() * len(x)
            n += len(x)
    return {k: v / n for k, v in totals.items()}


def train(
    data_dir: str | Path,
    out: str | Path,
    latent_dim: int = 16,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    object_weight: float = 10.0,
    beta: float = 1.0,
    seed: int = 0,
    log=print,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    split = load_split(data_dir, seed=seed)
    log(f"Trening: {len(split.train)} bilder, validering: {len(split.val)} bilder")

    model = ConvVAE(VAEConfig(latent_dim=latent_dim))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, split.train, opt, batch_size, object_weight, beta, rng)
        va = run_epoch(model, split.val, None, batch_size, object_weight, beta)
        history.append({"epoch": epoch, "train": tr, "val": va})
        log(
            f"epoke {epoch:2d}  tren {tr['loss']:8.2f} (rek {tr['recon']:7.2f}, kl {tr['kl']:5.2f})  "
            f"val {va['loss']:8.2f}  [{time.time() - t0:.0f}s]"
        )

    hparams = {
        "latent_dim": latent_dim, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "object_weight": object_weight, "beta": beta, "seed": seed, "data_dir": str(data_dir),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out, hparams=hparams, history=history)
    log(f"Lagret {out}")
    return {"hparams": hparams, "history": history}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tren VAE på GridDodge-rollouts")
    p.add_argument("--data", default="data/rollouts")
    p.add_argument("--out", default="checkpoints/vae.pt")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--object-weight", type=float, default=10.0, help="1 = vanlig MSE uten vekting")
    p.add_argument("--beta", type=float, default=1.0, help="vekt på KL-leddet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    result = train(
        args.data, args.out, latent_dim=args.latent_dim, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, object_weight=args.object_weight, beta=args.beta, seed=args.seed,
    )
    print(json.dumps(result["history"][-1]))


if __name__ == "__main__":
    main()
