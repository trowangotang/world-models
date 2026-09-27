"""Vurder hvor godt en trent VAE forstår verden, ikke bare hvor pene bildene er.

Tre typer mål, alle på valideringsepisodene:
  1. Rekonstruksjon: vanlig MSE per piksel.
  2. Rutenett-nøyaktighet: les rekonstruksjonen som et rutenett (parse_cells) og sjekk
     om hver celle, og spesielt agentens celle, er riktig.
  3. Lineær probe: tren en logistisk regresjon fra z (mu) til agentens og målets celle.
     Hvis en *lineær* modell klarer det, er informasjonen lett tilgjengelig for en
     enkel controller senere. Tilfeldig gjetting gir 1/64 ≈ 1,6 %.

    python -m worldmodels.vae.evaluate checkpoints/vae_z16_w10.pt checkpoints/vae_z16_w1.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from worldmodels.env.parse import AGENT, GOAL, find_cell, parse_cells
from worldmodels.env.preview import save_png
from worldmodels.vae.dataset import iterate_minibatches, load_split
from worldmodels.vae.model import ConvVAE


@torch.no_grad()
def encode_and_reconstruct(model: ConvVAE, frames: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Returnerer (mu (N, D) float32, rekonstruksjoner (N, H, W, 3) uint8)."""
    model.eval()
    mus, recons = [], []
    for x in iterate_minibatches(frames, batch_size, shuffle=False):
        mu, _ = model.encode(x)
        recon = model.decode(mu)
        mus.append(mu.numpy())
        recons.append((recon.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8))
    return np.concatenate(mus), np.concatenate(recons)


def grid_metrics(frames: np.ndarray, recons: np.ndarray) -> dict[str, float]:
    true_cells, rec_cells = parse_cells(frames), parse_cells(recons)
    true_agent, rec_agent = find_cell(true_cells, AGENT), find_cell(rec_cells, AGENT)
    return {
        "cell_accuracy": float((true_cells == rec_cells).mean()),
        "layout_exact": float((true_cells == rec_cells).all(axis=(1, 2)).mean()),
        "agent_cell_accuracy": float((true_agent == rec_agent).mean()),
        "agent_missing": float((rec_agent == -1).mean()),
    }


def linear_probe(
    z_train: np.ndarray, y_train: np.ndarray, z_val: np.ndarray, y_val: np.ndarray,
    num_classes: int = 64, epochs: int = 300, lr: float = 0.05, seed: int = 0,
) -> float:
    """Tren logistisk regresjon z -> klasse og returner nøyaktighet på valideringsdata."""
    torch.manual_seed(seed)
    mean, std = z_train.mean(0), z_train.std(0) + 1e-6
    zt = torch.from_numpy((z_train - mean) / std).float()
    zv = torch.from_numpy((z_val - mean) / std).float()
    yt = torch.from_numpy(y_train).long()
    probe = nn.Linear(zt.shape[1], num_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):  # fullbatch: datasettet er lite
        loss = nn.functional.cross_entropy(probe(zt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(zv).argmax(1).numpy()
    return float((pred == y_val).mean())


def evaluate_checkpoint(path: str | Path, split) -> dict:
    model, ckpt = ConvVAE.load(path)
    mu_tr, _ = encode_and_reconstruct(model, split.train)
    mu_va, rec_va = encode_and_reconstruct(model, split.val)
    labels_tr, labels_va = parse_cells(split.train), parse_cells(split.val)

    result = {"checkpoint": str(path), "hparams": ckpt.get("hparams", {})}
    diff = (rec_va.astype(np.float32) - split.val.astype(np.float32)) / 255.0
    result["val_mse"] = float((diff ** 2).mean())
    result.update(grid_metrics(split.val, rec_va))
    for name, kind in (("agent", AGENT), ("goal", GOAL)):
        result[f"probe_{name}_accuracy"] = linear_probe(
            mu_tr, find_cell(labels_tr, kind), mu_va, find_cell(labels_va, kind)
        )
    return result


def reconstruction_grid(split, checkpoints: list[str | Path], n: int = 10, scale: int = 2, seed: int = 0) -> np.ndarray:
    """Øverste rad: originaler. Deretter én rad per checkpoint."""
    from worldmodels.env.preview import episode_grid

    idx = np.random.default_rng(seed).choice(len(split.val), size=n, replace=False)
    originals = split.val[idx]
    rows = [originals]
    for path in checkpoints:
        model, _ = ConvVAE.load(path)
        rows.append(encode_and_reconstruct(model, originals)[1])
    return episode_grid(rows, frames=n, scale=scale)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluer én eller flere VAE-checkpointer")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--data", default="data/rollouts")
    p.add_argument("--image", default=None, help="lagre sammenligning av rekonstruksjoner som PNG")
    p.add_argument("--json", default=None, help="lagre målene som JSON")
    args = p.parse_args(argv)

    split = load_split(args.data)
    results = [evaluate_checkpoint(c, split) for c in args.checkpoints]

    cols = ["val_mse", "cell_accuracy", "layout_exact", "agent_cell_accuracy", "probe_agent_accuracy", "probe_goal_accuracy"]
    print("checkpoint".ljust(28) + "".join(c[:14].rjust(15) for c in cols))
    for r in results:
        print(Path(r["checkpoint"]).name.ljust(28) + "".join(f"{r[c]:15.4f}" for c in cols))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    if args.image:
        save_png(reconstruction_grid(split, args.checkpoints), args.image)
        print(f"Skrev {args.image}")


if __name__ == "__main__":
    main()
