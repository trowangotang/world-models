"""Vurder hvor godt en trent VAE forstår verden, ikke bare hvor pene bildene er.

Tre typer mål, alle på valideringsepisodene:
  1. Rekonstruksjon: vanlig MSE per piksel.
  2. Rutenett-nøyaktighet: les rekonstruksjonen som et rutenett (parse_cells) og sjekk
     om hver celle, og spesielt agentens celle, er riktig.
  3. Prober: tren en liten modell fra z (mu) til agentens og målets celle.
     En *lineær* probe sier om informasjonen er lett tilgjengelig for en lineær
     controller. En MLP-probe (ett skjult lag) sier om informasjonen finnes i z i det hele
     tatt. Tilfeldig gjetting gir 1/64 ≈ 1,6 %.

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


def train_probe(
    z_train: np.ndarray, y_train: np.ndarray, z_val: np.ndarray, y_val: np.ndarray,
    num_classes: int = 64, hidden: int | None = None, epochs: int = 20, lr: float | None = None,
    batch_size: int = 1024, seed: int = 0,
) -> float:
    """Tren en probe z -> klasse og returner nøyaktighet på valideringsdata.

    hidden=None gir en lineær probe (logistisk regresjon). Med hidden får proben ett
    skjult lag, som viser om informasjonen finnes i z selv om den ikke er lineært tilgjengelig.
    """
    torch.manual_seed(seed)
    mean, std = z_train.mean(0), z_train.std(0) + 1e-6
    zt = torch.from_numpy((z_train - mean) / std).float()
    zv = torch.from_numpy((z_val - mean) / std).float()
    yt = torch.from_numpy(y_train).long()
    d = zt.shape[1]
    if hidden is None:
        probe = nn.Linear(d, num_classes)
        lr = lr or 1e-2
    else:
        probe = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        lr = lr or 1e-3
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(zt))
        for i in range(0, len(zt), batch_size):
            b = perm[i:i + batch_size]
            loss = nn.functional.cross_entropy(probe(zt[b]), yt[b])
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
        # Målet er skjult i siste bilde når agenten står på det; de bildene hoppes over.
        y_tr, y_va = find_cell(labels_tr, kind), find_cell(labels_va, kind)
        tr, va = y_tr >= 0, y_va >= 0
        args = (mu_tr[tr], y_tr[tr], mu_va[va], y_va[va])
        result[f"probe_{name}_linear"] = train_probe(*args)
        result[f"probe_{name}_mlp"] = train_probe(*args, hidden=256)
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
    p.add_argument("--frames-per-episode", type=int, default=None)
    args = p.parse_args(argv)

    split = load_split(args.data, frames_per_episode=args.frames_per_episode)
    results = [evaluate_checkpoint(c, split) for c in args.checkpoints]

    cols = [
        "val_mse", "cell_accuracy", "layout_exact", "agent_cell_accuracy",
        "probe_agent_linear", "probe_agent_mlp", "probe_goal_linear", "probe_goal_mlp",
    ]
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
