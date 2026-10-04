"""Har M lært hvordan hindringene beveger seg? (steg 10, decisions.md D65)

For valideringsepisodene i verdenen med bevegelige hindringer: M spår neste z (komponenten med
høyest vekt, som i drømmen), VAE-en tegner det, og vi leser av hvor hindringene står. Det
sammenlignes med det neste ekte bildet (også gjennom VAE-en, så bare M sin feil måles) og med å bare
gjenta dette bildet ("stillstand"). Bare skritt fra og med skritt 2 telles, så M har sett bevegelse.

    PYTHONPATH=. python docs/experiments/scripts/motion_check.py checkpoints/mdnrnn_moving.pt
"""

import sys

import numpy as np
import torch

from worldmodels.env.parse import OBSTACLE, parse_cells
from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN, most_likely_mean
from worldmodels.mdnrnn.train import make_batch, split_indices
from worldmodels.vae.model import ConvVAE


@torch.no_grad()
def obstacle_maps(vae, z):
    img = vae.decode(z).permute(0, 2, 3, 1).numpy()
    return parse_cells(img) == OBSTACLE


@torch.no_grad()
def main():
    rnn_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/mdnrnn_moving.pt"
    vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
    rnn, ckpt = MDNRNN.load(rnn_path)
    vae.eval(), rnn.eval()
    seqs = ZSequences.load_many(["data/moving/zseq_20k.npz", "data/moving/zseq_agent.npz"])
    _, val = split_indices(len(seqs), seed=ckpt.get("hparams", {}).get("seed", 0))
    val = [i for i in val if seqs.lengths[i] >= 4][:1500]
    b = make_batch(seqs, val)
    pred = most_likely_mean(rnn(b.z_mu[:, :-1], b.actions))           # (B, T, D): spådd z_{t+1}
    rows = {"M": [0, 0, 0], "stillstand": [0, 0, 0]}                   # riktige, spådde, ekte hindringsceller
    moved = [0, 0]
    for j, i in enumerate(val):
        T = int(seqs.lengths[i])
        ts = np.arange(2, T - 1)                                       # siste skritt kan ende i krasj/mål
        if len(ts) == 0:
            continue
        true_next = obstacle_maps(vae, b.z_mu[j, ts + 1])
        now = obstacle_maps(vae, b.z_mu[j, ts])
        guess = obstacle_maps(vae, pred[j, ts])
        for name, m in (("M", guess), ("stillstand", now)):
            rows[name][0] += int((m & true_next).sum())
            rows[name][1] += int(m.sum())
            rows[name][2] += int(true_next.sum())
        # Cellene som endret seg: der en hindring kom til
        arrived = true_next & ~now
        moved[0] += int((guess & arrived).sum())
        moved[1] += int(arrived.sum())
    for name, (hit, p, t) in rows.items():
        print(f"{name:11s} treff {hit / t:.1%}, presisjon {hit / max(1, p):.1%}")
    print(f"celler en hindring flyttet seg inn i: M fant {moved[0] / max(1, moved[1]):.1%} av {moved[1]}")


if __name__ == "__main__":
    main()
