"""Ser nærsynet bevegelse bedre om det får forrige bilde direkte? (steg 10, decisions.md D67)

Nærsynet med minne (D66) får M sitt minne h, og må lese hvor hindringene er på vei derfra. Her
prøver vi å gi det forrige bildes z i tillegg (null før første bilde), slik at det kan sammenligne to
bilder selv. Trener nærsynet på nytt med samme oppsett og måler treff på hindringer som står der nå
og som er på vei inn, på valideringsbildene.

    PYTHONPATH=. python docs/experiments/scripts/motion_eye.py
"""

import numpy as np
import torch

from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.neighbours import NeighbourEye, _logits, neighbour_data, train_neighbours
from worldmodels.mdnrnn.train import split_indices

DATA = ["data/moving/zseq_20k.npz", "data/moving/zseq_agent.npz"]


def previous_z(seqs: ZSequences) -> torch.Tensor:
    z = torch.from_numpy(seqs.mu).float()
    prev = torch.zeros_like(z)
    prev[1:] = z[:-1]
    prev[seqs.obs_offsets] = 0
    return prev


@torch.no_grad()
def report(name, net, val, near, danger):
    p = torch.cat([torch.sigmoid(_logits(net, val, slice(i, i + 8192))[:, 1]) for i in range(0, len(near), 8192)])
    hit, arriving = p > 0.5, danger & ~near
    print(f"{name}: står der nå {float((hit & near).sum() / near.sum()):.1%}, "
          f"på vei inn {float((hit & arriving).sum() / arriving.sum()):.1%}, "
          f"falsk alarm {float((hit & ~danger).sum() / (~danger).sum()):.2%}", flush=True)


def main():
    rnn, ckpt = MDNRNN.load("checkpoints/mdnrnn_moving_core30.pt")
    rnn.eval()
    seqs = ZSequences.load_many(DATA)
    _, val_idx = split_indices(len(seqs), seed=ckpt.get("hparams", {}).get("seed", 0))
    train, val = neighbour_data(rnn, seqs, val_idx, memory=True)

    # Samme utvalg av bilder som neighbour_data, for å hente forrige z og fasit til målingen
    episode = np.repeat(np.arange(len(seqs)), seqs.lengths + 1)
    last = np.zeros(len(episode), bool)
    last[seqs.obs_offsets + seqs.lengths] = True
    labels_known = (seqs.danger >= 0).all(1) & (seqs.agent_cell >= 0) & (seqs.goal_cell >= 0)
    keep = ~last & labels_known
    is_val = np.isin(episode, val_idx)
    prev = previous_z(seqs)
    val_mask = keep & is_val
    near = torch.from_numpy((seqs.near_obstacle[val_mask] == 1))
    danger = torch.from_numpy((seqs.danger[val_mask] == 1))
    assert len(near) == len(val[0])

    with_prev = lambda d, m: (*d[:4], torch.cat([d[4], prev[m]], 1))  # noqa: E731
    train2, val2 = with_prev(train, keep & ~is_val), with_prev(val, val_mask)
    H, D = rnn.config.hidden_dim, rnn.config.latent_dim
    torch.manual_seed(0)
    net = NeighbourEye(D, 32, context_dim=H + D)
    train_neighbours(net, train2, val2, epochs=8)
    net.eval()
    report("h + forrige z", net, val2, near, danger)


if __name__ == "__main__":
    main()
