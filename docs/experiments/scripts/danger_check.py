"""Ser nærsynet hindringer som er på vei inn i nabocellen? (steg 10, decisions.md D66)

Faren i en retning er enten en hindring som står i nabocellen nå (lett, den syns i bildet), eller en
som flytter seg dit i neste skritt (krever å vite hvor den er på vei). Vi måler treff hver for seg på
valideringsbildene, for hver M som gis på kommandolinjen.

    PYTHONPATH=. python docs/experiments/scripts/danger_check.py checkpoints/mdnrnn_moving_nomem.pt \\
        checkpoints/mdnrnn_moving.pt
"""

import sys

import numpy as np
import torch

from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.neighbours import _logits, neighbour_data
from worldmodels.mdnrnn.train import split_indices


@torch.no_grad()
def main():
    seqs = ZSequences.load_many(["data/moving/zseq_20k.npz", "data/moving/zseq_agent.npz"])
    episode = np.repeat(np.arange(len(seqs)), seqs.lengths + 1)
    last = np.zeros(len(episode), bool)
    last[seqs.obs_offsets + seqs.lengths] = True
    for path in sys.argv[1:]:
        rnn, ckpt = MDNRNN.load(path)
        rnn.eval()
        _, val_idx = split_indices(len(seqs), seed=ckpt.get("hparams", {}).get("seed", 0))
        _, val = neighbour_data(rnn, seqs, val_idx, memory=rnn.config.neighbour_memory)
        keep = ~last & np.isin(episode, val_idx)
        keep &= (seqs.danger >= 0).all(1) & (seqs.near_obstacle >= 0).all(1) & (seqs.agent_cell >= 0) & (seqs.goal_cell >= 0)
        near = torch.from_numpy(seqs.near_obstacle[keep] == 1)
        danger = torch.from_numpy(seqs.danger[keep] == 1)
        assert len(near) == len(val[0]), (len(near), len(val[0]))
        p = torch.cat([torch.sigmoid(_logits(rnn.neighbours, val, slice(i, i + 8192))[:, 1]) for i in range(0, len(near), 8192)])
        hit = p > 0.5
        arriving = danger & ~near
        print(f"{path}: står der nå {float((hit & near).sum() / near.sum()):.1%} av {int(near.sum())}, "
              f"på vei inn {float((hit & arriving).sum() / arriving.sum()):.1%} av {int(arriving.sum())}, "
              f"falsk alarm {float((hit & ~danger).sum() / (~danger).sum()):.2%}")


if __name__ == "__main__":
    main()
