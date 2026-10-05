"""Steg 14: hvor godt vet kompasset hvor målet er, og hva klarer en håndlaget agent på kompasset?

    python docs/experiments/scripts/fp_compass_check.py --vae checkpoints/vae_fp.pt \
        --rnn checkpoints/mdnrnn_fp20k.pt --data data/zseq_fp_4k.npz data/zseq_fp_16k.npz

Den håndlagde agenten er en diagnose, ikke et resultat: den viser om kompasset og fremsynet er
gode nok til at en controller kan lære det samme. Regel: ser den en hindring rett fram (fremsyn),
snur den. Har den ikke sett målet, går den fram og snur av og til (ellers kan den gå fast i en vegg). Ellers går den fram så lenge målet er foran
(også på skrå), og snur mot siden målet er på når det ikke er det lenger.
"""

import argparse

import numpy as np
import torch

from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.env.gridworld import FORWARD, TURN_LEFT, TURN_RIGHT
from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import COMPASS_SCALE, MDNRNN
from worldmodels.mdnrnn.train import make_batch, split_indices
from worldmodels.vae.dataset import to_tensor
from worldmodels.vae.model import ConvVAE


@torch.no_grad()
def compass_quality(rnn, seqs, idx):
    b = make_batch(seqs, idx)
    out = rnn(b.z_mu[:, :-1], b.actions)
    h_before = torch.cat([torch.zeros_like(out.h[:, :1]), out.h[:, :-1]], 1)
    f = rnn.compass_features(b.z_mu[:, :-1], h_before)
    m = b.mask
    known = b.goal_known[:, :-1] == 1
    target = b.goal_ego[:, :-1]
    err = (f[..., :2] * COMPASS_SCALE - target).abs().sum(-1)
    side = torch.sign(target[..., 1])
    right_side = (torch.sign(f[..., 1].round(decimals=1)) == side)
    for name, sel in (("målet sett", m & known), ("ikke sett ennå", m & ~known)):
        print(f"{name:15s}: {int(sel.sum()):6d} bilder, avvik {err[sel].mean():.2f} celler, "
              f"riktig side {right_side[sel & (side != 0)].float().mean():.0%}, "
              f"tror sett {f[..., 2][sel].mean():.2f}")


@torch.no_grad()
def handcrafted(vae, rnn, n=500, seed=100_000, moving=0):
    config = GridConfig(first_person=True, moving_obstacles=moving)
    envs = [GridDodgeEnv(config) for _ in range(n)]
    obs = np.stack([e.reset(seed=seed + i) for i, e in enumerate(envs)])
    H = rnn.config.hidden_dim
    hidden = (torch.zeros(1, n, H), torch.zeros(1, n, H))
    rng = np.random.default_rng(0)
    active = np.ones(n, bool)
    outcome = np.array(["truncated"] * n, dtype=object)
    while active.any():
        z, _ = vae.encode(to_tensor(obs))
        comp = rnn.compass_features(z, hidden[0][-1]).numpy()
        danger = rnn.lookahead_obstacle(z, hidden).numpy()
        fwd, right, known = comp[:, 0] * COMPASS_SCALE, comp[:, 1] * COMPASS_SCALE, comp[:, 2]
        a = np.where(rng.random(n) < 0.3, rng.choice([TURN_LEFT, TURN_RIGHT], n), FORWARD)
        turn = np.where(right > 0, TURN_RIGHT, TURN_LEFT)
        a[(known > 0.5) & (fwd >= 0.5)] = FORWARD
        goal_side = (known > 0.5) & (fwd < 0.5)
        a[goal_side] = turn[goal_side]
        blocked = danger[:, FORWARD] > 0.5
        a[blocked] = np.where(danger[blocked, TURN_RIGHT] < danger[blocked, TURN_LEFT], TURN_RIGHT, TURN_LEFT)
        at = torch.from_numpy(a)
        hidden = rnn(z.unsqueeze(1), at.unsqueeze(1), hidden).hidden
        for i in np.flatnonzero(active):
            o, _, te, tr, info = envs[i].step(int(a[i]))
            obs[i] = o
            if te or tr:
                active[i] = False
                if te:
                    outcome[i] = info["event"]
    return {k: float((outcome == k).mean()) for k in ("goal", "obstacle", "truncated")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="checkpoints/vae_fp.pt")
    p.add_argument("--rnn", default="checkpoints/mdnrnn_fp20k.pt")
    p.add_argument("--data", nargs="+", default=["data/zseq_fp_4k.npz", "data/zseq_fp_16k.npz"])
    p.add_argument("--moving", type=int, default=0)
    a = p.parse_args()
    vae, _ = ConvVAE.load(a.vae)
    rnn, ckpt = MDNRNN.load(a.rnn)
    seqs = ZSequences.load_many(a.data)
    _, val = split_indices(len(seqs), seed=ckpt["hparams"]["seed"])
    compass_quality(rnn, seqs, val[:500])
    print("Håndlaget agent på kompass + fremsyn:", handcrafted(vae.eval(), rnn, moving=a.moving))


if __name__ == "__main__":
    main()
