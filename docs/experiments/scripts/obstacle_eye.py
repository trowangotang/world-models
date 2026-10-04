"""Kan øyet lese hindringene, og kan bevegelsen regnes ut fra to bilder? (steg 11, decisions.md D70)

Nærsynet med minne (D66) ser 28 % av hindringene som er på vei inn i nabocellen. Her prøver vi en
annen vei, på agentepisodene fra steg 10 (rask test, ingen ny M):

1. **Hindringsøyet** leser et 8x8-kart over hindringene fra z, formet som øyet fra steg 6.
2. **Bevegelsen** spås av et lite konvolusjonsnett fra kartene i forrige og dette bildet. Det er
   alt som trengs: en hindring som har flyttet seg, fortsetter, og snur bare ved noe den ser.
3. **Fare** i en retning er da hindring i nabocellen nå, eller i det spådde neste kartet.

Det måles på de samme valideringsepisodene som M ikke er trent på, sammen med nærsynet fra steg 10.
Fasitkartene leses fra de ekte bildene, så vi ser også taket: bevegelsen fra sanne kart.

    PYTHONPATH=. python docs/experiments/scripts/obstacle_eye.py
"""

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from worldmodels.data import episode_paths, load_episode
from worldmodels.env.gridworld import ACTIONS
from worldmodels.env.parse import OBSTACLE, parse_cells
from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.neighbours import _logits, neighbour_data
from worldmodels.mdnrnn.train import split_indices

SIDE = 8
RANDOM_EPISODES = 20_000   # zseq_20k ligger først i M sitt datasett, agentepisodene etter


class ObstacleEye(nn.Module):
    """z (B, D) -> logits (B, 64): er det en hindring i cellen?"""

    def __init__(self, latent_dim=32, channels=32):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(latent_dim, channels * 16)
        self.net = nn.Sequential(
            nn.ReLU(), nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Conv2d(channels, 1, 1),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, self.channels, 4, 4)).flatten(1)


class Motion(nn.Module):
    """(forrige kart, dette kartet) (B, 2, 64) -> logits for neste kart (B, 64). Bare lokale regler."""

    def __init__(self, channels=32):
        super().__init__()
        # Tre lag 3x3 ser tre celler ut: nok til å se hvor en hindring kom fra og om den må snu
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, maps):
        return self.net(maps.view(-1, 2, SIDE, SIDE)).flatten(1)


def obstacle_maps(paths):
    return np.concatenate([(parse_cells(load_episode(p).obs) == OBSTACLE).reshape(-1, SIDE * SIDE) for p in paths])


def fit(net, inputs, targets, epochs=8, lr=1e-3, batch=512, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(len(targets)))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.binary_cross_entropy_with_logits(net(inputs[b]), targets[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"  epoke {epoch}: tap {total / len(perm):.4f}  [{time.time() - t0:.0f}s]", flush=True)
    net.eval()
    return net


def pairs(maps, seqs):
    """(forrige, dette) for hvert bilde; i første bilde er forrige = dette (ingen bevegelse sett)."""
    prev = maps.clone()
    prev[1:] = maps[:-1]
    prev[seqs.obs_offsets] = maps[seqs.obs_offsets]
    return torch.stack([prev, maps], 1)


def neighbour_cells(agent_cells):
    """(N,) -> (N, 4) celleindeks for naboen i hver retning, -1 utenfor kanten."""
    r, c = agent_cells // SIDE, agent_cells % SIDE
    out = np.full((len(agent_cells), len(ACTIONS)), -1)
    for k, (dr, dc) in enumerate(ACTIONS):
        nr, nc = r + dr, c + dc
        inside = (nr >= 0) & (nr < SIDE) & (nc >= 0) & (nc < SIDE)
        out[inside, k] = (nr * SIDE + nc)[inside]
    return out


def danger_from_maps(now, after, cells):
    """Fare (N, 4) fra sannsynlighetskart nå og etter neste skritt."""
    idx = torch.from_numpy(np.maximum(cells, 0))
    p = torch.maximum(now.gather(1, idx), after.gather(1, idx))
    return torch.where(torch.from_numpy(cells >= 0), p, torch.zeros_like(p))


def report(name, p, near, danger):
    hit, arriving = p > 0.5, danger & ~near
    print(f"{name}: står der nå {float((hit & near).sum() / near.sum()):.1%}, "
          f"på vei inn {float((hit & arriving).sum() / arriving.sum()):.1%}, "
          f"falsk alarm {float((hit & ~danger).sum() / (~danger).sum()):.2%}", flush=True)


@torch.no_grad()
def predict(net, x, batch=8192):
    return torch.cat([torch.sigmoid(net(x[i:i + batch])) for i in range(0, len(x), batch)])


def main():
    torch.set_num_threads(4)
    seqs = ZSequences.load("data/moving/zseq_agent.npz")
    paths = episode_paths("data/moving/rollouts_agent")
    assert len(paths) == len(seqs)
    rnn, ckpt = MDNRNN.load("checkpoints/mdnrnn_moving.pt")
    rnn.eval()
    _, val_all = split_indices(RANDOM_EPISODES + len(seqs), seed=ckpt.get("hparams", {}).get("seed", 0))
    val_idx = val_all[val_all >= RANDOM_EPISODES] - RANDOM_EPISODES
    train_idx = np.setdiff1d(np.arange(len(seqs)), val_idx)
    train, val = seqs.subset(train_idx), seqs.subset(val_idx)
    print(f"{len(train)} episoder / {len(train.mu)} bilder til trening, {len(val)} / {len(val.mu)} til validering")

    maps = {k: torch.from_numpy(obstacle_maps([paths[i] for i in idx])).float()
            for k, idx in (("train", train_idx), ("val", val_idx))}
    z = {"train": torch.from_numpy(train.mu).float(), "val": torch.from_numpy(val.mu).float()}

    print("Hindringsøyet:")
    eye = fit(ObstacleEye(), z["train"], maps["train"])
    seen = {k: predict(eye, z[k]) for k in z}
    hit, true = seen["val"] > 0.5, maps["val"].bool()
    print(f"  celler riktig {float((hit == true).float().mean()):.2%}, hindringer funnet "
          f"{float((hit & true).sum() / true.sum()):.1%}, hele kartet riktig {float((hit == true).all(1).float().mean()):.1%}")

    # Bevegelse: trenes på øyets egne kart, så den lærer å leve med feilene; neste kart er fasit.
    # Siste bilde i hver episode har ikke noe neste bilde og tas ut.
    def motion_data(m, s):
        x = pairs(m, s)
        last = np.zeros(len(m), bool)
        last[s.obs_offsets + s.lengths] = True
        y = torch.zeros_like(m)
        y[:-1] = maps_of[id(s)][1:]
        return x[~last], y[~last], ~last

    maps_of = {id(train): maps["train"], id(val): maps["val"]}
    print("Bevegelse fra øyets kart:")
    xt, yt, _ = motion_data(seen["train"], train)
    motion = fit(Motion(), xt, yt, epochs=20)
    print("Bevegelse fra sanne kart (tak):")
    xt, yt, _ = motion_data(maps["train"], train)
    motion_true = fit(Motion(), xt, yt, epochs=20)

    # Måling på valideringsbildene, samme utvalg som nærsynet (kjent agent, mål og fare, ikke siste bilde)
    last = np.zeros(len(val.mu), bool)
    last[val.obs_offsets + val.lengths] = True
    keep = ~last & (val.danger >= 0).all(1) & (val.agent_cell >= 0) & (val.goal_cell >= 0)
    near = torch.from_numpy(val.near_obstacle[keep] == 1)
    danger = torch.from_numpy(val.danger[keep] == 1)
    cells = neighbour_cells(val.agent_cell[keep])
    k = torch.from_numpy(keep)

    _, nb_val = neighbour_data(rnn, val, np.arange(len(val)), memory=True)
    assert len(nb_val[0]) == len(near)
    p = torch.cat([torch.sigmoid(_logits(rnn.neighbours, nb_val, slice(i, i + 8192))[:, 1])
                   for i in range(0, len(near), 8192)])
    report("Nærsynet fra steg 10        ", p, near, danger)
    report("Hindringsøyet, bare nå      ", danger_from_maps(seen["val"][k], seen["val"][k], cells), near, danger)
    after = predict(motion, pairs(seen["val"], val))
    report("Hindringsøyet + bevegelse   ", danger_from_maps(seen["val"][k], after[k], cells), near, danger)
    after_true = predict(motion_true, pairs(maps["val"], val))
    report("Sanne kart + bevegelse (tak)", danger_from_maps(maps["val"][k], after_true[k], cells), near, danger)
    # Fra og med andre bilde, der bevegelsen kan ha blitt sett
    later = torch.from_numpy(np.concatenate([np.arange(n + 1) > 0 for n in val.lengths])[keep])
    report("  ... samme, fra skritt 1   ", danger_from_maps(seen["val"][k], after[k], cells)[later],
           near[later], danger[later])
    report("  ... steg 10, fra skritt 1 ", p[later], near[later], danger[later])


if __name__ == "__main__":
    main()
