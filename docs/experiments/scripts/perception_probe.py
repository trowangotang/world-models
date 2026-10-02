"""Steg 6: hvor godt kan agentens og målets celle leses fra z alene, med ulike lesere? (D48)"""
import numpy as np, torch, time
from torch import nn
from torch.nn import functional as F
from worldmodels.mdnrnn.encode import ZSequences

torch.manual_seed(0)
s = ZSequences.load("data/v3/zseq_20k.npz")
n_obs = s.lengths + 1
ep = np.repeat(np.arange(len(s)), n_obs)
train = ep < int(0.9 * len(s))
Z = torch.from_numpy(s.mu).float()
Y = torch.from_numpy(np.stack([s.agent_cell, s.goal_cell], 1)).long()
ok = (Y >= 0).all(1).numpy()
tr, va = np.flatnonzero(train & ok), np.flatnonzero(~train & ok)


class Spatial(nn.Module):
    """z -> 8x8-kart med logits for agent og mål, formet som starten på en dekoder."""
    def __init__(self, D=32, C=64):
        super().__init__()
        self.fc = nn.Linear(D, C * 4 * 4)
        self.net = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(C, C, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(C, C, 3, padding=1), nn.ReLU(), nn.Conv2d(C, 2, 1))
        self.C = C
    def forward(self, z):
        return self.net(self.fc(z).view(-1, self.C, 4, 4)).flatten(2)  # (B, 2, 64)


def mlp(D=32, H=512):
    return nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 128), nn.Unflatten(1, (2, 64)))


for name, model in [("linear", nn.Sequential(nn.Linear(32, 128), nn.Unflatten(1, (2, 64)))), ("mlp", mlp()), ("spatial", Spatial())]:
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(6):
        perm = np.random.default_rng(epoch).permutation(tr)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(model(Z[b]).reshape(-1, 64), Y[b].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = torch.cat([model(Z[va[i:i + 4096]]).argmax(-1) for i in range(0, len(va), 4096)])
    acc = (pred == Y[va]).float().mean(0).numpy()
    print(f"{name}: agent {acc[0]:.3f} mål {acc[1]:.3f}  [{time.time() - t0:.0f}s]", flush=True)
