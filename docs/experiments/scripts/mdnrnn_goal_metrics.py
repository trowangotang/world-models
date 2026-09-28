import json, sys
import numpy as np, torch
from worldmodels.data import episode_paths
from worldmodels.mdnrnn.encode import encode_episodes
from worldmodels.mdnrnn.evaluate import one_step_metrics
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.train import make_batch
from worldmodels.vae.evaluate import train_probe
from worldmodels.vae.model import ConvVAE

vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
sets = {k: encode_episodes(vae, episode_paths(f"logs_tmp/heldout_{k}")) for k in ("rand", "ctrl")}
out = {}
for path in sys.argv[1:]:
    m, _ = MDNRNN.load(path)
    res = {}
    for k, s in sets.items():
        r = one_step_metrics(m, vae, s, np.arange(len(s)))
        res[k] = {x: round(r[x], 3) for x in ("goal_recall", "goal_precision", "obstacle_recall", "obstacle_precision", "agent_cell_acc")}
    s = sets["rand"]; H, A, G = [], [], []
    with torch.no_grad():
        for st in range(0, len(s), 256):
            b = make_batch(s, range(st, min(st + 256, len(s))))
            o = m(b.inputs(False), b.actions); k = b.mask
            H.append(o.h[k].numpy()); A.append(b.agent_cell[:, 1:][k].numpy()); G.append(b.goal_cell[:, 1:][k].numpy())
    H, A, G = map(np.concatenate, (H, A, G))
    tr = np.arange(len(H)) < int(0.8 * len(H))
    for nm, y in (("agent", A), ("goal", G)):
        ok = y >= 0
        res[f"probe_{nm}_linear_h"] = round(train_probe(H[tr & ok], y[tr & ok], H[~tr & ok], y[~tr & ok]), 3)
    out[path] = res
    print(path, json.dumps(res), flush=True)
json.dump(out, open("logs_tmp/m_eval4c.json", "w"), indent=2)
