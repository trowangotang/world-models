import numpy as np, torch
from torch.nn import functional as F
from worldmodels.data import episode_paths, load_episode
from worldmodels.env.parse import GOAL, AGENT, find_cell, parse_cells
from worldmodels.mdnrnn.encode import encode_episodes, EVENT_GOAL
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.train import make_batch
from worldmodels.vae.evaluate import train_probe
from worldmodels.vae.model import ConvVAE

vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
paths = episode_paths("logs_tmp/heldout_rand")
seqs = encode_episodes(vae, paths)
goal = []  # goal cell per step (from first frame)
for p in paths:
    ep = load_episode(p)
    g = find_cell(parse_cells(ep.obs[:1]), GOAL)[0]
    goal.append(np.full(len(ep), g))
goal = np.concatenate(goal)
# per-step: z_t, agent cell t, action
idx = np.arange(len(seqs))
for name in ["checkpoints/mdnrnn_direct_k5.pt", "checkpoints/iter/mdnrnn_r3.pt"]:
    m, _ = MDNRNN.load(name)
    H, Z, A, C, E, P = [], [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            b = make_batch(seqs, idx[s:s+256])
            out = m(b.inputs(False), b.actions)
            k = b.mask
            H.append(out.h[k].numpy()); Z.append(b.z_mu[:, :-1][k].numpy()); A.append(b.actions[k].numpy())
            C.append(b.agent_cell[:, :-1][k].numpy()); E.append(b.events[k].numpy())
            P.append(F.softmax(out.event_logits, -1)[..., EVENT_GOAL][k].numpy())
    H, Z, A, C, E, P = map(np.concatenate, (H, Z, A, C, E, P))
    # target cell of the move
    r, c = C // 8, C % 8
    dr = np.array([-1, 1, 0, 0])[A]; dc = np.array([0, 0, -1, 1])[A]
    tr, tc = np.clip(r + dr, 0, 7), np.clip(c + dc, 0, 7)
    toward = (tr * 8 + tc) == goal
    adjacent = (np.abs(r - goal // 8) + np.abs(c - goal % 8)) == 1
    print(name)
    print(" goal events", (E == EVENT_GOAL).sum(), " steps toward goal", toward.sum(), " agree", ((E == EVENT_GOAL) == toward).mean())
    print(" mean p_goal: toward %.3f, adjacent-not-toward %.3f, elsewhere %.3f" % (P[toward].mean(), P[adjacent & ~toward].mean(), P[~adjacent].mean()))
    n = len(Z); tr_i = np.arange(n) < int(n * 0.8)
    ok = goal >= 0
    for nm, X in [("z", Z), ("z,h", np.hstack([Z, H]))]:
        print(f" probe goal cell from {nm}: linear %.2f" % train_probe(X[tr_i & ok], goal[tr_i & ok], X[~tr_i & ok], goal[~tr_i & ok]),
              " mlp %.2f" % train_probe(X[tr_i & ok], goal[tr_i & ok], X[~tr_i & ok], goal[~tr_i & ok], hidden=256))
