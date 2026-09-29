import numpy as np, torch
from worldmodels.controller.dream import DreamConfig, dream_fitness, make_warm_starts
from worldmodels.controller.policy import LinearController
from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.train import split_indices
rnn, _ = MDNRNN.load("checkpoints/mdnrnn_pos.pt")
seqs = ZSequences.load_many(["data/v2/zseq_20k.npz","data/v2/zseq_r1.npz","data/v2/zseq_r2.npz","data/v2/zseq_r3.npz"])
tr, _ = split_indices(len(seqs))
starts = make_warm_starts(rnn, seqs, tr, context=5).sample(512, np.random.default_rng(3))
c = LinearController(32, 256, 4, extra_dim=4)
n_w = c.in_dim * 4
sub = np.concatenate([np.arange((32 + 256) * 4, n_w), np.arange(n_w, n_w + 4)])
def embed(x):
    full = np.zeros((len(x), c.num_params), np.float32); full[:, sub] = x; return full
cfg = DreamConfig(horizon=10, temperature=0.0, remaining_steps=35)
hc = np.zeros(20); W = np.zeros((4, 4)); W[0] = [1, -1, 0, 0]; W[2] = [-1, 1, 0, 0]; W[1] = [0, 0, 1, -1]; W[3] = [0, 0, -1, 1]; hc[:16] = W.flatten()
d = dream_fitness(c, embed(np.stack([hc, np.zeros(20)])), rnn, starts, cfg, return_details=True)
print("håndlaget", d["fitness"][0], "null (alltid opp)", d["fitness"][1])

u = hc / np.linalg.norm(hc)
rng = np.random.default_rng(1)
for eps in [0.1, 0.25, 0.5, 0.75, 1.0]:
    N = rng.normal(size=(100, 20)); N /= np.linalg.norm(N, axis=1, keepdims=True)
    X = u + eps * N
    f = dream_fitness(c, embed(X), rnn, starts, cfg)
    print("støy %.2f: median %.3f  andel > -0.40: %.2f" % (eps, np.median(f), (f > -0.40).mean()), flush=True)
