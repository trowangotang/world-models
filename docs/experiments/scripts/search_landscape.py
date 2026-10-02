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
X = np.random.default_rng(0).normal(size=(2000, 20))
f = np.concatenate([dream_fitness(c, embed(X[i:i+200]), rnn, starts, cfg) for i in range(0, 2000, 200)])
print("tilfeldige: median %.3f  90%% %.3f  maks %.3f  andel > -0.40: %.4f" % (np.median(f), np.quantile(f, .9), f.max(), (f > -0.40).mean()))
best = X[np.argsort(-f)[:5]]
d = dream_fitness(c, embed(best), rnn, starts, cfg, return_details=True)
print("beste 5:", np.round(d["fitness"], 3), "mål", np.round(d["p_goal"], 2), "hindr", np.round(d["p_obstacle"], 2))
# cosine similarity of best to hc in the 16 weight dims (sign structure)
print("cos til håndlaget:", np.round((best[:, :16] @ hc[:16]) / np.linalg.norm(best[:, :16], axis=1) / np.linalg.norm(hc[:16]), 2))
