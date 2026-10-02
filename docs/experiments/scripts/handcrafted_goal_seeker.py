import numpy as np, torch
from worldmodels.controller.agent import WorldModelAgent, run_real_episodes
from worldmodels.controller.dream import DreamConfig, dream_fitness, make_warm_starts
from worldmodels.controller.policy import LinearController
from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.train import split_indices
from worldmodels.vae.model import ConvVAE
vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
rnn, _ = MDNRNN.load("checkpoints/mdnrnn_pos.pt")
seqs = ZSequences.load_many(["data/v2/zseq_20k.npz","data/v2/zseq_r1.npz","data/v2/zseq_r2.npz","data/v2/zseq_r3.npz"])
tr, _ = split_indices(len(seqs))
starts = make_warm_starts(rnn, seqs, tr, context=5).sample(1024, np.random.default_rng(1))
c = LinearController(32, 256, 4, extra_dim=4)
W, b = c.unflatten(c.params)
W = torch.zeros_like(W)
# features: ar, ac, gr, gc ; actions: up, down, left, right
off = 32 + 256
W[off + 0] = torch.tensor([1., -1., 0., 0.])   # ar
W[off + 2] = torch.tensor([-1., 1., 0., 0.])   # gr
W[off + 1] = torch.tensor([0., 0., 1., -1.])   # ac
W[off + 3] = torch.tensor([0., 0., -1., 1.])   # gc
c.params = torch.cat([W.flatten(), torch.zeros(4)]).numpy()
learned = LinearController.load("checkpoints/controller_pos.npz")
for name, ctrl in [("håndlaget: gå mot målet", c), ("lært på [z,h]", learned)]:
    d = dream_fitness(ctrl, ctrl.params[None], rnn, starts, DreamConfig(horizon=10, temperature=0.0), return_details=True)
    real = run_real_episodes(WorldModelAgent(vae, rnn, ctrl), 1000)
    print(name, "drøm fitness %.3f mål %.3f hindring %.3f" % (d["fitness"][0], d["p_goal"][0], d["p_obstacle"][0]), "| ekte", real, flush=True)
