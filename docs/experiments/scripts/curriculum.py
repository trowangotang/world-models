import sys, numpy as np, torch
from worldmodels.controller.agent import WorldModelAgent, run_real_episodes
from worldmodels.controller.dream import DreamConfig, dream_fitness, make_warm_starts
from worldmodels.controller.es import EvolutionStrategy
from worldmodels.controller.policy import LinearController
from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.train import split_indices
from worldmodels.vae.model import ConvVAE
feat = sys.argv[1] == "feat"
vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
rnn, _ = MDNRNN.load("checkpoints/mdnrnn_pos.pt")
seqs = ZSequences.load_many(["data/v2/zseq_20k.npz","data/v2/zseq_r1.npz","data/v2/zseq_r2.npz","data/v2/zseq_r3.npz"])
tr, _ = split_indices(len(seqs))
starts = make_warm_starts(rnn, seqs, tr, context=5)
# avstand agent-mål i ekte data etter oppvarming
dist = []
for i in starts.episodes:
    e = seqs.episode(i); a, g = e["agent_cell"][5], e["goal_cell"][5]
    dist.append(abs(a // 8 - g // 8) + abs(a % 8 - g % 8) if a >= 0 and g >= 0 else 99)
dist = np.array(dist); near_idx = np.flatnonzero(dist <= 3); print("nær-starter", len(near_idx), "av", len(dist), flush=True)
c = LinearController(32, 256, 4, extra_dim=4 if feat else 0)
es = EvolutionStrategy(c.num_params, population=32, sigma=0.1, lr=0.03, seed=0)
cfg = DreamConfig(horizon=10, temperature=0.0, remaining_steps=35)
rng = np.random.default_rng(0)
from worldmodels.controller.dream import WarmStarts
def pick(n):
    k = n // 2
    i = np.concatenate([rng.choice(near_idx, k, replace=False), rng.choice(len(starts), n - k, replace=False)])
    return starts.subset(i)
for g in range(1, 301):
    es.tell(dream_fitness(c, es.ask(), rnn, pick(256), cfg))
    if g % 50 == 0:
        c.params = es.theta.astype(np.float32)
        print(g, run_real_episodes(WorldModelAgent(vae, rnn, c), 500), flush=True)
c.save(f"checkpoints/controller_curriculum_{sys.argv[1]}.npz")
