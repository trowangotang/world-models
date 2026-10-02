import json, numpy as np, torch
from worldmodels.controller.agent import WorldModelAgent, run_real_episodes
from worldmodels.controller.policy import LinearController
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.model import ConvVAE
vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
hc = LinearController(32, 256, 4, extra_dim=4)
W, _ = hc.unflatten(hc.params); W = torch.zeros_like(W); o = 288
W[o] = torch.tensor([1., -1, 0, 0]); W[o + 2] = torch.tensor([-1., 1, 0, 0]); W[o + 1] = torch.tensor([0., 0, 1, -1]); W[o + 3] = torch.tensor([0., 0, -1, 1])
hc.params = torch.cat([W.flatten(), torch.zeros(4)]).numpy()
from worldmodels.data import RandomPolicy
runs = {
  "steg 4: ES, [z,h]": ("checkpoints/mdnrnn_direct_k5.pt", LinearController.load("checkpoints/controller_s256.npz")),
  "steg 4c: ES, posisjonstro, læreplan": ("checkpoints/mdnrnn_pos.pt", LinearController.load("checkpoints/controller_curriculum_feat.npz")),
  "ES + formet belønning": ("checkpoints/mdnrnn_pos.pt", LinearController.load("checkpoints/controller_es_shaped.npz")),
  "CMA-ES + formet belønning": ("checkpoints/mdnrnn_pos.pt", LinearController.load("checkpoints/controller_cma_shaped.npz")),
  "CMA-ES + formet, liten spredning z/h (0,1)": ("checkpoints/mdnrnn_pos.pt", LinearController.load("checkpoints/controller_cma_zh01.npz")),
  "CMA-ES + formet, liten spredning z/h (0,03)": ("checkpoints/mdnrnn_pos.pt", LinearController.load("checkpoints/controller_cma_zh003.npz")),
  "CMA-ES + formet, bare posisjonstro (20 parametre)": ("checkpoints/mdnrnn_pos.pt", LinearController.load("checkpoints/controller_cma_zh0.npz")),
  "diagnostikk: håndlaget": ("checkpoints/mdnrnn_pos.pt", hc),
}
out = {}
for k, (rp, c) in runs.items():
    rnn, _ = MDNRNN.load(rp)
    out[k] = run_real_episodes(WorldModelAgent(vae, rnn, c), 1000)
    print(k, {x: round(v, 3) for x, v in out[k].items()}, flush=True)
json.dump(out, open("logs_tmp/final4d.json", "w"), indent=2)
