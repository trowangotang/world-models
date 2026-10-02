"""Steg 6: en håndlaget lineær controller på øyets syn (+ fremsyn), i det ekte miljøet (D49).

Viser hva den samme regelen som steg 4d/4e prøver å lære, klarer når inndata kommer fra øyet."""
import numpy as np, torch
from worldmodels.controller.agent import WorldModelAgent, run_real_episodes
from worldmodels.controller.policy import LinearController
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.model import ConvVAE

vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
rnn, _ = MDNRNN.load("checkpoints/mdnrnn_eye.pt")


def handcrafted(lookahead_weight: float) -> LinearController:
    c = LinearController(32, 256, 4, extra_dim=8 if lookahead_weight else 4, lookahead=bool(lookahead_weight), sight=True)
    W, b = c.unflatten(c.params)
    W = torch.zeros_like(W)
    off = 32 + 256  # syn: ar, ac, gr, gc ; handlinger: opp, ned, venstre, høyre
    W[off + 0] = torch.tensor([1., -1., 0., 0.])
    W[off + 2] = torch.tensor([-1., 1., 0., 0.])
    W[off + 1] = torch.tensor([0., 0., 1., -1.])
    W[off + 3] = torch.tensor([0., 0., -1., 1.])
    if lookahead_weight:
        for a in range(4):  # fremsyn: straff handlingen med høy sannsynlighet for krasj
            W[off + 4 + a, a] = -lookahead_weight
    c.params = torch.cat([W.flatten(), torch.zeros(4)]).numpy()
    return c


if __name__ == "__main__":
    for name, w in [("syn, gå mot målet", 0.0), ("syn + fremsyn (vekt 5)", 5.0)]:
        print(name, run_real_episodes(WorldModelAgent(vae, rnn, handcrafted(w)), 1000, seed=100_000), flush=True)
    # Brukes som diagnostikk i evalueringen (python -m worldmodels.evaluation)
    handcrafted(0.0).save("checkpoints/controller_eye_handcrafted.npz")
