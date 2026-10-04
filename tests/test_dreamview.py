import base64

import numpy as np
import torch

from worldmodels.controller.policy import LinearController
from worldmodels.dreamview.__main__ import TEMPLATE, board_json
from worldmodels.dreamview.record import record_episode
from worldmodels.mdnrnn.model import MDNRNN, MDNRNNConfig
from worldmodels.vae.model import ConvVAE, VAEConfig


def tiny_world():
    torch.manual_seed(0)
    vae = ConvVAE(VAEConfig(latent_dim=4, base_channels=8)).eval()
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, eye=True, eye_channels=4,
                              neighbours=True, neighbour_channels=4)).eval()
    c = LinearController(4, 8, extra_dim=12, lookahead=True, sight=True, track=True)
    c.params = np.random.default_rng(0).normal(size=c.num_params).astype(np.float32)
    return vae, rnn, c


def test_record_episode_follows_the_real_env_and_dreams_from_each_step():
    vae, rnn, c = tiny_world()
    rec = record_episode(vae, rnn, c, seed=3, horizon=3)
    assert 1 <= len(rec.steps) <= 50 and rec.outcome in ("goal", "obstacle", "truncated")
    for s in rec.steps:
        assert s.recon.shape == (64, 64, 3) and s.recon.dtype == np.uint8
        assert np.isclose(s.agent_belief.sum(), 1, atol=1e-4) and np.isclose(s.goal_belief.sum(), 1, atol=1e-4)
        assert 1 <= len(s.dream) <= 3
        first = s.dream[0]
        assert first.real_pos is not None and first.real_event in ("move", "goal", "obstacle")
        assert 0 <= first.alive <= 1 and first.frame.shape == (64, 64, 3)
    # Den ekte episoden går skritt for skritt fra startcellen
    path = [s.pos for s in rec.steps] + [rec.final_pos]
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1 for a, b in zip(path, path[1:]))


def test_board_json_indexes_every_frame_in_the_sheet():
    vae, rnn, c = tiny_world()
    data = board_json(record_episode(vae, rnn, c, seed=5, horizon=2), "crash")
    png = base64.b64decode(data["sheet"].split(",", 1)[1])
    height = int.from_bytes(png[20:24], "big")         # IHDR: bredde, så høyde
    frames = [s["recon"] for s in data["steps"]] + [d["f"] for s in data["steps"] for d in s["dream"]]
    assert sorted(frames) == list(range(len(frames))) and height == 64 * len(frames)
    assert "/*DATA*/null" in TEMPLATE.read_text()
