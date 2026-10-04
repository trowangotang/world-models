import numpy as np
import torch

from worldmodels.data import collect_rollouts, episode_paths
from worldmodels.env import GridConfig
from worldmodels.mdnrnn.encode import ZSequences, encode_episodes
from worldmodels.mdnrnn.model import MDNRNN, MDNRNNConfig
from worldmodels.mdnrnn.obstacles import danger_by_direction, previous_maps, train_obstacles
from worldmodels.mdnrnn.tracker import EyeMemory, observe
from worldmodels.vae.model import ConvVAE, VAEConfig


def cell_map(*cells):
    m = torch.zeros(1, 64)
    for c in cells:
        m[0, c] = 1.0
    return m


def test_danger_by_direction_reads_neighbours_now_and_next():
    agent = cell_map(3 * 8 + 3)
    now = cell_map(2 * 8 + 3)          # over agenten
    after = cell_map(3 * 8 + 4)        # flytter seg til høyre for agenten
    assert danger_by_direction(now, after, agent)[0].tolist() == [1, 0, 0, 1]   # opp, ned, venstre, høyre
    # I hjørnet er kanten ingen fare
    assert danger_by_direction(torch.zeros(1, 64), torch.zeros(1, 64), cell_map(0))[0].tolist() == [0, 0, 0, 0]


def test_previous_maps_start_each_episode_without_motion():
    maps = torch.arange(5.0).unsqueeze(1).repeat(1, 64)
    prev = previous_maps(maps, np.array([0, 3]))
    assert prev[:, 0].tolist() == [0, 0, 1, 3, 3]


def test_encode_stores_obstacle_maps_and_old_files_load(tmp_path):
    collect_rollouts(tmp_path / "r", num_episodes=4, seed=0, config=GridConfig(moving_obstacles=3))
    torch.manual_seed(0)
    seqs = encode_episodes(ConvVAE(VAEConfig(latent_dim=4)), episode_paths(tmp_path / "r"))
    assert seqs.obstacles.shape == (len(seqs.mu), 64)
    # Seks hindringer, minus den agenten eventuelt dekker
    assert set(seqs.obstacles.sum(1)) <= {5, 6}
    d = {k: v for k, v in seqs.__dict__.items() if k != "obstacles"}
    np.savez(tmp_path / "old.npz", **d)
    assert (ZSequences.load(tmp_path / "old.npz").obstacles == -1).all()


def test_obstacle_eye_trains_and_memory_keeps_the_previous_map(tmp_path):
    collect_rollouts(tmp_path / "r", num_episodes=6, seed=0, config=GridConfig(moving_obstacles=3))
    torch.manual_seed(0)
    seqs = encode_episodes(ConvVAE(VAEConfig(latent_dim=4)), episode_paths(tmp_path / "r"))
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, eye=True, eye_channels=4,
                              neighbours=True, neighbour_channels=4, obstacle_eye=True, obstacle_channels=4)).eval()
    metrics = train_obstacles(rnn, seqs, np.array([5]), eye_epochs=2, motion_epochs=2, log=lambda m: None)
    assert metrics["eye_loss"][-1] < metrics["eye_loss"][0]
    z = torch.from_numpy(seqs.mu[:3]).float()
    _, _, m1 = observe(rnn, z[:1], None)
    assert m1.obstacles.shape == (1, 64) and m1.obstacles_before is None
    _, _, m2 = observe(rnn, z[1:2], m1.moved(torch.tensor([0])))
    assert torch.equal(m2.obstacles_before, m1.obstacles)
    # Faren i event_probs kommer fra kartene og er en sannsynlighet
    p = rnn.event_probs(None, z[1:2], m2.goal, torch.tensor([0]), m2.agent, None, m2.obstacles_before)
    assert torch.isclose(p.sum(), torch.tensor(1.0)) and (p >= 0).all()


def test_memory_without_obstacles_still_concats_and_repeats():
    m = EyeMemory.fresh(2, 64)
    assert len(EyeMemory.concat([m, m])) == 4 and m.repeat(3).obstacles is None
