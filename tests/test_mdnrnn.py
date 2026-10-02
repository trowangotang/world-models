import numpy as np
import pytest
import torch
from torch.nn import functional as F

from worldmodels.data import RandomPolicy, collect_rollouts, episode_paths, run_episode
from worldmodels.env import GridDodgeEnv
from worldmodels.mdnrnn.encode import (
    EVENT_GOAL,
    EVENT_MOVE,
    EVENT_OBSTACLE,
    ZSequences,
    encode_episodes,
    episode_events,
)
from worldmodels.mdnrnn.model import MDNRNN, MDNRNNConfig, mdn_nll, most_likely_mean, sample_next
from worldmodels.mdnrnn.train import batch_loss, make_batch, oversample_goal_episodes, split_indices
from worldmodels.vae.model import ConvVAE, VAEConfig


@pytest.fixture(scope="module")
def seqs(tmp_path_factory) -> ZSequences:
    d = tmp_path_factory.mktemp("rollouts")
    collect_rollouts(d, num_episodes=6, seed=0)
    torch.manual_seed(0)
    return encode_episodes(ConvVAE(VAEConfig(latent_dim=4)), episode_paths(d))


def test_episode_events_marks_goal_obstacle_and_ignores_truncation():
    env = GridDodgeEnv()
    kinds = set()
    for seed in range(40):
        ep = run_episode(env, RandomPolicy(seed=seed), seed=seed)
        ev = episode_events(ep)
        assert (ev[:-1] == EVENT_MOVE).all()
        if ep.truncated[-1]:
            assert ev[-1] == EVENT_MOVE
        elif ep.rewards[-1] > 0:
            assert ev[-1] == EVENT_GOAL
        else:
            assert ev[-1] == EVENT_OBSTACLE
        kinds.add(int(ev[-1]))
    assert kinds == {EVENT_MOVE, EVENT_GOAL, EVENT_OBSTACLE}


def test_encoded_sequences_have_consistent_lengths(seqs):
    assert len(seqs) == 6
    assert len(seqs.mu) == len(seqs.agent_cell) == int((seqs.lengths + 1).sum())
    assert len(seqs.actions) == len(seqs.events) == int(seqs.lengths.sum())
    ep = seqs.episode(2)
    assert len(ep["mu"]) == len(ep["actions"]) + 1
    assert (ep["agent_cell"] >= 0).all()


def test_zsequences_save_load_and_subset(seqs, tmp_path):
    path = tmp_path / "z.npz"
    seqs.save(path)
    loaded = ZSequences.load(path)
    assert np.array_equal(loaded.mu, seqs.mu)
    sub = seqs.subset([1, 3])
    assert np.array_equal(sub.episode(1)["actions"], seqs.episode(3)["actions"])


def test_make_batch_pads_and_masks(seqs):
    batch = make_batch(seqs, [0, 1, 2])
    T = int(seqs.lengths[:3].max())
    assert batch.z_mu.shape == (3, T + 1, 4)
    assert batch.mask.sum().item() == int(seqs.lengths[:3].sum())
    for b in range(3):
        n = int(seqs.lengths[b])
        assert batch.mask[b, :n].all() and not batch.mask[b, n:].any()
    assert torch.equal(batch.inputs(sample=False), batch.z_mu[:, :-1])
    assert not torch.equal(batch.inputs(sample=True), batch.z_mu[:, :-1])


def test_split_indices_is_disjoint_and_complete():
    tr, va = split_indices(50, val_fraction=0.2)
    assert len(va) == 10 and len(tr) == 40
    assert sorted(np.concatenate([tr, va]).tolist()) == list(range(50))


def test_model_output_shapes_and_residual_means():
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=16, num_mixtures=3))
    z = torch.randn(2, 5, 4)
    a = torch.randint(0, 4, (2, 5))
    out = model(z, a)
    assert out.logit_pi.shape == (2, 5, 3)
    assert out.mu.shape == out.log_sigma.shape == (2, 5, 3, 4)
    assert out.event_logits.shape == (2, 5, 3)
    assert out.h.shape == (2, 5, 16)
    # Med nullstilt mu-hode er alle komponentene lik inndata (residualparametrisering)
    torch.nn.init.zeros_(model.mu_head.weight)
    torch.nn.init.zeros_(model.mu_head.bias)
    assert torch.allclose(model(z, a).mu, z.unsqueeze(2).expand(-1, -1, 3, -1))


def test_mdn_nll_matches_single_gaussian():
    model = MDNRNN(MDNRNNConfig(latent_dim=2, hidden_dim=8, num_mixtures=1))
    out = model(torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.long))
    target = out.mu[:, :, 0] + 0.3
    expected = -torch.distributions.Normal(out.mu[:, :, 0], out.log_sigma[:, :, 0].exp()).log_prob(target).sum(-1) / 2
    assert torch.allclose(mdn_nll(out, target), expected, atol=1e-5)


def test_sampling_temperature_zero_is_most_likely_mean():
    model = MDNRNN(MDNRNNConfig(latent_dim=3, hidden_dim=8, num_mixtures=4))
    out = model(torch.randn(2, 3, 3), torch.randint(0, 4, (2, 3)))
    assert torch.equal(sample_next(out, temperature=0), most_likely_mean(out))
    g = torch.Generator().manual_seed(0)
    s = sample_next(out, temperature=1.0, generator=g)
    assert s.shape == (2, 3, 3) and torch.isfinite(s).all()


def test_training_reduces_loss_on_small_data(seqs):
    torch.manual_seed(0)
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=32, num_mixtures=2))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = make_batch(seqs, range(len(seqs)))
    first = batch_loss(model, batch, sample_inputs=False, event_weight=1.0)["loss"].item()
    for _ in range(80):
        loss = batch_loss(model, batch, sample_inputs=False, event_weight=1.0)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first - 1.0


def test_save_and_load_roundtrip(tmp_path):
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2)).eval()
    model.save(tmp_path / "m.pt")
    loaded, _ = MDNRNN.load(tmp_path / "m.pt")
    z, a = torch.randn(1, 3, 4), torch.randint(0, 4, (1, 3))
    assert torch.allclose(model(z, a).mu, loaded(z, a).mu)


def test_evaluation_runs_end_to_end(seqs):
    from worldmodels.mdnrnn.evaluate import dream_metrics, one_step_metrics, probe_metrics

    torch.manual_seed(0)
    vae = ConvVAE(VAEConfig(latent_dim=4)).eval()
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2)).eval()
    idx = np.arange(len(seqs))
    one = one_step_metrics(model, vae, seqs, idx)
    assert one["steps"] == int(seqs.lengths.sum())
    assert 0.0 <= one["agent_cell_acc"] <= 1.0
    dream = dream_metrics(model, vae, seqs, idx, horizon=3)
    assert len(dream["dream_agent_cell_acc"]) == 3
    assert dream["dream_episodes_alive"][0] == len(seqs)
    warm = dream_metrics(model, vae, seqs, idx, horizon=3, context=2)
    assert warm["dream_episodes_alive"][0] == int((seqs.lengths > 2).sum())
    probes = probe_metrics(model, seqs, idx[:4], idx[4:])
    assert set(probes) == {"probe_agent_linear_z", "probe_agent_linear_h", "probe_agent_linear_z_and_h"}


def test_event_class_weights_upweight_rare_events():
    from worldmodels.mdnrnn.train import event_class_weights

    events = np.array([0] * 100 + [1] * 1 + [2] * 4)
    w = event_class_weights(events)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(10.0)
    assert w[2] == pytest.approx(5.0)


def test_linear_input_variant_has_no_input_mlp():
    linear = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, input_mlp=False))
    mlp = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, input_mlp=True))
    assert isinstance(linear.input_net, torch.nn.Identity)
    assert linear.lstm.input_size == 4 + 4 and mlp.lstm.input_size == 8


def test_direct_path_lets_first_step_depend_on_input():
    """Med direkte vei påvirker inndata prediksjonen også gjennom trunk, ikke bare via LSTM-en."""
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, direct_path=True))
    assert model.trunk is not None
    assert MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, direct_path=False)).trunk is None
    out = model(torch.randn(2, 3, 4), torch.randint(0, 4, (2, 3)))
    assert out.mu.shape == (2, 3, 5, 4)


def test_old_checkpoint_config_loads_without_new_layers(tmp_path):
    old = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, input_mlp=False, direct_path=False))
    torch.save({"config": {"latent_dim": 4, "num_actions": 4, "hidden_dim": 8, "num_mixtures": 5},
                "state_dict": old.state_dict()}, tmp_path / "old.pt")
    loaded, _ = MDNRNN.load(tmp_path / "old.pt")
    assert loaded.trunk is None and isinstance(loaded.input_net, torch.nn.Identity)


def test_dream_image_has_two_rows_per_episode(seqs):
    from worldmodels.mdnrnn.evaluate import dream_image

    vae = ConvVAE(VAEConfig(latent_dim=4)).eval()
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8)).eval()
    long_eps = [i for i in range(len(seqs)) if seqs.lengths[i] >= 4][:2]
    img = dream_image(model, vae, seqs, long_eps, context=2, steps=2)
    assert img.dtype == np.uint8 and img.shape[2] == 3
    assert img.shape[0] == 2 * len(long_eps) * (128 + 2) + 2


def test_goal_cell_is_constant_per_episode_and_old_files_load(seqs, tmp_path):
    for i in range(len(seqs)):
        g = seqs.episode(i)["goal_cell"]
        assert (g == g[0]).all() and g[0] >= 0
    old = {k: v for k, v in seqs.__dict__.items() if k != "goal_cell"}
    np.savez(tmp_path / "old.npz", **old)
    assert (ZSequences.load(tmp_path / "old.npz").goal_cell == -1).all()


def test_position_head_loss_is_learnable(seqs):
    torch.manual_seed(0)
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=16, num_mixtures=2, position_head=True))
    batch = make_batch(seqs, range(len(seqs)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        parts = batch_loss(model, batch, False, 1.0, position_weight=1.0)
        first = first if first is not None else parts["position_ce"].item()
        opt.zero_grad()
        parts["loss"].backward()
        opt.step()
    assert parts["position_ce"].item() < 0.7 * first


def test_models_without_position_head_have_zero_position_loss(seqs):
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2))
    parts = batch_loss(model, make_batch(seqs, [0, 1]), False, 1.0, position_weight=5.0)
    assert parts["position_ce"].item() == 0.0


def test_oversampling_repeats_only_goal_episodes(seqs):
    idx = np.arange(len(seqs))
    goal = {i for i in idx if (seqs.episode(i)["events"] == EVENT_GOAL).any()}
    out = oversample_goal_episodes(seqs, idx, 3)
    assert len(out) == len(idx) + 2 * len(goal)
    assert set(out[len(idx):]) == goal
    assert np.array_equal(oversample_goal_episodes(seqs, idx, 1), idx)


def test_near_obstacles_marks_neighbours_and_ignores_walls():
    from worldmodels.mdnrnn.encode import near_obstacles

    grid = np.zeros((8, 8), bool)
    grid[2, 3] = True   # over (3, 3)
    grid[3, 4] = True   # til høyre for (3, 3)
    out = near_obstacles(grid, np.array([3 * 8 + 3, 0, -1]))
    assert out[0].tolist() == [1, 0, 0, 1]      # opp, ned, venstre, høyre
    assert out[1].tolist() == [0, 0, 0, 0]      # hjørnet: kantene er ikke hindringer
    assert out[2].tolist() == [-1, -1, -1, -1]


def test_encoded_near_obstacles_match_environment(seqs):
    assert seqs.near_obstacle.shape == (len(seqs.mu), 4)
    assert set(np.unique(seqs.near_obstacle)) <= {0, 1}
    # En hindrings-hendelse betyr at agenten gikk inn i en hindring i den retningen
    for i in range(len(seqs)):
        e = seqs.episode(i)
        for t in np.flatnonzero(e["events"] == EVENT_OBSTACLE):
            assert e["near_obstacle"][t, e["actions"][t]] == 1


def test_obstacle_head_loss_is_learnable(seqs):
    torch.manual_seed(0)
    model = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=16, num_mixtures=2, obstacle_head=True))
    batch = make_batch(seqs, range(len(seqs)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        parts = batch_loss(model, batch, False, 1.0, obstacle_weight=1.0)
        first = first if first is not None else parts["obstacle_bce"].item()
        opt.zero_grad()
        parts["loss"].backward()
        opt.step()
    assert parts["obstacle_bce"].item() < 0.8 * first


def test_training_can_add_heads_to_existing_checkpoint(seqs, tmp_path):
    from worldmodels.mdnrnn.train import train

    base = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, position_head=True))
    base.save(tmp_path / "base.pt")
    seqs.save(tmp_path / "z.npz")
    train(tmp_path / "z.npz", tmp_path / "out.pt", epochs=1, init_from=tmp_path / "base.pt",
          position_weight=1.0, obstacle_weight=1.0, log=lambda m: None)
    model, _ = MDNRNN.load(tmp_path / "out.pt")
    assert model.config.obstacle_head and model.config.position_head
    assert model.num_belief_features == 8
    f = model.belief_features(torch.randn(3, 8))
    assert f.shape == (3, 8) and (f[:, 4:] >= 0).all() and (f[:, 4:] <= 1).all()


def test_eye_shapes_positions_and_learns_from_z():
    from worldmodels.mdnrnn.eye import SpatialEye, expected_positions, eye_accuracy, train_eye

    eye = SpatialEye(latent_dim=6, channels=8)
    assert eye(torch.randn(5, 6)).shape == (5, 2, 64)
    assert eye(torch.randn(2, 3, 6)).shape == (2, 3, 2, 64)
    # Alt på én celle gir den cellens koordinater; jevn fordeling gir midten.
    logits = torch.full((1, 2, 64), -1e4)
    logits[0, 0, 0] = 0      # agent i (0, 0)
    logits[0, 1, 63] = 0     # mål i (7, 7)
    assert torch.allclose(expected_positions(logits, 8), torch.tensor([[-1.0, -1.0, 1.0, 1.0]]))
    assert torch.allclose(expected_positions(torch.zeros(1, 2, 64), 8), torch.zeros(1, 4), atol=1e-6)
    # Lærer en enkel avbildning: z er en one-hot-koding av radene og kolonnene.
    g = torch.Generator().manual_seed(0)
    cells = torch.randint(0, 64, (2000, 2), generator=g)
    z = torch.cat([F.one_hot(cells[:, 0] // 8, 8), F.one_hot(cells[:, 0] % 8, 8), F.one_hot(cells[:, 1] // 8, 8),
                   F.one_hot(cells[:, 1] % 8, 8)], 1).float()
    eye = SpatialEye(latent_dim=32, channels=16)
    train_eye(eye, (z[:1600], cells[:1600]), (z[1600:], cells[1600:]), epochs=15, lr=3e-3, batch=64, log=lambda m: None)
    agent, goal = eye_accuracy(eye, z[1600:], cells[1600:])
    assert agent > 0.9 and goal > 0.9


def test_eye_is_saved_with_the_model(tmp_path):
    m = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, eye=True, eye_channels=4))
    m.save(tmp_path / "m.pt")
    loaded, _ = MDNRNN.load(tmp_path / "m.pt")
    z = torch.randn(3, 4)
    assert torch.allclose(loaded.seen_positions(z), m.eval().seen_positions(z))
    assert MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8)).eye is None
