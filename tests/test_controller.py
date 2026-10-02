import numpy as np
import pytest
import torch

from worldmodels.controller.agent import WorldModelAgent, run_real_episodes
from worldmodels.controller.dream import DreamConfig, dream_fitness, make_warm_starts
from worldmodels.controller.es import EvolutionStrategy, centered_ranks
from worldmodels.controller.policy import LinearController
from worldmodels.data import collect_rollouts, episode_paths
from worldmodels.mdnrnn.encode import encode_episodes
from worldmodels.mdnrnn.model import MDNRNN, MDNRNNConfig
from worldmodels.vae.model import ConvVAE, VAEConfig


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("rollouts")
    collect_rollouts(d, num_episodes=12, seed=0)
    vae = ConvVAE(VAEConfig(latent_dim=4)).eval()
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2)).eval()
    seqs = encode_episodes(vae, episode_paths(d))
    return vae, rnn, seqs


def test_controller_param_layout_and_batched_matches_single():
    c = LinearController(z_dim=3, h_dim=5, num_actions=4)
    assert c.num_params == (3 + 5 + 1) * 4
    rng = np.random.default_rng(0)
    pop = rng.normal(size=(2, c.num_params)).astype(np.float32)
    z, h = torch.randn(2, 6, 3), torch.randn(2, 6, 5)
    batched = c.batched_logits(pop, z, h)
    for p in range(2):
        c.params = pop[p]
        assert torch.allclose(batched[p], c.logits(z[p], h[p]), atol=1e-5)


def test_controller_bias_decides_action_when_weights_are_zero():
    c = LinearController(z_dim=2, h_dim=2, num_actions=4)
    c.params[-4:] = [0, 0, 3, 0]  # bias favoriserer handling 2
    assert c.act(torch.randn(5, 2), torch.randn(5, 2)).tolist() == [2] * 5


def test_controller_save_load(tmp_path):
    c = LinearController(z_dim=2, h_dim=3)
    c.params = np.arange(c.num_params, dtype=np.float32)
    c.save(tmp_path / "c.npz")
    loaded = LinearController.load(tmp_path / "c.npz")
    assert np.array_equal(loaded.params, c.params) and loaded.h_dim == 3


def test_centered_ranks():
    assert centered_ranks(np.array([10.0, -5.0, 3.0])).tolist() == [0.5, -0.5, 0.0]


def test_es_requires_even_population_and_ask_before_tell():
    with pytest.raises(ValueError):
        EvolutionStrategy(3, population=5)
    with pytest.raises(RuntimeError):
        EvolutionStrategy(3, population=4).tell(np.zeros(4))


def test_es_maximizes_simple_quadratic():
    target = np.array([1.0, -2.0, 0.5])
    es = EvolutionStrategy(3, population=20, sigma=0.1, lr=0.1, seed=0)
    for _ in range(300):
        cands = es.ask()
        es.tell(-((cands - target) ** 2).sum(1))
    assert np.allclose(es.theta, target, atol=0.1)


def test_warm_starts_skip_short_episodes(world):
    _, rnn, seqs = world
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=3)
    assert len(starts) == int((seqs.lengths > 3).sum())
    assert starts.z.shape == (len(starts), 4) and starts.h.shape == (1, len(starts), 8)
    sub = starts.sample(2, np.random.default_rng(0))
    assert len(sub) == 2 and sub.h.shape == (1, 2, 8)


def test_dream_fitness_shapes_and_bounds(world):
    _, rnn, seqs = world
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    c = LinearController(4, 8)
    pop = np.random.default_rng(0).normal(size=(4, c.num_params))
    cfg = DreamConfig(horizon=5, temperature=0.0)
    f = dream_fitness(c, pop, rnn, starts, cfg)
    assert f.shape == (4,)
    # Avkastningen ligger mellom "treffer hindring i første skritt" og "når målet i første skritt"
    assert (f >= cfg.reward_obstacle - 1e-6).all() and (f <= cfg.reward_goal + 1e-6).all()
    d = dream_fitness(c, pop, rnn, starts, cfg, return_details=True)
    assert np.allclose(d["fitness"], f)
    assert d["action_share"].sum() == pytest.approx(1.0)


def test_dream_is_deterministic_at_zero_temperature(world):
    _, rnn, seqs = world
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    c = LinearController(4, 8)
    pop = np.random.default_rng(1).normal(size=(2, c.num_params))
    cfg = DreamConfig(horizon=4, temperature=0.0)
    assert np.array_equal(dream_fitness(c, pop, rnn, starts, cfg), dream_fitness(c, pop, rnn, starts, cfg))


def test_candidate_fitness_does_not_depend_on_rest_of_population(world):
    """Hver kandidat drømmer uavhengig, selv om alle kjøres i samme batch."""
    _, rnn, seqs = world
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    c = LinearController(4, 8)
    pop = np.random.default_rng(2).normal(size=(3, c.num_params))
    cfg = DreamConfig(horizon=4, temperature=0.0)
    together = dream_fitness(c, pop, rnn, starts, cfg)
    alone = np.array([dream_fitness(c, pop[i:i + 1], rnn, starts, cfg)[0] for i in range(3)])
    assert np.allclose(together, alone, atol=1e-5)


def test_real_episodes_run_and_report_rates(world):
    vae, rnn, _ = world
    c = LinearController(4, 8)
    stats = run_real_episodes(WorldModelAgent(vae, rnn, c), num_episodes=6)
    assert stats["episodes"] == 6
    assert stats["goal_rate"] + stats["obstacle_rate"] + stats["truncated_rate"] == pytest.approx(1.0)
    assert 1 <= stats["mean_length"] <= 50
