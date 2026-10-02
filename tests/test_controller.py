import numpy as np
import pytest
import torch

from worldmodels.controller.agent import WorldModelAgent, collect_agent_rollouts, play_episodes, run_real_episodes
from worldmodels.controller.dream import DreamConfig, dream_fitness, make_warm_starts
from worldmodels.controller.cma import SepCMAES
from worldmodels.controller.es import EvolutionStrategy, centered_ranks
from worldmodels.controller.policy import LinearController
from worldmodels.data import collect_rollouts, episode_paths, load_episode
from worldmodels.mdnrnn.encode import ZSequences, encode_episodes
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


def test_recorded_episodes_match_stats_and_format(world):
    vae, rnn, _ = world
    c = LinearController(4, 8)
    stats, episodes = play_episodes(WorldModelAgent(vae, rnn, c), 5, seed=7, epsilon=0.5, record=True)
    assert len(episodes) == 5
    for ep in episodes:
        ep.validate()
        assert ep.terminated[-1] or ep.truncated[-1]
    assert np.mean([ep.rewards.sum() for ep in episodes]) == pytest.approx(stats["mean_return"], abs=1e-5)


def test_full_exploration_ignores_controller(world):
    vae, rnn, _ = world
    c = LinearController(4, 8)
    c.params[:] = 0
    c.params[-4:] = [0, 0, 0, 10]  # bias: alltid "høyre" uten utforsking
    agent = WorldModelAgent(vae, rnn, c)
    agent.reset(200)
    obs = np.zeros((200, 64, 64, 3), np.uint8)
    assert (agent.act(obs) == 3).all()
    agent.reset(200)
    counts = np.bincount(agent.act(obs, epsilon=1.0, rng=np.random.default_rng(0)), minlength=4)
    assert counts.min() > 20


def test_collect_agent_rollouts_can_be_encoded(world, tmp_path):
    vae, rnn, _ = world
    stats = collect_agent_rollouts(WorldModelAgent(vae, rnn, LinearController(4, 8)), tmp_path, 7, seed=3, batch_size=3)
    paths = episode_paths(tmp_path)
    assert len(paths) == 7 and stats["episodes"] == 7
    load_episode(paths[0])
    seqs = encode_episodes(vae, paths)
    both = ZSequences.concat([seqs, seqs])
    assert len(both) == 14 and len(both.mu) == 2 * len(seqs.mu)
    assert np.array_equal(both.episode(7)["actions"], seqs.episode(0)["actions"])


def test_iterate_runs_one_tiny_round(world, tmp_path):
    from worldmodels.controller.iterate import iterate

    vae, rnn, seqs = world
    vae.save(tmp_path / "vae.pt")
    rnn.save(tmp_path / "rnn.pt")
    LinearController(4, 8).save(tmp_path / "c.npz")
    seqs.save(tmp_path / "zseq.npz")
    result = iterate(
        tmp_path / "vae.pt", tmp_path / "rnn.pt", tmp_path / "c.npz", [tmp_path / "zseq.npz"], tmp_path / "iter",
        rounds=1, episodes_per_round=6, rnn_epochs=1, generations=2, starts=4, eval_episodes=4, log=lambda m: None,
    )
    assert [r["round"] for r in result["rounds"]] == [0, 1]
    assert (tmp_path / "iter" / "controller_r1.npz").exists()
    assert len(result["rounds"][1]["data"]) == 2


def test_position_features_shape_range_and_center():
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, position_head=True))
    f = rnn.position_features(torch.randn(5, 3, 8))
    assert f.shape == (5, 3, 4) and f.abs().max() <= 1.0
    with torch.no_grad():
        rnn.position_head.weight.zero_()
        rnn.position_head.bias.zero_()
    assert torch.allclose(rnn.position_features(torch.randn(2, 8)), torch.zeros(2, 4), atol=1e-6)
    assert MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8)).position_features(torch.randn(2, 8)).shape == (2, 0)


def test_controller_with_position_features_runs_in_dream_and_real_env(world, tmp_path):
    vae, _, seqs = world
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, position_head=True)).eval()
    c = LinearController(4, 8, extra_dim=rnn.num_position_features)
    assert c.num_params == (4 + 8 + 4 + 1) * 4
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    pop = np.random.default_rng(0).normal(size=(2, c.num_params))
    assert dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3)).shape == (2,)
    c.save(tmp_path / "c.npz")
    assert LinearController.load(tmp_path / "c.npz").extra_dim == 4
    stats = run_real_episodes(WorldModelAgent(vae, rnn, c), num_episodes=3)
    assert stats["episodes"] == 3


def test_remaining_steps_charges_only_survivors(world):
    _, rnn, seqs = world
    c = LinearController(4, 8)
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    pop = np.zeros((1, c.num_params))
    base = dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3, temperature=0.0), return_details=True)
    charged = dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3, temperature=0.0, remaining_steps=10))
    expected = base["fitness"] + base["p_alive_end"] * 10 * DreamConfig.reward_step
    assert np.allclose(charged, expected, atol=1e-5)


def test_shaping_is_zero_without_position_head_and_rewards_approach(world):
    from worldmodels.controller.dream import goal_distance

    _, rnn, seqs = world
    c = LinearController(4, 8)
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    pop = np.zeros((1, c.num_params))
    plain = dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3, temperature=0.0))
    shaped = dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3, temperature=0.0, shaping=1.0))
    assert np.allclose(plain, shaped)

    pos = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, position_head=True)).eval()
    h = torch.randn(5, 8)
    d = goal_distance(pos, h)
    assert d.shape == (5,) and (d >= 0).all() and (d <= 14 + 1e-5).all()
    starts = make_warm_starts(pos, seqs, range(len(seqs)), context=2)
    details = dream_fitness(c, pop, pos, starts, DreamConfig(horizon=3, temperature=0.0, shaping=1.0),
                            return_details=True)
    base = dream_fitness(c, pop, pos, starts, DreamConfig(horizon=3, temperature=0.0))
    assert np.allclose(details["fitness"], base + details["shaping"], atol=1e-5)


def test_cma_minimizes_shifted_sphere_and_ill_conditioned_ellipsoid():
    target = np.linspace(-1, 1, 10)
    es = SepCMAES(10, sigma=1.0, seed=0)
    for _ in range(200):
        x = es.ask()
        es.tell(-((x - target) ** 2).sum(1))
    assert np.abs(es.theta - target).max() < 1e-3
    scale = 10 ** np.linspace(0, 3, 10)
    es = SepCMAES(10, sigma=1.0, seed=1, init=np.ones(10))
    for _ in range(600):
        x = es.ask()
        es.tell(-((scale * x) ** 2).sum(1))
    assert ((scale * es.theta) ** 2).sum() < 1e-6


def test_cma_normalization_keeps_unit_mean_and_same_direction():
    def f(x):  # skalainvariant, som argmax-controlleren
        return (x / np.linalg.norm(x, axis=1, keepdims=True)) @ np.ones(6)

    a = SepCMAES(6, sigma=0.3, seed=3, init=np.ones(6) * 2.0)
    b = SepCMAES(6, sigma=0.3, seed=3, init=np.ones(6) * 2.0, normalize=True)
    for _ in range(20):
        a.tell(f(a.ask()))
        b.tell(f(b.ask()))
    assert np.linalg.norm(b.theta) == pytest.approx(1.0)
    assert np.allclose(a.theta / np.linalg.norm(a.theta), b.theta, atol=1e-6)


def test_cma_requires_ask_before_tell():
    with pytest.raises(RuntimeError):
        SepCMAES(3).tell(np.zeros(7))


def test_cma_per_parameter_std_controls_initial_spread():
    es = SepCMAES(4, sigma=1.0, population=4000, seed=0, std=np.array([1.0, 1.0, 0.1, 0.1]))
    spread = es.ask().std(0)
    assert np.allclose(spread, [1.0, 1.0, 0.1, 0.1], rtol=0.1)


def test_cma_zero_std_freezes_parameters():
    es = SepCMAES(5, sigma=0.5, seed=0, std=np.array([0.0, 0.0, 1.0, 1.0, 1.0]))
    for _ in range(30):
        x = es.ask()
        es.tell(-((x - 3.0) ** 2).sum(1))
    assert np.allclose(es.theta[:2], 0.0)


def test_lookahead_matches_running_each_action_and_keeps_state(world):
    _, rnn, seqs = world
    torch.manual_seed(0)
    z = torch.randn(3, 4)
    hidden = (torch.randn(1, 3, 8), torch.randn(1, 3, 8))
    before = tuple(x.clone() for x in hidden)
    p = rnn.lookahead_obstacle(z, hidden)
    assert p.shape == (3, 4) and ((p >= 0) & (p <= 1)).all()
    for a in range(4):
        out = rnn(z.unsqueeze(1), torch.full((3, 1), a), hidden)
        expected = torch.softmax(out.event_logits[:, 0], -1)[:, 2]
        assert torch.allclose(p[:, a], expected, atol=1e-6)
    assert all(torch.equal(x, y) for x, y in zip(hidden, before))


def test_controller_with_lookahead_only_saves_flags_and_runs(world, tmp_path):
    from worldmodels.controller.features import num_world_features

    vae, rnn, seqs = world
    extra = num_world_features(rnn, beliefs=False, lookahead=True)
    c = LinearController(4, 8, extra_dim=extra, beliefs=False, lookahead=True)
    c.save(tmp_path / "c.npz")
    loaded = LinearController.load(tmp_path / "c.npz")
    assert (loaded.beliefs, loaded.lookahead, loaded.extra_dim) == (False, True, 4)
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    pop = np.random.default_rng(0).normal(size=(2, c.num_params))
    assert dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3)).shape == (2,)
    assert run_real_episodes(WorldModelAgent(vae, rnn, loaded), num_episodes=2)["episodes"] == 2


def test_controller_with_sight_saves_flag_and_runs_with_eye_shaping(world, tmp_path):
    from worldmodels.controller.features import num_world_features, world_features

    vae, _, seqs = world
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, eye=True, eye_channels=4)).eval()
    extra = num_world_features(rnn, beliefs=False, lookahead=True, sight=True)
    assert extra == 8
    c = LinearController(4, 8, extra_dim=extra, lookahead=True, sight=True)
    assert c.beliefs is False
    c.save(tmp_path / "c.npz")
    loaded = LinearController.load(tmp_path / "c.npz")
    assert (loaded.beliefs, loaded.lookahead, loaded.sight) == (False, True, True)
    z = torch.randn(3, 4)
    f, memory = world_features(rnn, z, (torch.zeros(1, 3, 8), torch.zeros(1, 3, 8)), False, True, sight=True)
    assert torch.allclose(f[:, :4], rnn.seen_positions(z)) and memory.goal.shape == (3, 64)
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    assert starts.memory.goal.shape == (len(starts), 64) and starts.sample(3, np.random.default_rng(0)).memory.agent.shape == (3, 64)
    pop = np.random.default_rng(0).normal(size=(2, c.num_params))
    d = dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3, temperature=0.0, shaping=0.1), return_details=True)
    assert d["fitness"].shape == (2,) and np.abs(d["shaping"]).sum() > 0
    assert run_real_episodes(WorldModelAgent(vae, rnn, loaded), num_episodes=2)["episodes"] == 2


def test_goal_memory_outvotes_a_single_wrong_frame():
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, eye=True, eye_channels=4)).eval()

    class FixedEye(torch.nn.Module):
        def forward(self, z):  # z[:, 0] sier hvilken celle målet er i, med stor sikkerhet
            logits = torch.full((len(z), 2, 64), -20.0)
            logits[torch.arange(len(z)), 1, z[:, 0].long()] = 20.0
            return logits

    rnn.eye = FixedEye()
    z = torch.zeros(1, 4)
    memory = None
    for cell in (63, 63, 63, 0):          # tre bilder sier (7, 7), det siste sier (0, 0)
        z[0, 0] = cell
        seen, memory = rnn.see(z, memory)
    assert torch.allclose(seen[0, 2:], torch.tensor([1.0, 1.0]), atol=1e-2)
    # Uten minne følger troen bildet, litt dratt mot midten av gulvet i ett enkelt bilde.
    assert torch.allclose(rnn.see(z)[0][0, 2:], torch.tensor([-1.0, -1.0]), atol=0.1)


def test_neighbour_labels_and_event_probs():
    from worldmodels.mdnrnn.neighbours import event_probs_from_logits, goal_direction_labels

    # agent (0, 0); målet til høyre, under, langt unna og ukjent
    labels = goal_direction_labels(np.array([0, 0, 0, 0]), np.array([1, 8, 63, -1]))
    assert labels.tolist() == [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1]]
    logits = torch.tensor([[[10.0, -10, -10, -10], [10.0, 10, -10, -10]]])  # mål opp; hindring opp og ned
    p = event_probs_from_logits(logits.repeat(3, 1, 1), torch.tensor([0, 1, 2]))
    assert torch.allclose(p.sum(1), torch.ones(3))
    assert p[0, 1] > 0.99 and p[1, 2] > 0.99 and p[2, 0] > 0.99   # opp: mål, ned: krasj, venstre: flytt


def test_neighbour_eye_drives_dream_events_and_lookahead(world, tmp_path):
    from worldmodels.mdnrnn.neighbours import neighbour_data, neighbour_metrics

    vae, _, seqs = world
    cfg = MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, eye=True, eye_channels=4, neighbours=True,
                       neighbour_channels=4)
    rnn = MDNRNN(cfg).eval()
    z, hidden = torch.randn(3, 4), (torch.zeros(1, 3, 8), torch.zeros(1, 3, 8))
    _, memory = rnn.see(z)
    p = rnn.event_probs(torch.zeros(3, 3), z, memory, torch.tensor([0, 1, 2]))
    assert torch.allclose(p.sum(1), torch.ones(3))
    logits = rnn.neighbour_logits(z, memory)
    look = rnn.lookahead_obstacle(z, hidden, memory)
    assert torch.allclose(look, (1 - torch.sigmoid(logits[:, 0])) * torch.sigmoid(logits[:, 1]))
    rnn.save(tmp_path / "m.pt")
    loaded, _ = MDNRNN.load(tmp_path / "m.pt")
    assert torch.allclose(loaded.neighbour_logits(z, memory), logits)

    train, val = neighbour_data(rnn, seqs, np.array([0, 1]))
    assert train[0].shape[1] == 4 and train[1].shape[1] == train[2].shape[1] == 64 and train[3].shape[1:] == (2, 4)
    assert torch.allclose(train[1].sum(1), torch.ones(len(train[1])))
    assert torch.allclose(train[2].sum(1), torch.ones(len(train[2])))
    assert set(neighbour_metrics(rnn.neighbours, val)) == {"goal", "obstacle"}

    c = LinearController(4, 8, extra_dim=8, lookahead=True, sight=True)
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    pop = np.random.default_rng(0).normal(size=(2, c.num_params))
    d = dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3, temperature=0.0), return_details=True)
    assert d["fitness"].shape == (2,) and (d["p_alive_end"] <= 1).all()
    assert run_real_episodes(WorldModelAgent(vae, rnn, c), num_episodes=2)["episodes"] == 2


def test_controller_with_tracking_saves_flag_and_runs_in_dream_and_reality(world, tmp_path):
    from worldmodels.controller.features import num_world_features, world_features

    vae, _, seqs = world
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, eye=True, eye_channels=4)).eval()
    extra = num_world_features(rnn, beliefs=False, lookahead=True, sight=True, track=True)
    assert extra == 12
    c = LinearController(4, 8, extra_dim=extra, lookahead=True, sight=True, track=True)
    c.save(tmp_path / "c.npz")
    assert LinearController.load(tmp_path / "c.npz").track is True
    with pytest.raises(ValueError):
        world_features(rnn, torch.randn(2, 4), (torch.zeros(1, 2, 8), torch.zeros(1, 2, 8)), False, False, track=True)
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    assert torch.allclose(starts.memory.agent.sum(-1), torch.ones(len(starts)))
    pop = np.random.default_rng(0).normal(size=(2, c.num_params))
    assert dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3, temperature=0.0, shaping=0.1)).shape == (2,)
    assert run_real_episodes(WorldModelAgent(vae, rnn, c), num_episodes=2)["episodes"] == 2
