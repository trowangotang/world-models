import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from worldmodels.controller.agent import WorldModelAgent, play_episodes
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridDodgeEnv
from worldmodels.evaluation import figures
from worldmodels.evaluation.beliefs import belief_accuracy
from worldmodels.evaluation.dream_check import auc, dream_vs_real
from worldmodels.evaluation.policies import (
    GreedyGoalBaseline, GreedySafeBaseline, RandomBaseline, SafeRandomBaseline, ShortestPathBaseline,
    WorldModelPolicy, bfs_distances,
)
from worldmodels.evaluation.report import write_report
from worldmodels.evaluation.run import evaluate_policy, is_oscillating, obstacles_between
from worldmodels.evaluation.stats import paired_difference, summarize, wilson
from worldmodels.mdnrnn.model import MDNRNN, MDNRNNConfig
from worldmodels.vae.model import ConvVAE, VAEConfig

SEEDS = range(500, 560)


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    vae = ConvVAE(VAEConfig(latent_dim=4)).eval()
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, position_head=True)).eval()
    c = LinearController(4, 8, 4, extra_dim=4, beliefs=True)
    c.params = np.random.default_rng(0).normal(size=c.num_params).astype(np.float32)
    return WorldModelAgent(vae, rnn, c)


def test_shortest_path_always_reaches_goal_in_shortest_steps():
    records, _ = evaluate_policy(ShortestPathBaseline(), SEEDS)
    assert all(r.outcome == "goal" and r.length == r.shortest for r in records)


def test_safe_policies_never_hit_obstacles():
    for policy in (SafeRandomBaseline(seed=1), GreedySafeBaseline(seed=1)):
        records, _ = evaluate_policy(policy, SEEDS)
        assert not any(r.outcome == "obstacle" for r in records)


def test_greedy_reaches_goal_when_no_obstacle_is_between():
    records, _ = evaluate_policy(GreedyGoalBaseline(seed=0), SEEDS)
    free = [r for r in records if r.obstacles_between == 0]
    assert free and all(r.outcome == "goal" and r.length == r.manhattan for r in free)


def test_records_are_consistent_with_the_environment():
    records, frames = evaluate_policy(RandomBaseline(seed=3), SEEDS, keep_frames=[SEEDS[0]], batch_size=25)
    assert [r.seed for r in records] == list(SEEDS)
    for r in records:
        env = GridDodgeEnv()
        env.reset(seed=r.seed)
        assert r.path[0] == env.agent_pos and len(r.path) == r.length + 1 == len(r.actions) + 1
        assert r.shortest == bfs_distances(env, env.agent_pos)[env.goal_pos] >= r.manhattan
        expected_end = {"goal": env.goal_pos}.get(r.outcome)
        if expected_end:
            assert r.path[-1] == expected_end
        if r.outcome == "obstacle":
            assert r.path[-1] in env.obstacles
        if r.outcome == "truncated":
            assert r.length == env.config.max_steps
    assert frames[SEEDS[0]].shape[0] == records[0].length + 1


def test_world_model_policy_matches_agent_stats(agent):
    records, _ = evaluate_policy(WorldModelPolicy("test", agent), SEEDS)
    stats, _ = play_episodes(agent, len(SEEDS), seed=SEEDS[0])
    assert np.isclose(np.mean([r.outcome == "goal" for r in records]), stats["goal_rate"])
    assert np.isclose(np.mean([r.episode_return for r in records]), stats["mean_return"])


def test_oscillation_and_obstacles_between():
    assert is_oscillating([(0, 0)] * 5 + [(1, 1), (1, 2)] * 10)
    assert not is_oscillating([(0, i % 8) for i in range(25)])
    assert not is_oscillating([(0, 0), (0, 1)])  # for kort
    env = GridDodgeEnv()
    env.reset(seed=0)
    env.obstacles = frozenset({(2, 2), (5, 5)})
    assert obstacles_between(env, (0, 0), (3, 3)) == 1
    assert obstacles_between(env, (6, 6), (0, 0)) == 2


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and np.isclose(lo + hi, 1.0)
    assert np.allclose(wilson(81, 100), (0.7222, 0.8749), atol=1e-3)
    lo, hi = wilson(0, 30)
    assert lo == 0.0 and 0.05 < hi < 0.15


def test_summary_and_paired_difference():
    random_records, _ = evaluate_policy(RandomBaseline(seed=0), SEEDS)
    best, _ = evaluate_policy(ShortestPathBaseline(), SEEDS)
    s = summarize(random_records)
    assert np.isclose(sum(s[o]["rate"] for o in ("goal", "obstacle", "truncated")), 1.0)
    assert sum(b["goal"]["n"] for b in s["by_distance"]) == len(SEEDS)
    assert s["return"]["lo"] <= s["return"]["mean"] <= s["return"]["hi"]
    d = paired_difference(best, random_records)
    assert d["only_b"] == 0 and d["diff"] > 0 and d["lo"] > 0
    assert np.isclose(d["diff"], 1 - s["goal"]["rate"])


def test_auc():
    assert auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([True, True, False, False])) == 1.0
    assert auc(np.array([0.5, 0.5]), np.array([True, False])) == 0.5
    assert auc(np.array([0.5]), np.array([True])) is None


def test_dream_vs_real_and_beliefs_run_with_tiny_models(agent):
    d = dream_vs_real(agent, SEEDS, start_step=2, horizon=4)
    assert 0 < d["starts"] <= len(SEEDS)
    for event in ("goal", "obstacle"):
        assert 0 <= d[event]["dream"] <= 1 and sum(b["count"] for b in d[event]["calibration"]) == d["starts"]
    b = belief_accuracy(agent, SEEDS)
    assert b["by_step"][0]["n"] == len(SEEDS) and 0 <= b["all"]["agent_exact"] <= 1
    no_head = WorldModelAgent(agent.vae, MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8)), LinearController(4, 8))
    assert belief_accuracy(no_head, SEEDS) is None


def test_figures_are_valid_svg_and_images(agent, tmp_path):
    rows = [(p.name, summarize(evaluate_policy(p, SEEDS)[0])) for p in (RandomBaseline(), ShortestPathBaseline())]
    dream = {"a": dream_vs_real(agent, SEEDS, start_step=2, horizon=4)}
    for svg in (figures.outcome_bars_svg(rows), figures.goal_by_distance_svg(rows),
                figures.calibration_svg(list(dream.items()), "goal")):
        assert ET.fromstring(svg).tag.endswith("svg")
    env = GridDodgeEnv()
    img = figures.path_overlay(env.reset(seed=1), [(0, 0), (0, 1), (1, 1)])
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    assert figures.tile([img] * 5, columns=4).shape == (2 * 260 + 4, 4 * 260 + 4, 3)

    results = {"seeds": [SEEDS[0], SEEDS[-1]], "policies": [{"name": n, "cheat": False} for n, _ in rows],
               "summaries": dict(rows), "paired": [], "dream_vs_real": dream, "beliefs": {}}
    json.dumps(results)
    write_report(tmp_path / "report.md", results, {"outcomes": "o.svg", "goal_by_distance": "g.svg",
                                                   "dream_goal": "d.svg", "dream_obstacle": "e.svg"})
    text = (tmp_path / "report.md").read_text()
    assert "## Hovedtabell" in text and "Korteste vei (juks)" in text
