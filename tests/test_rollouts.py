import numpy as np
import pytest

from worldmodels.data import (
    RandomPolicy,
    collect_rollouts,
    episode_paths,
    load_episode,
    run_episode,
    save_episode,
)
from worldmodels.env import GridConfig, GridDodgeEnv


def test_random_policy_returns_valid_actions_and_is_seeded():
    a = RandomPolicy(4, repeat_prob=0.3, seed=7)
    b = RandomPolicy(4, repeat_prob=0.3, seed=7)
    seq_a = [a.act() for _ in range(200)]
    seq_b = [b.act() for _ in range(200)]
    assert seq_a == seq_b
    assert set(seq_a) <= {0, 1, 2, 3}
    assert len(set(seq_a)) == 4


def test_random_policy_repeat_prob_increases_repeats():
    def repeat_rate(p):
        pol = RandomPolicy(4, repeat_prob=p, seed=0)
        acts = [pol.act() for _ in range(5000)]
        return np.mean([x == y for x, y in zip(acts, acts[1:])])

    assert repeat_rate(0.0) == pytest.approx(0.25, abs=0.03)
    assert repeat_rate(0.8) == pytest.approx(0.8 + 0.2 * 0.25, abs=0.03)


def test_random_policy_rejects_invalid_repeat_prob():
    with pytest.raises(ValueError):
        RandomPolicy(4, repeat_prob=1.0)


def test_run_episode_produces_consistent_shapes():
    env = GridDodgeEnv(GridConfig(max_steps=20))
    ep = run_episode(env, RandomPolicy(seed=0), seed=3)
    ep.validate()
    T = len(ep)
    assert 1 <= T <= 20
    assert ep.obs.shape == (T + 1, 64, 64, 3)
    # Bare siste skritt kan avslutte episoden
    assert not (ep.terminated[:-1] | ep.truncated[:-1]).any()
    assert ep.terminated[-1] or ep.truncated[-1]


def test_transitions_pair_obs_with_next_obs():
    env = GridDodgeEnv()
    ep = run_episode(env, RandomPolicy(seed=1), seed=1)
    trans = list(ep.transitions())
    assert len(trans) == len(ep)
    for t, (o, a, o_next) in enumerate(trans):
        assert np.array_equal(o, ep.obs[t])
        assert a == ep.actions[t]
        assert np.array_equal(o_next, ep.obs[t + 1])


def test_episode_replays_identically_in_env():
    """De lagrede observasjonene skal være nøyaktig det miljøet produserer."""
    env = GridDodgeEnv()
    ep = run_episode(env, RandomPolicy(seed=2), seed=11)
    replay = GridDodgeEnv()
    assert np.array_equal(replay.reset(seed=11), ep.obs[0])
    for t, a in enumerate(ep.actions):
        obs, r, *_ = replay.step(int(a))
        assert np.array_equal(obs, ep.obs[t + 1])
        assert r == pytest.approx(ep.rewards[t])


def test_save_and_load_roundtrip(tmp_path):
    env = GridDodgeEnv()
    ep = run_episode(env, RandomPolicy(seed=0), seed=0)
    path = tmp_path / "episode_00000.npz"
    save_episode(ep, path)
    loaded = load_episode(path)
    for field in ("obs", "actions", "rewards", "terminated", "truncated"):
        assert np.array_equal(getattr(ep, field), getattr(loaded, field))


def test_collect_rollouts_writes_files_and_stats(tmp_path):
    stats = collect_rollouts(tmp_path, num_episodes=5, seed=0)
    paths = episode_paths(tmp_path)
    assert len(paths) == 5
    assert stats["episodes"] == 5
    assert sum(stats["outcomes"].values()) == 5
    assert stats["transitions"] == sum(len(load_episode(p)) for p in paths)


def test_collect_rollouts_is_reproducible(tmp_path):
    collect_rollouts(tmp_path / "a", num_episodes=3, seed=5)
    collect_rollouts(tmp_path / "b", num_episodes=3, seed=5)
    for pa, pb in zip(episode_paths(tmp_path / "a"), episode_paths(tmp_path / "b")):
        assert np.array_equal(load_episode(pa).obs, load_episode(pb).obs)
