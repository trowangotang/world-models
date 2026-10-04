import numpy as np
import pytest
import torch

from worldmodels.controller.agent import WorldModelAgent, play_episodes
from worldmodels.controller.dream import DreamConfig, dream_fitness, make_warm_starts
from worldmodels.controller.features import num_world_features
from worldmodels.controller.policy import LinearController
from worldmodels.data import collect_rollouts, episode_paths, load_episode
from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.env.gridworld import BACK, COLOR_OBSTACLE, FORWARD, TURN_LEFT, TURN_RIGHT
from worldmodels.env.raycast import COLOR_WALL
from worldmodels.mdnrnn.encode import ZSequences, encode_episodes, goal_in_view
from worldmodels.mdnrnn.model import COMPASS_SCALE, MDNRNN, MDNRNNConfig
from worldmodels.mdnrnn.train import batch_loss, make_batch
from worldmodels.vae.loss import object_weight_map
from worldmodels.vae.model import ConvVAE, VAEConfig

FP = GridConfig(first_person=True)


def place(env, agent, heading, goal=(7, 7), obstacles=()):
    env.reset(seed=0)
    env.agent_pos, env.heading, env.goal_pos, env.obstacles = agent, heading, goal, frozenset(obstacles)
    env.movers = []


def test_first_person_uses_the_same_boards_as_top_down():
    top, fp = GridDodgeEnv(GridConfig()), GridDodgeEnv(FP)
    for seed in range(20):
        top.reset(seed=seed)
        fp.reset(seed=seed)
        assert (top.agent_pos, top.goal_pos, top.obstacles) == (fp.agent_pos, fp.goal_pos, fp.obstacles)


def test_turning_takes_a_step_without_moving():
    env = GridDodgeEnv(FP)
    place(env, (4, 4), heading=0)  # ser opp
    env.step(TURN_RIGHT)
    assert env.agent_pos == (4, 4) and env.heading == 3 and env.steps == 1  # høyre
    env.step(TURN_RIGHT)
    env.step(TURN_RIGHT)
    env.step(TURN_RIGHT)
    assert env.heading == 0
    env.step(TURN_LEFT)
    assert env.heading == 2  # venstre


def test_forward_and_back_follow_heading():
    env = GridDodgeEnv(FP)
    place(env, (4, 4), heading=3)  # ser mot høyre
    env.step(FORWARD)
    assert env.agent_pos == (4, 5)
    env.step(BACK)
    env.step(BACK)
    assert env.agent_pos == (4, 3) and env.heading == 3


def test_walking_into_obstacle_ends_episode():
    env = GridDodgeEnv(FP)
    place(env, (4, 4), heading=1, obstacles=[(5, 4)])  # ser ned mot hindringen
    obs, r, terminated, _, info = env.step(FORWARD)
    assert terminated and info["event"] == "obstacle" and r == FP.reward_obstacle
    assert (obs == np.array(COLOR_OBSTACLE, np.uint8)).all()  # står inne i hindringen: helt rødt


def test_wall_right_in_front_fills_the_view():
    env = GridDodgeEnv(FP)
    place(env, (0, 4), heading=0)  # ved øvre kant, ser opp
    obs = env.render()
    assert obs.shape == (64, 64, 3)
    column = obs[:, 32]
    assert (column == column[0]).all()                                   # bare vegg, ikke tak eller gulv
    assert np.allclose(column[0] / np.array(COLOR_WALL), column[0, 0] / COLOR_WALL[0], atol=0.02)  # grå, bare skyggelagt


def test_goal_ahead_is_visible_and_centred():
    env = GridDodgeEnv(FP)
    place(env, (6, 2), heading=0, goal=(2, 2))  # målet fire celler rett fram
    obs = env.render()
    assert goal_in_view(obs[None])[0] == 1
    green = (obs[..., 1].astype(int) > obs[..., 0].astype(int) + 30).any(0)
    cols = np.flatnonzero(green)
    assert abs(cols.mean() - 31.5) < 1 and len(cols) < 20  # i midten, og smal på avstand
    assert env.egocentric(env.goal_pos) == (4, 0)
    env.step(TURN_RIGHT)
    assert goal_in_view(env.render()[None])[0] == 0
    assert env.egocentric(env.goal_pos) == (0, -4)  # nå fire celler til venstre


def test_colourful_weighting_ignores_grey_walls():
    x = torch.zeros(1, 3, 1, 3)
    x[0, :, 0, 0] = torch.tensor(COLOR_WALL) / 255
    x[0, :, 0, 1] = torch.tensor(COLOR_OBSTACLE) / 255 * 0.4  # mørk, langt unna
    w = object_weight_map(x, 10.0, colourful=True)
    assert w[0, 0, 0].tolist() == [1.0, 10.0, 1.0]


@pytest.fixture(scope="module")
def fp_world(tmp_path_factory):
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("fp")
    collect_rollouts(d, num_episodes=10, seed=0, config=FP)
    vae = ConvVAE(VAEConfig(latent_dim=4)).eval()
    rnn = MDNRNN(MDNRNNConfig(latent_dim=4, hidden_dim=8, num_mixtures=2, compass=True)).eval()
    return d, vae, rnn, encode_episodes(vae, episode_paths(d))


def test_fp_episodes_store_truth_and_labels_match_env(fp_world):
    d, _, _, seqs = fp_world
    ep = load_episode(episode_paths(d)[0])
    assert ep.topdown.shape == ep.obs.shape and len(ep.heading) == len(ep.obs)
    env = GridDodgeEnv(FP)
    env.reset(seed=0)
    first = seqs.episode(0)
    assert tuple(first["goal_ego"][0]) == env.egocentric(env.goal_pos)
    assert first["heading"][0] == env.heading
    assert set(np.unique(seqs.goal_visible)) <= {0, 1}


def test_compass_loss_and_features(fp_world, tmp_path):
    _, vae, rnn, seqs = fp_world
    batch = make_batch(seqs, range(4))
    parts = batch_loss(rnn, batch, False, 1.0, compass_weight=1.0)
    assert torch.isfinite(parts["compass"]) and parts["compass"] > 0
    f = rnn.compass_features(torch.randn(5, 4), torch.zeros(5, 8))
    assert f.shape == (5, 3) and ((f[:, 2] >= 0) & (f[:, 2] <= 1)).all()
    seqs.save(tmp_path / "z.npz")
    assert np.array_equal(ZSequences.load(tmp_path / "z.npz").goal_ego, seqs.goal_ego)
    assert COMPASS_SCALE == 7.0


def test_compass_controller_runs_in_dream_and_real_world(fp_world):
    _, vae, rnn, seqs = fp_world
    extra = num_world_features(rnn, False, True, compass=True)
    assert extra == 4 + 3
    c = LinearController(4, 8, 4, extra, lookahead=True, compass=True)
    starts = make_warm_starts(rnn, seqs, range(len(seqs)), context=2)
    pop = np.random.default_rng(0).normal(size=(3, c.num_params)).astype(np.float32)
    fit = dream_fitness(c, pop, rnn, starts, DreamConfig(horizon=3, shaping=0.05))
    assert fit.shape == (3,) and np.isfinite(fit).all()
    c.params = pop[0]
    stats, episodes = play_episodes(WorldModelAgent(vae, rnn, c), 3, config=FP, record=True)
    assert stats["episodes"] == 3 and episodes[0].topdown is not None


def test_habits_count_turns_and_notice_walking_into_wall():
    from worldmodels.controller.features import Habits

    h = Habits.fresh(2)
    z = torch.zeros(2, 4)
    h = h.after(z, torch.tensor([TURN_LEFT, FORWARD]))
    f = h.features(torch.tensor([[1.0, 0, 0, 0], [0.0, 0, 0, 0]]))
    assert f[:, 0].tolist() == pytest.approx([1 / 3, 0])   # ett snu på rad
    assert f[:, 1].tolist() == [0.0, 1.0]                  # fram uten at bildet endret seg = gikk fast


def test_shortest_path_cheat_reaches_goal_and_counts_turns():
    from worldmodels.evaluation.firstperson import ShortestPath, run, shortest_first_action

    env = GridDodgeEnv(FP)
    place(env, (4, 4), heading=1, goal=(2, 4))  # målet to celler bak
    assert shortest_first_action(env)[1] == 4    # snu, snu, fram, fram
    summary, outcome, _ = run(ShortestPath(), 20, 500, FP)
    assert summary["goal"]["rate"] == 1.0 and summary["path_efficiency"] == pytest.approx(1.0)
