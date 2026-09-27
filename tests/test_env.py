import numpy as np
import pytest

from worldmodels.env import GridConfig, GridDodgeEnv


def make_env(**kwargs) -> GridDodgeEnv:
    return GridDodgeEnv(GridConfig(**kwargs), seed=0)


def test_reset_returns_image_with_expected_shape_and_dtype():
    env = make_env()
    obs = env.reset(seed=1)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8


def test_reset_is_deterministic_for_same_seed():
    env = make_env()
    a = env.reset(seed=42)
    layout_a = (env.agent_pos, env.goal_pos, env.obstacles)
    b = env.reset(seed=42)
    assert np.array_equal(a, b)
    assert (env.agent_pos, env.goal_pos, env.obstacles) == layout_a


def test_layout_has_distinct_cells_and_reachable_goal():
    env = make_env(num_obstacles=20)
    for seed in range(50):
        env.reset(seed=seed)
        assert env.agent_pos != env.goal_pos
        assert env.agent_pos not in env.obstacles
        assert env.goal_pos not in env.obstacles
        assert len(env.obstacles) == 20
        assert env._goal_reachable()


def test_moving_into_wall_keeps_agent_in_place():
    env = make_env()
    env.reset(seed=0)
    env.agent_pos, env.goal_pos, env.obstacles = (0, 0), (7, 7), frozenset()
    _, reward, terminated, truncated, info = env.step(0)  # opp, inn i veggen
    assert env.agent_pos == (0, 0)
    assert reward == pytest.approx(env.config.reward_step)
    assert not terminated and not truncated
    assert info["event"] == "move"


def test_hitting_obstacle_terminates_with_penalty():
    env = make_env()
    env.reset(seed=0)
    env.agent_pos, env.goal_pos, env.obstacles = (3, 3), (7, 7), frozenset({(3, 4)})
    _, reward, terminated, _, info = env.step(3)  # høyre
    assert terminated
    assert reward == env.config.reward_obstacle
    assert info["event"] == "obstacle"


def test_reaching_goal_terminates_with_reward():
    env = make_env()
    env.reset(seed=0)
    env.agent_pos, env.goal_pos, env.obstacles = (3, 3), (4, 3), frozenset()
    _, reward, terminated, _, info = env.step(1)  # ned
    assert terminated
    assert reward == env.config.reward_goal
    assert info["event"] == "goal"


def test_episode_is_truncated_after_max_steps():
    env = make_env(max_steps=3)
    env.reset(seed=0)
    env.agent_pos, env.goal_pos, env.obstacles = (0, 0), (7, 7), frozenset()
    results = [env.step(0) for _ in range(3)]
    assert [r[3] for r in results] == [False, False, True]
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_action_raises():
    env = make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(4)


def test_render_draws_agent_goal_and_obstacle_colors():
    env = make_env()
    env.reset(seed=0)
    env.agent_pos, env.goal_pos, env.obstacles = (0, 0), (1, 1), frozenset({(2, 2)})
    img = env.render()
    px = env.config.cell_px
    centre = px // 2
    assert tuple(img[centre, centre]) == (70, 130, 240)                      # agent
    assert tuple(img[px + centre, px + centre]) == (60, 200, 90)             # mål
    assert tuple(img[2 * px + centre, 2 * px + centre]) == (220, 60, 60)     # hindring


def test_render_ascii_matches_layout():
    env = make_env(grid_size=3, num_obstacles=1)
    env.reset(seed=0)
    env.agent_pos, env.goal_pos, env.obstacles = (0, 0), (2, 2), frozenset({(1, 1)})
    assert env.render_ascii() == "A..\n.#.\n..G"


def test_save_png_writes_valid_png_header(tmp_path):
    from worldmodels.env.preview import save_png

    img = np.zeros((4, 5, 3), dtype=np.uint8)
    path = tmp_path / "x.png"
    save_png(img, path)
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert data[12:16] == b"IHDR"
    assert int.from_bytes(data[16:20], "big") == 5   # bredde
    assert int.from_bytes(data[20:24], "big") == 4   # høyde


def test_parse_cells_recovers_layout_from_render():
    from worldmodels.env.parse import AGENT, EMPTY, GOAL, OBSTACLE, find_cell, parse_cells

    env = make_env()
    for seed in range(20):
        obs = env.reset(seed=seed)
        cells = parse_cells(obs)
        g = env.config.grid_size
        for r in range(g):
            for c in range(g):
                p = (r, c)
                expected = (
                    AGENT if p == env.agent_pos
                    else GOAL if p == env.goal_pos
                    else OBSTACLE if p in env.obstacles
                    else EMPTY
                )
                assert cells[r, c] == expected
        assert find_cell(cells, AGENT) == env.agent_pos[0] * g + env.agent_pos[1]


def test_parse_cells_accepts_float_batches_and_reports_missing():
    from worldmodels.env.parse import AGENT, find_cell, parse_cells

    env = make_env()
    obs = env.reset(seed=0)
    batch = np.stack([obs, np.zeros_like(obs)]).astype(np.float32) / 255.0
    cells = parse_cells(batch)
    assert cells.shape == (2, 8, 8)
    idx = find_cell(cells, AGENT)
    assert idx[0] == env.agent_pos[0] * 8 + env.agent_pos[1]
    assert idx[1] == -1
