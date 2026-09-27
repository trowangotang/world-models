"""GridDodge: en liten 2D-verden der agenten skal nå et mål uten å treffe hindringer.

Verden er et N x N-rutenett. Hver episode plasseres agent, mål og hindringer
tilfeldig, med garanti for at målet kan nås. Observasjonen er et RGB-bilde
(standard 64x64x3, uint8), slik at VAE-en senere må lære å komprimere piksler,
akkurat som i Ha & Schmidhuber (2018).

API-et følger Gymnasium-konvensjonen uten å avhenge av pakken:
    obs = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Handlinger: indeks -> (drow, dcol)
ACTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # 0: opp
    (1, 0),   # 1: ned
    (0, -1),  # 2: venstre
    (0, 1),   # 3: høyre
)
ACTION_NAMES: tuple[str, ...] = ("opp", "ned", "venstre", "høyre")
NUM_ACTIONS = len(ACTIONS)

# Farger (RGB) valgt så de er lette å skille, også etter VAE-komprimering.
COLOR_BACKGROUND = (20, 20, 30)
COLOR_OBSTACLE = (220, 60, 60)
COLOR_GOAL = (60, 200, 90)
COLOR_AGENT = (70, 130, 240)


@dataclass(frozen=True)
class GridConfig:
    grid_size: int = 8          # rutenettet er grid_size x grid_size celler
    cell_px: int = 8            # piksler per celle -> bilde på grid_size * cell_px
    num_obstacles: int = 6
    max_steps: int = 50
    reward_goal: float = 1.0
    reward_obstacle: float = -1.0
    reward_step: float = -0.01

    @property
    def image_size(self) -> int:
        return self.grid_size * self.cell_px


class GridDodgeEnv:
    """Unngå hindringer og nå målet.

    Regler:
      * Å gå inn i kanten av verden: agenten blir stående (ingen straff utover skrittkost).
      * Å gå inn i en hindring: episoden avsluttes med reward_obstacle.
      * Å nå målet: episoden avsluttes med reward_goal.
      * Etter max_steps skritt avkortes episoden (truncated=True).
    """

    def __init__(self, config: GridConfig | None = None, seed: int | None = None):
        self.config = config or GridConfig()
        c = self.config
        if c.num_obstacles > c.grid_size * c.grid_size - 2:
            raise ValueError("For mange hindringer for rutenettet")
        self._rng = np.random.default_rng(seed)
        self.agent_pos: tuple[int, int] = (0, 0)
        self.goal_pos: tuple[int, int] = (0, 0)
        self.obstacles: frozenset[tuple[int, int]] = frozenset()
        self.steps = 0
        self._done = True

    # ------------------------------------------------------------------ API
    @property
    def observation_shape(self) -> tuple[int, int, int]:
        s = self.config.image_size
        return (s, s, 3)

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Trekk nye layouter til målet er nåbart. Med få hindringer lykkes
        # dette nesten alltid på første forsøk.
        while True:
            self._sample_layout()
            if self._goal_reachable():
                break
        self.steps = 0
        self._done = False
        return self.render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("Episoden er ferdig, kall reset() først")
        if not 0 <= int(action) < NUM_ACTIONS:
            raise ValueError(f"Ugyldig handling {action}, må være 0..{NUM_ACTIONS - 1}")

        c = self.config
        dr, dc = ACTIONS[int(action)]
        r, col = self.agent_pos
        nr = min(max(r + dr, 0), c.grid_size - 1)
        ncol = min(max(col + dc, 0), c.grid_size - 1)
        self.agent_pos = (nr, ncol)
        self.steps += 1

        reward = c.reward_step
        terminated = False
        event = "move"
        if self.agent_pos in self.obstacles:
            reward = c.reward_obstacle
            terminated = True
            event = "obstacle"
        elif self.agent_pos == self.goal_pos:
            reward = c.reward_goal
            terminated = True
            event = "goal"
        truncated = not terminated and self.steps >= c.max_steps
        self._done = terminated or truncated

        info = {"event": event, "steps": self.steps, "agent_pos": self.agent_pos}
        return self.render(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------ rendering
    def render(self) -> np.ndarray:
        """Tegn verden som et uint8 RGB-bilde med form observation_shape."""
        c = self.config
        img = np.empty(self.observation_shape, dtype=np.uint8)
        img[:] = COLOR_BACKGROUND
        for pos in self.obstacles:
            self._fill_cell(img, pos, COLOR_OBSTACLE)
        self._fill_cell(img, self.goal_pos, COLOR_GOAL)
        # Agenten tegnes litt mindre enn en hel celle, slik at den er synlig
        # også når den står oppå målet eller en hindring i siste bilde.
        self._fill_cell(img, self.agent_pos, COLOR_AGENT, margin=max(1, c.cell_px // 4))
        return img

    def render_ascii(self) -> str:
        """Tekstversjon av verden, nyttig i tester og feilsøking."""
        g = self.config.grid_size
        rows = []
        for r in range(g):
            row = []
            for col in range(g):
                p = (r, col)
                if p == self.agent_pos:
                    row.append("A")
                elif p == self.goal_pos:
                    row.append("G")
                elif p in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)

    # -------------------------------------------------------------- interne
    def _fill_cell(self, img: np.ndarray, pos: tuple[int, int], color, margin: int = 0) -> None:
        px = self.config.cell_px
        r, c = pos
        img[r * px + margin:(r + 1) * px - margin, c * px + margin:(c + 1) * px - margin] = color

    def _sample_layout(self) -> None:
        c = self.config
        n_cells = c.grid_size * c.grid_size
        cells = self._rng.choice(n_cells, size=c.num_obstacles + 2, replace=False)
        coords = [divmod(int(i), c.grid_size) for i in cells]
        self.agent_pos = coords[0]
        self.goal_pos = coords[1]
        self.obstacles = frozenset(coords[2:])

    def _goal_reachable(self) -> bool:
        """Bredde-først-søk fra agent til mål rundt hindringene."""
        g = self.config.grid_size
        seen = {self.agent_pos}
        queue = deque([self.agent_pos])
        while queue:
            pos = queue.popleft()
            if pos == self.goal_pos:
                return True
            for dr, dc in ACTIONS:
                nxt = (pos[0] + dr, pos[1] + dc)
                if 0 <= nxt[0] < g and 0 <= nxt[1] < g and nxt not in seen and nxt not in self.obstacles:
                    seen.add(nxt)
                    queue.append(nxt)
        return False
