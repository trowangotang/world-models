"""Policyer som kan spille mange miljøer samtidig: grunnlinjer og verdensmodell-agenter.

Alle har samme grensesnitt:
    policy.reset(envs)              før første skritt
    actions = policy.act(obs, envs) obs: (B, 64, 64, 3), returnerer (B,) handlinger

Grunnlinjene merket "juks" leser miljøets interne tilstand (posisjoner og hindringer, og hvor de
bevegelige hindringene står etter neste skritt). De er
ikke konkurrenter, men målestokker: hvor godt kan man gjøre det med full kunnskap?
"""

from __future__ import annotations

from collections import deque

import numpy as np

from worldmodels.data.rollouts import RandomPolicy
from worldmodels.env.gridworld import ACTIONS, NUM_ACTIONS, GridDodgeEnv


def next_cell(env: GridDodgeEnv, pos: tuple[int, int], action: int) -> tuple[int, int]:
    """Cellen agenten havner i, med samme kantregel som miljøet (blir stående ved kanten)."""
    g = env.config.grid_size
    dr, dc = ACTIONS[action]
    return min(max(pos[0] + dr, 0), g - 1), min(max(pos[1] + dc, 0), g - 1)


def bfs_distances(env: GridDodgeEnv, source: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Korteste antall skritt fra source til hver nåbar celle, rundt hindringene."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        pos = queue.popleft()
        for a in range(NUM_ACTIONS):
            nxt = next_cell(env, pos, a)
            if nxt not in dist and nxt not in env.obstacles:
                dist[nxt] = dist[pos] + 1
                queue.append(nxt)
    return dist


class RandomBaseline:
    """Den samme klebrige tilfeldige policyen som samlet treningsdataene."""
    name = "Tilfeldig"
    cheat = False

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self, envs):
        self.policies = [RandomPolicy(seed=self.seed + i) for i in range(len(envs))]

    def act(self, obs, envs):
        return np.array([p.act() for p in self.policies])


class SafeRandomBaseline:
    """Juks: tilfeldig blant handlingene som ikke fører inn i en hindring."""
    name = "Tilfeldig som unngår hindringer (juks)"
    cheat = True

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, envs):
        pass

    def act(self, obs, envs):
        out = []
        for env in envs:
            if env._done:  # ferdige episoder får en handling som ignoreres
                out.append(0)
                continue
            safe = [a for a in range(NUM_ACTIONS) if next_cell(env, env.agent_pos, a) not in env.danger()]
            out.append(self.rng.choice(safe or list(range(NUM_ACTIONS))))
        return np.array(out)


class GreedyGoalBaseline:
    """Juks: går rett mot målet med sanne posisjoner, men ser ikke hindringene.

    Det er det steg 4d-controlleren prøver å gjøre med M sin tro, så den viser hvor langt
    "bare gå mot målet" rekker når troen er perfekt."""
    name = "Rett mot målet (juks)"
    cheat = True

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, envs):
        pass

    def act(self, obs, envs):
        out = []
        for env in envs:
            if env._done:
                out.append(0)
                continue
            (r, c), (gr, gc) = env.agent_pos, env.goal_pos
            good = [a for a, (dr, dc) in enumerate(ACTIONS) if (dr and np.sign(gr - r) == dr) or (dc and np.sign(gc - c) == dc)]
            out.append(self.rng.choice(good))
        return np.array(out)


class GreedySafeBaseline:
    """Juks: som "rett mot målet", men tar aldri et skritt inn i en hindring.

    Det er det steg 4e-controlleren prøver å gjøre med M sin tro og fremsyn. Finnes ingen trygg
    handling mot målet, velges en tilfeldig trygg handling. Den husker ingenting, akkurat som C."""
    name = "Rett mot målet, unngår hindringer (juks)"
    cheat = True

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, envs):
        pass

    def act(self, obs, envs):
        out = []
        for env in envs:
            if env._done:
                out.append(0)
                continue
            (r, c), (gr, gc) = env.agent_pos, env.goal_pos
            safe = [a for a in range(NUM_ACTIONS) if next_cell(env, env.agent_pos, a) not in env.danger()]
            good = [a for a in safe if (ACTIONS[a][0] and np.sign(gr - r) == ACTIONS[a][0])
                    or (ACTIONS[a][1] and np.sign(gc - c) == ACTIONS[a][1])]
            out.append(self.rng.choice(good or safe or list(range(NUM_ACTIONS))))  # helt innestengt: alt er like ille
        return np.array(out)


class ShortestPathBaseline:
    """Juks: følger korteste vei rundt hindringene. Taket for hva som er mulig."""
    name = "Korteste vei (juks)"
    cheat = True

    def reset(self, envs):
        self.dist = [bfs_distances(env, env.goal_pos) for env in envs]

    def act(self, obs, envs):
        out = []
        for env, dist in zip(envs, self.dist):
            # Med bevegelige hindringer er avstandene fra starten bare et anslag, og den holder seg
            # unna cellene som blir farlige i neste skritt (D64).
            danger = env.danger()
            options = [(next_cell(env, env.agent_pos, a) in danger, dist.get(next_cell(env, env.agent_pos, a), 10 ** 6), a)
                       for a in range(NUM_ACTIONS)]
            out.append(min(options)[2])
        return np.array(out)


class WorldModelPolicy:
    """Hele agenten: bildet gjennom V, minnet i M, handlingen fra C. Ser bare pikslene."""
    cheat = False

    def __init__(self, name: str, agent):
        self.name, self.agent = name, agent

    def reset(self, envs):
        self.agent.reset(len(envs))

    def act(self, obs, envs):
        return self.agent.act(obs)
