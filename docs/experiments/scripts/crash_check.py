"""Hvorfor krasjer agentene i verdenen med bevegelige hindringer? (steg 10, decisions.md D67)

Spiller hver agent på 1000 brett (seeds 100000–100999) med tre bevegelige hindringer og deler
krasjene i to: gikk agenten inn i en hindring som allerede sto der (synlig i bildet), eller flyttet en
hindring seg inn der agenten havnet ("på vei inn", krever å vite hvor den er på vei)? Vi teller også
krasj i første skritt, der ingen kan vite hvor hindringene er på vei, og krasj uten noen trygg handling.

    PYTHONPATH=. python docs/experiments/scripts/crash_check.py mdnrnn_sense.pt:controller_track.npz \\
        mdnrnn_moving.pt:controller_moving.npz
"""

import sys
from collections import Counter

import numpy as np

from worldmodels.controller.agent import WorldModelAgent
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridConfig, GridDodgeEnv
from worldmodels.env.gridworld import ACTIONS
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.model import ConvVAE

EPISODES, SEED = 1000, 100_000


def target(env, action):
    g = env.config.grid_size
    (r, c), (dr, dc) = env.agent_pos, ACTIONS[action]
    return min(max(r + dr, 0), g - 1), min(max(c + dc, 0), g - 1)


def main():
    vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
    for spec in sys.argv[1:]:
        rnn_name, ctrl_name = spec.split(":")
        rnn, _ = MDNRNN.load(f"checkpoints/{rnn_name}")
        agent = WorldModelAgent(vae, rnn, LinearController.load(f"checkpoints/{ctrl_name}"))
        envs = [GridDodgeEnv(GridConfig(moving_obstacles=3)) for _ in range(EPISODES)]
        obs = np.stack([env.reset(seed=SEED + i) for i, env in enumerate(envs)])
        agent.reset(EPISODES)
        active = np.ones(EPISODES, bool)
        kinds, goals = Counter(), 0
        while active.any():
            actions = agent.act(obs)
            for i in np.flatnonzero(active):
                env, a = envs[i], int(actions[i])
                cell, now, soon = target(env, a), env.obstacles, env.next_obstacles()
                trapped = all(target(env, b) in env.danger() for b in range(len(ACTIONS)))
                first = env.steps == 0
                obs[i], _, terminated, truncated, info = env.step(a)
                if terminated and info["event"] == "obstacle":
                    kind = "sto der" if cell in now else "på vei inn"
                    kinds[kind] += 1
                    kinds["første skritt"] += first
                    kinds["ingen trygg handling"] += trapped
                goals += terminated and info["event"] == "goal"
                active[i] = not (terminated or truncated)
        crashes = kinds["sto der"] + kinds["på vei inn"]
        print(f"{ctrl_name}: mål {goals / EPISODES:.1%}, krasj {crashes / EPISODES:.1%}  "
              + ", ".join(f"{k} {v}" for k, v in kinds.items()))


if __name__ == "__main__":
    main()
