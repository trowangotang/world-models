"""Samle episoder med en ferdig agent (og litt tilfeldighet), f.eks. i verdenen med bevegelige hindringer.

Steg 10 bruker dem sammen med de tilfeldige rolloutene: den tilfeldige policyen krasjer etter få
skritt, mens agenten lever lenger og ser hindringene bevege seg rundt seg (decisions.md D65).

    PYTHONPATH=. python docs/experiments/scripts/agent_rollouts.py --moving 3 --episodes 10000 \\
        --out data/moving/rollouts_agent
"""

import argparse

from worldmodels.controller.agent import WorldModelAgent, collect_agent_rollouts
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridConfig
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.model import ConvVAE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="checkpoints/vae_z32_w10.pt")
    p.add_argument("--rnn", default="checkpoints/mdnrnn_sense.pt")
    p.add_argument("--controller", default="checkpoints/controller_track.npz")
    p.add_argument("--moving", type=int, default=3)
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=500_000)
    p.add_argument("--out", default="data/moving/rollouts_agent")
    a = p.parse_args()
    vae, _ = ConvVAE.load(a.vae)
    rnn, _ = MDNRNN.load(a.rnn)
    agent = WorldModelAgent(vae, rnn, LinearController.load(a.controller))
    stats = collect_agent_rollouts(agent, a.out, a.episodes, seed=a.seed, epsilon=a.epsilon,
                                   config=GridConfig(moving_obstacles=a.moving))
    print(stats)


if __name__ == "__main__":
    main()
