"""Steg 8: hvorfor pendler steg 7-agenten, og hva hjelper? (decisions.md D56, D57)

Kjører steg 7-agenten på 1000 brett (seeds 100000–100999) og ser på de siste 20 skrittene i episodene
som pendler: ser øyet agenten og målet riktig, og stemmer fremsynet? Deretter prøver vi steg 7-
controlleren med sporing (filteret i mdnrnn/tracker.py) og en håndlagt besøksvekt, og lagrer den
som checkpoints/controller_track_handcrafted.npz (diagnostikk i evalueringen).

    PYTHONPATH=. python docs/experiments/scripts/oscillation.py
"""

import numpy as np
import torch

from worldmodels.controller.agent import WorldModelAgent, run_real_episodes
from worldmodels.controller.features import world_features
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridDodgeEnv
from worldmodels.evaluation.run import is_oscillating
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.dataset import to_tensor
from worldmodels.vae.model import ConvVAE

VISIT_WEIGHT = 0.1
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def cell(p):
    return tuple(np.clip(np.rint((p + 1) / 2 * 7), 0, 7).astype(int))


@torch.no_grad()
def diagnose(vae, rnn, controller, seeds):
    n = len(seeds)
    envs = [GridDodgeEnv() for _ in seeds]
    obs = np.stack([e.reset(seed=s) for e, s in zip(envs, seeds)])
    H = rnn.config.hidden_dim
    hidden, memory = (torch.zeros(1, n, H), torch.zeros(1, n, H)), None
    log = [[] for _ in range(n)]
    while not all(e._done for e in envs):
        z, _ = vae.encode(to_tensor(obs))
        extra, memory = world_features(rnn, z, hidden, False, True, True, memory)
        a = controller.act(z, hidden[0][-1], extra)
        hidden = rnn(z.unsqueeze(1), a.unsqueeze(1), hidden).hidden
        memory = memory.moved(a)
        for i, e in enumerate(envs):
            if e._done:
                continue
            f = extra[i].numpy()
            true_obstacle = [tuple(np.add(e.agent_pos, d)) in e.obstacles for d in DIRECTIONS]
            log[i].append((tuple(e.agent_pos), tuple(e.goal_pos), cell(f[0:2]), cell(f[2:4]),
                           bool(all((f[4 + k] > 0.5) == true_obstacle[k] for k in range(4)))))
            obs[i] = e.step(int(a[i]))[0]
    stuck = [i for i in range(n) if len(log[i]) >= 50 and is_oscillating([s[0] for s in log[i]])]
    print(f"{len(stuck)} av {n} episoder pendler")

    def share(rows):
        rows = list(rows)
        agent = np.mean([s[2] == s[0] for s in rows])
        goal = np.mean([s[3] == s[1] for s in rows])
        look = np.mean([s[4] for s in rows])
        return f"øyet ser agenten riktig {agent:.0%}, målet {goal:.0%}, fremsynet riktig i alle retninger {look:.0%}"

    print("  siste 20 skritt i pendlende episoder:", share(s for i in stuck for s in log[i][-20:]))
    print("  alle skritt i de andre episodene:   ", share(s for i in range(n) if i not in stuck for s in log[i]))


def with_tracking(controller, visit_weight):
    """Steg 7-controlleren med sporing: samme vekter, pluss -visit_weight på besøk i hver retning."""
    c = LinearController(controller.z_dim, controller.h_dim, controller.num_actions, controller.extra_dim + 4,
                         lookahead=True, sight=True, track=True)
    W, b = controller.unflatten(controller.params)
    Wn = np.zeros((c.in_dim, c.num_actions), np.float32)
    Wn[: controller.in_dim] = W.numpy()
    Wn[controller.in_dim:] = -visit_weight * np.eye(c.num_actions)
    c.params = np.concatenate([Wn.ravel(), b.numpy()]).astype(np.float32)
    return c


def main():
    torch.set_grad_enabled(False)
    vae, _ = ConvVAE.load("checkpoints/vae_z32_w10.pt")
    rnn, _ = MDNRNN.load("checkpoints/mdnrnn_sense.pt")
    step7 = LinearController.load("checkpoints/controller_sense.npz")
    diagnose(vae.eval(), rnn.eval(), step7, list(range(100_000, 101_000)))
    for name, c in [("steg 7", step7), ("steg 7 + sporing, besøksvekt 0", with_tracking(step7, 0.0)),
                    (f"steg 7 + sporing, besøksvekt {VISIT_WEIGHT}", with_tracking(step7, VISIT_WEIGHT))]:
        r = run_real_episodes(WorldModelAgent(vae, rnn, c), 1000)
        print(f"{name}: mål {r['goal_rate']:.1%}, krasj {r['obstacle_rate']:.1%}, avkortet {r['truncated_rate']:.1%}")
    with_tracking(step7, VISIT_WEIGHT).save("checkpoints/controller_track_handcrafted.npz")


if __name__ == "__main__":
    main()
