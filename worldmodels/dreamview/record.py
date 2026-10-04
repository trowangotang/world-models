"""Ta opp hva agenten ser, tror og drømmer i hvert skritt av en ekte episode (steg 9).

For hvert skritt t i en ekte episode lagrer vi:

  * hva V ser: bildet etter koding og dekoding (z_t gjennom dekoderen),
  * hva øyet tror: agentcellen fra bildet alene og fra filteret, og målcellen fra målminnet,
  * hva controlleren får og gjør: fremsyn, besøk, skårene og handlingen,
  * en drøm: fra akkurat denne tilstanden lar vi M drømme `horizon` skritt fram med controlleren
    ved roret, slik som under trening (temperatur 0, se controller/dream.py). For hvert drømmeskritt
    lagrer vi det dekodede drømmebildet, handlingen og sannsynlighetene for mål og krasj,
  * virkeligheten for de samme handlingene: en kopi av miljøet tar handlingene drømmen valgte, så
    drøm og virkelighet kan sammenlignes skritt for skritt.

Den ekte verden er så enkel at den tegnes i nettleseren fra posisjonene. Bare bildene som kommer
fra modellene (V sin rekonstruksjon og drømmen) lagres som bilder.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from worldmodels.controller.features import world_features
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridDodgeEnv
from worldmodels.mdnrnn.encode import EVENT_GOAL, EVENT_MOVE, EVENT_OBSTACLE
from worldmodels.mdnrnn.model import MDNRNN, most_likely_mean
from worldmodels.vae.dataset import to_tensor
from worldmodels.vae.model import ConvVAE


MIN_ALIVE = 0.02  # drømmen regnes som over når den lever med mindre sannsynlighet enn dette


@dataclass
class DreamStep:
    action: int
    p_goal: float          # sannsynlighet for mål i dette skrittet, gitt at drømmen fortsatt lever
    p_obstacle: float
    alive: float           # sannsynlighet for at drømmen fortsatt lever etter skrittet
    frame: np.ndarray      # (64, 64, 3) uint8: M sitt bilde etter handlingen
    agent_belief: np.ndarray  # (64,) hva controlleren trodde om agentcellen før handlingen
    real_pos: tuple[int, int] | None   # hvor agenten faktisk havnet med samme handling (None: allerede over)
    real_event: str | None             # "move", "goal", "obstacle" eller None


@dataclass
class RealStep:
    pos: tuple[int, int]
    recon: np.ndarray         # (64, 64, 3) uint8: bildet slik V gjenskaper det
    eye_agent: np.ndarray     # (64,) agentcellen fra dette bildet alene
    agent_belief: np.ndarray  # (64,) etter filteret
    goal_belief: np.ndarray   # (64,) fra målminnet
    lookahead: np.ndarray     # (4,) sannsynlighet for krasj per handling
    visits: np.ndarray        # (4,) besøk i cellen hver handling fører til
    logits: np.ndarray        # (4,) controllerens skår
    action: int
    event: str                # hva som skjedde da handlingen ble tatt
    dream: list[DreamStep] = field(default_factory=list)


@dataclass
class EpisodeRecord:
    seed: int
    goal: tuple[int, int]
    obstacles: list[tuple[int, int]]
    steps: list[RealStep]
    outcome: str
    final_pos: tuple[int, int]   # der agenten stod da episoden sluttet


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) i [0, 1] -> (H, W, 3) uint8."""
    return (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def dream_from(vae, rnn, controller, z, hidden, memory, env, horizon: int) -> list[DreamStep]:
    """Drøm horizon skritt fra (z, hidden, memory), der memory ennå ikke har sett z.

    Samme regler som controller/dream.py: temperatur 0, hendelser fra nærsynet, sporingen følger med."""
    real = copy.deepcopy(env)
    alive = 1.0
    out_steps = []
    for _ in range(horizon):
        extra, memory = world_features(rnn, z, hidden, controller.beliefs, controller.lookahead, controller.sight,
                                       memory, track_goal=True, track=controller.track)
        a = controller.act(z, hidden[0][-1], extra)
        out = rnn(z.unsqueeze(1), a.unsqueeze(1), hidden)
        p = rnn.event_probs(out.event_logits[:, 0], z, memory.goal, a, memory.agent)[0]
        belief = memory.agent[0].numpy().copy()
        memory = memory.moved(a)
        hidden = out.hidden
        z = most_likely_mean(out)[:, 0]
        real_pos = real_event = None
        if not real._done:
            _, _, _, _, info = real.step(int(a))
            real_pos, real_event = tuple(int(x) for x in real.agent_pos), info["event"]
        out_steps.append(DreamStep(
            action=int(a), p_goal=float(p[EVENT_GOAL]), p_obstacle=float(p[EVENT_OBSTACLE]),
            alive=alive * float(p[EVENT_MOVE]), frame=to_uint8(vae.decode(z)[0]),
            agent_belief=belief, real_pos=real_pos, real_event=real_event,
        ))
        alive *= float(p[EVENT_MOVE])
        if alive < MIN_ALIVE and real._done:
            break   # drømmen er over (mål eller krasj), og virkeligheten også
    return out_steps


@torch.no_grad()
def record_episode(vae: ConvVAE, rnn: MDNRNN, controller: LinearController, seed: int, horizon: int = 10) -> EpisodeRecord:
    """Spill én ekte episode med agenten og ta opp alt (se modulens beskrivelse)."""
    vae, rnn = vae.eval(), rnn.eval()
    env = GridDodgeEnv()
    obs = env.reset(seed=seed)
    H = rnn.config.hidden_dim
    hidden = (torch.zeros(1, 1, H), torch.zeros(1, 1, H))
    memory = None
    steps = []
    outcome = "truncated"
    while not env._done:
        z, _ = vae.encode(to_tensor(obs[None]))
        dream = dream_from(vae, rnn, controller, z, hidden, memory, env, horizon)
        extra, memory = world_features(rnn, z, hidden, controller.beliefs, controller.lookahead, controller.sight,
                                       memory, track=controller.track)
        logits = controller.logits(z, hidden[0][-1], extra)[0]
        a = int(logits.argmax())
        k = 4 if controller.sight else 0
        steps.append(RealStep(
            pos=tuple(int(x) for x in env.agent_pos), recon=to_uint8(vae.decode(z)[0]),
            eye_agent=F.softmax(rnn.eye(z)[0, 0], -1).numpy(), agent_belief=memory.agent[0].numpy().copy(),
            goal_belief=F.softmax(memory.goal[0], -1).numpy(),
            lookahead=extra[0, k:k + 4].numpy() if controller.lookahead else np.zeros(4),
            visits=extra[0, k + 4:k + 8].numpy() if controller.track else np.zeros(4),
            logits=logits.numpy(), action=a, event="move", dream=dream,
        ))
        hidden = rnn(z.unsqueeze(1), torch.tensor([[a]]), hidden).hidden
        memory = memory.moved(torch.tensor([a]))
        obs, _, terminated, _, info = env.step(a)
        steps[-1].event = info["event"]
        if terminated:
            outcome = info["event"]
    return EpisodeRecord(seed=seed, goal=tuple(int(x) for x in env.goal_pos),
                         obstacles=sorted(tuple(int(x) for x in o) for o in env.obstacles), steps=steps, outcome=outcome,
                         final_pos=tuple(int(x) for x in env.agent_pos))
