"""Drømmen: et simulert miljø der MDN-RNN-en spiller verden.

Controlleren trenes her i stedet for i det ekte miljøet. Oppsettet følger funnene fra steg 3
(decisions.md D22):

  * **Oppvarming.** Hver drøm starter fra en ekte episode. RNN-en ser de første `context`
    ekte skrittene (fra datasettet med tilfeldige handlinger), slik at minnet vet hvor agenten er.
    Deretter overtar controlleren, og alt videre er drøm.
  * **Korte drømmer.** Drømmen sporer av etter 6–8 skritt, så vi drømmer `horizon` skritt (10).
  * **Forventet belønning.** I stedet for å trekke én hendelse per skritt bruker vi
    sannsynlighetene fra hendelseshodet: forventet belønning i skrittet, og sannsynligheten for
    at episoden fortsatt lever. Det gir mye mindre støy i fitness enn å trekke utfall.
  * **Resten av episoden.** Med `remaining_steps` betaler en agent som overlever drømmen for
    skrittene som ville gjenstått i en ekte episode. Ellers er det nesten gratis å gjemme seg.
  * **Formet belønning.** Med `shaping` gir drømmen litt belønning for hver celle agenten
    kommer nærmere målet, ut fra M sin egen tro om posisjonene.
  * **Temperatur.** Neste z trekkes fra blandingen med temperatur tau. Litt støy gjør det
    vanskeligere for controlleren å utnytte feil i drømmen (et poeng fra artikkelen).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from worldmodels.controller.features import Habits, world_features
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridConfig
from worldmodels.mdnrnn.encode import EVENT_GOAL, EVENT_MOVE, EVENT_OBSTACLE, ZSequences
from worldmodels.mdnrnn.model import MDNRNN, most_likely_mean, sample_next
from worldmodels.mdnrnn.tracker import EyeMemory, observe
from worldmodels.mdnrnn.train import make_batch


@dataclass
class WarmStarts:
    """Tilstanden etter oppvarming for et sett ekte episoder."""
    z: torch.Tensor        # (N, D) modellens prediksjon av z etter oppvarming
    h: torch.Tensor        # (1, N, H) LSTM-tilstand
    c: torch.Tensor        # (1, N, H)
    episodes: np.ndarray   # (N,) hvilke episoder startene kom fra
    memory: EyeMemory | None = None  # øyets hukommelse fra oppvarmingen (mål, agent, besøk)

    def __len__(self) -> int:
        return len(self.episodes)

    def sample(self, n: int, rng: np.random.Generator) -> "WarmStarts":
        idx = rng.choice(len(self), size=min(n, len(self)), replace=False)
        return self.subset(idx)

    def subset(self, idx) -> "WarmStarts":
        memory = None if self.memory is None else self.memory.subset(idx)
        return WarmStarts(self.z[idx], self.h[:, idx], self.c[:, idx], self.episodes[idx], memory)

    @classmethod
    def concat(cls, parts: list["WarmStarts"]) -> "WarmStarts":
        memory = None if parts[0].memory is None else EyeMemory.concat([p.memory for p in parts])
        return cls(torch.cat([p.z for p in parts]), torch.cat([p.h for p in parts], dim=1),
                   torch.cat([p.c for p in parts], dim=1), np.concatenate([p.episodes for p in parts]), memory)


@torch.no_grad()
def make_warm_starts(model: MDNRNN, seqs: ZSequences, indices, context: int = 5, batch_size: int = 1024) -> WarmStarts:
    """Kjør RNN-en over de første `context` ekte skrittene av hver episode som varer så lenge.

    context=0 starter drømmen i episodens første bilde, med tomt minne, slik agenten selv starter.
    Det trengs når hindringene beveger seg: der er de første skrittene de farligste (D67)."""
    model.eval()
    if context == 0:
        keep = np.asarray(list(indices))
        z = torch.from_numpy(seqs.mu[seqs.obs_offsets[keep]]).float()
        zeros = torch.zeros(1, len(keep), model.config.hidden_dim)
        memory = None if model.eye is None else EyeMemory.fresh(len(keep), model.config.grid_cells)
        return WarmStarts(z, zeros, zeros.clone(), keep, memory)
    keep = np.array([i for i in indices if seqs.lengths[i] > context])
    zs, hs, cs, ms = [], [], [], []
    for start in range(0, len(keep), batch_size):
        b = make_batch(seqs, keep[start:start + batch_size])
        out = model(b.z_mu[:, :context], b.actions[:, :context])
        zs.append(most_likely_mean(out)[:, -1])
        hs.append(out.hidden[0])
        cs.append(out.hidden[1])
        if model.eye is not None:
            # Øyet har sett de ekte oppvarmingsbildene: det husker hvor målet var (D49), hvor
            # agenten står og hvor den har vært (D56, D57).
            memory = None
            for t in range(context):
                _, _, memory = observe(model, b.z_mu[:, t], memory)
                memory = memory.moved(b.actions[:, t])
            ms.append(memory)
    memory = None if not ms else EyeMemory.concat(ms)
    return WarmStarts(torch.cat(zs), torch.cat(hs, dim=1), torch.cat(cs, dim=1), keep, memory)


def goal_distance(model: MDNRNN, h: torch.Tensor) -> torch.Tensor:
    """Manhattan-avstand i celler mellom der M tror agenten og målet er. (B, H) -> (B,)"""
    return position_distance(model, model.position_features(h))


def position_distance(model: MDNRNN, f: torch.Tensor) -> torch.Tensor:
    """Manhattan-avstand i celler fra posisjoner (B, 4) i [-1, 1], 8 celler per akse."""
    cells_per_unit = (model.config.grid_cells ** 0.5 - 1) / 2
    return ((f[:, 0] - f[:, 2]).abs() + (f[:, 1] - f[:, 3]).abs()) * cells_per_unit


def compass_distance(f: torch.Tensor) -> torch.Tensor:
    """Manhattan-avstand i celler til målet fra kompasset (B, 3) = [fram, høyre, sett]."""
    from worldmodels.mdnrnn.model import COMPASS_SCALE

    return (f[:, 0].abs() + f[:, 1].abs()) * COMPASS_SCALE


@dataclass(frozen=True)
class DreamConfig:
    horizon: int = 10
    temperature: float = 1.0
    reward_goal: float = GridConfig.reward_goal
    reward_obstacle: float = GridConfig.reward_obstacle
    reward_step: float = GridConfig.reward_step
    # Skritt som gjenstår av en ekte episode når drømmen slutter. Overlever agenten drømmen,
    # betaler den skrittkostnaden for dem også, fordi det å vente ut tiden ikke er gratis i
    # virkeligheten (decisions.md D33). 0 = gammel oppførsel.
    remaining_steps: int = 0
    # Formet belønning: så mye per celle agenten kommer nærmere målet, målt med M sin egen tro
    # om posisjonene (krever posisjonshode). Potensialbasert, så den endrer ikke hvilken
    # politikk som er best (Ng m.fl. 1999), men gir søket en bakke å klatre (decisions.md D37).
    shaping: float = 0.0


@torch.no_grad()
def dream_fitness(
    controller: LinearController,
    population: np.ndarray,
    model: MDNRNN,
    starts: WarmStarts,
    config: DreamConfig = DreamConfig(),
    generator: torch.Generator | None = None,
    return_details: bool = False,
):
    """Forventet drømmeavkastning for hver kandidat, snittet over alle startene.

    population: (P, num_params). Alle P kandidater drømmer fra de samme B startene, i én batch
    med P * B drømmer. Returnerer (P,) numpy, eller en dict med detaljer.
    """
    model.eval()
    P, B = len(population), len(starts)
    D = starts.z.shape[1]
    z = starts.z.repeat(P, 1)                                          # (P*B, D)
    hidden = (starts.h.repeat(1, P, 1), starts.c.repeat(1, P, 1))
    alive = torch.ones(P * B)
    total = torch.zeros(P * B)
    p_goal_sum = torch.zeros(P * B)
    p_obstacle_sum = torch.zeros(P * B)
    action_counts = torch.zeros(controller.num_actions)
    # Med syn måles avstanden i det øyet ser i drømmebildene, med kompass i det kompasset sier (førsteperson),
    # ellers i minnets tro.
    compass_shaping = config.shaping > 0 and controller.compass
    eye_shaping = config.shaping > 0 and (controller.sight or compass_shaping)
    use_shaping = config.shaping > 0 and (eye_shaping or model.num_position_features > 0)
    shaped = torch.zeros(P * B)
    memory = None if starts.memory is None else starts.memory.repeat(P)
    seen_at = model.num_belief_features if controller.beliefs else 0   # hvor synet står i extra
    prev_distance = None
    # Vanene starter blanke i drømmen; oppvarmingen er tilfeldige skritt som ikke er agentens egne
    habits = Habits.fresh(P * B) if controller.habits else None
    for _ in range(config.horizon):
        h = hidden[0][-1]
        extra, memory = world_features(model, z, hidden, controller.beliefs, controller.lookahead,
                                       controller.sight, memory, track_goal=model.neighbours is not None,
                                       track=controller.track, compass=controller.compass)
        if habits is not None:
            extra = habits.features(z) if extra is None else torch.cat([extra, habits.features(z)], dim=-1)
        if eye_shaping:
            # Gevinsten for forrige skritt, nå som vi ser bildet det førte til. alive er sannsynligheten
            # for at det skrittet var en vanlig flytting; etter mål eller krasj betyr bildet ingenting.
            at = extra.shape[1] - (2 if habits is not None else 0)
            distance = (compass_distance(extra[:, at - 3:at]) if compass_shaping
                        else position_distance(model, extra[:, seen_at:seen_at + 4]))
            if prev_distance is not None:
                shaped += alive * config.shaping * (prev_distance - distance)
            prev_distance = distance
        logits = controller.batched_logits(
            population, z.view(P, B, D), h.view(P, B, -1), None if extra is None else extra.view(P, B, -1)
        )
        a = logits.argmax(-1).view(P * B)
        action_counts += torch.bincount(a, minlength=controller.num_actions)
        out = model(z.unsqueeze(1), a.unsqueeze(1), hidden)
        hidden = out.hidden
        p = model.event_probs(out.event_logits[:, 0], z, None if memory is None else memory.goal, a,
                              None if memory is None else memory.agent, h,
                              None if memory is None else memory.obstacles_before)
        if memory is not None:
            memory = memory.moved(a)
        if habits is not None:
            habits = habits.after(z, a)
        expected = (
            p[:, EVENT_MOVE] * config.reward_step
            + p[:, EVENT_GOAL] * config.reward_goal
            + p[:, EVENT_OBSTACLE] * config.reward_obstacle
        )
        total += alive * expected
        if use_shaping and not eye_shaping:
            gain = goal_distance(model, h) - goal_distance(model, hidden[0][-1])
            shaped += alive * config.shaping * gain
        p_goal_sum += alive * p[:, EVENT_GOAL]
        p_obstacle_sum += alive * p[:, EVENT_OBSTACLE]
        alive = alive * p[:, EVENT_MOVE]
        nxt = sample_next(out, config.temperature, generator) if config.temperature > 0 else most_likely_mean(out)
        z = nxt[:, 0]
    if compass_shaping:
        last = compass_distance(model.compass_features(z, hidden[0][-1]))
        shaped += alive * config.shaping * (prev_distance - last)
    elif eye_shaping:
        seen, tracked, _ = observe(model, z, memory)
        seen = tracked if controller.track else seen
        shaped += alive * config.shaping * (prev_distance - position_distance(model, seen))
    total += alive * config.remaining_steps * config.reward_step
    fitness = (total + shaped).view(P, B).mean(1).numpy()
    if not return_details:
        return fitness
    return {
        "fitness": fitness,
        "p_goal": p_goal_sum.view(P, B).mean(1).numpy(),
        "p_obstacle": p_obstacle_sum.view(P, B).mean(1).numpy(),
        "p_alive_end": alive.view(P, B).mean(1).numpy(),
        "shaping": shaped.view(P, B).mean(1).numpy(),
        "action_share": (action_counts / action_counts.sum()).numpy(),
        # Per kandidat og start, (P, B): brukes til å sammenligne drøm og virkelighet (steg 5).
        "p_goal_per_start": p_goal_sum.view(P, B).numpy(),
        "p_obstacle_per_start": p_obstacle_sum.view(P, B).numpy(),
    }
