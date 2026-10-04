"""Øyets hukommelse: hvor målet er, hvor agenten er, og hvor agenten har vært (steg 8).

Øyet (eye.py) leser hvert bilde for seg. Det bommer på agenten når den står tett inntil målet
eller en hindring, fordi VAE-en gjør bildet uskarpt der (decisions.md D56). Da tror agenten den
står et annet sted enn den gjør, og controlleren velger et skritt som var riktig for den andre
cellen. Neste bilde leses riktig, controlleren går tilbake, og agenten pendler.

Men agenten vet noe øyet ikke bruker: hvilken handling den nettopp tok. Et skritt flytter den
høyst én celle. Så vi holder en tro om agentcellen og oppdaterer den som et Bayes-filter:

  * **flytt** (`EyeMemory.moved`): troen flyttes med handlingen. Går den i veggen, blir den stående.
  * **se** (`observe`): troen ganges med det øyet ser i bildet, og med at agenten ikke står på
    målet (da hadde episoden vært over).

I tillegg teller vi hvor agenten har vært, med glemsel (`visits`). Controlleren får vite hvor mye
den har vært i cellen hver handling fører til (`visit_features`). Et skritt tilbake dit den nettopp
kom fra, eller rett inn i veggen, gir et høyt tall, og controlleren kan lære å la være (D57).

Alt dette er regnet ut fra M sitt øye og agentens egne handlinger, så det virker likt i drømmen og
i det ekte miljøet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

AGENT_EVIDENCE_FLOOR = 1e-3   # samme rolle som GOAL_EVIDENCE_FLOOR: ett bilde kan ikke utelukke en celle
VISIT_DECAY = 0.9             # besøk for 10 skritt siden teller ca. en tredjedel


@dataclass
class EyeMemory:
    goal: torch.Tensor     # (B, celler) summen av log-sannsynlighetene for målet (MDNRNN.see)
    agent: torch.Tensor    # (B, celler) troen om agentcellen, summerer til 1
    visits: torch.Tensor   # (B, celler) troen fra tidligere skritt, summert med glemsel

    def __len__(self) -> int:
        return len(self.goal)

    @classmethod
    def fresh(cls, n: int, cells: int) -> "EyeMemory":
        """Hukommelsen før første bilde: virker i observe som memory=None, men kan slås sammen med andre."""
        return cls(torch.zeros(n, cells), torch.full((n, cells), 1 / cells), torch.zeros(n, cells))

    @classmethod
    def concat(cls, parts: list["EyeMemory"]) -> "EyeMemory":
        return cls(*(torch.cat([getattr(m, f) for m in parts]) for f in ("goal", "agent", "visits")))

    def subset(self, idx) -> "EyeMemory":
        return EyeMemory(self.goal[idx], self.agent[idx], self.visits[idx])

    def repeat(self, n: int) -> "EyeMemory":
        """Samme hukommelse n ganger etter hverandre (én kopi per kandidat i drømmen)."""
        return EyeMemory(self.goal.repeat(n, 1), self.agent.repeat(n, 1), self.visits.repeat(n, 1))

    def clone(self) -> "EyeMemory":
        return EyeMemory(self.goal.clone(), self.agent.clone(), self.visits.clone())

    def moved(self, actions: torch.Tensor) -> "EyeMemory":
        """Flytt troen om agenten med handlingene (B,). Kalles etter at handlingen er valgt."""
        agent = move_belief(self.agent)[torch.arange(len(actions)), actions]
        return EyeMemory(self.goal, agent, self.visits)


def move_belief(belief: torch.Tensor) -> torch.Tensor:
    """(B, celler) -> (B, 4, celler): troen etter opp, ned, venstre, høyre. Går man i veggen,
    blir man stående, akkurat som i GridDodge."""
    side = int(round(belief.shape[-1] ** 0.5))
    g = belief.view(-1, side, side)
    up, down, left, right = (torch.zeros_like(g) for _ in range(4))
    up[:, :-1] = g[:, 1:]
    up[:, 0] += g[:, 0]
    down[:, 1:] = g[:, :-1]
    down[:, -1] += g[:, -1]
    left[:, :, :-1] = g[:, :, 1:]
    left[:, :, 0] += g[:, :, 0]
    right[:, :, 1:] = g[:, :, :-1]
    right[:, :, -1] += g[:, :, -1]
    return torch.stack([up, down, left, right], dim=1).flatten(2)


def belief_positions(belief: torch.Tensor) -> torch.Tensor:
    """(B, celler) -> (B, 2) forventet [rad, kolonne] i [-1, 1], samme skala som øyet."""
    side = int(round(belief.shape[-1] ** 0.5))
    g = belief.view(-1, side, side)
    coords = torch.linspace(-1, 1, side, device=belief.device)
    return torch.stack([(g.sum(2) * coords).sum(1), (g.sum(1) * coords).sum(1)], dim=-1)


def observe(model, z: torch.Tensor, memory: EyeMemory | None) -> tuple[torch.Tensor, torch.Tensor, EyeMemory]:
    """Se bildet z og oppdater hukommelsen.

    Returnerer (seen, tracked, memory): seen er det øyet ser i dette bildet alene (agenten) og
    målminnet (steg 6), tracked er det samme med filtrert agentposisjon. Begge (B, 4) i [-1, 1]."""
    seen, goal = model.see(z, None if memory is None else memory.goal)
    likelihood = F.softmax(model.eye(z)[..., 0, :], dim=-1) + AGENT_EVIDENCE_FLOOR
    prior = torch.full_like(likelihood, 1.0) if memory is None else memory.agent
    # Agenten står ikke på målet. Gulvet hindrer at alt blir null når målminnet er helt sikkert
    # på cellen troen står i (det skjer i drømmen, der M kan tegne agenten oppå målet).
    not_goal = (1 - F.softmax(goal, dim=-1)).clamp_min(AGENT_EVIDENCE_FLOOR)
    agent = prior * likelihood * not_goal
    agent = agent / agent.sum(-1, keepdim=True)
    visits = agent if memory is None else VISIT_DECAY * memory.visits + agent
    tracked = torch.cat([belief_positions(agent), seen[:, 2:]], dim=-1)
    return seen, tracked, EyeMemory(goal, agent, visits)


def visit_features(memory: EyeMemory) -> torch.Tensor:
    """(B, 4): hvor mye agenten har vært i cellen hver handling fører til, inkludert dette skrittet."""
    return (move_belief(memory.agent) * memory.visits.unsqueeze(1)).sum(-1)
