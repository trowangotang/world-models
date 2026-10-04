"""Ekstra inndata til controlleren: det verdensmodellen tror om verden, utover [z, h].

  * **Tro** (`beliefs`): hvor M tror agenten og målet er (4 tall), og med hindringshode, om det
    står en hindring i hver nabocelle (4 tall). Leses lineært fra h (decisions.md D32, D40).
  * **Syn** (`sight`): hvor øyet ser agenten og målet i bildet akkurat nå (4 tall). Samme format
    som troen, men lest fra z i stedet for minnet, så den virker fra første skritt (D48).
  * **Fremsyn** (`lookahead`): for hver handling, M sin sannsynlighet for å treffe en hindring
    om agenten tar den nå (4 tall). M spør seg selv "hva skjer om jeg går opp?" (D41).
  * **Sporing** (`track`, krever syn): agentposisjonen i synet kommer fra et filter som også vet
    hvilken handling agenten tok, og controlleren får for hver handling hvor mye agenten nylig
    har vært i cellen den fører til (4 tall). Mot pendling (D56, D57, se mdnrnn/tracker.py).

  * **Kompass** (`compass`, førsteperson): hvor målet er sett fra agenten, (fram, høyre), og
    sannsynligheten for at den har sett det (3 tall). Leses fra minnet før skrittet og bildet nå (D77).
    I førsteperson er dette nesten det en lineær controller trenger: mål til høyre betyr snu til høyre.

  * **Vaner** (`habits`, førsteperson): hvor mange ganger på rad agenten har snudd, og om forrige
    skritt fram ikke endret bildet (gikk i veggen). To tall agenten vet om seg selv, mot å spinne
    rundt eller gå fast (D80).

Alt kommer fra M. Controlleren er fortsatt en lineær avbildning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.tracker import EyeMemory, observe, visit_features


def num_world_features(model: MDNRNN, beliefs: bool, lookahead: bool, sight: bool = False, track: bool = False,
                       compass: bool = False, habits: bool = False) -> int:
    return ((model.num_belief_features if beliefs else 0) + (4 if sight else 0)
            + (model.config.num_actions if lookahead else 0) + (model.config.num_actions if track else 0)
            + (3 if compass else 0) + (2 if habits else 0))


STUCK_DISTANCE = 0.5   # z endrer seg minst ~0,8 når agenten flytter seg, og ikke i det hele tatt når den står fast
TURN_STREAK_FULL = 3   # så mange snu på rad gir vane-tallet 1


@dataclass
class Habits:
    """Det agenten vet om sine egne siste skritt i førsteperson (D80)."""
    streak: torch.Tensor        # (B,) antall snu på rad
    last_forward: torch.Tensor  # (B,) bool: forrige handling var fram
    prev_z: torch.Tensor | None = None  # (B, D) bildet før forrige handling

    @classmethod
    def fresh(cls, n: int) -> "Habits":
        return cls(torch.zeros(n), torch.zeros(n, dtype=torch.bool))

    def features(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 2): snu på rad (0..1) og om forrige skritt fram ikke flyttet agenten."""
        if self.prev_z is None:
            stuck = torch.zeros(len(z))
        else:
            stuck = (self.last_forward & ((z - self.prev_z).norm(dim=-1) < STUCK_DISTANCE)).float()
        return torch.stack([(self.streak / TURN_STREAK_FULL).clamp(max=1), stuck], dim=-1)

    def after(self, z: torch.Tensor, actions: torch.Tensor) -> "Habits":
        from worldmodels.env.gridworld import FORWARD, TURN_LEFT, TURN_RIGHT

        turned = (actions == TURN_LEFT) | (actions == TURN_RIGHT)
        return Habits(torch.where(turned, self.streak + 1, torch.zeros_like(self.streak)), actions == FORWARD, z)

    def repeat(self, p: int) -> "Habits":
        return Habits(self.streak.repeat(p), self.last_forward.repeat(p),
                      None if self.prev_z is None else self.prev_z.repeat(p, 1))


def world_features(
    model: MDNRNN, z: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], beliefs: bool, lookahead: bool,
    sight: bool = False, memory: EyeMemory | None = None, track_goal: bool = False, track: bool = False,
    compass: bool = False,
) -> tuple[torch.Tensor | None, EyeMemory | None]:
    """z: (B, D), hidden: LSTM-tilstanden før skrittet -> ((B, antall) eller None, ny hukommelse).

    memory er øyets hukommelse (se tracker.py). Den oppdateres med z når noe trenger det:
    syn, fremsyn fra nærsynet, eller track_goal (drømmen, når den spår hendelser med nærsyn).
    Etter at handlingen er valgt, må den som kaller flytte den med memory.moved(handlinger)."""
    if track and not sight:
        raise ValueError("track krever sight")
    parts = []
    if sight or track_goal or (lookahead and model.neighbours is not None):
        seen, tracked, memory = observe(model, z, memory)
    if beliefs:
        parts.append(model.belief_features(hidden[0][-1]))
    if sight:
        parts.append(tracked if track else seen)
    if lookahead:
        agent_map = memory.agent if track else None
        parts.append(model.lookahead_obstacle(z, hidden, None if memory is None else memory.goal, agent_map,
                                              None if memory is None else memory.obstacles_before))
    if track:
        parts.append(visit_features(memory))
    if compass:
        parts.append(model.compass_features(z, hidden[0][-1]))
    return (torch.cat(parts, dim=-1) if parts else None), memory
