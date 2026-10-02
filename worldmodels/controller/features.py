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

Alt kommer fra M. Controlleren er fortsatt en lineær avbildning.
"""

from __future__ import annotations

import torch

from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.tracker import EyeMemory, observe, visit_features


def num_world_features(model: MDNRNN, beliefs: bool, lookahead: bool, sight: bool = False, track: bool = False) -> int:
    return ((model.num_belief_features if beliefs else 0) + (4 if sight else 0)
            + (model.config.num_actions if lookahead else 0) + (model.config.num_actions if track else 0))


def world_features(
    model: MDNRNN, z: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], beliefs: bool, lookahead: bool,
    sight: bool = False, memory: EyeMemory | None = None, track_goal: bool = False, track: bool = False,
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
        parts.append(model.lookahead_obstacle(z, hidden, None if memory is None else memory.goal, agent_map))
    if track:
        parts.append(visit_features(memory))
    return (torch.cat(parts, dim=-1) if parts else None), memory
