"""Ekstra inndata til controlleren: det verdensmodellen tror om verden, utover [z, h].

  * **Tro** (`beliefs`): hvor M tror agenten og målet er (4 tall), og med hindringshode, om det
    står en hindring i hver nabocelle (4 tall). Leses lineært fra h (decisions.md D32, D40).
  * **Fremsyn** (`lookahead`): for hver handling, M sin sannsynlighet for å treffe en hindring
    om agenten tar den nå (4 tall). M spør seg selv "hva skjer om jeg går opp?" (D41).

Alt kommer fra M. Controlleren er fortsatt en lineær avbildning.
"""

from __future__ import annotations

import torch

from worldmodels.mdnrnn.model import MDNRNN


def num_world_features(model: MDNRNN, beliefs: bool, lookahead: bool) -> int:
    return (model.num_belief_features if beliefs else 0) + (model.config.num_actions if lookahead else 0)


def world_features(
    model: MDNRNN, z: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], beliefs: bool, lookahead: bool
) -> torch.Tensor | None:
    """z: (B, D), hidden: LSTM-tilstanden før skrittet -> (B, antall) eller None."""
    parts = []
    if beliefs:
        parts.append(model.belief_features(hidden[0][-1]))
    if lookahead:
        parts.append(model.lookahead_obstacle(z, hidden))
    return torch.cat(parts, dim=-1) if parts else None
