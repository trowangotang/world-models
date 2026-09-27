"""Les rutenettet tilbake fra et bilde.

Brukes til to ting i senere steg:
  * etiketter (agentens og målets celle) for å teste hva VAE-ens latente vektor vet,
  * å vurdere rekonstruksjoner objektivt: står agenten i riktig celle i det VAE-en tegner?

Hver celle klassifiseres etter hvilken palettfarge sentrumspikselen ligger nærmest.
Agenten tegnes med en kant, så sentrum er alltid agentfarge når agenten står der.
"""

from __future__ import annotations

import numpy as np

from worldmodels.env.gridworld import (
    COLOR_AGENT,
    COLOR_BACKGROUND,
    COLOR_GOAL,
    COLOR_OBSTACLE,
    GridConfig,
)

EMPTY, OBSTACLE, GOAL, AGENT = 0, 1, 2, 3
PALETTE = np.array([COLOR_BACKGROUND, COLOR_OBSTACLE, COLOR_GOAL, COLOR_AGENT], dtype=np.float32)


def parse_cells(img: np.ndarray, config: GridConfig | None = None) -> np.ndarray:
    """Klassifiser hver celle i ett bilde eller en batch.

    img: (H, W, 3) eller (N, H, W, 3), uint8 i [0, 255] eller float i [0, 1].
    Returnerer (G, G) eller (N, G, G) med verdier EMPTY/OBSTACLE/GOAL/AGENT.
    """
    c = config or GridConfig()
    x = np.asarray(img)
    single = x.ndim == 3
    if single:
        x = x[None]
    x = x.astype(np.float32)
    if x.max() <= 1.0 + 1e-6 and np.asarray(img).dtype != np.uint8:
        x = x * 255.0
    centre = c.cell_px // 2
    centres = x[:, centre::c.cell_px, centre::c.cell_px, :]           # (N, G, G, 3)
    dist = ((centres[..., None, :] - PALETTE) ** 2).sum(-1)            # (N, G, G, 4)
    cells = dist.argmin(-1)
    return cells[0] if single else cells


def find_cell(cells: np.ndarray, kind: int) -> np.ndarray:
    """Indeks (r * G + c) for første celle av gitt type, eller -1 hvis den mangler.

    cells: (G, G) eller (N, G, G). Returnerer skalar eller (N,).
    """
    flat = cells.reshape(*cells.shape[:-2], -1) == kind
    idx = flat.argmax(-1)
    return np.where(flat.any(-1), idx, -1)
