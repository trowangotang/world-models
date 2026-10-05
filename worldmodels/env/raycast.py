"""Førstepersonsbilde av rutenettet med raycasting, som i Wolfenstein 3D og gamle Doom.

Kameraet står midt i agentens celle og ser i retningen agenten vender. For hver bildekolonne
sendes én stråle ut i verden. Strålen går celle for celle (DDA) til den treffer en hindring,
målet eller kanten av verden, og avstanden bestemmer hvor høy veggbiten blir i bildet.
Alt er numpy og regnes for alle kolonnene samtidig.

Hindringer og målet er hele kuber som fyller cellen sin. Kanten av verden er en grå vegg.
"""

from __future__ import annotations

import numpy as np

# Celletyper i kartet raycasteren får
EMPTY_CELL, OBSTACLE_CELL, GOAL_CELL = 0, 1, 2
WALL_CELL = 3  # utenfor rutenettet

COLOR_CEILING = (20, 20, 30)   # samme som bakgrunnen ovenfra
COLOR_FLOOR = (50, 45, 40)
COLOR_WALL = (95, 95, 110)
FOV_DEGREES = 90.0
SIDE_SHADE = 0.75              # vegger som vender sideveis tegnes litt mørkere, så hjørner synes


def _cell_colors(obstacle, goal) -> np.ndarray:
    colors = np.zeros((4, 3), dtype=np.float32)
    colors[OBSTACLE_CELL] = obstacle
    colors[GOAL_CELL] = goal
    colors[WALL_CELL] = COLOR_WALL
    return colors


def cast_rays(cells: np.ndarray, pos: tuple[int, int], heading: tuple[int, int], width: int,
              fov_degrees: float = FOV_DEGREES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Send `width` stråler fra midten av cellen pos i retningen heading = (drad, dkolonne).

    Returnerer (avstand vinkelrett på kameraet, celletype som ble truffet, side 0/1), alle (width,)."""
    G = cells.shape[0]
    py, px = pos[0] + 0.5, pos[1] + 0.5
    dir_x, dir_y = float(heading[1]), float(heading[0])
    half = np.tan(np.radians(fov_degrees) / 2)
    plane_x, plane_y = -dir_y * half, dir_x * half
    cam = 2 * (np.arange(width) + 0.5) / width - 1
    ray_x, ray_y = dir_x + plane_x * cam, dir_y + plane_y * cam
    with np.errstate(divide="ignore"):
        delta_x = np.where(ray_x == 0, 1e30, np.abs(1 / ray_x))
        delta_y = np.where(ray_y == 0, 1e30, np.abs(1 / ray_y))
    map_x = np.full(width, pos[1])
    map_y = np.full(width, pos[0])
    step_x = np.where(ray_x < 0, -1, 1)
    step_y = np.where(ray_y < 0, -1, 1)
    side_x = np.where(ray_x < 0, px - map_x, map_x + 1 - px) * delta_x
    side_y = np.where(ray_y < 0, py - map_y, map_y + 1 - py) * delta_y

    hit = np.zeros(width, dtype=np.int64)
    side = np.zeros(width, dtype=np.int64)
    dist = np.zeros(width)
    active = np.ones(width, dtype=bool)
    for _ in range(2 * G + 2):  # lengste vei gjennom rutenettet
        go_x = side_x < side_y
        mx = active & go_x
        my = active & ~go_x
        side_x[mx] += delta_x[mx]
        map_x[mx] += step_x[mx]
        side_y[my] += delta_y[my]
        map_y[my] += step_y[my]
        side[mx], side[my] = 0, 1
        outside = (map_x < 0) | (map_x >= G) | (map_y < 0) | (map_y >= G)
        kind = np.where(outside, WALL_CELL, cells[np.clip(map_y, 0, G - 1), np.clip(map_x, 0, G - 1)])
        done = active & (kind != EMPTY_CELL)
        hit[done] = kind[done]
        dist[done] = np.where(side[done] == 0, side_x[done] - delta_x[done], side_y[done] - delta_y[done])
        active &= ~done
        if not active.any():
            break
    return np.maximum(dist, 1e-3), hit, side


def render_first_person(cells: np.ndarray, pos: tuple[int, int], heading: tuple[int, int], size: int,
                        obstacle_color, goal_color) -> np.ndarray:
    """Tegn det agenten ser: (size, size, 3) uint8.

    cells: (G, G) med EMPTY_CELL / OBSTACLE_CELL / GOAL_CELL. Står agenten selv i en hindring eller i
    målet (siste bilde i en episode), fyller den fargen hele bildet."""
    colors = _cell_colors(obstacle_color, goal_color)
    own = cells[pos]
    if own != EMPTY_CELL:
        return np.broadcast_to(colors[own].astype(np.uint8), (size, size, 3)).copy()
    dist, hit, side = cast_rays(cells, pos, heading, size)
    column = colors[hit] * np.where(side == 1, SIDE_SHADE, 1.0)[:, None]
    # Litt tåke: det som er langt unna blir mørkere, så dybden synes også i fargen
    column *= np.clip(1.1 - 0.07 * dist, 0.45, 1.0)[:, None]
    line = size / dist
    y = np.arange(size)[:, None] + 0.5
    wall = np.abs(y - size / 2) < line[None, :] / 2                    # (size, size)
    img = np.empty((size, size, 3), dtype=np.float32)
    img[: size // 2] = COLOR_CEILING
    img[size // 2:] = COLOR_FLOOR
    img = np.where(wall[..., None], column[None, :, :], img)
    return np.clip(img, 0, 255).astype(np.uint8)
