"""En enkel evolusjonsstrategi (ES) for å optimere controllerens vekter.

Artikkelen bruker CMA-ES. Vi bruker en enklere variant (Salimans m.fl. 2017):
  * trekk støy eps_i ~ N(0, I) og evaluer theta + sigma * eps_i og theta - sigma * eps_i
    (antitetisk, halverer variansen),
  * ranger fitnessverdiene og gjør dem om til tall i [-0.5, 0.5] (robust mot uteliggere),
  * estimer gradienten som et vektet snitt av støyen og ta et Adam-skritt.

Den er ~40 linjer, lett å teste og skalerer fint til de ~1000 parametrene vi har.
Fitness beregnes utenfor klassen, slik at samme ES kan brukes i drømmen og i virkeligheten.
"""

from __future__ import annotations

import numpy as np


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Ranger x og skaler til [-0.5, 0.5]. Høyest verdi får 0.5."""
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1) - 0.5


class EvolutionStrategy:
    def __init__(
        self,
        num_params: int,
        population: int = 32,
        sigma: float = 0.1,
        lr: float = 0.03,
        seed: int = 0,
        init: np.ndarray | None = None,
    ):
        if population % 2:
            raise ValueError("population må være et partall (antitetisk sampling)")
        self.theta = np.zeros(num_params) if init is None else np.asarray(init, dtype=np.float64).copy()
        self.population, self.sigma, self.lr = population, sigma, lr
        self.rng = np.random.default_rng(seed)
        self._m = np.zeros(num_params)
        self._v = np.zeros(num_params)
        self._t = 0
        self._eps: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        """Returner (population, num_params) kandidater som skal evalueres."""
        half = self.rng.standard_normal((self.population // 2, len(self.theta)))
        self._eps = np.concatenate([half, -half])
        return self.theta + self.sigma * self._eps

    def tell(self, fitness: np.ndarray) -> None:
        """Oppdater theta fra fitness for kandidatene fra siste ask(). Høyere er bedre."""
        if self._eps is None:
            raise RuntimeError("Kall ask() før tell()")
        weights = centered_ranks(np.asarray(fitness, dtype=np.float64))
        grad = weights @ self._eps / (len(weights) * self.sigma)
        # Adam-skritt (maksimering)
        self._t += 1
        b1, b2 = 0.9, 0.999
        self._m = b1 * self._m + (1 - b1) * grad
        self._v = b2 * self._v + (1 - b2) * grad ** 2
        m_hat = self._m / (1 - b1 ** self._t)
        v_hat = self._v / (1 - b2 ** self._t)
        self.theta = self.theta + self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        self._eps = None
