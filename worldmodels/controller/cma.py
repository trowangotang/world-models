"""CMA-ES med diagonal kovarians (sep-CMA-ES, Ros & Hansen 2008).

Hvorfor: den enkle ES-en i es.py følger en gradient fra ett punkt. I drømmen fra steg 4c er
"vent ved veggen" en bred, trygg dal, og gradienten fører dit hver gang (decisions.md D35).
CMA-ES velger i stedet de beste kandidatene i hver generasjon og flytter hele søkefordelingen
mot dem. Den tilpasser også steglengden og hvor mye hver parameter varierer. Det gjør den
mindre avhengig av en jevn gradient.

Full CMA-ES har en n x n kovariansmatrise. Med ~1200 parametre blir det tregt å oppdatere,
og den lærer for sakte. Den diagonale varianten lærer én varians per parameter og skalerer
lineært. Grensesnittet er det samme som EvolutionStrategy: ask() / tell(fitness) / theta.
"""

from __future__ import annotations

import numpy as np


class SepCMAES:
    def __init__(
        self,
        num_params: int,
        sigma: float = 0.5,
        population: int | None = None,
        seed: int = 0,
        init: np.ndarray | None = None,
        normalize: bool = False,
    ):
        n = self.n = num_params
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros(n) if init is None else np.asarray(init, dtype=np.float64).copy()
        self.sigma = float(sigma)
        self.population = lam = population or 4 + int(3 * np.log(n))
        self.mu = mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = w / w.sum()
        self.mueff = mueff = 1.0 / (self.weights ** 2).sum()

        # Standardverdier fra Hansens veiledning. c1 og cmu skaleres opp med (n + 2) / 3 for
        # den diagonale varianten, fordi den bare har n tall å lære i stedet for n^2.
        self.cs = (mueff + 2) / (n + mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        scale = (n + 2) / 3
        self.c1, self.cmu = min(c1 * scale, 0.5), min(cmu * scale, 0.5)
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        self.var = np.ones(n)   # diagonalen i C
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.generation = 0
        self._z: np.ndarray | None = None

    @property
    def theta(self) -> np.ndarray:
        return self.mean

    def ask(self) -> np.ndarray:
        """Returner (population, num_params) kandidater."""
        self._z = self.rng.standard_normal((self.population, self.n))
        return self.mean + self.sigma * self._z * np.sqrt(self.var)

    def tell(self, fitness: np.ndarray) -> None:
        """Oppdater fordelingen fra fitness for kandidatene fra siste ask(). Høyere er bedre."""
        if self._z is None:
            raise RuntimeError("Kall ask() før tell()")
        order = np.argsort(-np.asarray(fitness, dtype=np.float64))[: self.mu]
        z = self._z[order]
        y = z * np.sqrt(self.var)
        z_w, y_w = self.weights @ z, self.weights @ y
        self.mean = self.mean + self.sigma * y_w

        self.generation += 1
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_w
        ps_norm = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
        hsig = float(ps_norm / self.chi_n < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
        rank_mu = self.weights @ (y ** 2)
        self.var = (
            (1 - self.c1 - self.cmu) * self.var
            + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.var)
            + self.cmu * rank_mu
        )
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        if self.normalize:
            # Controlleren velger argmax av logits, så fitness endres ikke om alle vektene
            # skaleres. Uten normalisering vokser både middelverdi og sigma uten grense, og søket
            # blir i praksis tilfeldig. Vi holder middelverdien på lengde 1 og skalerer sigma
            # likt, så fordelingen beskriver de samme politikkene (decisions.md D36).
            norm = np.linalg.norm(self.mean)
            if norm > 0:
                self.mean /= norm
                self.sigma /= norm
        self._z = None
