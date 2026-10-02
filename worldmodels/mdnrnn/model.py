"""MDN-RNN: en LSTM som predikerer neste latente tilstand og hva som skjer.

Gitt z_t og handling a_t gir modellen:
  * en blanding av K gaussiske fordelinger over z_{t+1} (MDN-hodet),
  * sannsynligheter for hendelsen i skrittet: vanlig flytt, mål eller hindring.

To bevisste avvik fra artikkelen (se decisions.md):
  * Hver komponent er en gaussisk fordeling over *hele* z-vektoren, ikke én blanding per
    dimensjon. I et rutenett er utfallene diskrete (agenten havner i én av noen få celler), og
    alle dimensjonene må endre seg sammen. Én komponent per utfall passer bedre.
  * Middelverdiene er residualer: mu_k = z_t + delta_k. Det meste av scenen står stille, så
    modellen trenger bare lære endringen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from worldmodels.env import NUM_ACTIONS
from worldmodels.mdnrnn.encode import NUM_EVENTS

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -7.0, 2.0


@dataclass(frozen=True)
class MDNRNNConfig:
    latent_dim: int = 32
    num_actions: int = NUM_ACTIONS
    hidden_dim: int = 256
    num_mixtures: int = 5
    input_mlp: bool = True
    direct_path: bool = True
    position_head: bool = False
    grid_cells: int = 64
    obstacle_head: bool = False
    # Øyet leser posisjonene fra z, altså fra bildet som sees nå, ikke fra minnet (eye.py, D48).
    eye: bool = False
    eye_channels: int = 32
    # Nærsynet spår hendelsene fra bildet og øyets målminne i stedet for fra h (neighbours.py, D52).
    neighbours: bool = False
    neighbour_channels: int = 32


@dataclass
class MDNOutput:
    logit_pi: torch.Tensor   # (B, T, K)
    mu: torch.Tensor         # (B, T, K, D)
    log_sigma: torch.Tensor  # (B, T, K, D)
    event_logits: torch.Tensor  # (B, T, NUM_EVENTS)
    hidden: tuple[torch.Tensor, torch.Tensor]
    h: torch.Tensor          # (B, T, H) skjult tilstand etter hvert skritt
    position_logits: torch.Tensor | None = None  # (B, T, 2, celler): agent og mål etter skrittet
    obstacle_logits: torch.Tensor | None = None  # (B, T, 4): hindring over/under/venstre/høyre etter skrittet


class MDNRNN(nn.Module):
    def __init__(self, config: MDNRNNConfig | None = None):
        super().__init__()
        self.config = c = config or MDNRNNConfig()
        in_dim = c.latent_dim + c.num_actions
        # Posisjonen er ikke lineært tilgjengelig i z (decisions.md D14). Et lite MLP før LSTM-en
        # lar modellen rette ut z allerede i første skritt, i stedet for over flere skritt.
        self.input_net = (
            nn.Sequential(nn.Linear(in_dim, c.hidden_dim), nn.ReLU(), nn.Linear(c.hidden_dim, c.hidden_dim), nn.ReLU())
            if c.input_mlp
            else nn.Identity()
        )
        self.lstm = nn.LSTM(c.hidden_dim if c.input_mlp else in_dim, c.hidden_dim, batch_first=True)
        # Direkte vei: hodene ser både minnet h og inndata for dette skrittet, gjennom et skjult
        # lag. Første skritt i en episode har ingen historikk, og da må prediksjonen komme fra
        # z_t alene. Uten denne veien lærte LSTM-en å lene seg på historikken og bare kopierte
        # z i første skritt.
        head_in = c.hidden_dim
        if c.direct_path:
            x_dim = c.hidden_dim if c.input_mlp else c.latent_dim + c.num_actions
            self.trunk = nn.Sequential(nn.Linear(c.hidden_dim + x_dim, c.hidden_dim), nn.ReLU())
        else:
            self.trunk = None
        self.pi_head = nn.Linear(head_in, c.num_mixtures)
        self.mu_head = nn.Linear(head_in, c.num_mixtures * c.latent_dim)
        self.sigma_head = nn.Linear(head_in, c.num_mixtures * c.latent_dim)
        self.event_head = nn.Linear(head_in, NUM_EVENTS)
        # Hjelpehode: hvor er agenten og målet? Det er lineært fra h med vilje, slik at
        # posisjonene blir lineært lesbare fra minnet, som er det den lineære controlleren ser.
        # Brukes bare som ekstra tap under trening (decisions.md D31).
        self.position_head = nn.Linear(c.hidden_dim, 2 * c.grid_cells) if c.position_head else None
        # Hjelpehode nr. 2: står det en hindring i nabocellen i hver retning? Også lineært fra h,
        # av samme grunn (decisions.md D40).
        self.obstacle_head = nn.Linear(c.hidden_dim, c.num_actions) if c.obstacle_head else None
        if c.eye:
            from worldmodels.mdnrnn.eye import SpatialEye

            self.eye = SpatialEye(c.latent_dim, c.eye_channels, int(round(c.grid_cells ** 0.5)))
        else:
            self.eye = None
        if c.neighbours:
            from worldmodels.mdnrnn.neighbours import NeighbourEye

            self.neighbours = NeighbourEye(c.latent_dim, c.neighbour_channels, int(round(c.grid_cells ** 0.5)))
        else:
            self.neighbours = None

    def forward(
        self, z: torch.Tensor, actions: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> MDNOutput:
        """z: (B, T, D) float, actions: (B, T) long."""
        c = self.config
        a = F.one_hot(actions, c.num_actions).float()
        x = self.input_net(torch.cat([z, a], dim=-1))
        h, hidden = self.lstm(x, hidden)
        feat = self.trunk(torch.cat([h, x], dim=-1)) if self.trunk is not None else h
        B, T, _ = h.shape
        delta = self.mu_head(feat).view(B, T, c.num_mixtures, c.latent_dim)
        log_sigma = self.sigma_head(feat).view(B, T, c.num_mixtures, c.latent_dim)
        return MDNOutput(
            logit_pi=self.pi_head(feat),
            mu=z.unsqueeze(2) + delta,
            log_sigma=log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX),
            event_logits=self.event_head(feat),
            hidden=hidden,
            h=h,
            position_logits=(
                self.position_head(h).view(B, T, 2, c.grid_cells) if self.position_head is not None else None
            ),
            obstacle_logits=self.obstacle_head(h) if self.obstacle_head is not None else None,
        )

    @property
    def num_position_features(self) -> int:
        return 4 if self.position_head is not None else 0

    def position_features(self, h: torch.Tensor) -> torch.Tensor:
        """Modellens tro om hvor agenten og målet er, som forventet (rad, kolonne) i [-1, 1].

        h: (..., H) -> (..., 4) = [agent rad, agent kolonne, mål rad, mål kolonne]. Tom (..., 0)
        uten posisjonshode. Retningen til målet blir da en lineær funksjon av inndata, noe en
        lineær controller kan bruke direkte (decisions.md D32).
        """
        if self.position_head is None:
            return h.new_zeros(*h.shape[:-1], 0)
        G = self.config.grid_cells
        side = int(round(G ** 0.5))
        probs = F.softmax(self.position_head(h).view(*h.shape[:-1], 2, G), dim=-1)
        coords = torch.linspace(-1, 1, side, device=h.device)
        rows = (probs.view(*probs.shape[:-1], side, side).sum(-1) * coords).sum(-1)  # (..., 2)
        cols = (probs.view(*probs.shape[:-1], side, side).sum(-2) * coords).sum(-1)
        return torch.stack([rows[..., 0], cols[..., 0], rows[..., 1], cols[..., 1]], dim=-1)

    def seen_positions(self, z: torch.Tensor) -> torch.Tensor:
        """Øyets tro om posisjonene, fra z alene: (..., D) -> (..., 4) i [-1, 1], samme format
        som position_features. Krever eye=True."""
        return self.see(z)[0]

    def see(self, z: torch.Tensor, goal_memory: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Det controlleren ser: agenten fra bildet nå, målet fra alle bildene så langt.

        Målet står stille, så hvert bilde er en stemme om hvor det er. goal_memory (B, celler) er
        summen av log-sannsynlighetene fra bildene før, None ved start. Et bilde der øyet bommer
        (oftest når agenten står like ved målet) blir nedstemt av de andre (decisions.md D49).
        Returnerer (posisjoner (B, 4) i [-1, 1], ny goal_memory)."""
        from worldmodels.mdnrnn.eye import GOAL_EVIDENCE_FLOOR, expected_positions

        logits = self.eye(z)
        evidence = torch.log(F.softmax(logits[..., 1, :], dim=-1) + GOAL_EVIDENCE_FLOOR)
        memory = evidence if goal_memory is None else goal_memory + evidence
        side = int(round(self.config.grid_cells ** 0.5))
        combined = torch.stack([logits[..., 0, :], memory], dim=-2)
        return expected_positions(combined, side), memory

    @property
    def num_belief_features(self) -> int:
        return self.num_position_features + (self.config.num_actions if self.obstacle_head is not None else 0)

    def belief_features(self, h: torch.Tensor) -> torch.Tensor:
        """Alt M tror om verden som controlleren kan få: posisjonene (4 tall) og, med
        hindringshode, sannsynligheten for hindring i hver retning (4 tall i [0, 1])."""
        parts = [self.position_features(h)]
        if self.obstacle_head is not None:
            parts.append(torch.sigmoid(self.obstacle_head(h)))
        return torch.cat(parts, dim=-1)

    def event_probs(
        self, event_logits: torch.Tensor, z: torch.Tensor, goal_memory: torch.Tensor | None, actions: torch.Tensor,
        agent_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sannsynlighet for [flytt, mål, hindring] (B, 3) når handlingen tas fra bildet z.

        Med nærsyn kommer de fra bildet og øyets målminne (etter at z er sett), ellers fra
        hendelseshodet på h (event_logits, (B, 3)). agent_map er en filtrert tro om agentcellen
        (tracker.py); uten den brukes det øyet ser i bildet alene."""
        if self.neighbours is None:
            return F.softmax(event_logits, dim=-1)
        from worldmodels.mdnrnn.neighbours import event_probs_from_logits

        return event_probs_from_logits(self.neighbour_logits(z, goal_memory, agent_map), actions)

    def neighbour_logits(
        self, z: torch.Tensor, goal_memory: torch.Tensor | None, agent_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Nærsynets logits (B, 2, 4). goal_memory skal allerede inneholde bildet z (se see)."""
        if goal_memory is None:
            goal_memory = self.see(z)[1]
        if agent_map is None:
            agent_map = F.softmax(self.eye(z)[..., 0, :], dim=-1)
        return self.neighbours(z, F.softmax(goal_memory, dim=-1), agent_map)

    @torch.no_grad()
    def lookahead_obstacle(
        self, z: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], goal_memory: torch.Tensor | None = None,
        agent_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fremsyn ett skritt: sannsynligheten for å treffe en hindring for hver handling.

        z: (B, D), hidden: LSTM-tilstanden før skrittet. Kjører modellen én gang per handling
        (i én batch) uten å endre tilstanden. Returnerer (B, num_actions) (decisions.md D41).
        Med nærsyn leses det rett fra bildet, samme kilde som drømmen bruker (D52).
        """
        from worldmodels.mdnrnn.encode import EVENT_OBSTACLE

        if self.neighbours is not None:
            # Samme regel som event_probs: mål vinner over hindring
            logits = self.neighbour_logits(z, goal_memory, agent_map)
            return (1 - torch.sigmoid(logits[:, 0])) * torch.sigmoid(logits[:, 1])

        B, A = z.shape[0], self.config.num_actions
        zz = z.repeat(A, 1).unsqueeze(1)                                    # (A*B, 1, D)
        aa = torch.arange(A).repeat_interleave(B).unsqueeze(1)              # (A*B, 1)
        hh = tuple(x.repeat(1, A, 1) for x in hidden)
        p = F.softmax(self(zz, aa, hh).event_logits[:, 0], dim=-1)[:, EVENT_OBSTACLE]
        return p.view(A, B).T

    # ------------------------------------------------------------ lagring
    def save(self, path, **extra) -> None:
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> tuple["MDNRNN", dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        # Eldre checkpointer ble lagret før input-MLP og direkte vei fantes
        config = {"input_mlp": False, "direct_path": False, **ckpt["config"]}
        model = cls(MDNRNNConfig(**config))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt


def mdn_nll(out: MDNOutput, target: torch.Tensor) -> torch.Tensor:
    """Negativ log-sannsynlighet for target (B, T, D) under blandingen, per skritt (B, T).

    Delt på D, slik at tallet er sammenlignbart mellom ulike størrelser på z.
    """
    t = target.unsqueeze(2)                                           # (B, T, 1, D)
    log_prob = -0.5 * (((t - out.mu) / out.log_sigma.exp()) ** 2) - out.log_sigma - 0.5 * torch.log(
        torch.tensor(2 * torch.pi)
    )
    log_prob = log_prob.sum(-1)                                       # (B, T, K)
    log_mix = torch.logsumexp(F.log_softmax(out.logit_pi, dim=-1) + log_prob, dim=-1)
    return -log_mix / target.shape[-1]


def most_likely_mean(out: MDNOutput) -> torch.Tensor:
    """Middelverdien til komponenten med høyest vekt, (B, T, D). Deterministisk prediksjon."""
    k = out.logit_pi.argmax(-1)                                        # (B, T)
    idx = k[..., None, None].expand(*k.shape, 1, out.mu.shape[-1])
    return out.mu.gather(2, idx).squeeze(2)


def sample_next(out: MDNOutput, temperature: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Trekk z_{t+1} fra blandingen, (B, T, D).

    temperature skalerer både valg av komponent og spredningen. 0 gir most_likely_mean.
    Høyere temperatur gir en mer uforutsigbar drøm, som i artikkelen brukes for å gjøre
    det vanskeligere for controlleren å utnytte feil i modellen.
    """
    if temperature <= 0:
        return most_likely_mean(out)
    probs = F.softmax(out.logit_pi / temperature, dim=-1)
    k = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator).view(probs.shape[:-1])
    idx = k[..., None, None].expand(*k.shape, 1, out.mu.shape[-1])
    mu = out.mu.gather(2, idx).squeeze(2)
    sigma = out.log_sigma.exp().gather(2, idx).squeeze(2)
    noise = torch.randn(mu.shape, generator=generator)
    return mu + sigma * noise * (temperature ** 0.5)
