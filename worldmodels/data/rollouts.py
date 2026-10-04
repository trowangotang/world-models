"""Datainnsamling: kjør en tilfeldig policy i miljøet og lagre rollouts til disk.

Hver episode lagres som én komprimert .npz-fil:
    obs         (T+1, H, W, 3) uint8   observasjon før hvert skritt + siste observasjon
    actions     (T,)           int64
    rewards     (T,)           float32
    terminated  (T,)           bool    True på skrittet der agenten traff mål/hindring
    truncated   (T,)           bool    True hvis episoden ble avkortet av max_steps

Overgang t er (obs[t], actions[t], obs[t+1]). Vi lagrer altså ikke "neste
observasjon" som en egen kopi; den er bare obs forskjøvet ett hakk. Det halverer
lagringsbehovet, og sekvensformen er akkurat det MDN-RNN-en trenger i steg 3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from worldmodels.env import NUM_ACTIONS, GridConfig, GridDodgeEnv


class RandomPolicy:
    """Tilfeldig policy med "klebrige" handlinger.

    Med sannsynlighet repeat_prob gjentas forrige handling, ellers trekkes en ny
    uniformt. Det gir lengre rette strekk enn ren uniform støy, slik at
    agenten kommer lenger ut i verden og dataene dekker flere tilstander.
    """

    def __init__(self, num_actions: int = NUM_ACTIONS, repeat_prob: float = 0.5, seed: int | None = None):
        if not 0.0 <= repeat_prob < 1.0:
            raise ValueError("repeat_prob må være i [0, 1)")
        self.num_actions = num_actions
        self.repeat_prob = repeat_prob
        self._rng = np.random.default_rng(seed)
        self._last: int | None = None

    def reset(self) -> None:
        self._last = None

    def act(self, obs: np.ndarray | None = None) -> int:
        if self._last is None or self._rng.random() >= self.repeat_prob:
            self._last = int(self._rng.integers(self.num_actions))
        return self._last


@dataclass
class Episode:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    def transitions(self) -> Iterator[tuple[np.ndarray, int, np.ndarray]]:
        """Gi (obs, action, next_obs) for hvert skritt."""
        for t in range(len(self)):
            yield self.obs[t], int(self.actions[t]), self.obs[t + 1]

    def validate(self) -> None:
        T = len(self.actions)
        assert self.obs.shape[0] == T + 1, "obs må ha én rad mer enn actions"
        assert self.rewards.shape == (T,) and self.terminated.shape == (T,) and self.truncated.shape == (T,)
        assert self.obs.dtype == np.uint8


def run_episode(env: GridDodgeEnv, policy: RandomPolicy, seed: int | None = None) -> Episode:
    obs = env.reset(seed=seed)
    policy.reset()
    frames, actions, rewards, terms, truncs = [obs], [], [], [], []
    done = False
    while not done:
        a = policy.act(obs)
        obs, r, terminated, truncated, _ = env.step(a)
        frames.append(obs)
        actions.append(a)
        rewards.append(r)
        terms.append(terminated)
        truncs.append(truncated)
        done = terminated or truncated
    return Episode(
        obs=np.stack(frames).astype(np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terms, dtype=bool),
        truncated=np.asarray(truncs, dtype=bool),
    )


def save_episode(episode: Episode, path: str | Path) -> None:
    episode.validate()
    np.savez_compressed(
        path,
        obs=episode.obs,
        actions=episode.actions,
        rewards=episode.rewards,
        terminated=episode.terminated,
        truncated=episode.truncated,
    )


def load_episode(path: str | Path) -> Episode:
    with np.load(path) as d:
        ep = Episode(**{k: d[k] for k in ("obs", "actions", "rewards", "terminated", "truncated")})
    ep.validate()
    return ep


def episode_paths(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("episode_*.npz"))


def collect_rollouts(
    out_dir: str | Path,
    num_episodes: int,
    seed: int = 0,
    config: GridConfig | None = None,
    repeat_prob: float = 0.5,
) -> dict:
    """Samle num_episodes episoder til out_dir og returner enkel statistikk.

    Episode i bruker env-seed seed + i, slik at hele datasettet kan
    reproduseres eksakt fra én seed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = GridDodgeEnv(config)
    policy = RandomPolicy(env.num_actions, repeat_prob=repeat_prob, seed=seed)

    lengths, outcomes = [], {"goal": 0, "obstacle": 0, "truncated": 0}
    for i in range(num_episodes):
        ep = run_episode(env, policy, seed=seed + i)
        save_episode(ep, out / f"episode_{i:05d}.npz")
        lengths.append(len(ep))
        if ep.truncated[-1]:
            outcomes["truncated"] += 1
        elif ep.rewards[-1] > 0:
            outcomes["goal"] += 1
        else:
            outcomes["obstacle"] += 1

    return {
        "episodes": num_episodes,
        "transitions": int(sum(lengths)),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "outcomes": outcomes,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Samle rollouts med tilfeldig policy i GridDodge")
    p.add_argument("--out", default="data/rollouts", help="mappe for episode_*.npz")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat-prob", type=float, default=0.5, help="sannsynlighet for å gjenta forrige handling")
    args = p.parse_args(argv)

    stats = collect_rollouts(args.out, args.episodes, seed=args.seed, repeat_prob=args.repeat_prob)
    o = stats["outcomes"]
    print(
        f"Lagret {stats['episodes']} episoder / {stats['transitions']} overganger i {args.out}\n"
        f"Snittlengde {stats['mean_length']:.1f} skritt. "
        f"Utfall: mål {o['goal']}, hindring {o['obstacle']}, avkortet {o['truncated']}"
    )


if __name__ == "__main__":
    main()
