"""Tren controlleren i drømmen med en evolusjonsstrategi.

    python -m worldmodels.controller.train --vae checkpoints/vae_z32_w10.pt \
        --rnn checkpoints/mdnrnn_direct_k5.pt --data data/zseq_20k.npz --out checkpoints/controller.npz

Hver generasjon trekkes nye oppvarmingsstarter, slik at controlleren ikke tilpasser seg noen få
drømmer. Med jevne mellomrom sjekkes den beste controlleren i det ekte miljøet. Den sjekken
brukes bare til logging, aldri til å velge vekter: treningen skjer kun i drømmen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from worldmodels.controller.agent import WorldModelAgent, run_real_episodes
from worldmodels.controller.dream import DreamConfig, WarmStarts, dream_fitness, make_warm_starts
from worldmodels.controller.cma import SepCMAES
from worldmodels.controller.es import EvolutionStrategy
from worldmodels.controller.features import num_world_features
from worldmodels.controller.policy import LinearController
from worldmodels.env import GridConfig
from worldmodels.mdnrnn.encode import ZSequences
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.train import split_indices
from worldmodels.vae.model import ConvVAE


def train(
    vae_path, rnn_path, data, out,
    generations: int = 200,
    population: int = 32,
    starts_per_generation: int = 64,
    sigma: float = 0.1,
    lr: float = 0.03,
    context: int | list[int] = 5,
    horizon: int = 10,
    temperature: float = 1.0,
    real_check_every: int = 25,
    real_check_episodes: int = 200,
    init_from: str | Path | None = None,
    use_positions: bool = False,
    lookahead: bool = False,
    sight: bool = False,
    track: bool = False,
    charge_remaining: bool = False,
    shaping: float = 0.0,
    optimizer: str = "es",
    zh_std: float = 1.0,
    seed: int = 0,
    moving: int = 0,
    compass: bool = False,
    first_person: bool = False,
    no_back: bool = False,
    habits: bool = False,
    log=lambda msg: print(msg, flush=True),
) -> dict:
    torch.manual_seed(seed)
    real_config = GridConfig(moving_obstacles=moving, first_person=first_person)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    vae, _ = ConvVAE.load(vae_path)
    rnn, rnn_ckpt = MDNRNN.load(rnn_path)
    seqs = ZSequences.load_many(data)
    train_idx, _ = split_indices(len(seqs), seed=rnn_ckpt.get("hparams", {}).get("seed", 0))
    # Flere oppvarmingslengder gir drømmer som starter på ulike tidspunkt i episoden (D67)
    contexts = [context] if isinstance(context, int) else list(context)
    starts = WarmStarts.concat([make_warm_starts(rnn, seqs, train_idx, context=k) for k in contexts])
    log(f"{len(starts)} oppvarmingsstarter fra treningsepisodene")

    if use_positions and not rnn.num_belief_features:
        raise ValueError("use_positions krever en MDN-RNN med posisjonshode")
    if sight and rnn.eye is None:
        raise ValueError("sight krever en MDN-RNN med øye (python -m worldmodels.mdnrnn.eye)")
    if track and not sight:
        raise ValueError("track krever sight (--use-eye)")
    if compass and rnn.compass_net is None:
        raise ValueError("compass krever en MDN-RNN med kompass (--compass-weight)")
    extra_dim = num_world_features(rnn, use_positions, lookahead, sight, track, compass, habits)
    # Uten rygging (førsteperson) kan controlleren bare velge fram og snu: den må se dit den går (D79).
    # Rygg er siste handling, så de andre beholder nummeret sitt, og M kjenner fortsatt alle fire.
    num_actions = rnn.config.num_actions - (1 if no_back else 0)
    controller = LinearController(seqs.mu.shape[1], rnn.config.hidden_dim, num_actions, extra_dim,
                                  beliefs=use_positions, lookahead=lookahead, sight=sight, track=track, compass=compass,
                                  habits=habits)
    init = LinearController.load(init_from).params if init_from else None
    if init is not None:
        log(f"Fortsetter fra {init_from}")
    if optimizer == "cma":
        # Vektene for z og h kan starte med mindre spredning enn posisjonstroen og bias (D38)
        std = np.ones(controller.num_params)
        std[: (controller.z_dim + controller.h_dim) * controller.num_actions] = zh_std
        es = SepCMAES(controller.num_params, sigma=sigma, population=population, seed=seed, init=init,
                      normalize=True, std=std)
    elif optimizer == "es":
        es = EvolutionStrategy(controller.num_params, population=population, sigma=sigma, lr=lr, seed=seed, init=init)
    else:
        raise ValueError(f"ukjent optimizer: {optimizer}")
    remaining = max(0, GridConfig.max_steps - max(contexts) - horizon) if charge_remaining else 0
    dream_cfg = DreamConfig(horizon=horizon, temperature=temperature, remaining_steps=remaining, shaping=shaping)
    if shaping and not (rnn.num_position_features or sight or compass):
        raise ValueError("shaping krever en MDN-RNN med posisjonshode")
    # Fast sett med starter for å måle fremgang i drømmen på samme måte hver gang
    eval_starts = starts.sample(512, np.random.default_rng(seed + 1))
    history = []
    for g in range(1, generations + 1):
        t0 = time.time()
        batch = starts.sample(starts_per_generation, rng)
        candidates = es.ask()
        fitness = dream_fitness(controller, candidates, rnn, batch, dream_cfg, gen)
        es.tell(fitness)
        entry = {"generation": g, "fitness_mean": float(fitness.mean()), "fitness_max": float(fitness.max())}
        if g % real_check_every == 0 or g == generations:
            details = dream_fitness(controller, es.theta[None], rnn, eval_starts, dream_cfg, gen, return_details=True)
            controller.params = es.theta.astype(np.float32)
            real = run_real_episodes(WorldModelAgent(vae, rnn, controller), real_check_episodes, config=real_config)
            entry.update({
                "dream_fitness": float(details["fitness"][0]),
                "dream_p_goal": float(details["p_goal"][0]),
                "dream_p_obstacle": float(details["p_obstacle"][0]),
                "action_share": details["action_share"].round(3).tolist(),
                "real": real,
            })
            log(
                f"gen {g:4d}  drøm {entry['dream_fitness']:+.3f} (mål {entry['dream_p_goal']:.2f}, "
                f"hindring {entry['dream_p_obstacle']:.2f})  ekte: mål {real['goal_rate']:.0%}, "
                f"hindring {real['obstacle_rate']:.0%}, avkortet {real['truncated_rate']:.0%}, "
                f"avkastning {real['mean_return']:+.3f}  [{time.time() - t0:.1f}s]"
            )
        history.append(entry)

    controller.params = es.theta.astype(np.float32)
    hparams = {
        "generations": generations, "population": population, "starts_per_generation": starts_per_generation,
        "sigma": sigma, "lr": lr, "context": contexts, "horizon": horizon, "temperature": temperature, "seed": seed,
        "vae": str(vae_path), "rnn": str(rnn_path), "init_from": str(init_from) if init_from else None,
        "use_positions": use_positions, "remaining_steps": remaining, "shaping": shaping, "optimizer": optimizer,
        "zh_std": zh_std, "lookahead": lookahead, "sight": sight, "track": track, "moving": moving,
        "compass": compass, "first_person": first_person, "no_back": no_back, "habits": habits,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    controller.save(out, hparams=json.dumps(hparams))
    log(f"Lagret {out}")
    return {"hparams": hparams, "history": history}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tren controlleren i drømmen")
    p.add_argument("--vae", default="checkpoints/vae_z32_w10.pt")
    p.add_argument("--rnn", default="checkpoints/mdnrnn_direct_k5.pt")
    p.add_argument("--data", nargs="+", default=["data/zseq_20k.npz"], help="z-sekvenser for oppvarmingsstarter")
    p.add_argument("--out", default="checkpoints/controller.npz")
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--starts", type=int, default=64, help="drømmer per kandidat per generasjon")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--context", type=int, nargs="+", default=[5],
                   help="ekte oppvarmingsskritt før drømmen; flere tall gir starter fra hvert av dem")
    p.add_argument("--horizon", type=int, default=10, help="drømte skritt")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--real-check-every", type=int, default=25)
    p.add_argument("--history", default=None, help="lagre treningshistorikk som JSON")
    p.add_argument("--init-from", default=None, help="start fra vektene i denne controlleren")
    p.add_argument("--use-positions", action="store_true",
                   help="gi controlleren M sin tro om hvor agent og mål er (krever posisjonshode)")
    p.add_argument("--lookahead", action="store_true",
                   help="gi controlleren M sitt fremsyn: sannsynlighet for hindring for hver handling")
    p.add_argument("--use-eye", action="store_true",
                   help="gi controlleren det øyet ser: agent og mål i bildet akkurat nå (krever M med øye)")
    p.add_argument("--track", action="store_true",
                   help="spor agenten med et filter og gi controlleren hvor den nylig har vært (krever --use-eye)")
    p.add_argument("--charge-remaining", action="store_true",
                   help="den som overlever drømmen betaler skrittkostnaden for resten av en ekte episode")
    p.add_argument("--shaping", type=float, default=0.0,
                   help="belønning per celle nærmere målet i drømmen, ut fra M sin tro (krever posisjonshode)")
    p.add_argument("--optimizer", choices=("es", "cma"), default="es")
    p.add_argument("--zh-std", type=float, default=1.0,
                   help="CMA-ES: startspredning for vektene på z og h, relativt til resten")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--moving", type=int, default=0, help="de ekte kontrollene kjøres med så mange bevegelige hindringer")
    p.add_argument("--compass", action="store_true", help="gi controlleren M sitt kompass mot målet (førsteperson)")
    p.add_argument("--first-person", action="store_true", help="de ekte kontrollene kjøres i førsteperson")
    p.add_argument("--no-back", action="store_true", help="controlleren kan ikke rygge (førsteperson)")
    p.add_argument("--habits", action="store_true",
                   help="gi controlleren snu på rad og om den gikk fast (førsteperson)")
    a = p.parse_args(argv)
    result = train(
        a.vae, a.rnn, a.data, a.out, generations=a.generations, population=a.population,
        starts_per_generation=a.starts, sigma=a.sigma, lr=a.lr, context=a.context, horizon=a.horizon,
        temperature=a.temperature, real_check_every=a.real_check_every, init_from=a.init_from,
        use_positions=a.use_positions, lookahead=a.lookahead, sight=a.use_eye, track=a.track, charge_remaining=a.charge_remaining,
        shaping=a.shaping, optimizer=a.optimizer, zh_std=a.zh_std, seed=a.seed, moving=a.moving,
        compass=a.compass, first_person=a.first_person, no_back=a.no_back, habits=a.habits,
    )
    if a.history:
        Path(a.history).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
