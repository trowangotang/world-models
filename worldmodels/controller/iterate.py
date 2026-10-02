"""Iterativ trening: la controlleren samle nye ekte data, og tren verdensmodellen videre på dem.

    python -m worldmodels.controller.iterate --rounds 3 --out-dir checkpoints/iter

Én runde:
    1. Spill episoder i det ekte miljøet med nåværende controller (+ litt tilfeldig utforsking).
    2. Kod dem til z-sekvenser med den faste VAE-en.
    3. Tren MDN-RNN-en videre fra forrige runde, på alle data så langt.
    4. Tren controlleren videre i den nye drømmen, fra forrige rundes vekter.
    5. Mål controlleren i det ekte miljøet (de samme evalueringsseedene hver runde).

Tanken er at drømmen blir lurt der controlleren faktisk går. Når de tilstandene kommer med i
dataene, lærer MDN-RNN-en hva som egentlig skjer der, og utnyttelsen av drømmen slutter å lønne seg.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from worldmodels.controller import train as controller_train
from worldmodels.controller.agent import WorldModelAgent, collect_agent_rollouts, run_real_episodes
from worldmodels.controller.policy import LinearController
from worldmodels.data import episode_paths
from worldmodels.mdnrnn import train as mdnrnn_train
from worldmodels.mdnrnn.encode import encode_episodes
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.vae.model import ConvVAE

# Tilfeldige rollouts bruker seeds fra 0, evaluering fra 100 000. Innsamling i runde r bruker
# seeds fra COLLECT_SEED + r * 100 000, så ingen layout går igjen på tvers.
COLLECT_SEED = 300_000


def evaluate(vae, rnn_path, controller_path, episodes: int) -> dict:
    rnn, _ = MDNRNN.load(rnn_path)
    return run_real_episodes(WorldModelAgent(vae, rnn, LinearController.load(controller_path)), episodes)


def iterate(
    vae_path, rnn_path, controller_path, data: list, out_dir,
    rounds: int = 3,
    episodes_per_round: int = 5000,
    epsilon: float = 0.3,
    rnn_epochs: int = 8,
    rnn_lr: float = 5e-4,
    generations: int = 150,
    starts: int = 256,
    eval_episodes: int = 1000,
    log=lambda msg: print(msg, flush=True),
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vae, _ = ConvVAE.load(vae_path)
    data = [str(d) for d in data]
    results = [{"round": 0, "rnn": str(rnn_path), "controller": str(controller_path),
                "real": evaluate(vae, rnn_path, controller_path, eval_episodes)}]
    log(f"runde 0: {results[0]['real']}")

    for r in range(1, rounds + 1):
        t0 = time.time()
        rnn, _ = MDNRNN.load(rnn_path)
        agent = WorldModelAgent(vae, rnn, LinearController.load(controller_path))
        rollout_dir = out / f"rollouts_r{r}"
        collected = collect_agent_rollouts(agent, rollout_dir, episodes_per_round, seed=COLLECT_SEED + r * 100_000,
                                           epsilon=epsilon)
        log(f"runde {r}: samlet {episodes_per_round} episoder {collected}")

        zseq = out / f"zseq_r{r}.npz"
        encode_episodes(vae, episode_paths(rollout_dir)).save(zseq)
        data.append(str(zseq))

        new_rnn = out / f"mdnrnn_r{r}.pt"
        rnn_result = mdnrnn_train.train(data, new_rnn, epochs=rnn_epochs, lr=rnn_lr, init_from=rnn_path, log=log)

        new_controller = out / f"controller_r{r}.npz"
        c_result = controller_train.train(
            vae_path, new_rnn, data, new_controller, generations=generations, starts_per_generation=starts,
            init_from=controller_path, seed=r, log=log,
        )
        last = c_result["history"][-1]
        real = evaluate(vae, new_rnn, new_controller, eval_episodes)
        results.append({
            "round": r, "rnn": str(new_rnn), "controller": str(new_controller), "data": list(data),
            "collected": collected, "rnn_best_epoch": rnn_result["hparams"]["best_epoch"],
            "rnn_val_last": rnn_result["history"][-1]["val"],
            "dream_p_goal": last["dream_p_goal"], "dream_p_obstacle": last["dream_p_obstacle"],
            "action_share": last["action_share"], "real": real, "seconds": round(time.time() - t0),
        })
        log(f"runde {r}: drøm mål {last['dream_p_goal']:.2f}, ekte {real}  [{time.time() - t0:.0f}s]")
        (out / "iterate_results.json").write_text(json.dumps(results, indent=2))
        rnn_path, controller_path = new_rnn, new_controller
    return {"rounds": results}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Iterativ trening av MDN-RNN og controller")
    p.add_argument("--vae", default="checkpoints/vae_z32_w10.pt")
    p.add_argument("--rnn", default="checkpoints/mdnrnn_direct_k5.pt")
    p.add_argument("--controller", default="checkpoints/controller_s256.npz")
    p.add_argument("--data", nargs="+", default=["data/zseq_20k.npz"], help="z-sekvenser fra tilfeldige rollouts")
    p.add_argument("--out-dir", default="checkpoints/iter")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=5000, help="nye ekte episoder per runde")
    p.add_argument("--epsilon", type=float, default=0.3, help="sannsynlighet for tilfeldig handling ved innsamling")
    p.add_argument("--rnn-epochs", type=int, default=8)
    p.add_argument("--rnn-lr", type=float, default=5e-4)
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--starts", type=int, default=256)
    p.add_argument("--eval-episodes", type=int, default=1000)
    a = p.parse_args(argv)
    iterate(
        a.vae, a.rnn, a.controller, a.data, a.out_dir, rounds=a.rounds, episodes_per_round=a.episodes,
        epsilon=a.epsilon, rnn_epochs=a.rnn_epochs, rnn_lr=a.rnn_lr, generations=a.generations,
        starts=a.starts, eval_episodes=a.eval_episodes,
    )


if __name__ == "__main__":
    main()
