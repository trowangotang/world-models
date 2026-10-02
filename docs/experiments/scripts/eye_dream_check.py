"""Steg 6: rangerer drømmen den håndlagde "gå mot målet" over det søket fant? (D50)"""
import sys
import numpy as np, torch
sys.path.insert(0, "docs/experiments/scripts")
from handcrafted_eye import handcrafted, rnn  # noqa: E402
from worldmodels.controller.dream import DreamConfig, dream_fitness, make_warm_starts  # noqa: E402
from worldmodels.controller.policy import LinearController  # noqa: E402
from worldmodels.mdnrnn.encode import ZSequences  # noqa: E402
from worldmodels.mdnrnn.train import split_indices  # noqa: E402

seqs = ZSequences.load_many([f"data/v3/zseq_{n}.npz" for n in ("20k", "r1", "r2", "r3")])
tr, _ = split_indices(len(seqs))
starts = make_warm_starts(rnn, seqs, tr, context=5).sample(1024, np.random.default_rng(1))
cfg = DreamConfig(horizon=10, temperature=0.0, remaining_steps=35, shaping=0.05)
for name, c in [("håndlaget: syn", handcrafted(0.0)), ("håndlaget: syn + fremsyn", handcrafted(5.0)),
                ("lært: syn", LinearController.load("checkpoints/controller_eye_plain.npz"))] + (
               [("lært: syn + fremsyn", LinearController.load(sys.argv[1]))] if len(sys.argv) > 1 else []):
    d = dream_fitness(c, c.params[None], rnn, starts, cfg, return_details=True)
    print(f"{name}: fitness {d['fitness'][0]:+.3f}, mål {d['p_goal'][0]:.2f}, hindring {d['p_obstacle'][0]:.2f}, "
          f"lever {d['p_alive_end'][0]:.2f}, forming {d['shaping'][0]:+.3f}, handlinger {d['action_share'].round(2)}", flush=True)
