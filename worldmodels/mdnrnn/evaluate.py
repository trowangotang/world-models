"""Vurder MDN-RNN-en på valideringsepisodene.

Alle posisjonsmål dekoder den predikerte z-en med VAE-en og leser av agentens celle,
så de måler det som betyr noe: havner agenten der den faktisk havnet?

  * Ett skritt fram: predikert z_{t+1} gitt ekte z_t og a_t. Sammenlignes med en
    grunnlinje som sier "ingenting endrer seg" (z_{t+1} = z_t).
  * Drøm over k skritt: gi modellen de ekte handlingene, men mat den med sine *egne*
    prediksjoner. Enten fra første bilde (ctx0), eller etter 5 ekte skritt som varmer opp
    minnet (ctx5). Viser hvor fort drømmen sporer av.
  * Hendelser: hvor godt treffer modellen mål- og hinder-skrittene.
  * Prober på h: kan en lineær modell lese agentens posisjon ut av RNN-ens skjulte
    tilstand, og er det lettere enn fra z alene? (Spørsmålet fra D14.)

    python -m worldmodels.mdnrnn.evaluate checkpoints/mdnrnn.pt --vae checkpoints/vae_z32_w10.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from worldmodels.env.parse import AGENT, find_cell, parse_cells
from worldmodels.mdnrnn.encode import EVENT_GOAL, EVENT_MOVE, EVENT_OBSTACLE, ZSequences
from worldmodels.mdnrnn.model import MDNRNN, most_likely_mean
from worldmodels.mdnrnn.train import make_batch, split_indices
from worldmodels.vae.evaluate import train_probe
from worldmodels.vae.model import ConvVAE


@torch.no_grad()
def decode_agent_cells(vae: ConvVAE, z: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """Dekod z (N, D) og returner agentens celle i hvert bilde (-1 hvis den mangler)."""
    cells = []
    for start in range(0, len(z), batch_size):
        img = vae.decode(z[start:start + batch_size]).permute(0, 2, 3, 1).numpy()
        cells.append(find_cell(parse_cells(img), AGENT))
    return np.concatenate(cells)


@torch.no_grad()
def one_step_metrics(model: MDNRNN, vae: ConvVAE, seqs: ZSequences, indices, batch_size: int = 256) -> dict:
    pred_z, copy_z, true_cells, events, pred_events, moved = [], [], [], [], [], []
    for start in range(0, len(indices), batch_size):
        b = make_batch(seqs, indices[start:start + batch_size])
        out = model(b.inputs(sample=False), b.actions)
        m = b.mask
        pred_z.append(most_likely_mean(out)[m])
        copy_z.append(b.z_mu[:, :-1][m])
        true_cells.append(b.agent_cell[:, 1:][m].numpy())
        moved.append((b.agent_cell[:, 1:] != b.agent_cell[:, :-1])[m].numpy())
        events.append(b.events[m].numpy())
        pred_events.append(out.event_logits.argmax(-1)[m].numpy())
    true_cells = np.concatenate(true_cells)
    moved = np.concatenate(moved)
    pred_cells = decode_agent_cells(vae, torch.cat(pred_z))
    copy_cells = decode_agent_cells(vae, torch.cat(copy_z))
    events, pred_events = np.concatenate(events), np.concatenate(pred_events)

    def recall(kind):
        sel = events == kind
        return float((pred_events[sel] == kind).mean()) if sel.any() else float("nan")

    def precision(kind):
        sel = pred_events == kind
        return float((events[sel] == kind).mean()) if sel.any() else float("nan")

    return {
        "steps": int(len(true_cells)),
        "agent_cell_acc": float((pred_cells == true_cells).mean()),
        "agent_cell_acc_when_moved": float((pred_cells == true_cells)[moved].mean()),
        "baseline_copy_acc": float((copy_cells == true_cells).mean()),
        "baseline_copy_acc_when_moved": float((copy_cells == true_cells)[moved].mean()),
        "event_accuracy": float((pred_events == events).mean()),
        "goal_recall": recall(EVENT_GOAL),
        "goal_precision": precision(EVENT_GOAL),
        "obstacle_recall": recall(EVENT_OBSTACLE),
        "obstacle_precision": precision(EVENT_OBSTACLE),
        "move_recall": recall(EVENT_MOVE),
    }


@torch.no_grad()
def dream_metrics(
    model: MDNRNN, vae: ConvVAE, seqs: ZSequences, indices, horizon: int = 10, context: int = 0
) -> dict:
    """Agent-celle-nøyaktighet etter k drømte skritt, for k = 1..horizon.

    Modellen får først se `context` ekte skritt (oppvarming av minnet), og drømmer deretter
    videre med de ekte handlingene og sine egne prediksjoner. Bare episoder som varer lenge
    nok teller for hvert k. Grunnlinjen sier at agenten blir stående der den var da drømmen
    startet.
    """
    keep = [i for i in indices if seqs.lengths[i] > context]
    b = make_batch(seqs, keep)
    hidden = None
    if context > 0:
        warm = model(b.z_mu[:, :context], b.actions[:, :context])
        hidden = warm.hidden
        z = most_likely_mean(warm)[:, -1:]                            # prediksjon av z_context
        # Vi mater modellens egen prediksjon av z_context videre, ikke den ekte. Da er alt etter
        # oppvarmingen ren drøm, akkurat som når controlleren skal trenes.
        first_action = context
    else:
        z = b.z_mu[:, 0:1]
        first_action = 0
    start_cells = decode_agent_cells(vae, b.z_mu[:, context])
    steps = min(horizon, b.actions.shape[1] - first_action)
    correct, copy_correct, counts = np.zeros(horizon), np.zeros(horizon), np.zeros(horizon)
    for k in range(steps):
        t = first_action + k
        out = model(z, b.actions[:, t:t + 1], hidden)
        hidden = out.hidden
        z = most_likely_mean(out)
        alive = b.mask[:, t].numpy()
        cells = decode_agent_cells(vae, z[:, 0])
        truth = b.agent_cell[:, t + 1].numpy()
        correct[k] = ((cells == truth) & alive).sum()
        copy_correct[k] = ((start_cells == truth) & alive).sum()
        counts[k] = alive.sum()
    n = np.maximum(counts, 1)
    return {
        "dream_agent_cell_acc": [float(a) for a in correct / n],
        "dream_baseline_copy_acc": [float(a) for a in copy_correct / n],
        "dream_episodes_alive": [int(c) for c in counts],
    }


@torch.no_grad()
def hidden_states(model: MDNRNN, seqs: ZSequences, indices, batch_size: int = 256):
    """Returnerer (h_t, z_{t+1}, agentens celle i t+1) for alle ekte skritt.

    h_t er tilstanden etter å ha sett z_t og a_t, altså modellens bilde av verden rett
    før neste observasjon. Det er dette en controller ville fått sammen med z.
    """
    hs, zs, cells = [], [], []
    for start in range(0, len(indices), batch_size):
        b = make_batch(seqs, indices[start:start + batch_size])
        out = model(b.inputs(sample=False), b.actions)
        m = b.mask
        hs.append(out.h[m].numpy())
        zs.append(b.z_mu[:, 1:][m].numpy())
        cells.append(b.agent_cell[:, 1:][m].numpy())
    return np.concatenate(hs), np.concatenate(zs), np.concatenate(cells)


def probe_metrics(model: MDNRNN, seqs: ZSequences, train_idx, val_idx, max_train: int = 60000) -> dict:
    h_tr, z_tr, y_tr = hidden_states(model, seqs, train_idx)
    h_va, z_va, y_va = hidden_states(model, seqs, val_idx)
    if len(y_tr) > max_train:  # hold probetreningen rask
        sel = np.random.default_rng(0).choice(len(y_tr), max_train, replace=False)
        h_tr, z_tr, y_tr = h_tr[sel], z_tr[sel], y_tr[sel]
    ok_tr, ok_va = y_tr >= 0, y_va >= 0
    results = {}
    for name, (a, b) in {
        "z": (z_tr, z_va),
        "h": (h_tr, h_va),
        "z_and_h": (np.hstack([z_tr, h_tr]), np.hstack([z_va, h_va])),
    }.items():
        results[f"probe_agent_linear_{name}"] = train_probe(a[ok_tr], y_tr[ok_tr], b[ok_va], y_va[ok_va])
    return results


def evaluate(checkpoint, vae_path, data, dream_horizon: int = 10, contexts=(0, 5)) -> dict:
    model, ckpt = MDNRNN.load(checkpoint)
    vae, _ = ConvVAE.load(vae_path)
    seqs = ZSequences.load(data)
    train_idx, val_idx = split_indices(len(seqs), seed=ckpt.get("hparams", {}).get("seed", 0))
    result = {"checkpoint": str(checkpoint), "hparams": ckpt.get("hparams", {})}
    result.update(one_step_metrics(model, vae, seqs, val_idx))
    for c in contexts:
        d = dream_metrics(model, vae, seqs, val_idx, dream_horizon, context=c)
        result.update({f"{k}_ctx{c}": v for k, v in d.items()})
    result.update(probe_metrics(model, seqs, train_idx, val_idx))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluer en MDN-RNN")
    p.add_argument("checkpoint")
    p.add_argument("--vae", default="checkpoints/vae_z32_w10.pt")
    p.add_argument("--data", default="data/zseq_20k.npz")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    r = evaluate(args.checkpoint, args.vae, args.data, args.horizon)
    for k, v in r.items():
        if k not in ("hparams", "checkpoint"):
            print(f"{k:32s} {v if isinstance(v, list) else round(v, 4)}")
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
