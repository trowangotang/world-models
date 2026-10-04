"""Tren MDN-RNN-en på z-sekvenser.

    python -m worldmodels.mdnrnn.train --data data/zseq_20k.npz --out checkpoints/mdnrnn.pt

Inndata i skritt t er en *trukket* z_t ~ q(z | x_t), ikke bare middelverdien. Da lærer
modellen å tåle litt støy i sin egen inndata, noe den møter når den senere drømmer og
mater sine egne prediksjoner tilbake. Målet er middelverdien mu_{t+1}.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from worldmodels.mdnrnn.encode import EVENT_GOAL, NUM_EVENTS, ZSequences
from worldmodels.mdnrnn.model import MDNRNN, MDNRNNConfig, mdn_nll


@dataclass
class Batch:
    z_mu: torch.Tensor       # (B, T+1, D) middelverdier for alle observasjoner
    z_logvar: torch.Tensor   # (B, T+1, D)
    actions: torch.Tensor    # (B, T)
    events: torch.Tensor     # (B, T)
    agent_cell: torch.Tensor # (B, T+1)
    goal_cell: torch.Tensor  # (B, T+1), -1 der den er ukjent
    near_obstacle: torch.Tensor  # (B, T+1, 4), -1 der det er ukjent
    goal_ego: torch.Tensor   # (B, T+1, 2) målet sett fra agenten (førsteperson)
    goal_known: torch.Tensor # (B, T+1) 1 når målet har vært synlig i dette eller et tidligere bilde, -1 = ukjent
    mask: torch.Tensor       # (B, T) True for ekte skritt, False for utfylling

    def inputs(self, sample: bool, generator: torch.Generator | None = None) -> torch.Tensor:
        mu = self.z_mu[:, :-1]
        if not sample:
            return mu
        std = (0.5 * self.z_logvar[:, :-1]).exp()
        return mu + std * torch.randn(mu.shape, generator=generator)

    @property
    def targets(self) -> torch.Tensor:
        return self.z_mu[:, 1:]


def make_batch(seqs: ZSequences, indices) -> Batch:
    """Fyll opp episodene til samme lengde og lag en maske for de ekte skrittene."""
    eps = [seqs.episode(i) for i in indices]
    T = max(len(e["actions"]) for e in eps)
    B, D = len(eps), seqs.mu.shape[1]
    z_mu = np.zeros((B, T + 1, D), np.float32)
    z_logvar = np.zeros((B, T + 1, D), np.float32)
    actions = np.zeros((B, T), np.int64)
    events = np.zeros((B, T), np.int64)
    agent = np.full((B, T + 1), -1, np.int64)
    goal = np.full((B, T + 1), -1, np.int64)
    near = np.full((B, T + 1, eps[0]["near_obstacle"].shape[-1]), -1, np.int64)
    ego = np.zeros((B, T + 1, 2), np.float32)
    known = np.full((B, T + 1), -1, np.int64)
    mask = np.zeros((B, T), bool)
    for b, e in enumerate(eps):
        n = len(e["actions"])
        z_mu[b, :n + 1] = e["mu"]
        z_logvar[b, :n + 1] = e["logvar"]
        actions[b, :n] = e["actions"]
        events[b, :n] = e["events"]
        agent[b, :n + 1] = e["agent_cell"]
        goal[b, :n + 1] = e["goal_cell"]
        near[b, :n + 1] = e["near_obstacle"]
        ego[b, :n + 1] = e["goal_ego"]
        visible = e["goal_visible"]
        known[b, :n + 1] = np.maximum.accumulate(visible) if (visible >= 0).all() else -1
        mask[b, :n] = True
    t = torch.from_numpy
    return Batch(t(z_mu), t(z_logvar), t(actions), t(events), t(agent), t(goal), t(near), t(ego), t(known), t(mask))


def split_indices(n: int, val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def oversample_goal_episodes(seqs: ZSequences, indices, factor: int) -> np.ndarray:
    """Gjenta episodene som ender i mål factor ganger. Bare ~10 % av episodene når målet,
    så modellen ser ellers svært få eksempler på hvordan det ser ut."""
    indices = np.asarray(indices)
    if factor <= 1:
        return indices
    goal_eps = np.array([i for i in indices if (seqs.episode(i)["events"] == EVENT_GOAL).any()], dtype=np.int64)
    return np.concatenate([indices] + [goal_eps] * (factor - 1))


def event_class_weights(events: np.ndarray, power: float = 0.5) -> torch.Tensor:
    """Vekt per hendelsestype, proporsjonal med frekvens^-power og normert slik at "flytt" = 1.

    Mål skjer i under 1 % av skrittene og hindringer i ~5 %. Uten vekting lærer modellen
    å alltid si "flytt". Kvadratroten demper vektene så de sjeldne klassene ikke tar over.
    """
    counts = np.bincount(events, minlength=NUM_EVENTS).astype(np.float64)
    w = np.where(counts > 0, np.maximum(counts, 1) ** -power, 0.0)
    return torch.tensor(w / w[0], dtype=torch.float32)


def batch_loss(
    model: MDNRNN,
    batch: Batch,
    sample_inputs: bool,
    event_weight: float,
    class_weights: torch.Tensor | None = None,
    mse_weight: float = 0.0,
    position_weight: float = 0.0,
    obstacle_weight: float = 0.0,
    compass_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """nll + event_weight * hendelsestap + mse_weight * kvadratfeil for forventet z_{t+1}.

    MSE-leddet tvinger blandingens forventning til å treffe neste z. Uten det kan modellen
    få lav nll ved å treffe de mange dimensjonene som ikke endrer seg, og være slapp på
    de få som koder hvor agenten flyttet seg (se decisions.md).
    """
    inputs = batch.inputs(sample_inputs)
    out = model(inputs, batch.actions)
    m = batch.mask
    nll = mdn_nll(out, batch.targets)[m].mean()
    ce = F.cross_entropy(out.event_logits[m], batch.events[m], weight=class_weights)
    expected = (F.softmax(out.logit_pi, dim=-1).unsqueeze(-1) * out.mu).sum(2)
    mse = ((expected - batch.targets) ** 2)[m].mean()
    position = position_loss(out, batch) if out.position_logits is not None else torch.zeros(())
    obstacle = obstacle_loss(out, batch) if out.obstacle_logits is not None else torch.zeros(())
    compass = compass_loss(model, out, inputs, batch) if model.compass_net is not None else torch.zeros(())
    loss = (nll + event_weight * ce + mse_weight * mse + position_weight * position + obstacle_weight * obstacle
            + compass_weight * compass)
    return {"loss": loss, "nll": nll, "event_ce": ce, "mse": mse, "position_ce": position, "obstacle_bce": obstacle,
            "compass": compass}


def compass_loss(model: MDNRNN, out, inputs: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Kompasset for bilde t leses fra minnet før skrittet (h_{t-1}, nuller i første skritt) og z_t.

    Kvadratfeil for målets plass sett fra agenten, pluss kryssentropi for om målet er sett. Plassen trenes
    også før målet er sett: da lærer kompasset det beste gjettet ut fra hva som ikke synes (D77)."""
    from worldmodels.mdnrnn.model import COMPASS_SCALE

    ok = batch.mask & (batch.goal_known[:, :-1] >= 0)
    if not ok.any():
        return torch.zeros(())
    h_before = torch.cat([torch.zeros_like(out.h[:, :1]), out.h[:, :-1]], dim=1)
    pred = model.compass_logits(inputs, h_before)
    target = batch.goal_ego[:, :-1] / COMPASS_SCALE
    mse = ((pred[..., :2] - target) ** 2).sum(-1)[ok].mean()
    bce = F.binary_cross_entropy_with_logits(pred[..., 2][ok], batch.goal_known[:, :-1][ok].float())
    return mse + bce


def obstacle_loss(out, batch: Batch) -> torch.Tensor:
    """Binær kryssentropi for hindring i hver nabocelle etter skrittet. Ukjente (-1) hoppes over."""
    target = batch.near_obstacle[:, 1:]
    ok = batch.mask.unsqueeze(-1) & (target >= 0)
    if not ok.any():
        return torch.zeros(())
    return F.binary_cross_entropy_with_logits(out.obstacle_logits[ok], target[ok].float())


def position_loss(out, batch: Batch) -> torch.Tensor:
    """Kryssentropi for agentens og målets celle etter hvert skritt. Ukjente celler (-1) hoppes over."""
    losses = []
    for i, cells in enumerate((batch.agent_cell[:, 1:], batch.goal_cell[:, 1:])):
        ok = batch.mask & (cells >= 0)
        if ok.any():
            losses.append(F.cross_entropy(out.position_logits[:, :, i][ok], cells[ok]))
    return torch.stack(losses).mean() if losses else torch.zeros(())


def run_epoch(
    model, seqs, indices, opt, batch_size, event_weight, rng=None, class_weights=None, mse_weight=0.0,
    position_weight=0.0, obstacle_weight=0.0, compass_weight=0.0,
) -> dict[str, float]:
    training = opt is not None
    model.train(training)
    order = rng.permutation(indices) if training else indices
    totals, n = {"loss": 0.0, "nll": 0.0, "event_ce": 0.0, "mse": 0.0, "position_ce": 0.0, "obstacle_bce": 0.0,
                 "compass": 0.0}, 0
    with torch.set_grad_enabled(training):
        for start in range(0, len(order), batch_size):
            batch = make_batch(seqs, order[start:start + batch_size])
            parts = batch_loss(model, batch, training, event_weight, class_weights, mse_weight, position_weight,
                               obstacle_weight, compass_weight)
            if training:
                opt.zero_grad()
                parts["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            k = int(batch.mask.sum())
            for key in totals:
                totals[key] += parts[key].item() * k
            n += k
    return {k: v / n for k, v in totals.items()}


def train(
    data: str | Path | list,
    out: str | Path,
    hidden_dim: int = 256,
    num_mixtures: int = 5,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    event_weight: float = 1.0,
    balance_events: bool = True,
    input_mlp: bool = True,
    direct_path: bool = True,
    mse_weight: float = 10.0,
    position_weight: float = 0.0,
    obstacle_weight: float = 0.0,
    goal_oversample: int = 1,
    compass_weight: float = 0.0,
    init_from: str | Path | None = None,
    seed: int = 0,
    log=lambda msg: print(msg, flush=True),
) -> dict:
    """Tren MDN-RNN-en. Med init_from fortsetter treningen fra en tidligere checkpoint
    (arkitekturen hentes derfra), slik som i iterativ trening."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seqs = ZSequences.load_many(data)
    train_idx, val_idx = split_indices(len(seqs), seed=seed)
    log(f"Trening: {len(train_idx)} episoder, validering: {len(val_idx)} episoder")

    if init_from is not None:
        model, _ = MDNRNN.load(init_from)
        log(f"Fortsetter fra {init_from}")
        # Nye hjelpehoder kan legges til en eksisterende modell; resten av vektene beholdes
        wanted = replace(model.config, position_head=model.config.position_head or position_weight > 0,
                         obstacle_head=model.config.obstacle_head or obstacle_weight > 0,
                         compass=model.config.compass or compass_weight > 0)
        if wanted != model.config:
            old = model.state_dict()
            model = MDNRNN(wanted)
            missing, _ = model.load_state_dict(old, strict=False)
            log(f"La til nye hoder: {sorted({k.split('.')[0] for k in missing})}")
    else:
        model = MDNRNN(MDNRNNConfig(
            latent_dim=seqs.mu.shape[1], hidden_dim=hidden_dim, num_mixtures=num_mixtures, input_mlp=input_mlp,
            direct_path=direct_path, position_head=position_weight > 0, obstacle_head=obstacle_weight > 0,
            compass=compass_weight > 0,
        ))
    class_weights = event_class_weights(seqs.subset(train_idx).events) if balance_events else None
    if class_weights is not None:
        log(f"Hendelsesvekter (flytt, mål, hindring): {[round(w, 2) for w in class_weights.tolist()]}")
    epoch_idx = oversample_goal_episodes(seqs, train_idx, goal_oversample)
    if goal_oversample > 1:
        log(f"Mål-episoder vises {goal_oversample} ganger per epoke: {len(epoch_idx)} episoder per epoke")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_val, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, seqs, epoch_idx, opt, batch_size, event_weight, rng, class_weights, mse_weight,
                       position_weight, obstacle_weight, compass_weight)
        va = run_epoch(model, seqs, val_idx, None, batch_size, event_weight, None, class_weights, mse_weight,
                       position_weight, obstacle_weight, compass_weight)
        history.append({"epoch": epoch, "train": tr, "val": va})
        # Tidlig stopp på dynamikken alene. Hendelseshodet overtilpasser seg tidligere enn resten,
        # og ville ellers stoppet treningen før z-prediksjonen er ferdig lært.
        val_dynamics = va["nll"] + mse_weight * va["mse"]
        if val_dynamics < best_val:
            best_val, best_epoch = val_dynamics, epoch
            best_state = copy.deepcopy(model.state_dict())
        log(
            f"epoke {epoch:2d}  tren nll {tr['nll']:7.3f} hendelse {tr['event_ce']:.3f}  "
            f"mse {tr['mse']:.4f}  val nll {va['nll']:7.3f} hendelse {va['event_ce']:.3f} mse {va['mse']:.4f} "
            f"posisjon {va['position_ce']:.3f} hindring {va['obstacle_bce']:.3f} kompass {va['compass']:.3f}  "
            f"[{time.time() - t0:.0f}s]"
        )

    hparams = {
        "hidden_dim": hidden_dim, "num_mixtures": num_mixtures, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "event_weight": event_weight, "balance_events": balance_events, "input_mlp": input_mlp,
        "direct_path": direct_path, "mse_weight": mse_weight, "position_weight": position_weight,
        "obstacle_weight": obstacle_weight, "compass_weight": compass_weight,
        "goal_oversample": goal_oversample, "best_epoch": best_epoch,
        "seed": seed, "data": [str(d) for d in data] if isinstance(data, (list, tuple)) else str(data),
        "init_from": str(init_from) if init_from else None,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    model.save(out, hparams=hparams, history=history)
    log(f"Lagret {out} (beste epoke: {best_epoch})")
    return {"hparams": hparams, "history": history}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Tren MDN-RNN på z-sekvenser")
    p.add_argument("--data", nargs="+", default=["data/zseq_20k.npz"], help="én eller flere z-sekvensfiler")
    p.add_argument("--out", default="checkpoints/mdnrnn.pt")
    p.add_argument("--init-from", default=None, help="fortsett fra denne checkpointen")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--mixtures", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--event-weight", type=float, default=1.0)
    p.add_argument("--no-balance-events", action="store_true", help="ikke vekt sjeldne hendelser opp")
    p.add_argument("--linear-input", action="store_true", help="z og handling rett inn i LSTM-en, som i artikkelen")
    p.add_argument("--mse-weight", type=float, default=10.0, help="vekt på kvadratfeil for forventet z")
    p.add_argument("--no-direct-path", action="store_true", help="hodene ser bare h, som i artikkelen")
    p.add_argument("--position-weight", type=float, default=0.0,
                   help="vekt på hjelpehodet som lærer minnet hvor agent og mål er (0 = av)")
    p.add_argument("--obstacle-weight", type=float, default=0.0,
                   help="vekt på hjelpehodet for hindringer i nabocellene (0 = av)")
    p.add_argument("--compass-weight", type=float, default=0.0,
                   help="vekt på kompasset som lærer hvor målet er sett fra agenten (førsteperson, 0 = av)")
    p.add_argument("--goal-oversample", type=int, default=1, help="vis mål-episoder så mange ganger per epoke")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    result = train(
        args.data, args.out, hidden_dim=args.hidden_dim, num_mixtures=args.mixtures, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, event_weight=args.event_weight,
        balance_events=not args.no_balance_events, input_mlp=not args.linear_input,
        direct_path=not args.no_direct_path, mse_weight=args.mse_weight, position_weight=args.position_weight,
        obstacle_weight=args.obstacle_weight,
        goal_oversample=args.goal_oversample, compass_weight=args.compass_weight, init_from=args.init_from, seed=args.seed,
    )
    print(json.dumps(result["history"][-1]))


if __name__ == "__main__":
    main()
