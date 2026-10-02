"""Steg 7: hvem spår hendelsene best for handlingen som faktisk ble tatt, M sitt hendelseshode
(fra minnet h) eller nærsynet (fra bildet og øyets målminne)? Valideringsepisodene. (D52)"""
import numpy as np, torch
from worldmodels.mdnrnn.encode import EVENT_GOAL, EVENT_OBSTACLE, ZSequences
from worldmodels.mdnrnn.model import MDNRNN
from worldmodels.mdnrnn.neighbours import eye_maps, event_probs_from_logits
from worldmodels.mdnrnn.train import make_batch, split_indices

seqs = ZSequences.load_many([f"data/v3/zseq_{n}.npz" for n in ("20k", "r1", "r2", "r3")])
_, val = split_indices(len(seqs))
val_seqs = seqs.subset(val)
model, _ = MDNRNN.load("checkpoints/mdnrnn_sense.pt")

with torch.no_grad():
    # M: lærertvunget gjennom hver episode, sannsynligheten for hendelsen etter hvert skritt
    head = []
    for s in range(0, len(val_seqs), 256):
        b = make_batch(val_seqs, np.arange(s, min(s + 256, len(val_seqs))))
        out = model(b.z_mu[:, :-1], b.actions)
        head.append(torch.softmax(out.event_logits, -1)[b.mask])
    head = torch.cat(head)
    # Nærsynet: fra bildet før hvert skritt og øyets minne så langt
    agent_maps, goal_maps = eye_maps(model, val_seqs)
    obs_before = np.concatenate([o + np.arange(n) for o, n in zip(val_seqs.obs_offsets, val_seqs.lengths)])
    z = torch.from_numpy(val_seqs.mu[obs_before]).float()
    a = torch.from_numpy(val_seqs.actions).long()
    logits = torch.cat([model.neighbours(z[i:i + 8192], goal_maps[obs_before][i:i + 8192], agent_maps[obs_before][i:i + 8192])
                        for i in range(0, len(z), 8192)])
    sense = event_probs_from_logits(logits, a)

events = torch.from_numpy(val_seqs.events)
print(f"{len(events)} skritt, {int((events == EVENT_GOAL).sum())} mål, {int((events == EVENT_OBSTACLE).sum())} krasj")
for name, p in [("hendelseshodet (h)", head), ("nærsynet", sense)]:
    for ev, label in [(EVENT_GOAL, "mål"), (EVENT_OBSTACLE, "krasj")]:
        true, pred = events == ev, p[:, ev]
        hit = pred > 0.5
        print(f"{name:20s} {label:6s} treff {float((hit & true).sum() / true.sum()):.2f}  presisjon "
              f"{float((hit & true).sum() / max(1, hit.sum())):.2f}  snitt spådd {float(pred.mean()):.4f} mot ekte {float(true.float().mean()):.4f}")
