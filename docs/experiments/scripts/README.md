# Diagnoseskript fra steg 4c

Kjøres fra rotmappen med `PYTHONPATH=. python docs/experiments/scripts/<navn>.py`. De forutsetter
checkpointene og dataene fra steg 4b og 4c (se README og decisions.md D31–D35), og mappene
`logs_tmp/heldout_rand` og `logs_tmp/heldout_ctrl` med 1000 nye episoder hver.

| Skript | Hva det viser |
|---|---|
| `goal_probe.py` | Om målets plass kan leses fra z og fra RNN-minnet (D31) |
| `mdnrnn_goal_metrics.py` | Hvor mange mål- og hindringshendelser en MDN-RNN forutser |
| `handcrafted_goal_seeker.py` | En håndlaget "gå mot målet"-controller på M sin tro, i drøm og virkelighet (D33) |
| `dream_rewards_hiding.py` | Hvordan drømmen rangerer "gå mot målet" mot "vent" for ulike oppsett (D34) |
| `curriculum.py` | Controller med læreplan: halvparten av drømmene starter nær målet (D35) |
| `final_eval_4c.py` | Sluttabellen i README |

## Steg 4d

| Skript | Hva det viser |
|---|---|
| `search_landscape.py` | Fitness for 2000 tilfeldige controllere på de 20 posisjonsparametrene (D36) |
| `search_basin.py` | Hvor bred toppen rundt "gå mot målet" er (D36) |
| `final_eval_4d.py` | Sluttabellen for steg 4d |

## Steg 6

| Skript | Hva det viser |
|---|---|
| `perception_probe.py` | Hvor godt agent og mål kan leses fra z med lineær, MLP og romlig leser (D48) |
| `handcrafted_eye.py` | Håndlaget "gå mot målet" på øyet, med og uten fremsyn; lagrer diagnostikk-controlleren (D49, D50) |
| `eye_dream_check.py` | Drømmens fitness for håndlaget og lærte controllere med syn (D51) |
