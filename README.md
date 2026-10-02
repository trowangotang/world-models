# World Models i en liten 2D-verden

Et demoprosjekt inspirert av [Ha & Schmidhuber (2018), *World Models*](https://worldmodels.github.io/).
Målet er å trene en agent nesten helt inne i sin egen "drøm": en lært modell av verden,
i stedet for i det ekte miljøet. Prosjektet er også et utstillingsvindu for vibekoding:
hvert steg bygges, testes og godkjennes før neste, og viktige valg logges i
[`decisions.md`](decisions.md).

![Fem episoder med tilfeldig policy](docs/preview.png)

*Fem episoder med tilfeldig policy, ett tidssteg per kolonne. Blå = agent, grønn = mål, rød = hindring.*

## Arkitektur

```
 ekte miljø ──► (1) rollouts ──► (2) VAE  ──► z_t        (V: komprimerer bilde til latent vektor)
  GridDodge     tilfeldig         64x64x3      │
                policy            → z          ▼
                                   (3) MDN-RNN: p(z_{t+1} | z_t, a_t, h_t)   (M: lærer dynamikken)
                                               │
                                   (4) controller a_t = f(z_t, h_t)          (C: trenes i drømmen)
                                               │
                                   (5) evaluering i ekte miljø
```

| Steg | Modul | Status |
|------|-------|--------|
| 1. Miljø + tilfeldige rollouts | `worldmodels/env`, `worldmodels/data` | ✅ ferdig og godkjent |
| 2. VAE | `worldmodels/vae` | ✅ ferdig |
| 3. MDN-RNN | `worldmodels/mdnrnn` | ✅ ferdig |
| 4. Controller trent i drømmen | `worldmodels/controller` | ⚠️ virker, men når sjelden målet |
| 4b. Iterativ trening | `worldmodels/controller/iterate.py` | ⚠️ fjerner utnyttelsen av drømmen, ikke flere mål |
| 4c. Mål-bevisst drøm | `worldmodels/mdnrnn`, `worldmodels/controller` | ⚠️ drømmen kjenner igjen målet, men søket finner det ikke |
| 4d. Bedre søk | `worldmodels/controller/cma.py` | ✅ når målet i 41 %, men krasjer i 55 % |
| 4e. Unngå hindringer | `worldmodels/controller/features.py` | ✅ krasj ned til 34 %, men pendler i 29 % |
| 5. Evaluering i ekte miljø | `worldmodels/evaluation` | ✅ ferdig: [rapport](docs/evaluation/report.md) |
| 6. Bedre syn | `worldmodels/mdnrnn/eye.py` | ⚠️ øyet ser riktig, men drømmen tror ikke på det |
| 7. Drømmen ser | `worldmodels/mdnrnn/neighbours.py` | ✅ når målet i 75 %, krasjer i 11 % |

## Miljøet: GridDodge

Et 8x8-rutenett rendret som et 64x64 RGB-bilde (8 piksler per celle, samme oppløsning som i artikkelen).

- **Hver episode**: agent, mål og 6 hindringer plasseres tilfeldig. Et bredde-først-søk garanterer at målet kan nås.
- **Handlinger**: 4 diskrete (opp, ned, venstre, høyre). Å gå inn i kanten gjør at agenten blir stående.
- **Belønning**: +1 for å nå målet, −1 for å treffe en hindring (begge avslutter episoden), −0.01 per skritt.
- **Maks lengde**: 50 skritt, deretter avkortes episoden.
- **Avhengigheter**: bare NumPy.

Hvorfor akkurat dette miljøet er forklart i [`decisions.md`](decisions.md).

## Rollout-data

Hver episode lagres som `data/rollouts/episode_XXXXX.npz`:

| Felt | Form | Type | Betydning |
|------|------|------|-----------|
| `obs` | (T+1, 64, 64, 3) | uint8 | observasjon før hvert skritt, pluss siste observasjon |
| `actions` | (T,) | int64 | handling tatt i skritt t |
| `rewards` | (T,) | float32 | belønning for skritt t |
| `terminated` | (T,) | bool | traff mål eller hindring |
| `truncated` | (T,) | bool | avkortet etter 50 skritt |

Overgang *t* er `(obs[t], actions[t], obs[t+1])`, og `Episode.transitions()` gir disse direkte.

## Kom i gang

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Kjør testene
pytest

# Steg 1: samle 1000 episoder med tilfeldig policy (~2 sekunder, ~4 MB)
python -m worldmodels.data --episodes 1000 --out data/rollouts

# Lag et forhåndsvisningsbilde
python -m worldmodels.env.preview --out preview.png --episodes 5 --frames 12
```

Med 1000 episoder og seed 0 gir den tilfeldige policyen 15 532 overganger
(snitt 15,5 skritt): 11 % når målet, 82 % treffer en hindring, 7 % avkortes.

### Steg 2: tren og evaluer VAE-en

VAE-en trenger mange *layouter*, ikke bare mange bilder (se D12 i `decisions.md`).
Derfor samler vi 20 000 episoder og bruker høyst 4 bilder fra hver.

```bash
python -m worldmodels.data --episodes 20000 --out data/rollouts_20k                # ~35 s, 84 MB

python -m worldmodels.vae.train --data data/rollouts_20k --frames-per-episode 4 \
    --latent-dim 32 --object-weight 10 --epochs 12 --out checkpoints/vae_z32_w10.pt  # ~45 min på CPU

python -m worldmodels.vae.evaluate checkpoints/vae_z32_w10.pt \
    --data data/rollouts_20k --frames-per-episode 4 --image recon.png
```

### Steg 3: kod z-sekvenser, tren og evaluer MDN-RNN-en

```bash
python -m worldmodels.mdnrnn.encode --vae checkpoints/vae_z32_w10.pt \
    --data data/rollouts_20k --out data/zseq_20k.npz                           # ~3 min

python -m worldmodels.mdnrnn.train --data data/zseq_20k.npz \
    --out checkpoints/mdnrnn.pt                                                # 30 epoker, ~15 min

python -m worldmodels.mdnrnn.evaluate checkpoints/mdnrnn.pt \
    --vae checkpoints/vae_z32_w10.pt --image dream.png
```

### Steg 4: tren controlleren i drømmen

```bash
python -m worldmodels.controller.train --vae checkpoints/vae_z32_w10.pt \
    --rnn checkpoints/mdnrnn_direct_k5.pt --data data/zseq_20k.npz \
    --generations 300 --starts 256 --out checkpoints/controller.npz            # ~20 min
```

Loggen viser både hvor godt controlleren gjør det i drømmen og i det ekte miljøet.

### Steg 4b: iterativ trening

```bash
python -m worldmodels.controller.iterate --vae checkpoints/vae_z32_w10.pt \
    --rnn checkpoints/mdnrnn_direct_k5.pt --controller checkpoints/controller_s256.npz \
    --data data/zseq_20k.npz --rounds 3 --out-dir checkpoints/iter          # ~8 min per runde
```

Hver runde samler 5000 ekte episoder med controlleren (30 % tilfeldige handlinger), trener
MDN-RNN-en og controlleren videre, og måler resultatet. Alt lagres i `checkpoints/iter/`.

### Steg 4c: mål-bevisst drøm

```bash
# Kod dataene på nytt, så målets celle kommer med (gjør det samme for rollouts fra 4b)
python -m worldmodels.mdnrnn.encode --data data/rollouts_20k --out data/v2/zseq_20k.npz

# MDN-RNN med hjelpehode for agentens og målets posisjon                    # ~55 min på CPU
python -m worldmodels.mdnrnn.train --data data/v2/zseq_20k.npz data/v2/zseq_r1.npz \
    data/v2/zseq_r2.npz data/v2/zseq_r3.npz --epochs 30 --position-weight 1 --out checkpoints/mdnrnn_pos.pt

# Controller med M sin tro om posisjoner, i en drøm som ikke belønner å gjemme seg
python -m worldmodels.controller.train --rnn checkpoints/mdnrnn_pos.pt --data data/v2/zseq_*.npz \
    --use-positions --charge-remaining --temperature 0 --generations 300 --starts 256
```

### Steg 4d: bedre søk

```bash
# CMA-ES med formet belønning. --zh-std 0 gir controlleren med bare posisjonstroen (20 parametre).
python -m worldmodels.controller.train --rnn checkpoints/mdnrnn_pos.pt --data data/v2/zseq_*.npz \
    --use-positions --charge-remaining --temperature 0 --shaping 0.05 \
    --optimizer cma --sigma 0.5 --zh-std 0.03 --generations 300 --starts 256   # ~20 min
```

### Steg 4e: unngå hindringer

```bash
# Posisjonstro + fremsyn (M spør "hva skjer om jeg går hit?" for hver handling)
python -m worldmodels.controller.train --rnn checkpoints/mdnrnn_pos.pt --data data/v2/zseq_*.npz \
    --use-positions --lookahead --charge-remaining --temperature 0 --shaping 0.05 \
    --optimizer cma --sigma 0.5 --zh-std 0 --generations 200 --starts 256   # ~35 min
```

### Steg 5: evaluering i ekte miljø

```bash
# Alle controllerne og fem grunnlinjer på de samme 2000 nye brettene (~4 min på 4 CPU-er).
# Skriver docs/evaluation/report.md, results.json og figurene. Mangler et sjekkpunkt, hoppes det over.
python -m worldmodels.evaluation --out docs/evaluation
```

### Steg 6: bedre syn

```bash
# Øyet: leser agentens og målets celle fra z, legges til M (~6 min)
python -m worldmodels.mdnrnn.eye --rnn checkpoints/mdnrnn_pos.pt --data data/v3/zseq_*.npz \
    --out checkpoints/mdnrnn_eye.pt
# Controller på syn + fremsyn, samme oppsett som 4e (~45 min)
python -m worldmodels.controller.train --rnn checkpoints/mdnrnn_eye.pt --data data/v3/zseq_*.npz \
    --use-eye --lookahead --charge-remaining --temperature 0 --shaping 0.05 \
    --optimizer cma --sigma 0.5 --zh-std 0 --generations 200 --starts 256 --out checkpoints/controller_eye_look.npz
```

### Steg 7: drømmen ser

```bash
# Nærsynet: spår mål og hindring i hver nabocelle fra bildet og øyets målminne (~9 min)
python -m worldmodels.mdnrnn.neighbours --rnn checkpoints/mdnrnn_eye.pt --data data/v3/zseq_*.npz \
    --out checkpoints/mdnrnn_sense.pt
# Samme controlleroppsett som steg 6, men i drømmen som nå spår hendelser fra nærsynet (~75 min)
python -m worldmodels.controller.train --rnn checkpoints/mdnrnn_sense.pt --data data/v3/zseq_*.npz \
    --use-eye --lookahead --charge-remaining --temperature 0 --shaping 0.05 \
    --optimizer cma --sigma 0.5 --zh-std 0 --generations 200 --starts 256 --seed 1 --out checkpoints/controller_sense.npz
```

## Resultater fra steg 2

![Rekonstruksjoner](docs/experiments/vae_20k_recon.png)

*Øverst: originaler fra valideringsepisoder (layouter VAE-en aldri har sett). Deretter
16 dim uten vekting, 16 dim med objektvekting og 32 dim med objektvekting.*

| Variant | Pikselfeil (MSE) | Riktige celler | Hele layouten riktig | Agent i riktig celle | Probe agent: lineær / MLP |
|---------|-----:|-----:|-----:|-----:|-----:|
| 16 dim, uten vekting | 0,0068 | 96,2 % | 0 % | 0 % | 3 % / 3 % |
| 16 dim, vekt 10 | 0,0036 | 98,2 % | 30 % | 77 % | 2 % / 34 % |
| **32 dim, vekt 10** | **0,0016** | **99,2 %** | **65 %** | **90 %** | **11 % / 53 %** |

Alle tall er på valideringsepisodene. Tilfeldig gjetting på agentens celle gir 1,6 %, og et bilde
med bare bakgrunn og hindringer gir ~87 % riktige celler.

- **Objektvektingen er avgjørende.** Uten den tegner VAE-en aldri agenten eller målet.
- **32 dimensjoner slår 16** på alle mål, så `vae_z32_w10` er modellen vi bygger videre på.
- **Posisjonen er i z, men ikke lineært.** En lineær probe finner agentens celle i 11 % av tilfellene,
  en liten MLP i 53 %. Det påvirker valget av controller senere (se D14).
- Det første forsøket med bare 1000 episoder overtilpasset seg til layoutene. Resultatene ligger i
  `docs/experiments/vae_1000ep_*`.

## Resultater fra steg 3

MDN-RNN-en tar inn z_t og handlingen, og predikerer neste z og om skrittet ender i mål, hindring
eller et vanlig flytt. Alle tall er på valideringsepisoder, og posisjonen leses av ved å dekode
prediksjonen med VAE-en.

| Mål | Artikkelens oppsett | Valgt modell | Grunnlinje: ingenting endrer seg |
|-----|-----:|-----:|-----:|
| Agenten i riktig celle etter ett skritt | 50 % | **58 %** | 28 % |
| ... når agenten faktisk flyttet seg | 33 % | **44 %** | 0 % |
| Drøm etter 5 ekte skritt: riktig celle etter 1 / 6 drømte skritt | 32 % / 21 % | **43 % / 29 %** | 27 % / 14 % |
| Treff på mål / hindring | 0 % / 14 % | 16 % / 42 % | – |
| Lineær probe: agentens celle fra (z, h) | 89 % | 84 % | fra z alene: 16 % |

![Drøm mot virkelighet](docs/experiments/mdnrnn_dream.png)

*To rader per episode: øverst det som faktisk skjedde, under drømmen. De fem første bildene er
ekte, deretter drømmer modellen med de samme handlingene. Drømmen holder seg nær virkeligheten
noen skritt, men sporer av: agenter blir borte, og mål dukker opp der det ikke var noe.*

- **Minnet løser problemet fra steg 2.** Agentens posisjon kan ikke leses lineært fra z (16 %), men
  fra RNN-ens skjulte tilstand kan den (84 %). En lineær controller på (z, h), som i artikkelen,
  er dermed realistisk.
- **Drømmer trenger oppvarming.** Fra kald start kopierer modellen bare z. Etter 5 ekte skritt
  slår den grunnlinjen tydelig.
- **Mål er vanskelige å forutse** (16 %). Det er den største svakheten før steg 4.
- Veien hit gikk gjennom 8 varianter. Se D20–D22 i `decisions.md`.

## Resultater fra steg 4

En lineær controller på [z, h] trenes med en evolusjonsstrategi, bare i drømmen. Drømmene starter
etter 5 ekte oppvarmingsskritt og varer 10 skritt. Deretter måles den i det ekte miljøet på
episoder den aldri har sett.

| Policy | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Tilfeldig | 11 % | 82 % | 7 % | −0,85 |
| **Controller trent i drømmen** | 6 % | 38 % | 57 % | −0,62 |
| Korteste vei (juks, kjenner kartet) | 100 % | 0 % | 0 % | +0,95 |

- **Den har lært å unngå hindringer.** Den krasjer halvparten så ofte som tilfeldig.
- **Men den finner ikke målet.** Den går mest mot høyre, blir stående ved veggen og venter ut tiden.
- **Drømmen blir lurt.** Drømmen tror controlleren når målet i opptil 36 % av tilfellene, mens det
  i virkeligheten skjer i 7 %. MDN-RNN-en forutser bare 16 % av mål-hendelsene, så controlleren
  lærer å utnytte feil i drømmen i stedet for å finne målet.
- Høyere temperatur i drømmen, som artikkelen foreslår, hjalp ikke. Se D26–D27.

## Resultater fra steg 4b

Controlleren samler nye ekte data, MDN-RNN-en trenes videre på dem, og controlleren trenes på
nytt i den bedre drømmen. Tre runder, 1000 ekte evalueringsepisoder per runde:

| Runde | Drøm: mål | Ekte: mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|---:|
| 0 (steg 4) | 36 % | 8 % | 35 % | 57 % | −0,57 |
| 1 | 5 % | 6 % | 33 % | 61 % | −0,58 |
| 3 | 8 % | 6 % | 30 % | 65 % | −0,57 |

- **Drømmen lures ikke lenger.** Allerede etter første runde stemmer drømmens målrate med virkeligheten.
- **Krasjene går ned** fra 35 % til 30 %, og M forutser nå 66 % av krasjene i controllerens
  data (før: 37 %).
- **Målet er fortsatt flaskehalsen.** M forutser bare 17 % av mål-hendelsene, og en helt ny
  controller trent i den nye drømmen når målet i bare 3 %. Se D28–D30.

## Resultater fra steg 4c

Vi lærte MDN-RNN-en å holde orden på hvor agenten og målet er, ga controlleren den troen som
4 ekstra tall, og rettet to feil i drømmen som gjorde det lønnsomt å gjemme seg.

| Controller (ekte miljø, 1000 episoder) | Mål | Hindring | Avkastning |
|---|---:|---:|---:|
| Tilfeldig | 11 % | 82 % | −0,85 |
| Steg 4 | 8 % | 35 % | −0,57 |
| Ny M + posisjonstro + læreplan | 10 % | 44 % | −0,61 |
| *Håndlaget "gå mot målet" på M sin tro (diagnostikk)* | *40 %* | *56 %* | *−0,24* |

- **M forutser dobbelt så mange mål** (23 % mot 10 %) og vet hvor målet er i 52 % av skrittene (før: 3 %).
- **Verdensmodellen holder nå.** En controller som bare går dit M tror målet er, når målet i 40 %,
  og drømmen rangerer den over "vent ved veggen".
- **Men søket finner den ikke.** Evolusjonsstrategien havner i "vent ved veggen" fra null, selv
  med 20 parametre og mer støy. Startet i den håndlagde løsningen blir den der. Se D31–D35.

## Resultater fra steg 4d

| Controller (ekte miljø, 1000 episoder) | Mål | Hindring | Avkastning |
|---|---:|---:|---:|
| Tilfeldig | 11 % | 82 % | −0,85 |
| Steg 4 | 8 % | 35 % | −0,57 |
| CMA-ES + formet belønning, z/h-spredning 0,03 | 40 % | 55 % | −0,25 |
| **CMA-ES + formet belønning, bare posisjonstro (20 parametre)** | **41 %** | **55 %** | **−0,22** |

- **Bedre søk alene var ikke nok.** CMA-ES, omstarter og flere drømmer per kandidat endte alle
  ved veggen. "Gå mot målet" er en smal topp i et flatt landskap, så ingen søkemetode fikk signal.
- **Formet belønning fra M sin egen tro ga søket en bakke.** Drømmen gir litt belønning per celle
  nærmere målet, ut fra der M *tror* agenten og målet er. Den er potensialbasert, så den endrer
  ikke hva som er best.
- **Den lærte controlleren slår den håndlagde**, og den beste bruker bare 20 parametre.
- **Neste svakhet er hindringer:** den går rett mot målet og krasjer i over halvparten av
  episodene. Se D36–D39.

## Resultater fra steg 4e

| Controller (ekte miljø, 1000 episoder) | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Tilfeldig | 11 % | 82 % | 7 % | −0,85 |
| Steg 4d: posisjonstro | 41 % | 55 % | 4 % | −0,22 |
| **Posisjonstro + fremsyn** | **38 %** | **34 %** | **29 %** | **−0,18** |

- **Et hjelpehode for hindringer lærte lite** (AUC 0,76). Minnet holder ikke oversikt over dem.
- **Fremsyn virket bedre.** For hver handling kjører M ett skritt frem og sier hvor sannsynlig et
  krasj er (AUC 0,94). Controlleren får de 4 tallene og er fortsatt lineær, med 36 parametre.
- **Krasjene falt fra 55 % til 34 %**, og avkastningen er den beste så langt.
- **Men agenten pendler.** I over halvparten av de avkortede episodene går den til side for en
  hindring og rett tilbake. En controller uten hukommelse om egne skritt kommer ikke rundt. Se D40–D42.

## Resultater fra steg 5: evaluering av hele kjeden

Alle policyer spilte de samme 2000 nye brettene (seeds 200 000–201 999), som ingen av dem har
sett før. Hele rapporten med intervaller, flere tabeller og bilder: [`docs/evaluation/report.md`](docs/evaluation/report.md).

![Utfall](docs/evaluation/outcomes.svg)

| Policy (2000 brett, 95 %-intervall) | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Tilfeldig | 13 % (12–14) | 80 % (79–82) | 7 % | −0,83 |
| Steg 4: lært i drømmen | 7 % (6–8) | 36 % (34–38) | 57 % | −0,58 |
| Steg 4b: iterativ trening | 6 % (5–7) | 28 % (26–30) | 66 % | −0,56 |
| Steg 4c: mål-bevisst drøm | 10 % (9–11) | 46 % (44–49) | 44 % | −0,62 |
| Steg 4d: bedre søk | 43 % (41–45) | 54 % (52–56) | 3 % | −0,19 |
| **Steg 4e: unngå hindringer** | **41 % (39–43)** | **32 % (30–34)** | **27 %** | **−0,12** |
| *Juks: rett mot målet* | 66 % | 34 % | 0 % | 0,30 |
| *Juks: rett mot målet, unngår hindringer* | 99 % | 0 % | 2 % | 0,92 |
| *Juks: korteste vei* | 100 % | 0 % | 0 % | 0,96 |

*Juks-policyene leser posisjonene rett fra miljøet. De er målestokker, ikke konkurrenter.*

- **Sluttagenten (4e) er den beste så langt**, med best avkastning. På de samme brettene krasjer
  den 22 prosentpoeng sjeldnere enn 4d (intervall 19–25), og når målet like ofte (−1,8 poeng,
  intervall −4,3 til +0,7). Det største enkeltspranget i hele prosjektet var bedre søk (4c → 4d: +33 poeng mål).
- **En controller uten hukommelse er nok.** "Gå mot målet, men aldri inn i en hindring" med sanne
  posisjoner når målet i 99 % og pendler nesten aldri. Det er nøyaktig regelen 4e prøver å lære.
  Pendlingen fra steg 4e skyldes altså ikke at C mangler minne, slik D42 antok, men at troen er feil.
- **Flaskehalsen er hva M vet, ikke hva C gjør.** Mens 4e spiller, peker M sin tro på riktig celle
  for agenten i 66 % av skrittene, men for målet bare i 26 % (snittfeil 2,2 celler). Og i første
  skritt har M ikke sett noe ennå, så C går i blinde: 29 % av krasjene til 4e skjer i skritt 1–2.
- **Drømmen har blitt ærligere.** Steg 4 drømte om fire ganger så mange mål som det fikk (10 % mot 2,5 %),
  altså en drøm som ble utnyttet. For 4e lover drømmen 35 % mål innen 10 skritt, og 32 % skjer.
  Men M skiller dårlig mellom enkeltsituasjoner (AUC 0,68–0,70), så tallet stemmer bare i snitt.

![Mål etter avstand](docs/evaluation/goal_by_distance.svg)

*Samme brett, to agenter: der 4d (øverst) krasjet og 4e (nederst) kom frem. Streken går fra gul til hvit.*

![Samme brett](docs/evaluation/compare_paths.png)

Se D43–D47 i [`decisions.md`](decisions.md).

## Resultater fra steg 6: bedre syn

Steg 5 viste at M bare visste hvor målet var i 26 % av skrittene. Steg 6 gir agenten et **øye**: en
liten modell formet som starten på en dekoder, som leser agentens og målets celle fra z, altså fra
bildet akkurat nå. Fordi målet står stille, husker øyet målet fra bilde til bilde.

| Under spilling (2000 brett) | Agent riktig | Mål riktig | Første skritt |
|---|---:|---:|---:|
| Minnet til M (steg 4e) | 66 % | 26 % | 1 % / 1 % |
| **Øyet (steg 6)** | **87 %** | **92 %** | **93 % / 89 %** |

| Policy (2000 brett) | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Steg 4e: unngå hindringer | 41 % | 32 % | 27 % | −0,12 |
| **Håndlaget «gå mot målet» på øyet** | **61 %** | 34 % | 4 % | **+0,22** |
| Steg 6: lært på syn + fremsyn | 28 % | 25 % | 47 % | −0,23 |

- **Synet er løst.** Øyet ser målet riktig i 92 % av skrittene, også i første skritt.
- **Med øyet er en enkel regel den beste agenten så langt.** Fire tall fra øyet og en håndlaget
  lineær regel gir positiv avkastning for første gang. Den er ikke lært, så den teller som
  diagnostikk, ikke som et steg.
- **Men den lærte controlleren ble dårligere enn 4e.** Drømmen tror den gode regelen når målet i
  28 % og krasjer i 45 % av drømmene; i virkeligheten er det 69 % og 17 %. Drømmen straffer altså
  akkurat den strategien som virker, og søket finner noe forsiktig som pendler.
- **Ny flaskehals: drømmen vet ikke det øyet vet.** Hendelsene i drømmen (mål, krasj) spås av M fra
  minnet h, som fortsatt ikke vet hvor målet er. Neste steg er å la M bruke øyet når den spår. Se D48–D51.

## Resultater fra steg 7: drømmen ser

Steg 6 ga agenten et øye, men drømmen spådde fortsatt mål og krasj fra minnet til M, som ikke visste
hvor målet var. I steg 7 spår drømmen hendelsene med **nærsyn**: for hver retning ser øyet om målet
eller en hindring står i nabocellen, ut fra bildet og minnet om målet.

| Hvem spår hendelsen (valideringsepisoder) | Mål: treff | Mål: presisjon | Krasj: treff | Krasj: presisjon |
|---|---:|---:|---:|---:|
| Hendelseshodet på minnet h (før) | 30 % | 19 % | 70 % | 41 % |
| **Nærsynet (steg 7)** | **85 %** | **94 %** | **77 %** | **95 %** |

![Utfall](docs/evaluation/outcomes.svg)

| Policy (2000 brett, 95 %-intervall) | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Tilfeldig | 13 % | 80 % | 7 % | −0,83 |
| Steg 4e: unngå hindringer | 41 % | 32 % | 27 % | −0,12 |
| Steg 6: syn | 28 % | 25 % | 47 % | −0,23 |
| Håndlaget på øyet (diagnostikk) | 61 % | 34 % | 4 % | +0,22 |
| **Steg 7: drømmen ser** | **75 % (73–77)** | **11 % (10–12)** | **14 %** | **+0,54** |
| *Juks: mot målet, unngår hindringer* | 99 % | 0 % | 2 % | 0,92 |

- **Den beste agenten i prosjektet, og den er lært i drømmen.** På de samme brettene når den målet
  i 35 prosentpoeng flere episoder enn 4e, og krasjer i 21 poeng færre. Den slår også den
  håndlagde regelen med 14 poeng.
- **Drømmen er ærlig nå.** For sluttagenten lover drømmen 59 % mål og 10 % krasj innen 10 skritt;
  i virkeligheten skjer 67 % og 5 %. I steg 6 lovet drømmen 28 % mål for en regel som fikk 69 %.
- **Det som gjenstår:** 14 % av episodene ender med at agenten går frem og tilbake ved en hindring.
  Juks-grunnlinjen viser at en controller uten minne kan komme nesten helt opp i 99 %, så det er
  fortsatt rom. Se D52–D55.

*Samme brett, to agenter: der steg 6 (øverst) krasjet og steg 7 (nederst) kom frem.*

![Samme brett](docs/evaluation/compare_paths.png)

## Prosjektstruktur

```
worldmodels/
  env/          GridDodge-miljøet og forhåndsvisning
  data/         tilfeldig policy og innsamling/lagring av rollouts
  vae/          modell, tap, datasett, trening og evaluering av VAE-en
  mdnrnn/       koding til z-sekvenser, MDN-RNN, øyet, trening og evaluering
  controller/   lineær controller og inndata, ES og CMA-ES, drømmemiljø, kjøring i ekte miljø, iterativ trening
  evaluation/   steg 5: grunnlinjer, evaluering med intervaller, drøm mot virkelighet, figurer og rapport
tests/          enhetstester (pytest)
docs/           bilder til README, evalueringsrapporten, resultater og diagnoseskript fra eksperimenter
decisions.md    logg over valg og begrunnelser
```
