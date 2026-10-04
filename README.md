# World Models i en liten 2D-verden

En agent som lærer å spille nesten bare inne i sin egen "drøm": en lært modell av verden. Inspirert av
[Ha & Schmidhuber (2018), *World Models*](https://worldmodels.github.io/), men med egne valg underveis.
Prosjektet er også et utstillingsvindu for vibekoding: hvert steg er bygget, testet og godkjent før
neste, og alle valg og blindveier står i [`decisions.md`](decisions.md).

![Sluttagenten med bevegelige hindringer](docs/evaluation_moving/final_paths.png)

*Sluttagenten på brett den aldri har sett, med tre hindringer som beveger seg. Blå = agent, grønn = mål, rød = hindring.*

## Resultatet

| Verden (2000 nye brett) | Når målet | Krasjer | Tilfeldig spill | Juks med fasit |
|---|---:|---:|---:|---:|
| Stille hindringer (steg 8) | **93 %** | 7 % | 13 % | 99 % |
| Tre bevegelige hindringer (steg 11) | **90 %** | 10 % | 10 % | 99 % |

Controlleren er en lineær regel med noen få dusin tall, trent bare i drømmen. Veien dit gikk gjennom
fem steg der agenten mest lærte å lure drømmen (rundt 10 % mål), før drømmen lærte å se.

- **[Oversiktssiden](docs/index.html)** viser reisen steg for steg, med tall og bilder.
- **[Drømmesiden](docs/drom/index.html)** spiller av episoder og viser hva agenten ser, tror og drømmer i hvert skritt.
- **[Steg for steg](docs/steg.md)** har kommandoene og alle resultatene for hvert steg.

GitHub viser HTML-filene som kode. Last dem ned og åpne dem i nettleseren; de har alt innebygd.

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
| 8. Slutt på pendlingen | `worldmodels/mdnrnn/tracker.py` | ✅ når målet i 93 %, krasjer i 7 %, pendler aldri |
| 9. Vis drømmen | `worldmodels/dreamview` | ✅ [interaktiv side](docs/drom/index.html) med drøm og virkelighet side om side |
| 10. Bevegelige hindringer | `worldmodels/env`, `worldmodels/mdnrnn/neighbours.py` | ✅ når målet i 83 % (steg 8: 79 %), drømmen er ærlig om krasj |
| 11. Øyet leser hindringene | `worldmodels/mdnrnn/obstacles.py` | ✅ når målet i 90 % med bevegelige hindringer, uten ny controller-trening |
| 12. Drømmesiden, bevegelig | `worldmodels/dreamview` | ✅ [siden](docs/drom/index.html) viser hindringskartet og hvor M tror hindringene går |
| 13. Porteføljeside | `docs/index.html`, `docs/steg.md` | ✅ [oversiktsside](docs/index.html) over hele reisen, kort README |

## Miljøet: GridDodge

Et 8x8-rutenett rendret som et 64x64 RGB-bilde (8 piksler per celle, samme oppløsning som i artikkelen).

- **Hver episode**: agent, mål og 6 hindringer plasseres tilfeldig. Et bredde-først-søk garanterer at målet kan nås.
- **Handlinger**: 4 diskrete (opp, ned, venstre, høyre). Å gå inn i kanten gjør at agenten blir stående.
- **Belønning**: +1 for å nå målet, −1 for å treffe en hindring (begge avslutter episoden), −0.01 per skritt.
- **Maks lengde**: 50 skritt, deretter avkortes episoden.
- **Avhengigheter**: bare NumPy.

Hvorfor akkurat dette miljøet er forklart i [`decisions.md`](decisions.md).

## Kom i gang

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Samle 1000 episoder med tilfeldig policy (~2 sekunder)
python -m worldmodels.data --episodes 1000 --out data/rollouts
```

Resten av kjeden, med kommandoer for hvert steg, står i [`docs/steg.md`](docs/steg.md).

## Prosjektstruktur

```
worldmodels/
  env/          GridDodge-miljøet og forhåndsvisning
  data/         tilfeldig policy og innsamling/lagring av rollouts
  vae/          modell, tap, datasett, trening og evaluering av VAE-en
  mdnrnn/       koding til z-sekvenser, MDN-RNN, øyet, nærsynet, sporing, hindringsøyet, trening og evaluering
  controller/   lineær controller og inndata, ES og CMA-ES, drømmemiljø, kjøring i ekte miljø, iterativ trening
  evaluation/   steg 5: grunnlinjer, evaluering med intervaller, drøm mot virkelighet, figurer og rapport
  dreamview/    steg 9: tar opp drøm og virkelighet og lager den interaktive siden
tests/          enhetstester (pytest)
docs/           oversiktssiden, steg.md, drømmesiden, evalueringsrapportene, bilder og diagnoseskript
decisions.md    logg over valg og begrunnelser
```
