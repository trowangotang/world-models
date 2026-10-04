# Beslutningslogg

Korte notater om valg som er tatt underveis, og hvorfor. Nyeste nederst.
Formålet er å vise tankeprosessen, ikke bare sluttresultatet.

---

## 2026-09-27 · Steg 1: miljø og tilfeldige rollouts

### D1. Eget rutenett-miljø ("GridDodge") i stedet for CarRacing eller et fysikkmiljø
**Valg:** 8x8-rutenett der agenten skal nå et mål og unngå hindringer, rendret som 64x64 RGB.
**Hvorfor:**
- *Raskt:* 1000 episoder samles på ~2 sekunder med bare NumPy. Hele pipelinen kan
  testes på minutter på en vanlig CPU.
- *Tro mot artikkelen der det teller:* observasjonen er et bilde på samme oppløsning som i
  Ha & Schmidhuber, så VAE-en må faktisk lære å komprimere piksler.
- *Lett å se om modellene har lært noe:* en rekonstruksjon eller en "drømt" rollout kan
  vurderes med øyet (står den blå firkanten der den skal?).
- *Et fysikkmiljø* (kontinuerlig posisjon og fart) ble vurdert. Det er mer likt CarRacing,
  men gir flere parametre å tune og gjør feilsøking tregere. Kan bli en senere utvidelse.

### D2. Ny tilfeldig layout per episode, med garantert nåbart mål
**Hvorfor:** Med én fast layout kunne controlleren bare pugget en rute, og verdensmodellen
ville ikke trengt å se på bildet. Tilfeldige layouter tvinger VAE-en til å kode *hvor* ting er,
og MDN-RNN-en til å bruke det. BFS-sjekken hindrer umulige episoder som ville forvirret evalueringen.

### D3. Deterministisk dynamikk
**Valg:** Gitt layout og handling er neste tilstand helt bestemt.
**Hvorfor:** Enklest å forstå og teste. MDN-RNN-en får likevel usikkerhet å modellere, fordi
den bare ser en komprimert latent vektor. Stokastikk (f.eks. hindringer som flytter seg) er en
mulig utvidelse hvis blandingsfordelingen i MDN-en skal vises tydeligere.

### D4. Fire diskrete handlinger
**Hvorfor:** Passer naturlig til et rutenett og gjør controlleren enkel (lineært lag + argmax).
Artikkelen bruker kontinuerlige handlinger; det er et bevisst avvik.

### D5. Klebrig tilfeldig policy (repeat_prob = 0.5)
**Valg:** Med 50 % sjanse gjentas forrige handling.
**Hvorfor:** Ren uniform støy får agenten til å vandre fram og tilbake nær start. Å gjenta
handlinger gir lengre rette strekk, så dataene dekker mer av verden. Parameteren kan
justeres med `--repeat-prob`.
**Resultat:** snitt 15,5 skritt per episode; 11 % mål, 82 % hindring, 7 % avkortet.
Få målepisoder er greit for VAE-en, men kan bli tynt for å lære belønning i MDN-RNN-en.
Da kan vi samle flere episoder eller senke antall hindringer.

### D6. Lagre obs som sekvens (T+1) i stedet for separate (obs, next_obs)-par
**Hvorfor:** Neste observasjon er bare obs forskjøvet ett skritt, så egne kopier ville doblet
lagringen. Sekvensformen er også det MDN-RNN-en trenger. `Episode.transitions()` gir
parene direkte for den som vil ha dem.

### D7. Én komprimert .npz per episode
**Hvorfor:** Ingen ekstra avhengigheter, enkel å inspisere, og episoder kan lastes enkeltvis.
1000 episoder tar ~4 MB fordi bildene er store ensfargede flater.

### D8. Agent tegnes som 6x6 piksler i en 8x8-celle
**Hvorfor:** Første versjon brukte 4x4, men så liten figur (16 av 4096 piksler) blir lett visket
ut av en VAE med rekonstruksjonstap. 6x6 er tydeligere, og kanten gjør agenten synlig
oppå mål og hindringer.

### D9. Reproduserbarhet via seeds
**Valg:** Episode *i* bruker miljø-seed `seed + i`, og policyen har sin egen seed.
**Hvorfor:** Hele datasettet kan gjenskapes eksakt fra én seed, og testene sjekker at
lagrede observasjoner kan spilles av identisk i miljøet.

---

## 2026-09-27 · Steg 2: VAE

Martin ønsket at prosjektet skal ta egne valg og ikke kopiere artikkelen. Valgene under er
derfor bevisste avvik der det passer verdenen vår bedre.

### D10. Objektvektet rekonstruksjonstap (object_weight = 10)
**Valg:** Piksler som ikke er bakgrunn i originalbildet teller 10 ganger mer i tapet.
**Hvorfor:** 85–90 % av hvert bilde er ensfarget bakgrunn. En vanlig VAE kan få lavt tap ved å
tegne bakgrunn og hindringer og droppe den lille agenten.
**Resultat:** Hypotesen holdt. Uten vekting (`object_weight=1`) tegner VAE-en **aldri** agenten
eller målet (0 % riktig agentcelle). Den får dessuten høyere pikselfeil enn den vektede
varianten, fordi den lærer det enkleste, altså bakgrunn og hindringer, og stopper der.
Med vekting havner agenten i riktig celle i 77 % (16 dim) og 90 % (32 dim) av rekonstruksjonene.
Se `docs/experiments/vae_20k_recon.png`.

### D11. Måle "forståelse", ikke bare pikselfeil
**Valg:** `worldmodels/vae/evaluate.py` leser rekonstruksjonen tilbake som et rutenett og måler
om hver celle, og særlig agentens celle, er riktig. I tillegg trenes prober fra z til agentens
og målets posisjon.
**Hvorfor:** Pikselfeil lyver her. Et bilde med bare bakgrunn og hindringer får ~87 % riktige
celler og lav MSE, men er ubrukelig for en agent. Rutenettmålene avslører det med en gang.

### D12. Første forsøk overtilpasset: flere layouter, færre bilder per episode
**Hva skjedde:** Første trening brukte de 1000 episodene fra steg 1. Treningstapet falt til ~86,
mens valideringstapet steg til ~1570. VAE-en hadde pugget de ~900 layoutene i treningssettet og
feilet på nye (bare 8–33 % riktig agentcelle).
**Innsikt:** Mangfoldet i dataene er antall *layouter*, ikke antall bilder. Bildene i en episode
er nesten like.
**Valg:** Samlet 20 000 episoder (35 s) og trekker høyst 4 bilder per episode
(`--frames-per-episode 4`). Det gir ~67 000 treningsbilder fra ~18 000 layouter.
Validering og trening ble da like gode (for eksempel 142 mot 114 for 32 dim), og riktig
agentcelle steg fra 33 % til 90 %. Det første forsøket er beholdt i
`docs/experiments/vae_1000ep_*` som dokumentasjon.

### D13. 32 latente dimensjoner, ikke 16
**Forslag i planen:** 16 dimensjoner, fordi verden er enkel.
**Resultat:** 16 holdt ikke helt. Med 32 dim er hele layouten eksakt riktig i 65 % av bildene
(mot 30 %), og agenten er i riktig celle i 90 % (mot 77 %). En layout har 6 hindringer, et mål
og en agent på 64 mulige celler, og det er mer informasjon enn det ser ut som.
**Valg:** `vae_z32_w10` er modellen vi går videre med, og 32 er nå standardverdien i koden.

### D14. Posisjon er i z, men ikke lineært tilgjengelig
**Funn:** En lineær probe leser agentens celle ut av z i bare 11 % av tilfellene (tilfeldig gjetting
gir 1,6 %). En liten MLP-probe (ett skjult lag) klarer 53 %, og dekoderen tegner agenten riktig
i 90 %. Informasjonen er altså i z, men viklet inn på en ikke-lineær måte.
**Hvorfor det betyr noe:** I artikkelen er controlleren lineær i (z, h). Hvis vi gjør det samme,
må MDN-RNN-ens skjulte tilstand h gjøre jobben med å rette ut posisjonen. Ellers trenger vi en
ikke-lineær controller. Dette tar vi stilling til i steg 3–4, og proben gir oss et mål på det.

### D15. Treningsoppsett
Adam (lr 1e-3), batch 128, beta = 1, 12 epoker (~4 min per epoke på CPU med 3 kjøringer
parallelt). Validering er 10 % av episodene, delt per episode. Checkpointer lagres utenfor git.

---

## 2026-09-27 · Steg 3: MDN-RNN

### D16. Z-sekvenser kodes én gang på forhånd
**Valg:** `python -m worldmodels.mdnrnn.encode` kjører VAE-en over alle 20 000 episodene og lagrer
mu, logvar, handlinger, hendelser og agentens celle i én fil (40 MB).
**Hvorfor:** RNN-en ser aldri bilder. Uten forhåndskoding ville hver epoke kjørt VAE-en på nytt.
Agentens celle lagres bare for evaluering. Modellen trenes aldri på den.

### D17. Hendelser i stedet for belønning
**Valg:** Modellen klassifiserer hvert skritt som *flytt*, *mål* eller *hindring*, i stedet for å
regrese belønningen og predikere "done" separat som i artikkelen.
**Hvorfor:** I GridDodge bestemmes både belønning og episodeslutt fullt ut av hendelsen, så én
klassifisering dekker begge. Avkorting etter 50 skritt regnes som et vanlig skritt, fordi det er en
tidsgrense og ikke noe som skjer i verden.

### D18. Hver blandingskomponent dekker hele z, og middelverdien er en endring fra z_t
**Hvorfor:** I artikkelen har hver dimensjon sin egen blanding. Her er utfallene diskrete (agenten
havner i én av få celler) og alle dimensjonene må endre seg sammen, så én komponent per utfall
passer bedre. Siden nesten hele scenen står stille, lærer modellen bare endringen.

### D19. Evaluering mot "ingenting endrer seg"
**Hvorfor:** Lav NLL sier lite i seg selv. Vi dekoder prediksjonen med VAE-en, leser av agentens
celle og sammenligner med en grunnlinje som kopierer z_t. Når agenten faktisk flytter seg, treffer
grunnlinjen aldri, så forskjellen viser om modellen har lært dynamikken.

### D20. Første forsøk: godt minne, men dårlig prediksjon av neste skritt
**Funn:** Med artikkelens oppsett (z og handling rett inn i LSTM-en) traff modellen agentens neste
celle i 50 % av skrittene, mot 28 % for kopiering. Men i **første** skritt av hver episode var den på
kopinivå (10 %), mål ble aldri forutsett (0 %), og drømmer fra kald start sporet av med en gang.
Samtidig kunne en lineær probe lese agentens posisjon ut av h i 80 % av tilfellene.
Minnet visste altså hvor agenten var, men prediksjonen av neste z var svak.
**Diagnose:** En vanlig MLP som predikerer z_{t+1} fra (z_t, a_t) med kvadratfeil, uten minne,
traff 35 % i første skritt og 53 % totalt. Dynamikken kan altså læres fra z alene. Svakheten lå
i hvordan MDN-RNN-en ble trent.

### D21. Tre endringer som hjalp, og to som ikke gjorde det
Målt som agentens celle etter ett skritt / når agenten flyttet seg (8 varianter, `docs/experiments/mdnrnn_*.json`):
- **MSE-ledd på blandingens forventning (`--mse-weight 10`)**: 38 % → 55 % / 17 % → 41 %. NLL alene
  lar modellen treffe de mange dimensjonene som ikke endrer seg og være slapp på de få som betyr noe.
- **Direkte vei fra inndata til hodene**: små gevinster (55 % → 58 %), og mål-treff 7 % → 16 %.
- **Tidlig stopp på dynamikken alene**: hendelseshodet overtilpasset seg etter ~10 epoker og ville
  stoppet treningen for tidlig.
- *Ikke hjulpet:* input-MLP alene og vekting av sjeldne hendelser. Vektingen økte treffene på
  hindringer (14 % → 45 %), men presisjonen falt fra 65 % til 39 %, og modellen overtilpasset seg raskere.
- *1 mot 5 komponenter:* nesten likt. Dynamikken er deterministisk, så blandingen gir lite.
  Vi beholder 5 fordi det koster lite og gir rom for usikkerhet i drømmen.

**Valgt modell:** `mdnrnn_direct_k5` (standardinnstillingene i `train.py`).

### D22. Drømmer trenger oppvarming
**Funn:** Fra kald start (bare z_0) er drømmen fortsatt på kopinivå. Hvis modellen først ser
5 ekte skritt, slår drømmen grunnlinjen tydelig: 43 % mot 27 % etter ett skritt og 29 % mot 14 %
etter seks skritt med egne prediksjoner. Men den sporer av over tid. I bildet
`docs/experiments/mdnrnn_dream.png` ser man agenter som forsvinner og mål som dukker opp.
**Konsekvens for steg 4:** Controlleren bør trenes i drømmer som starter etter noen ekte
oppvarmingsskritt, og drømmene bør være korte (rundt 5–10 skritt). Mål-hendelser forutses bare i
16 % av tilfellene, så en controller som trenes bare i drømmen vil få svakt signal om målet.
Det må vi ta stilling til før steg 4.

---

## 2026-09-28 · Steg 4: controller trent i drømmen

### D23. Lineær controller på [z, h], som i artikkelen
**Hvorfor:** Proben i steg 3 viste at agentens posisjon kan leses lineært fra (z, h) i 84 % av
tilfellene. Da bør en lineær controller ha det den trenger, og all "forståelse" ligger i V og M.
Den har 1156 parametre og velger handlingen med høyest verdi.

### D24. Enkel evolusjonsstrategi i stedet for CMA-ES
**Valg:** Antitetisk støy, rangerte fitnessverdier og Adam (Salimans m.fl. 2017), 32 kandidater
per generasjon, sigma 0,1. Omtrent 40 linjer og enhetstestet på en kvadratisk funksjon.
**Hvorfor:** CMA-ES krever en ekstern pakke eller mye kode, og med ~1000 parametre er den enkle
varianten god nok. En generasjon tar under ett sekund fordi alle kandidatene drømmer i én batch.

### D25. Drømmeoppsett basert på funnene fra steg 3
- **5 ekte oppvarmingsskritt** fra datasettet før controlleren tar over (D22).
- **10 drømte skritt**, fordi drømmen sporer av etter 6–8.
- **Forventet belønning** fra hendelseshodet i stedet for trukne utfall: belønningen i hvert skritt
  er vektet med sannsynligheten for at episoden fortsatt lever. Mindre støy i fitness.
- Den ekte sjekken underveis (200 episoder med nye seeds) brukes bare til logging, aldri til å
  velge vekter.

### D26. Resultat: færre krasj, men ikke flere mål
Ekte miljø, seeds fra 100 000 og oppover (ingen av dem finnes i treningsdataene):

| Policy | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Tilfeldig | 11 % | 82 % | 7 % | −0,85 |
| **Controller trent i drømmen** | 6 % | 38 % | 57 % | −0,62 |
| Tilfeldig som unngår hindringer (juks, kjenner kartet) | 29 % | 0 % | 72 % | −0,13 |
| Korteste vei (juks) | 100 % | 0 % | 0 % | +0,95 |

Controlleren har lært *noe* ekte: den krasjer halvparten så ofte som tilfeldig og får bedre
avkastning. Men den når målet sjeldnere. Den velger "høyre" i ~77 % av skrittene, går inn i
høyre vegg og blir stående til tiden går ut. Det er trygt både i drømmen og i virkeligheten.

**Drømmen blir lurt:** med temperatur 0 tror drømmen at controlleren når målet i 36 % av
drømmene, men i virkeligheten skjer det i 7 %. Controlleren utnytter feil i M, akkurat det
artikkelen advarer mot. Grunnen er at M bare treffer 16 % av mål-hendelsene (D22), så signalet
for "gå mot målet" er både svakt og upålitelig, mens "unngå hindringer" læres godt.

### D27. Høyere temperatur hjalp ikke
Artikkelen gjør drømmen mer uforutsigbar (tau 1,15) for å motvirke utnyttelse. Vi prøvde
tau 0 / 1 / 1,5 / 2,5 og 64 mot 256 starter per generasjon. Alle endte på 5–7 % mål og 30–40 %
hindring. Problemet er ikke at controlleren utnytter en for forutsigbar drøm, men at drømmen
ikke vet nok om mål. Rådata: `docs/experiments/controller_runs.json`.

## Steg 4b: iterativ trening

### D28. Iterativ trening som eget steg
Martin valgte å prøve artikkelens del 5: la controlleren samle nye ekte data, tren MDN-RNN-en
videre på dem, og tren controlleren på nytt i den forbedrede drømmen. Tanken er at drømmen blir
lurt akkurat der controlleren går, og at de tilstandene mangler i de tilfeldige dataene.

Våre valg (`worldmodels/controller/iterate.py`):
- **5000 nye episoder per runde**, med **epsilon 0,3**: hver handling byttes ut med en tilfeldig en
  med 30 % sannsynlighet. Uten utforsking ville dataene bare vist vegg-strategien.
- **Alle data beholdes**: M trenes på de tilfeldige dataene pluss alle rundene så langt, så den
  ikke glemmer resten av verden.
- **Fortsett fra forrige vekter** for både M (8 epoker, lr 5e-4) og C (150 generasjoner). Billigere
  enn å starte på nytt, og hver runde bygger på den forrige.
- **VAE-en holdes fast.** Den ser allerede alle layouter (D12), og en ny V ville gjort M og C ubrukelige.
- **Egne seeds**: innsamling i runde r bruker seeds fra 300 000 + r · 100 000, evaluering fra 100 000.

### D29. Resultat: drømmen lures ikke lenger, men målet er fortsatt vanskelig
1000 ekte episoder per runde, samme seeds hver gang:

| Runde | Drøm: mål | Ekte: mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|---:|
| 0 (steg 4) | 36 % | 8 % | 35 % | 57 % | −0,57 |
| 1 | 5 % | 6 % | 33 % | 61 % | −0,58 |
| 2 | 11 % | 6 % | 32 % | 62 % | −0,58 |
| 3 | 8 % | 6 % | 30 % | 65 % | −0,57 |

- **Utnyttelsen forsvant i første runde.** Drømmen tror ikke lenger at vegg-strategien når målet.
  Det er akkurat effekten artikkelen beskriver.
- **Krasjene går jevnt ned**, fra 35 % til 30 %.
- **Men målraten står stille på ~6 %**, og controlleren går fortsatt mot høyre i ~88 % av skrittene.

M målt på 1000 nye episoder den aldri har sett (andel av hendelsene den forutser):

| M | Tilfeldige: mål / hindring | Controllerens: mål / hindring |
|---|---:|---:|
| Original (steg 3) | 12 % / 44 % | 12 % / 37 % |
| Etter runde 3 | 10 % / 60 % | 17 % / 66 % |

M har lært mye mer om hindringer, men nesten ingenting nytt om mål. Controllerens data inneholder
bare ~10 % mål-episoder, så signalet blir ikke sterkere av å samle mer av samme slag.

### D30. En fersk controller slipper ikke ut heller
For å sjekke om controlleren bare satt fast i steg 4-vektene, trente vi en ny fra null (300
generasjoner) i drømmen fra runde 3. Den endte på 3 % mål, 24 % hindring og 74 % avkortet: enda
forsiktigere, men ikke bedre på mål. Flaskehalsen er altså drømmens svake kunnskap om mål, ikke
startpunktet til controlleren. Rådata: `docs/experiments/iterate_results.json`.

## Steg 4c: mål-bevisst drøm

### D31. Årsaken: minnet vet ikke hvor målet er
Martin valgte å gjøre drømmen mål-bevisst. Før vi endret noe, sjekket vi hvorfor M bommer på mål:
- VAE-en tegner målet i riktig celle i 81 % av bildene, så informasjonen *finnes* i z.
- Men verken z eller RNN-minnet h gjør den lesbar: en probe finner målets celle i bare 3 %
  (tilfeldig gjetting: 1,6 %). Agentens celle kan leses fra h i 73 %.
- En mål-hendelse skjer når agenten går inn i målets celle. Uten å vite hvor målet er, kan M
  bare gjette. Og en lineær controller på [z, h] kan heller ikke styre mot noe den ikke ser.

Målets celle lagres nå i z-sekvensene (`goal_cell`, fast gjennom episoden). Eldre filer lastes
med -1 (ukjent).

### D32. Hjelpehode for posisjoner, og M sin tro som inndata til C
- **Hjelpehode** (`--position-weight 1`): et lineært lag fra h som predikerer agentens og målets
  celle. Det er bare et ekstra tap under trening, men tvinger minnet til å holde orden på
  posisjonene. Merkelappene kommer fra fargene i bildet, samme kilde som hendelsene.
- **Overvekt av mål-episoder** (`--goal-oversample 4`) ble også prøvd. Det hjalp ikke tydelig
  (17 % mot 23 % av målene i tilfeldige data), så vi bruker varianten uten.
- **Posisjonstro til controlleren** (`--use-positions`): fra hjelpehodet regnes forventet
  (rad, kolonne) for agent og mål, 4 tall i [-1, 1]. Da er retningen til målet en lineær
  funksjon av inndata, og controlleren er fortsatt lineær (1172 parametre).

M målt på 1000 nye episoder (andel av hendelsene den forutser):

| M | Tilfeldige: mål / hindring | Controllerens: mål / hindring | Målets celle fra h |
|---|---:|---:|---:|
| Etter steg 4b | 10 % / 60 % | 17 % / 66 % | 3 % (probe) |
| **Med hjelpehode** | **23 % / 67 %** | **24 % / 72 %** | **52 %** (hjelpehodet) |

### D33. En håndlaget controller viser at verdensmodellen nå holder
For å skille "drømmen er for dårlig" fra "søket finner ikke løsningen" laget vi en controller for
hånd: gå i retningen der M tror målet er, bare ut fra de 4 posisjonstallene. Den er et
diagnoseverktøy, ikke en løsning, fordi vi har skrevet vektene selv.

I ekte miljø når den målet i **40 %** (tilfeldig: 11 %) med avkastning −0,24 (beste lærte
til nå: −0,57). Den krasjer fortsatt i 56 %, fordi den ikke ser etter hindringer. Drømmen
spår 30 % mål og 62 % hindring for den, så drømmen er nå rimelig ærlig om mål.

### D34. Drømmen belønnet å gjemme seg
Selv om drømmen nå kjente igjen "gå mot målet", ga den høyere fitness til å stå stille ved en
vegg. To grunner:
- **Venting var nesten gratis.** En drøm varer 10 skritt, så å overleve kostet bare 0,1. I
  virkeligheten koster det 0,45 å vente ut hele episoden. `--charge-remaining` lar den som
  overlever drømmen betale for skrittene som ville gjenstått.
- **Støy i drømmen overdrev krasj.** Med temperatur 1 steg andelen krasj for "gå mot målet" fra
  62 % til 69 %. Med temperatur 0 og kostnaden over scorer "gå mot målet" −0,38 mot −0,48 for
  å vente. Nå peker drømmen i samme retning som virkeligheten.

### D35. Resultat: søket er den nye flaskehalsen
Evolusjonsstrategien finner likevel ikke "gå mot målet" selv:

| Controller (ekte miljø, 1000 episoder) | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Tilfeldig | 11 % | 82 % | 7 % | −0,85 |
| Steg 4 | 8 % | 35 % | 57 % | −0,57 |
| Steg 4b, etter 3 runder | 6 % | 30 % | 65 % | −0,57 |
| Ny M, [z, h] | 5 % | 27 % | 69 % | −0,57 |
| Ny M, [z, h] + posisjonstro, rettet drøm | 7 % | 37 % | 56 % | −0,60 |
| Som over + læreplan | 10 % | 44 % | 46 % | −0,61 |
| *Håndlaget "gå mot målet" (diagnostikk)* | *40 %* | *56 %* | *4 %* | *−0,24* |

- Startet i den håndlagde løsningen blir ES der (41 % mål). Startet fra null havner den i
  "vent ved veggen" hver gang, også med bare de 20 relevante parametrene og 10 ganger mer støy.
  Det er altså et søkeproblem: "vent" er en bred, trygg dal, og halvveis mot målet er verre enn begge.
- En læreplan der halvparten av drømmene starter nær målet (høyst 3 skritt unna) hjelper litt:
  målraten stiger jevnt til 10 %, men den krasjer mer. Skriptene ligger i
  `docs/experiments/scripts/`, rådata i `docs/experiments/goal_aware.json`.

## Steg 4d: bedre søk

### D36. CMA-ES, og hvorfor søket alene ikke holdt
Martin valgte å forbedre søket. Vi la til **sep-CMA-ES** (`worldmodels/controller/cma.py`): CMA-ES
med diagonal kovarians, som skalerer til ~1200 parametre. Den er testet på standard testfunksjoner.

Fordi controlleren velger argmax, endres ikke fitness om alle vektene skaleres. Uten grep vokser
både middelverdi og sigma uten grense. Med `normalize=True` holdes middelverdien på lengde 1 og
sigma skaleres likt. Det endrer ikke søket, men gjør sigma lesbar.

Så prøvde vi søket på bare de 20 parametrene som betyr noe for "gå mot målet" (posisjonstroen og
bias), der løsningen fantes:

| Søk (20 parametre, drømmen fra D34) | Fitness i drømmen |
|---|---:|
| Håndlaget "gå mot målet" | −0,30 |
| Alltid "opp" (alle vekter 0) | −0,48 |
| Beste av 2000 tilfeldige controllere | −0,42 |
| ES / CMA-ES / 1024 drømmer per kandidat | −0,46 |
| Omstarter med populasjon 16, 32, 64, 128 (IPOP) | −0,47 til −0,48 |

Ingen metode fant den. Grunnen er landskapet: rundt den håndlagde løsningen er fitness god
bare innenfor en liten kjegle (84 % av punktene er gode med 25 % støy, 17 % med 50 %). Utenfor er
landskapet flatt rundt −0,47. Det finnes ingen bakke å klatre, uansett søkemetode.

### D37. Formet belønning fra M sin egen tro
Løsningen var å gi søket en bakke. Drømmen gir nå litt belønning (`--shaping 0.05`) for hver
celle agenten kommer nærmere målet, målt med M sin *egen* tro om posisjonene. Den bruker ikke
noe fra det ekte miljøet.

Belønningen er **potensialbasert** (Ng m.fl. 1999): den er forskjellen i avstand før og etter
skrittet. Summen over en episode avhenger bare av start og slutt, så den endrer ikke hvilken
politikk som er best, bare hvor lett den er å finne.

Med den fant CMA-ES "gå mot målet" fra null på de 20 parametrene (−0,24 i ekte miljø).

### D38. Liten startspredning på z og h
På hele controlleren (1172 parametre) gikk det tregt: 16 % mål etter 300 generasjoner. De 1152
vektene på z og h er for det meste støy for søket og drukner de 20 som betyr noe.

CMA-ES lar hver parameter ha sin egen startspredning (`--zh-std`). Med 0,03 for z og h og 1 for
resten finner den retningen først og kan så bruke z og h der det lønner seg. Med `--zh-std 0`
står z- og h-vektene stille på 0, og controlleren bruker bare posisjonstroen.

### D39. Resultat
Ekte miljø, 1000 episoder:

| Controller | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Tilfeldig | 11 % | 82 % | 7 % | −0,85 |
| Steg 4 (ES, [z, h]) | 8 % | 35 % | 57 % | −0,57 |
| Steg 4c (ES, posisjonstro, læreplan) | 10 % | 44 % | 46 % | −0,61 |
| ES + formet belønning | 17 % | 54 % | 29 % | −0,58 |
| CMA-ES + formet belønning | 16 % | 57 % | 28 % | −0,61 |
| CMA-ES + formet, z/h-spredning 0,1 | 35 % | 55 % | 10 % | −0,31 |
| CMA-ES + formet, z/h-spredning 0,03 | 40 % | 55 % | 6 % | −0,25 |
| **CMA-ES + formet, bare posisjonstro (20 parametre)** | **41 %** | **55 %** | **4 %** | **−0,22** |
| *Håndlaget "gå mot målet" (diagnostikk)* | *40 %* | *56 %* | *4 %* | *−0,24* |

- **Den formede belønningen er det som gjør forskjellen.** Uten den ender alle søk ved veggen.
- **CMA-ES med liten spredning på z og h** trengs for å finne løsningen i hele controlleren.
- **Den lærte controlleren slår den håndlagde**, og den beste bruker bare 20 parametre.
- **Hindringer er neste svakhet.** Alle går rett mot målet og krasjer i over halvparten av
  episodene. Informasjonen om hindringer ligger i z og h, men søket har ikke lært å bruke den.

Rådata: `docs/experiments/search.json`, skript i `docs/experiments/scripts/`.

## Steg 4e: unngå hindringer

### D40. Hjelpehode for hindringer lærte lite
Martin valgte å lære controlleren å unngå hindringer. Første forsøk fulgte oppskriften fra målet
(D32): z-sekvensene fikk en merkelapp for om det står en hindring i hver nabocelle
(`near_obstacle`), og M fikk et lineært hode fra h som skal forutsi den (`--obstacle-weight`).
M ble trent videre fra `mdnrnn_pos` i 10 epoker.

Hodet lærte lite: AUC 0,76, og det sier aldri over 50 % (bare 4,7 % av nabocellene er hindringer).
Minnet holder ikke oversikt over hindringene, akkurat som det ikke gjorde med målet før D32.
Hodet er beholdt i koden som valgfritt, men brukes ikke videre.

### D41. Fremsyn: M spør seg selv "hva skjer om jeg går hit?"
M *vet* likevel mye om hindringer: hendelseshodet forutså to av tre krasj allerede i steg 4c.
Det trenger bare å få vite hvilken handling det gjelder. Så for hver av de fire handlingene
kjører vi M ett skritt frem, uten å endre minnet, og leser av sannsynligheten for krasj
(`MDNRNN.lookahead_obstacle`, `--lookahead`). Det er 4 tall til controlleren.

Fremsynet skiller hindring fra ikke-hindring mye bedre enn hodet: AUC 0,94, og med grense 50 %
fanger det 67 % av hindringene med 6 % falske alarmer. Controlleren er fortsatt lineær, nå på
posisjonstro (4) + fremsyn (4) + bias: 36 parametre.

### D42. Resultat: færre krasj, men agenten pendler
Ekte miljø, 1000 episoder. Samme oppsett som D39 (CMA-ES, formet belønning, `--zh-std 0`):

| Controller | Mål | Hindring | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Tilfeldig | 11 % | 82 % | 7 % | −0,85 |
| Steg 4d: posisjonstro (20 parametre) | 41 % | 55 % | 4 % | −0,22 |
| **Posisjonstro + fremsyn (36 parametre)** | **38 %** | **34 %** | **29 %** | **−0,18** |
| Som over, z/h-spredning 0,03 | 35 % | 33 % | 32 % | −0,20 |

- **Krasjene falt fra 55 % til 34 %**, og avkastningen er den beste så langt.
- **Men nå blir 29 % av episodene avkortet.** Av 88 avkortede episoder (av 300) pendlet agenten
  mellom 2–3 celler i 51: den går til side for en hindring, og så rett tilbake mot målet. En
  lineær controller som bare ser nåtiden, har ingen måte å huske at den nettopp var der.
- Å gi controlleren også z og h (spredning 0,03) hjalp ikke.

Rådata: `docs/experiments/obstacles.json`.

## Steg 5: evaluering i det ekte miljøet

### D43. Én felles evaluering på 2000 nye brett, med intervaller og juks-grunnlinjer
Martin valgte å gå rett til steg 5 i stedet for å fikse pendlingen først. Tidligere steg ble målt
på 1000 episoder med litt ulike oppsett, så tallene var vanskelige å sammenligne. Nå spiller alle
controllerne og fem grunnlinjer de samme 2000 brettene (seeds 200 000–201 999, som ingen har sett).
`python -m worldmodels.evaluation` gjør alt og skriver `docs/evaluation/report.md`.

- **95 %-intervaller** (Wilson for rater, bootstrap for avkastning), og **parvise forskjeller** på
  de samme brettene, som er mye skarpere enn to separate intervaller.
- **Grunnlinjer med juks** som leser miljøets indre tilstand: tilfeldig som unngår hindringer, rett
  mot målet, rett mot målet som unngår hindringer, og korteste vei. De svarer på "hvor godt kunne
  denne typen policy gjort det med perfekt kunnskap?", og skiller feil i C fra feil i V og M.
- Vi la ikke til nye controllere. Steg 5 skal måle det vi har, ikke flytte målstreken.

Med flere brett ble 4e litt bedre enn i D42 (41 % mål mot 38 %), innenfor usikkerheten.

### D44. Sluttagenten er den beste, men 4d og 4e når målet like ofte
Parvis mot 4d på de samme brettene: 22 prosentpoeng færre krasj (intervall 19–25), og ingen sikker
forskjell i mål (−1,8 poeng, intervall −4,3 til +0,7). Fremsynet byttet krasj mot avkortede
episoder, ikke mot flere mål. Avkastningen er best for 4e (−0,12 mot −0,19), fordi et krasj koster
mer enn å bli stående.

### D45. Pendlingen skyldes feil tro, ikke at C mangler minne
D42 antok at 4e pendler fordi en controller uten hukommelse ikke kan komme rundt en hindring.
Juks-grunnlinjen "rett mot målet, unngår hindringer" er nettopp en slik controller, uten minne,
men med sanne posisjoner. Den når målet i 99 % og pendler i bare 21 av 2000 episoder.
Altså er regelen 4e prøver å lære god nok. Det som svikter, er inndataene.

### D46. Flaskehalsen er hva M vet om målet, og de første skrittene er blinde
Vi målte M sin tro mot fasiten i hvert ekte skritt (`evaluation/beliefs.py`). For 4e:

- Agentens posisjon: riktig celle i 66 % av skrittene, 73–81 % etter 5–10 skritt.
- **Målets posisjon: riktig celle i bare 26 %**, snittfeil 2,2 celler. Den blir ikke bedre over tid.
- **Skritt 0 er blindt.** Troen kommer fra minnet før bildet er lest, og minnet er tomt i starten.
  Første handling er derfor nesten den samme på alle brett. 29 % av krasjene til 4e skjer i skritt 1–2.

Det forklarer også hvorfor 4d og 4e bare når målet i 57 % av brettene der målet er 1–3 skritt unna,
mens juks-versjonen av samme regel når det i 100 %. Neste forbedring bør gjelde V og M (for eksempel
å lese posisjonene fra [z, h] etter at bildet er sett), ikke C.

### D47. Drømmen har blitt ærligere, men bare i snitt
Etter 5 ekte skritt drømte vi 10 skritt fra nøyaktig den tilstanden, og spilte så de samme skrittene
i virkeligheten (`evaluation/dream_check.py`). Steg 4 drømte om nesten fire ganger så mange mål som
den fikk (9,6 % mot 2,5 %): drømmen ble utnyttet, som i D26. For 4e lover drømmen 35 % mål og 32 %
skjer. 4d sin drøm er derimot for pessimistisk (31 % mål mot 50 %, 57 % krasj mot 34 %).
M rangerer enkeltsituasjoner svakt (AUC 0,65–0,70 for 4d og 4e), så drømmen er god til å sammenligne
policyer i snitt, men dårlig til å si hva som skjer på ett bestemt brett.

## Steg 6: bedre syn

### D48. Et øye formet som en dekoder
Martin valgte å forbedre synet, fordi steg 5 viste at M sin tro om målet var flaskehalsen (D46).
Informasjonen finnes i z (dekoderen tegner målet riktig i 81 %), men den er viklet inn. Vi målte
tre lesere på z alene, trent på 18 000 episoder og testet på 2000 andre:

| Leser av z | Agent | Mål |
|---|---:|---:|
| Lineær | 16 % | 2 % |
| MLP, to lag | 87 % | 66 % |
| **Romlig: z → 4x4-kart → 8x8-kart, én logit per celle** | **97 %** | **90 %** |

Den romlige leseren er bygd som starten på en dekoder, og ga et stort sprang. Rutenettet er
romlig, så en leser med romlig struktur trenger mye mindre for å lære det. Den ferdige versjonen
(`mdnrnn/eye.py`, 32 kanaler, 43 000 parametre) ble trent i 8 epoker på alle z-sekvensene med M fryst:
98,5 % agent og 94 % mål på nye episoder. Øyet lagres i M-sjekkpunktet (`mdnrnn_eye.pt`), slik at
drømmen, agenten og controlleren finner det der de allerede finner M. Controlleren får det øyet ser
med `--use-eye` (4 tall i samme format som posisjonstroen).

### D49. Øyet husker målet
En håndlaget "gå mot målet" på øyet nådde målet i 49 %, men ble avkortet i 15 %. Feilen var nesten
alltid den samme: når agenten står like ved målet, tegner VAE-en dem utydelig, øyet mister målet,
agenten snur, ser målet igjen, og snur tilbake. Målet står stille, så vi lar hvert bilde stemme:
øyet summerer log(p + 0,001) for målets celle over alle bildene i episoden (`MDNRNN.see`). Ett
bilde som bommer, blir nedstemt av de andre. Det ga 58 % mål og bare 4 % avkortet. Agentens
posisjon leses fortsatt bare fra bildet nå, siden den flytter seg. Minnet følger med inn i drømmen
fra oppvarmingsbildene.

### D50. Med øyet er en enkel regel den beste agenten så langt
Evaluert på de samme 2000 brettene som i steg 5 ser øyet målet riktig i 92 % av skrittene mens
agenten spiller, mot 26 % for minnet til M, og 89 % allerede i første skritt (mot 1 %). Den
håndlagde regelen på øyet når målet i 61 %, krasjer i 34 % og får avkastning +0,22, den første
positive. Den er ikke lært, så den står i evalueringen som diagnostikk.

### D51. Den lærte controlleren ble dårligere, fordi drømmen ikke vet det øyet vet
Samme oppsett som 4e, men med syn i stedet for minnets tro: 28 % mål, 25 % krasj, 47 % avkortet,
avkastning −0,23. Det er 13 poeng færre mål enn 4e på de samme brettene. Med bare syn (uten fremsyn)
ble det 22 % mål ved siste kontroll under treningen.

Drøm mot virkelighet forklarer hvorfor. For den håndlagde regelen lover drømmen 28 % mål og 45 %
krasj innen 10 skritt; i virkeligheten skjer 69 % og 17 %. I drømmen er den derfor dårligere enn det
søket fant (fitness −0,33 mot −0,16), selv om den er mye bedre i virkeligheten. Drømmens hendelser
spås av M fra minnet h, og h vet fortsatt ikke hvor målet er. Øyet hjelper controlleren å se, men
ikke drømmen å spå.

Neste naturlige steg er derfor å la M bruke øyet når den spår hendelser, og så trene controlleren
på nytt. Vi har ikke gjort det her, fordi det betyr å trene M på nytt. Det er et eget steg.

## Steg 7: drømmen ser

### D52. Nærsyn: hendelsene i drømmen spås fra bildet, ikke fra minnet
Martin sa ja til å la M bruke øyet når den spår (D51). Det som avgjør hva som skjer når agenten
tar en handling, er hva som står i nabocellen den veien. Nærsynet (`mdnrnn/neighbours.py`) svarer på
nettopp det: for hver av de fire retningene, er målet der, og står det en hindring der?

- **Inndata:** z (bildet nå), øyets minne om målet som et 8x8-kart, og øyets kart over hvor agenten er.
- **Oppbygning:** som øyet (z → 4x4 → 8x8), pluss målkartet som ekstra kanal, to konvolusjonslag og
  ett kart per retning og spørsmål ("om agenten står her, hva er ved siden av?"). Kartet leses av
  der øyet ser agenten. Et første forsøk som selv måtte finne agenten (maksimum over kartet), lærte
  ingenting på seks epoker; med øyets agentkart lærte det på én.
- **Merkelapper for alle retninger:** hvert bilde sier hva som er i alle fire nabocellene, ikke bare
  den agenten gikk til. Det gir mange flere eksempler enn hendelsene alene. Merkelappene kommer fra
  fargene i bildet, som før (D32).
- **Kalibrert:** vanlig binær kryssentropi uten vekting. Hendelseshodet til M ble trent med vektede
  klasser (standard i `mdnrnn.train`, se D21), og spådde dobbelt så mange krasj som det som skjedde (7,5 % mot 3,7 % av skrittene).
  Drømmen regner med sannsynlighetene som de er, så de må stemme.

I drømmen erstatter nærsynet hendelseshodet (`MDNRNN.event_probs`), og fremsynet til controlleren
leses fra samme sted, slik at controlleren og drømmen er enige. Dynamikken (neste z) kommer fortsatt
fra M. Nærsynet ble trent i 8 epoker med resten av M fryst (`mdnrnn_sense.pt`).

### D53. Drømmen spår hendelser mye bedre
På valideringsepisodene, for handlingen som faktisk ble tatt:

| Hvem spår | Mål: treff / presisjon | Krasj: treff / presisjon | Snitt spådd mot ekte (mål, krasj) |
|---|---:|---:|---:|
| Hendelseshodet på h | 30 % / 19 % | 70 % / 41 % | 1,5 % mot 0,7 %, 7,5 % mot 3,7 % |
| Nærsynet | 85 % / 94 % | 77 % / 95 % | 0,6 % mot 0,7 %, 3,0 % mot 3,7 % |

I den nye drømmen er rangeringen riktig: den håndlagde regelen med fremsyn får fitness +0,31, mot
−0,13 i steg 6, og ligger over controlleren fra steg 6 (+0,06).

### D54. Resultat: 75 % mål og 11 % krasj
Samme controlleroppsett som steg 6 (syn + fremsyn, 36 parametre, CMA-ES, formet belønning), trent i
den nye drømmen med to frø. Ved siste kontroll under trening nådde frø 0 målet i 62 % og frø 1 i
75 %. Vi valgte frø 1 fordi drømmen ga den høyest fitness (+0,26 mot +0,22), ikke ut fra
evalueringsbrettene.

På de samme 2000 brettene som i steg 5 og 6: **75 % mål, 11 % krasj, 14 % avkortet, avkastning
+0,54**. Parvis mot 4e: +35 poeng mål og −21 poeng krasj. Mot den håndlagde regelen på øyet:
+14 poeng mål. Drøm mot virkelighet: drømmen lover 59 % mål og 10 % krasj innen 10 skritt, og 67 %
og 5 % skjer. Drømmen er fortsatt litt forsiktig, men rangerer riktig.

### D55. Det som gjenstår
- **Pendling ved hindringer:** 266 av 271 avkortede episoder er pendling. Juks-grunnlinjen uten
  minne når 99 %, så en bedre controller kan fortsatt komme mye lenger.
- **Lange avstander:** 86 % mål når målet er 1–3 skritt unna, 62 % ved 8 eller flere.
- **Øyet mister agenten over tid:** agenten ses riktig i 92 % av de første skrittene, men 67 % etter
  10 skritt. De lange episodene er der agenten står inntil hindringer, som VAE-en tegner utydelig.

## Steg 8: slutt på pendlingen

Martin: "prøv å fiks problemet med at agenten går frem og tilbake ved en hindring".

### D56. Pendlingen har to årsaker: øyet mister agenten, og C husker ingenting
Vi kjørte steg 7-agenten på 1000 brett (`docs/experiments/scripts/oscillation.py`). 154 episoder
pendlet. I de siste 20 skrittene av dem så øyet agenten i riktig celle bare 58 % av gangene, mot
92 % i de andre episodene. Målet ble sett riktig i 78 % (93 %), og fremsynet stemte i 78 % (93 %).

Typiske tilfeller:
- **Øyet forveksler agenten med målet.** Agenten står i (3, 2), målet i (3, 1). Øyet ser agenten i
  (3, 1), altså oppå målet. Controlleren tror den er fremme og går til høyre. I (3, 3) ser øyet
  riktig igjen og går til venstre. Slik fortsetter det. VAE-en tegner to ruter som står inntil
  hverandre utydelig, og øyet leser hvert bilde for seg.
- **Rett inn i veggen.** Agenten står i (2, 7) med målet rett over. Øyet ser riktig, men den lineære
  controlleren gir "høyre" litt høyere skår enn "opp" og går inn i veggen. Den blir stående, ser
  det samme bildet igjen og gjør det samme i 50 skritt. Controlleren har ingen måte å merke at den
  står fast.

Juks-grunnlinjen som går mot målet og unngår hindringer, kommer seg løs fordi den velger tilfeldig
mellom like gode handlinger. Vi prøvde det samme på steg 7 (trekke handlingen fra softmax av
skårene): pendlingen forsvinner, men krasjene øker til 17–26 %. Tilfeldighet skjuler problemet
i stedet for å løse det.

### D57. Sporing: et Bayes-filter for agenten og et minne om hvor den har vært
Vår egen løsning, ikke fra artikkelen (`worldmodels/mdnrnn/tracker.py`). Agenten vet hvilken handling
den nettopp tok, og et skritt flytter den høyst én celle. Øyets hukommelse holder nå tre ting:

- **Mål** (som før, D49): summen av log-sannsynlighetene fra alle bildene.
- **Agent**: en tro om agentcellen. Etter hver handling flyttes troen med handlingen (inn i veggen
  betyr å bli stående). Når neste bilde kommer, ganges troen med det øyet ser, og med at agenten
  ikke står på målet (da hadde episoden vært over). Et bilde som sier "agenten hoppet to celler"
  blir dermed nesten ignorert.
- **Besøk**: troen om agentcellen fra hvert skritt, summert med glemsel 0,9 per skritt.

Controlleren får fire nye tall: for hver handling, hvor mye agenten nylig har vært i cellen den
handlingen fører til. Å gå tilbake dit den kom fra gir ca. 0,9. Å gå inn i veggen gir over 1, fordi
agenten blir stående der den er. Posisjonene i synet og agentkartet til nærsynet (fremsynet) kommer
nå fra filteret i stedet for fra enkeltbildet.

Alt regnes ut fra øyet til M og agentens egne handlinger. Derfor virker det likt i drømmen, der
bildene er M sine spådommer, og i det ekte miljøet. Controlleren er fortsatt lineær (52 parametre i bruk,
mot 36 i steg 7), og den lærer selv hvor mye besøk skal telle.

To små ting underveis:
- **Gulv på "ikke på målet".** I drømmen kan M tegne agenten oppå målet. Da ble troen null overalt
  og fitness NaN. Vi gir faktoren samme gulv som øyet (0,001).
- **Drømmen bruker også filteret.** Nærsynet spår hendelsene i drømmen ut fra filterets agentkart
  for alle controllere, siden det er M sin beste gjetning. Det endrer drømmetallene for steg 7 litt
  i evalueringen (lovet mål 59 % → 57 %, lovet krasj 10 % → 15 %), men ikke hvordan steg 7 spiller.

### D58. Først en håndlagt test, så lært i drømmen
Før vi trente noe, la vi sporing til steg 7-controlleren med uendrede vekter og en håndlagt vekt på
−0,1 for besøk (på 1000 brett, seeds 100000–100999):

| | Mål | Krasj | Avkortet |
|---|---:|---:|---:|
| Steg 7 | 74 % | 11 % | 16 % |
| + filteret alene | 83 % | 9 % | 8 % |
| + filteret og besøk | 89 % | 11 % | 0,2 % |

Begge delene hjelper. Filteret alene halverer pendlingen, og besøksminnet fjerner resten. Den
håndlagde varianten står i evalueringen som diagnostikk.

Deretter trente vi en ny controller fra null i drømmen, med samme oppsett som i steg 7 (CMA-ES,
200 generasjoner, formet belønning, to frø) og `--track`. Begge frøene nådde 88–93 % mål ved
kontrollene underveis. Vi valgte frø 1 fordi drømmen ga den høyest fitness (+0,99 mot +0,95).

### D59. Resultat: 93 % mål, 7 % krasj, ingen pendling
På de samme 2000 brettene som i steg 5–7:

| | Mål | Krasj | Avkortet | Avkastning |
|---|---:|---:|---:|---:|
| Steg 7 | 75,4 % | 11,1 % | 13,6 % | +0,54 |
| Steg 7 + sporing, håndlagt (diagnostikk) | 89,8 % | 10,1 % | 0,1 % | +0,75 |
| **Steg 8: sporing** | **92,7 %** | **7,0 %** | **0,3 %** | **+0,81** |
| Juks: mot målet, unngår hindringer | 98,5 % | 0 % | 1,6 % | +0,92 |

- Parvis mot steg 7: +17 poeng mål (369 brett bare steg 8 klarte, 24 bare steg 7) og −4 poeng
  krasj. Mot den håndlagde varianten: +3 poeng mål og −3 poeng krasj, så det lønte seg å la
  drømmen finne vektene.
- 6 avkortede episoder av 2000, ingen av dem pendling (mot 266).
- Øyet med filter ser agenten riktig i 98 % av skrittene, mot 74 % uten.
- Drømmen er nesten helt ærlig: den lover 95 % mål og 3,3 % krasj innen 10 skritt, og 95 % og
  2,7 % skjer.
- Ved 8 eller flere skritt til målet: 85 % mål, mot 62 % i steg 7.

### D60. Det som gjenstår
- **Krasj:** 7 % av episodene, de fleste i de første skrittene (76 i skritt 1–2, 50 i skritt 3–9).
  I de første skrittene har filteret bare sett ett bilde, så det kan ikke rette øyet ennå.
- **Avstand til juksen:** 6 poeng under grunnlinjen som leser miljøet direkte. Hullet er nesten
  bare krasj.

## Steg 9: vis drømmen

Martin: "vis drømmen først også kan vi legge til bevegelige hindringer etterpå".

### D61. Én selvstendig HTML-fil, uten byggeverktøy
`python -m worldmodels.dreamview` spiller noen brett med sluttagenten, tar opp alt underveis
(`dreamview/record.py`) og skriver dataene inn i malen `dreamview/page.html`. Resultatet er én fil,
`docs/drom/index.html` (0,7 MB), som virker uten server, rammeverk eller nett, bortsett fra skriftene.

- **Bare modellenes bilder lagres som bilder.** Det ekte brettet tegnes i nettleseren fra
  posisjonene. VAE-rekonstruksjonene og drømmebildene for ett brett ligger stablet i én PNG.
- **Fargene rundes til 16 nivåer.** Bildene fra modellene er uskarpe, så det synes ikke, men PNG-ene
  blir omtrent seks ganger mindre (384 kB → 61 kB for et brett med 55 bilder).
- **Ingen nye avhengigheter.** PNG-ene skrives med den samme lille zlib-koderen som før
  (`env/preview.py`, nå med `png_bytes`).

### D62. Hva siden viser, og hvordan drømmen sammenlignes med virkeligheten
I hvert ekte skritt starter en ny drøm fra agentens tilstand akkurat da, med samme regler som under
trening: temperatur 0, hendelser fra nærsynet og sporing. Drømmen varer i ti skritt, eller til den er
over både i drømmen og i virkeligheten.

Virkeligheten ved siden av tar **de samme handlingene** som drømmen valgte, i en kopi av miljøet. Den
spiller altså blindt etter drømmen og retter seg ikke etter det den ser. Det gjør sammenligningen
rettferdig mot drømmen: der de skiller lag, er det modellen som tok feil, ikke agenten som valgte
annerledes.

Brettene velges automatisk fra de 1000 første evalueringsbrettene (seeds fra 200000), de første i
hver gruppe: 4 omveier rundt hindringer, 3 brett steg 7 pendlet på, 2 rett fram og 2 krasj.

### D63. Hva vi ser i drømmene
Over alle 65 drømmene på de 11 brettene (et lite og ikke tilfeldig utvalg, så bare en pekepinn):

- Drømmen spår riktig utfall (mål, krasj eller ingen av delene innen ti skritt) i 83 %.
- Den lover mål i 85 % av drømmene, og de samme handlingene gir mål i 80 % i virkeligheten.
- Den undervurderer krasj: 5 % lovet mot 17 % i virkeligheten. 11 drømmer endte i krasj i den blinde
  virkeligheten. 6 av dem skjedde i drømmeskritt 1–2, der drømmen ga krasj 2–89 % (snitt 28 %), og
  5 skjedde senere (skritt 3–10), der drømmen ga krasj 0–14 %. Utover i drømmen blir bildet uskarpt,
  og agenten kan gå rett gjennom en hindring i drømmen. I det ekte spillet ser agenten et nytt bilde
  hvert skritt og retter seg, så dette er et mål på hvor langt drømmen holder, ikke krasjraten til
  agenten (7 %).
- Øyet alene så agenten i feil celle i 3 av 65 ekte skritt, alle på omveiene. Filteret fra steg 8
  var riktig i alle 65. Med valget "Øyet alene" kan man se de tre tilfellene.
