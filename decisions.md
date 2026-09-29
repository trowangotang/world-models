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
