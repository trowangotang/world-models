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
