"""Skriv report.md fra results.json-strukturen. Bare tabeller og figurer; tolkningen står i README."""

from __future__ import annotations

from pathlib import Path


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def ci(r: dict) -> str:
    return f"{pct(r['rate'])} ({pct(r['lo'], 0)}–{pct(r['hi'], 0)})".replace(" %)", " %)")


def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed_pp(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x) * 100:.1f}".replace(".", ",")


FAILURE_TEXT = {"obstacle": "krasjet", "truncated": "ble avkortet"}


def write_report(path: Path, results: dict, figures: dict) -> None:
    S = results["summaries"]
    cheat = {p["name"]: p["cheat"] for p in results["policies"]}
    lo, hi = results["seeds"]
    n = hi - lo + 1
    L = [
        "# Evaluering i det ekte miljøet", "",
        f"Alle policyer spilte de samme {n} nye brettene (seeds {lo}–{hi}), som ingen av dem har sett under "
        "trening. Tallene i parentes er 95 %-intervaller (Wilson for rater, bootstrap for avkastning). "
        "Policyer merket *juks* leser miljøets indre tilstand og er målestokker, ikke konkurrenter.", "",
        "Generert av `python -m worldmodels.evaluation`. Tolkningen står i README.", "",
        "## Hovedtabell", "",
        "| Policy | Mål | Hindring | Avkortet | Avkastning | Snittlengde |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in S.items():
        label = f"*{name}*" if cheat[name] else name
        r = s["return"]
        L.append(f"| {label} | {ci(s['goal'])} | {ci(s['obstacle'])} | {ci(s['truncated'])} | "
                 f"{num(r['mean'])} ({num(r['lo'])} – {num(r['hi'])}) | {num(s['mean_length'], 1)} |")
    L += ["", f"![Utfall]({figures['outcomes']})", ""]

    L += ["## Parvise sammenligninger på de samme brettene", "",
          "Forskjell i prosentpoeng (a − b). «Bare a» er brett der a lyktes og b ikke, og omvendt.", "",
          "| a | b | Mål, endring | Bare a / bare b | Hindring, endring |", "|---|---|---:|---:|---:|"]
    for p in results["paired"]:
        g, o = p["goal"], p["obstacle"]
        L.append(f"| {p['a']} | {p['b']} | {signed_pp(g['diff'])} ({signed_pp(g['lo'])} til {signed_pp(g['hi'])}) | "
                 f"{g['only_a']} / {g['only_b']} | {signed_pp(o['diff'])} ({signed_pp(o['lo'])} til {signed_pp(o['hi'])}) |")

    L += ["", "## Etter avstand til målet", "",
          "Målrate etter korteste vei fra start til mål (rundt hindringene).", ""]
    bins = next(iter(S.values()))["by_distance"]
    heads = [f"{b['bin'][0]}–{b['bin'][1]}" if b["bin"][1] < 30 else f"{b['bin'][0]}+" for b in bins]
    L += ["| Policy | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    L.append("| *Andel av brettene* | " + " | ".join(pct(b["goal"]["n"] / n, 0) for b in bins) + " |")
    for name, s in S.items():
        L.append(f"| {name} | " + " | ".join(pct(b["goal"]["rate"], 0) for b in s["by_distance"]) + " |")
    L += ["", f"![Mål etter avstand]({figures['goal_by_distance']})", ""]

    free_n = next(iter(S.values()))["by_blocking"]["free"]["goal"]["n"]
    L += ["## Når hindringer står i veien", "",
          f"«Fri vei»: ingen hindring i rektangelet mellom start og mål, så enhver rett vei er trygg "
          f"({pct(free_n / n, 0)} av brettene). «Hindring mellom»: minst én hindring står der en agent som går "
          "rett mot målet, kan treffe den.", "",
          "| Policy | Mål, fri vei | Mål, hindring mellom | Krasj, fri vei | Krasj, hindring mellom |", "|---|---:|---:|---:|---:|"]
    for name, s in S.items():
        f, b = s["by_blocking"]["free"], s["by_blocking"]["blocked"]
        L.append(f"| {name} | {pct(f['goal']['rate'], 0)} | {pct(b['goal']['rate'], 0)} | "
                 f"{pct(f['obstacle']['rate'], 0)} | {pct(b['obstacle']['rate'], 0)} |")

    L += ["", "## Hvordan episodene ender", "",
          "Pendling: avkortet, og innom høyst 3 ulike celler de siste 20 skrittene. Effektivitet: korteste vei "
          "delt på antall skritt brukt, for episoder som nådde målet (1,0 er perfekt).", "",
          "| Policy | Avkortet | Av dem pendling | Effektivitet | Krasj i skritt 1–2 | Krasj i skritt 3–9 | Krasj i skritt 10+ |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in S.items():
        h = {tuple(x["steps"]): x["count"] for x in s["obstacle_step"]}
        early = h[(1, 1)] + h[(2, 2)]
        mid = h[(3, 4)] + h[(5, 9)]
        late = h[(10, 19)] + h[(20, 50)]
        eff = num(s["path_efficiency"]) if s["path_efficiency"] is not None else "–"
        L.append(f"| {name} | {s['truncated_count']} | {s['truncated_with_oscillation']} | {eff} | {early} | {mid} | {late} |")

    if results.get("beliefs"):
        L += ["", "## Vet M hvor agenten og målet er?", "",
              "Mens agenten spiller: andel skritt der M sin tro peker på riktig celle, og snittfeil i celler. "
              "Kilde «minne»: troen M leser fra h før bildet er sett, så skritt 0 er en gjetning med tomt minne. "
              "Kilde «øye»: det øyet leser fra bildet i samme skritt (steg 6).", "",
              "| Agent | Kilde | Skritt | Agent riktig | Agent feil (celler) | Mål riktig | Mål feil (celler) | Episoder |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
        for name, b in results["beliefs"].items():
            source = {"eye": "øye", "tracked eye": "øye + sporing", "memory": "minne"}[b.get("source", "memory")]
            for t, r in [*b["by_step"].items(), ("alle", b["all"])]:
                L.append(f"| {name} | {source} | {t} | {pct(r['agent_exact'], 0)} | {num(r['agent_error'], 1)} | "
                         f"{pct(r['goal_exact'], 0)} | {num(r['goal_error'], 1)} | {r['n']} |")

    if results["dream_vs_real"]:
        d0 = next(iter(results["dream_vs_real"].values()))
        L += ["", "## Drøm mot virkelighet", "",
              f"Etter {d0['start_step']} ekte skritt drømmer agenten {d0['horizon']} skritt videre fra nøyaktig den "
              "tilstanden (temperatur 0, som i treningen). Så spiller den de samme skrittene i det ekte miljøet. "
              "Drømmen er ærlig når snittet av M sine sannsynligheter er nær den ekte raten. AUC sier hvor godt "
              "M skiller brettene der det skjer fra dem der det ikke skjer (0,5 er gjetting).", "",
              "| Agent | Starter | Mål: drøm | Mål: ekte | Mål: AUC | Hindring: drøm | Hindring: ekte | Hindring: AUC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, d in results["dream_vs_real"].items():
            g, o = d["goal"], d["obstacle"]
            auc = lambda x: num(x) if x is not None else "–"  # noqa: E731
            L.append(f"| {name} | {d['starts']} | {pct(g['dream'])} | {ci(g['real'])} | {auc(g['auc'])} | "
                     f"{pct(o['dream'])} | {ci(o['real'])} | {auc(o['auc'])} |")
        L += ["", f"![Drøm mot virkelighet, mål]({figures['dream_goal']})", "",
              f"![Drøm mot virkelighet, hindring]({figures['dream_obstacle']})", ""]

    if "final_paths" in figures:
        last = results.get("final") or list(results["dream_vs_real"])[-1]
        L += ["## Episoder fra sluttagenten", "",
              f"{last}. Øverste rad nådde målet, midterste traff en hindring, nederste ble avkortet. "
              "Streken går fra gul (start) til hvit (slutt), og det hvite kvadratet er der episoden endte.", "",
              f"![Veier]({figures['final_paths']})", "",
              "Den første episoden av hvert utfall, skritt for skritt (16 første bilder):", "",
              f"![Episoder]({figures['final_strips']})", ""]
    if "compare_paths" in figures:
        c = figures["compare"]
        L += ["## Samme brett, to agenter", "",
              f"Brett der {c['a']} (øverst) {FAILURE_TEXT[c.get('failure', 'obstacle')]} og {c['b']} (nederst) "
              "kom frem.", "",
              f"![Sammenligning]({figures['compare_paths']})", ""]
    path.write_text("\n".join(L))
