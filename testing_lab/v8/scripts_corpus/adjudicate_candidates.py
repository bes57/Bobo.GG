"""agent:corpus — fetch candidate event pages + year hubs; emit facts for the
inclusion decisions (exact dates with year, prize, participant orgs, and how
many participants are franchised VCT orgs). Sequential via vlr_client.fetch.
Output: scratchpad candidate_facts.json + hub_events.json
"""
import json
import os
import re
import sys
import time

ROOT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, ROOT)
from bs4 import BeautifulSoup  # noqa: E402
from scrapers.enriched.vlr_client import fetch  # noqa: E402
from scrapers.RefreshLiveData import VLR_NAME_TO_ORG  # noqa: E402
from scrapers.BuildMapRatings import TEAM_REGIONS  # noqa: E402

CACHE = ("/private/tmp/claude-501/-Users-benny-es1-PythonTest/"
         "1a0d507c-1768-436d-a3aa-05cca76a816a/scratchpad")

NAME_TO_ORG = dict(VLR_NAME_TO_ORG)
NAME_TO_ORG.update({
    "Giants Gaming": "GIA", "Giants": "GIA", "GiantX": "GX",
    "KOI": "MKOI", "Movistar KOI": "MKOI",
    "Apeks": "APK", "TALON": "TLN", "Talon Esports": "TLN", "TALON Esports": "TLN",
    "Bleed eSports": "BLD", "Bleed Esports": "BLD", "BLEED": "BLD",
    "2Game Esports": "2G", "Ulf Esports": "ULF", "ULF Esports": "ULF",
    "Nongshim RedForce Academy": "", "T1 Academy": "",
})
NAME_CI = {k.lower(): v for k, v in NAME_TO_ORG.items() if v}

# (vlr_event_id, slug) candidates needing facts
CANDIDATES = [
    ("2449", "esports-world-cup-2025"),
    ("1658", "champions-tour-2023-americas-last-chance-qualifier"),
    ("1659", "champions-tour-2023-emea-last-chance-qualifier"),
    ("1660", "champions-tour-2023-pacific-last-chance-qualifier"),
    ("1664", "champions-tour-2023-champions-china-qualifier"),
    ("2894", "china-evolution-series-2026-act-1"),
    ("2339", "china-evolution-series-act-1"),
    ("2450", "china-evolution-series-act-2-x-asian-champions-league"),
    ("2590", "china-evolution-series-act-3"),
    ("2720", "china-evolution-series-epilogue"),
    ("2402", "hero-esports-asian-champions-league-2025"),
    ("2602", "red-bull-home-ground-2025"),
    ("2666", "spotlight-series-2025-pacific-x-ges-asia"),
    ("2669", "spotlight-series-2025-americas"),
    ("2667", "ten-global-invitational-2025"),
    ("2740", "valorant-radiant-international-invitational"),
    ("2734", "shanghai-esports-masters-2025"),
    ("2735", "china-esports-festival-super-champions-cup"),
    ("2171", "red-bull-home-ground-5"),
    ("2212", "ludwig-x-tarik-invitational-3"),
    ("1953", "ludwig-x-tarik-invitational-2"),
    ("2291", "vct-off-season-spotlight-series-2024-pacific"),
    ("2260", "vct-off-season-spotlight-series-2024-americas"),
    ("2207", "vct-off-season-spotlight-series-2024-emea"),
    ("2219", "ten-valorant-asia-invitational"),
    ("2228", "valorant-radiant-asia-invitational"),
    ("2234", "fgc-invitational-2024"),
    ("2265", "shanghai-esports-masters-2024"),
    ("1453", "ludwig-x-tarik-invitational"),
    ("1752", "red-bull-home-ground-4"),
    ("1911", "convergence-2023"),
    ("1868", "sentinels-invitational-2023"),
    ("1915", "sean-gares-off-season-showdown"),
    ("1807", "ten-global-invitational-2023"),
    ("1747", "china-evolution-series-act-1-variation"),
    ("1825", "china-evolution-series-act-2-selection"),
    ("1880", "china-evolution-series-act-3-heritability"),
]

HUBS = ["vct-2021", "vct-2022", "vct-2023", "vct-2024", "vct-2026"]


def get_html(url, cache_name):
    path = os.path.join(CACHE, "html_cache", cache_name)
    if os.path.exists(path):
        return open(path).read()
    html = fetch(url)
    with open(path, "w") as f:
        f.write(html)
    time.sleep(1.0)
    return html


def parse_event_page(html):
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.select_one("h1.wf-title")
    title = title_el.get_text(strip=True) if title_el else ""
    desc = {}
    for item in soup.select(".event-desc-item"):
        lab = item.select_one(".event-desc-item-label")
        val = item.select_one(".event-desc-item-value")
        if lab and val:
            desc[lab.get_text(strip=True).lower()] = val.get_text(" ", strip=True)
    teams = []
    for tn in soup.select(".event-team-name"):
        t = tn.get_text(strip=True)
        if t and t not in teams:
            teams.append(t)
    return title, desc, teams


def main():
    facts = {}
    for eid, slug in CANDIDATES:
        url = f"https://www.vlr.gg/event/{eid}/{slug}"
        try:
            html = get_html(url, f"event_{eid}.html")
        except Exception as e:
            facts[eid] = {"error": str(e), "url": url}
            print(f"{eid} FETCH FAIL: {e}", flush=True)
            continue
        title, desc, teams = parse_event_page(html)
        orgs = []
        for t in teams:
            o = NAME_CI.get(t.lower())
            if o and o in TEAM_REGIONS and o not in orgs:
                orgs.append(o)
        facts[eid] = {
            "url": url, "title": title,
            "dates": desc.get("dates", ""), "prize": desc.get("prize pool", desc.get("prize", "")),
            "teams": teams, "n_teams": len(teams),
            "franchised_orgs": orgs, "n_franchised": len(orgs),
        }
        print(f"{eid} {title[:45]:45} | {desc.get('dates',''):28} | teams={len(teams):3} "
              f"franchised={len(orgs):2} {orgs[:8]}", flush=True)
    with open(os.path.join(CACHE, "candidate_facts.json"), "w") as f:
        json.dump(facts, f, indent=1)

    hubs = {}
    for hub in HUBS:
        try:
            html = get_html(f"https://www.vlr.gg/{hub}", f"hub_{hub}.html")
        except Exception as e:
            hubs[hub] = {"error": str(e)}
            continue
        soup = BeautifulSoup(html, "html.parser")
        evs = []
        for a in soup.select("a[href*='/event/']"):
            m = re.match(r"/event/(\d+)/([^/?#]+)", a.get("href", ""))
            if not m:
                continue
            txt = " ".join(a.get_text(" ", strip=True).split())
            evs.append({"vlr_event_id": m.group(1), "slug": m.group(2), "text": txt[:110]})
        hubs[hub] = evs
        print(f"{hub}: {len(evs)} events", flush=True)
    with open(os.path.join(CACHE, "hub_events.json"), "w") as f:
        json.dump(hubs, f, indent=1)


if __name__ == "__main__":
    main()
