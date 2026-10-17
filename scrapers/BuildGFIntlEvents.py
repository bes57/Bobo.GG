"""
International-event attendance for the Champions Shanghai Grand Finals preview.

Reads each player's VLR "Event Placements" list and keeps VCT internationals
only (Masters, Champions, LOCK//IN) from 2021 on. VLR does not list the ongoing
event, so Champions Shanghai 2026 is appended explicitly for everyone here (all
five are in the grand final). Writes "intl_events" into
data/article_gf_preview.json.

Run from the repo root:  python3 scrapers/BuildGFIntlEvents.py
"""
import os, sys, re, json, time
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scrapers.enriched import vlr_client as C   # noqa: E402

OUT = os.path.join(ROOT, "data", "article_gf_preview.json")
PLAYERS = [("Derke", 5022), ("Chronicle", 458), ("Jamppi", 9780), ("Sayonara", 33902), ("PROFEK", 5554)]
CURRENT = {"event": "Valorant Champions 2026", "short": "Champions", "year": 2026, "team": "", "placement": "Grand Final", "current": True}
TEAM = {"Derke": "Team Vitality", "Chronicle": "Team Vitality", "Jamppi": "Team Vitality", "Sayonara": "Team Vitality", "PROFEK": "Team Vitality",
        "erde": "LOUD", "Darker": "LOUD", "lukxo": "LOUD", "tkzin": "LOUD", "DaviH": "LOUD"}

INTL = re.compile(r"^(Champions Tour|Valorant|VCT)\b.*(Masters|Champions|LOCK//IN|LOCK\u2215\u2215IN)", re.I)
# 2021\'s regional "Stage 1: Masters" events (CIS/EU/NA/...) were not internationals.
REGIONAL = re.compile(r"Stage 1: Masters$", re.I)
LOUD_CHECK = [("erde", 29242), ("Darker", 10557), ("lukxo", 28716), ("tkzin", 41224), ("DaviH", 10885)]


def short_name(name):
    n = re.sub(r"\s+", " ", name.strip())
    if "Masters" in n:
        n = re.sub(r"^.*?Masters\s*-?\s*", "Masters ", n)
    n = re.sub(r"^Champions Tour\s*(\d{4})?:?\s*", "", n)
    n = re.sub(r"^Valorant\s+", "", n)
    n = re.sub(r"\s+\d{4}$", "", n).strip()
    return n


def parse(html):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a.player-event-item"):
        name_el = a.select_one(".text-of")
        name = name_el.get_text(" ", strip=True) if name_el else ""
        if not name or not INTL.search(name) or REGIONAL.search(name):
            continue
        divs = a.find_all("div", recursive=False)
        year = re.search(r"(20\d\d)", divs[-1].get_text(" ", strip=True)) if divs else None
        year = int(year.group(1)) if year else None
        if not year or year < 2021:
            continue
        light = a.select_one(".ge-text-light")
        placement = re.sub(r"\s+", " ", light.get_text(" ", strip=True)) if light else ""
        placement = placement.replace("–", "-").replace("&ndash;", "-")
        # team = trailing text of the first placement block
        block = a.select_one("div[style*='margin-top']")
        team = ""
        if block:
            texts = [t.strip() for t in block.find_all(string=True, recursive=False) if t.strip()]
            team = texts[-1] if texts else ""
        out.append({"event": name, "short": short_name(name), "year": year, "team": team,
                    "placement": placement, "href": a.get("href", "")})
    # oldest first
    # within a year: LOCK//IN, first Masters, second Masters, Champions
    order = {"Reykjavík": 1, "Berlin": 2, "Copenhagen": 2, "Tokyo": 1, "Madrid": 1, "Shanghai": 2,
             "Bangkok": 1, "Toronto": 2, "Santiago": 1, "London": 2}
    def _rank(e):
        sh = e["short"]
        if "LOCK//IN" in sh: return 0
        if sh.startswith("Masters"): return order.get(sh.replace("Masters", "").strip(), 1)
        return 9
    out.sort(key=lambda e: (e["year"], _rank(e)))
    return out


def main():
    heads = {}
    try:
        for url, img in json.load(open(os.path.join(ROOT, "data", "headshots.json"))).items():
            m = re.search(r"/player/(\d+)/", url)
            if m:
                heads[int(m.group(1))] = img
    except Exception:
        pass

    def collect(group):
        res = {}
        for name, pid in group:
            html = C.fetch(f"https://www.vlr.gg/player/{pid}/{name.lower()}")
            evs = parse(html)
            if not heads.get(pid):
                soup = BeautifulSoup(html, "html.parser")
                img = soup.select_one(".player-header img, .wf-avatar img")
                src = (img.get("src") or "") if img else ""
                if src and "/sil." not in src:
                    heads[pid] = ("https:" + src) if src.startswith("//") else src
            cur = dict(CURRENT); cur["team"] = TEAM.get(name, "")
            evs.append(cur)
            wins = sum(1 for e in evs if re.search(r"\b1st\b", e.get("placement", "")) and not e.get("current"))
            res[name] = {"vlr_id": pid, "count": len(evs), "titles": wins, "first": evs[0]["short"] if evs else None,
                         "head": heads.get(pid, ""), "events": evs}
            print(f"{name}: {len(evs)} intl events (incl. Champions Shanghai), {wins} titles")
            for e in evs:
                print(f"   {e['year']}  {e['short']:<32} {e.get('placement',''):<22} {e.get('team','')}")
            time.sleep(0.6)
        return res

    vit = collect(PLAYERS)
    loud = collect(LOUD_CHECK)
    d = json.load(open(OUT))
    d["intl_events"] = {"players": vit, "loud_players": loud,
                        "note": "VCT Masters/Champions/LOCK//IN since 2021, incl. Champions Shanghai 2026; 2021 regional Stage 1 Masters, EWC and off-season events excluded"}
    json.dump(d, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
