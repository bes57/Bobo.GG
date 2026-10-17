"""
Chronicle-on-Neon split for the Champions Shanghai Grand Finals preview.

For every map Vitality played at Champions 2026, record which agent Chronicle
played and whether VIT won the map, then split VIT's map record by
"Chronicle on Neon" vs "Chronicle on anything else". Agents come from the VLR
overview page per player cell (same parse the comps collector uses). Writes the
result into data/article_gf_preview.json under "chronicle".

Run from the repo root:  python3 scrapers/BuildGFChronicleNeon.py
"""
import os, sys, re, csv, json, time
from collections import defaultdict
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scrapers.enriched import vlr_client as C                       # noqa: E402
from scrapers.enriched.parse import _game_containers, _map_name, _player_cell  # noqa: E402

ORG, PLAYER, EVENT_CSV = "VIT", "Chronicle", os.path.join(ROOT, "data", "maps", "2026_champions.csv")
OUT = os.path.join(ROOT, "data", "article_gf_preview.json")
AGENT_LABEL = {"kayo": "KAY/O"}


def pretty_agent(a):
    return AGENT_LABEL.get(a, a.capitalize())


def main():
    # VIT's Champions matches, in play order (match id ascending ~ chronological, then by map num).
    mids = []
    for r in csv.DictReader(open(EVENT_CSV)):
        if r["Org"] == ORG and r["MatchID"] not in mids:
            mids.append(r["MatchID"])
    results = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "match_results.csv"))):
        results[(r["MatchID"], r["MapNum"])] = (r["WinnerOrg"], r["Score"], r["MatchName"])

    maps = []
    for mid in mids:
        html = C.fetch(f"{C.BASE}/{mid}")
        soup = BeautifulSoup(html, "html.parser")
        for gid, g in _game_containers(soup):
            agents = defaultdict(dict)          # org -> {player: agent}
            cells = g.select(".ovw-cell.mod-player, td.mod-player")
            imgs_all = g.select("img[src*='/agents/']")
            for idx, pcell in enumerate(cells):
                agimg = pcell.select_one(".ovw-agents img")
                if agimg is None:
                    row = pcell.find_parent("tr")
                    agimg = row.select_one("td.mod-agents img") if row else None
                if agimg is None:
                    # VLR's grid layout: the icon lives in a sibling cell of the
                    # same row, so climb to the smallest ancestor holding exactly
                    # one agent icon.
                    anc = pcell.parent
                    while anc is not None and anc is not g:
                        found = anc.select("img[src*='/agents/']")
                        if len(found) == 1:
                            agimg = found[0]; break
                        if len(found) > 1:
                            break
                        anc = anc.parent
                if agimg is None and len(imgs_all) == len(cells):
                    agimg = imgs_all[idx]        # last resort: same DOM order
                if agimg is None:
                    continue
                name, org, _pid = _player_cell(pcell)
                m = re.search(r"/agents/([^./]+)", agimg.get("src", ""))
                if org and name and m:
                    agents[org][name.strip()] = m.group(1)
            if ORG not in agents:
                continue
            res = results.get((mid, str(gid)))
            if not res:
                continue
            winner, score, stage = res
            opp = next((o for o in agents if o != ORG), "")
            ag = next((a for p, a in agents[ORG].items() if p.lower() == PLAYER.lower()), None)
            won = winner == ORG
            ws, ls = score.split("-") if "-" in score else (score, "")
            maps.append({"match_id": mid, "map_num": str(gid),
                         "map": re.sub(r"PICK$", "", _map_name(g) or "").strip(),
                         "opp": opp, "stage": stage, "agent": ag,
                         "agent_label": pretty_agent(ag) if ag else "?",
                         "won": won, "score": f"{ws}-{ls}" if won else f"{ls}-{ws}"})
        time.sleep(0.6)

    neon = [m for m in maps if m["agent"] == "neon"]
    other = [m for m in maps if m["agent"] != "neon"]
    def rec(lst):
        w = sum(1 for m in lst if m["won"]); n = len(lst)
        return {"w": w, "l": n - w, "n": n, "pct": round(w / n, 4) if n else None}
    chron = {"player": PLAYER, "org": ORG, "event": "Champions Shanghai",
             "neon": rec(neon), "other": rec(other), "maps": maps,
             "other_agents": sorted({m["agent_label"] for m in other})}
    d = json.load(open(OUT))
    d["chronicle"] = chron
    json.dump(d, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in chron.items() if k != "maps"}, indent=1))
    for m in maps:
        print(f'  {m["match_id"]} {m["stage"]:<32} {m["map"]:<8} vs {m["opp"]:<5} {m["agent_label"]:<8} {"W" if m["won"] else "L"} {m["score"]}')
    print("wrote", OUT)


if __name__ == "__main__":
    main()
