"""
LOUD's per-series VLR ratings in the Champions Shanghai playoffs, for the
Grand Finals preview's "My Prediction" graphic. Reads data/series/2026_champions.csv
(one row per player per series) + data/match_results.csv for stage/result and
writes "loud_po" into data/article_gf_preview.json.

Run from the repo root:  python3 scrapers/BuildGFLoudPlayoffs.py
"""
import os, csv, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "article_gf_preview.json")
ORG = "LOUD"
SHORT = {"Upper Quarterfinals": "Upper QF", "Upper Semifinals": "Upper SF", "Upper Final": "Upper Final",
         "Lower Final": "Lower Final", "Grand Final": "Grand Final"}


def main():
    results = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "match_results.csv"))):
        if r["MapNum"] == "all":
            results[r["MatchID"]] = r
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "series", "2026_champions.csv")))
            if r["MapNum"] == "all" and r["MatchID"] in results and results[r["MatchID"]]["MatchName"].startswith("Playoffs")]
    mids = sorted({r["MatchID"] for r in rows if r["Org"] == ORG}, key=int)
    matches = []
    for mid in mids:
        res = results[mid]
        opp = next((r["Org"] for r in rows if r["MatchID"] == mid and r["Org"] != ORG), "")
        stage = res["MatchName"].replace("Playoffs: ", "")
        won = res["WinnerOrg"] == ORG
        ws, ls = res["Score"].split("-")
        matches.append({"match_id": mid, "stage": stage, "short": SHORT.get(stage, stage), "opp": opp,
                        "won": won, "score": f"{ws}-{ls}" if won else f"{ls}-{ws}"})
    players = {}
    for r in rows:
        if r["Org"] != ORG:
            continue
        players.setdefault(r["Player"], {})[r["MatchID"]] = {"rating": float(r["R2.0"]), "acs": float(r["ACS"] or 0)}
    # top-2 per match
    top2 = {m["match_id"]: sorted(players, key=lambda p: -players[p].get(m["match_id"], {}).get("rating", 0))[:2] for m in matches}
    out_players = []
    for p, d in players.items():
        rs = [d[m["match_id"]]["rating"] for m in matches if m["match_id"] in d]
        out_players.append({"player": p, "ratings": {m["match_id"]: d.get(m["match_id"], {}).get("rating") for m in matches},
                            "avg": round(sum(rs) / len(rs), 3) if rs else None,
                            "top2_count": sum(1 for m in matches if p in top2[m["match_id"]])})
    out_players.sort(key=lambda x: -(x["avg"] or 0))
    blob = {"org": ORG, "matches": matches, "players": out_players, "top2": top2}
    d = json.load(open(OUT)); d["loud_po"] = blob
    json.dump(d, open(OUT, "w"), indent=2)
    for m in matches:
        print(m["short"], "vs", m["opp"], "W" if m["won"] else "L", m["score"], "| top2:", top2[m["match_id"]])
    for p in out_players:
        print(f'  {p["player"]:<8}', [p["ratings"][m["match_id"]] for m in matches], "avg", p["avg"], "top2x", p["top2_count"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
