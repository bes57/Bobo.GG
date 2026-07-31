"""agent:corpus — testing_lab/v8/data/prefranchise/match_results_prefranchise.csv
Built from the scrape-time parses (scrape_verify_<id>.json) of the prefranchise
events, in data/match_results.csv schema plus an EventID column. These rows are
deliberately NOT written into data/match_results.csv (separate corpus).
"""
import csv
import json
import os

ROOT = "/Users/benny_es1/PythonTest"
V8STATS = os.path.join(ROOT, "testing_lab", "v8", "stats")
PREFR = os.path.join(ROOT, "testing_lab", "v8", "data", "prefranchise")
OUT = os.path.join(PREFR, "match_results_prefranchise.csv")

reg = json.load(open(os.path.join(PREFR, "registry.json")))
rows = []
for ev in reg["events"]:
    p = os.path.join(V8STATS, f"scrape_verify_{ev['id']}.json")
    if not os.path.exists(p):
        print(f"MISSING verify for {ev['id']} — skipped (loud)")
        continue
    v = json.load(open(p))
    for mid, rec in sorted(v.items()):
        for mapnum, (w, score) in rec["maps"].items():
            rows.append({"MatchID": mid, "MapNum": mapnum, "WinnerOrg": w,
                         "Score": score, "MatchName": rec.get("match_name", ""),
                         "EventID": ev["id"]})
        sw, ss = rec["series"]
        rows.append({"MatchID": mid, "MapNum": "all", "WinnerOrg": sw,
                     "Score": ss, "MatchName": rec.get("match_name", ""),
                     "EventID": ev["id"]})

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["MatchID", "MapNum", "WinnerOrg", "Score",
                                      "MatchName", "EventID"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")
