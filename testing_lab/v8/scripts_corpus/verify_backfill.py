"""agent:corpus — verification per preregister.corpus.md.

Mechanical (every backfilled series):
  data-dest events:  scrape-time parse (scrape_verify_<id>.json, from cached
                     HTML) must equal BuildMatchResults' independently
                     re-fetched rows in data/match_results.csv — per-map
                     winner+score AND the series 'all' row. Date (ET day from
                     the scrape-time ts) must land inside the registry window
                     +/- 1 day.
  prefranchise:      per-map winner must be one of the two orgs on that map's
                     rows in maps_<id>.csv; series score must equal the tally
                     of per-map winners; ts within window +/- 1 day.

Sample (per event): min(10, n) series drawn without replacement, seed 20260728
(crn.json absent at prereg time — disclosed), re-fetched LIVE via
vlr_client.fetch and re-parsed; winner+score must match match_results.csv
(data) / scrape_verify json (prefranchise).

Usage: verify_backfill.py mechanical | sample
Output: testing_lab/v8/stats/verification_report.json (merged), loud non-zero
exit on any failure.
"""
import csv
import json
import os
import random
import re
import sys
import time
import datetime

ROOT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scrapers"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V8STATS = os.path.join(ROOT, "testing_lab", "v8", "stats")
PREFR = os.path.join(ROOT, "testing_lab", "v8", "data", "prefranchise")
REPORT = os.path.join(V8STATS, "verification_report.json")
SEED = 20260728

from MoreTestingMaybeFiles import ALL_EVENTS  # noqa: E402

DATA_EVENT_WINDOWS = {e["id"]: (e.get("start"), e.get("end")) for e in ALL_EVENTS}


def load_report():
    return json.load(open(REPORT)) if os.path.exists(REPORT) else {}


def save_report(rep):
    with open(REPORT, "w") as f:
        json.dump(rep, f, indent=1)


def load_verify_files():
    out = {}
    for fn in sorted(os.listdir(V8STATS)):
        m = re.match(r"scrape_verify_(.+)\.json$", fn)
        if m:
            out[m.group(1)] = json.load(open(os.path.join(V8STATS, fn)))
    return out


def window_for(eid, verify):
    if eid in DATA_EVENT_WINDOWS and DATA_EVENT_WINDOWS[eid][0]:
        s, e = DATA_EVENT_WINDOWS[eid]
    else:  # prefranchise: window from its registry file
        reg = json.load(open(os.path.join(PREFR, "registry.json")))
        ent = next(x for x in reg["events"] if x["id"] == eid)
        s, e = ent["start"], ent["end"]
    sd = datetime.date.fromisoformat(s) - datetime.timedelta(days=1)
    ed = datetime.date.fromisoformat(e) + datetime.timedelta(days=1)
    return sd, ed


def mechanical():
    import pandas as pd
    verify = load_verify_files()
    mr_path = os.path.join(ROOT, "data", "match_results.csv")
    mr = pd.read_csv(mr_path, dtype=str)
    mr["MatchID"] = mr["MatchID"].astype(str).str.strip()
    mr_idx = {}
    for _, r in mr.iterrows():
        mr_idx.setdefault(r["MatchID"], {})[str(r["MapNum"])] = (r["WinnerOrg"], r["Score"])

    prefr_ids = set()
    if os.path.exists(os.path.join(PREFR, "registry.json")):
        reg = json.load(open(os.path.join(PREFR, "registry.json")))
        prefr_ids = {x["id"] for x in reg["events"]}

    results = {}
    total_series = total_maps = 0
    failures = []
    for eid, matches in verify.items():
        is_prefr = eid in prefr_ids
        sd, ed = window_for(eid, verify)
        ok_series = 0
        maps_csv = {}
        if is_prefr:
            path = os.path.join(PREFR, f"maps_{eid}.csv")
            with open(path) as f:
                for row in csv.DictReader(f):
                    maps_csv.setdefault(str(row["MatchID"]), {}).setdefault(
                        str(row["MapNum"]), set()).add(row["Org"])
        for mid, rec in matches.items():
            errs = []
            # date-in-window (ET day of the scrape-time ts)
            ts = rec.get("utc_ts")
            if not ts:
                errs.append("no ts captured")
            else:
                try:
                    d = datetime.date.fromisoformat(ts[:10])
                    if not (sd <= d <= ed):
                        errs.append(f"date {ts[:10]} outside window {sd}..{ed}")
                except ValueError:
                    errs.append(f"bad ts {ts}")
            if is_prefr:
                for mapnum, (w, score) in rec["maps"].items():
                    total_maps += 1
                    orgs = maps_csv.get(mid, {}).get(mapnum, set())
                    if w not in orgs:
                        errs.append(f"map {mapnum}: winner {w} not in maps CSV orgs {sorted(orgs)}")
                tallied = {}
                for w, _ in rec["maps"].values():
                    tallied[w] = tallied.get(w, 0) + 1
                sw, ss = rec["series"]
                exp = f"{tallied.get(sw, 0)}-{sum(tallied.values()) - tallied.get(sw, 0)}"
                if max(tallied, key=tallied.get) != sw or exp != ss:
                    errs.append(f"series {sw} {ss} != map tally {tallied}")
            else:
                got = mr_idx.get(mid)
                if got is None:
                    errs.append("MatchID missing from match_results.csv")
                else:
                    for mapnum, (w, score) in rec["maps"].items():
                        total_maps += 1
                        g = got.get(mapnum)
                        if g is None:
                            errs.append(f"map {mapnum} missing in match_results")
                        elif (g[0], g[1]) != (w, score):
                            errs.append(f"map {mapnum}: mine {w} {score} vs builder {g[0]} {g[1]}")
                    sw, ss = rec["series"]
                    g = got.get("all")
                    if g is None:
                        errs.append("series 'all' row missing in match_results")
                    elif (g[0], g[1]) != (sw, ss):
                        errs.append(f"series: mine {sw} {ss} vs builder {g[0]} {g[1]}")
            total_series += 1
            if errs:
                failures.append({"event": eid, "match": mid, "errors": errs})
            else:
                ok_series += 1
        results[eid] = {"n_series": len(matches), "ok": ok_series}
    rep = load_report()
    rep["mechanical"] = {"per_event": results, "total_series": total_series,
                         "total_maps_checked": total_maps,
                         "failures": failures, "pass": not failures}
    save_report(rep)
    print(json.dumps(rep["mechanical"]["per_event"], indent=1))
    print(f"TOTAL series {total_series}, maps {total_maps}, failures {len(failures)}")
    for f_ in failures:
        print("FAIL", f_)
    sys.exit(1 if failures else 0)


def sample():
    from bs4 import BeautifulSoup
    from scrapers.enriched.vlr_client import fetch
    from scrape_backfill import parse_results
    verify = load_verify_files()
    rep = load_report()
    samp = rep.get("sample", {"per_event": {}, "failures": []})
    failures = samp["failures"]
    for eid in sorted(verify):
        if eid in samp["per_event"] and samp["per_event"][eid].get("done"):
            continue
        mids = sorted(verify[eid].keys())
        rng = random.Random(f"{SEED}:{eid}")
        pick = mids if len(mids) <= 10 else rng.sample(mids, 10)
        ok = 0
        for mid in pick:
            html = fetch(f"https://www.vlr.gg/{mid}/")
            soup = BeautifulSoup(html, "html.parser")
            per_map, series_res, _ = parse_results(soup)
            want = verify[eid][mid]
            errs = []
            if series_res != want["series"]:
                errs.append(f"series live {series_res} vs recorded {want['series']}")
            if per_map != want["maps"]:
                errs.append(f"maps live {per_map} vs recorded {want['maps']}")
            if errs:
                failures.append({"event": eid, "match": mid, "errors": errs})
            else:
                ok += 1
            time.sleep(0.9)
        samp["per_event"][eid] = {"n": len(pick), "ok": ok, "done": True}
        rep["sample"] = samp
        save_report(rep)
        print(f"{eid}: {ok}/{len(pick)}", flush=True)
    samp["pass"] = not failures
    rep["sample"] = samp
    save_report(rep)
    for f_ in failures:
        print("FAIL", f_)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    {"mechanical": mechanical, "sample": sample}[sys.argv[1]]()
