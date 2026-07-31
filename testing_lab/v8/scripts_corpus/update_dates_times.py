"""agent:corpus — incremental match_dates.json / match_times.json update for
NEW MatchIDs only, mirroring RefreshLiveData step 5 semantics exactly:
  match_dates[mid] = raw data-utc-ts[:10]      (ET calendar day, VLR bucketing)
  match_times[mid] = _et_walltime_to_utc(ts)    (true UTC instant)
The raw ts comes from the match HTML cached at scrape time (same page element
RefreshLiveData._scrape_date reads) — zero extra VLR load. Any new MatchID
without a cached ts is reported loudly.
"""
import json
import os
import sys

ROOT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scrapers"))
import pandas as pd  # noqa: E402
from scrapers.RefreshLiveData import _et_walltime_to_utc  # noqa: E402

V8STATS = os.path.join(ROOT, "testing_lab", "v8", "stats")
DATA = os.path.join(ROOT, "data")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    # every scrape_verify_<id>.json holds utc_ts per MatchID from my scrape
    ts_by_mid = {}
    for fn in sorted(os.listdir(V8STATS)):
        if not fn.startswith("scrape_verify_") or not fn.endswith(".json"):
            continue
        for mid, rec in json.load(open(os.path.join(V8STATS, fn))).items():
            if rec.get("utc_ts"):
                ts_by_mid[mid] = rec["utc_ts"]

    mr = pd.read_csv(os.path.join(DATA, "match_results.csv"), dtype=str)
    all_ids = {str(int(float(m))) for m in mr["MatchID"].dropna().unique()}

    dates_p = os.path.join(DATA, "match_dates.json")
    times_p = os.path.join(DATA, "match_times.json")
    dates = load(dates_p)
    times = load(times_p)

    new_ids = [m for m in all_ids if m not in dates]
    missing = []
    n_dates = n_times = 0
    for mid in sorted(new_ids):
        ts = ts_by_mid.get(mid)
        if not ts:
            missing.append(mid)
            continue
        dates[mid] = ts[:10]
        n_dates += 1
        if mid not in times:
            times[mid] = _et_walltime_to_utc(ts)
            n_times += 1

    with open(dates_p, "w") as f:
        json.dump(dates, f, indent=2)
    with open(times_p, "w") as f:
        json.dump(times, f, indent=2)
    print(f"new dates: {n_dates}, new times: {n_times}, "
          f"total dates: {len(dates)}, times: {len(times)}")
    if missing:
        print(f"LOUD FAILURE — {len(missing)} new MatchIDs lack a cached ts: {missing[:20]}")
        sys.exit(2)


if __name__ == "__main__":
    main()
