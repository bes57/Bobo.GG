"""Veto predictor eval: walk-forward top-1/top-3 accuracy of ban/pick steps
from map_vetos.csv, comparing (A) raw decayed rates (current style, +0.02
smoothing) vs (B) James-Stein-shrunk rates toward the concurrent global map
rate. 2025-2026 scored. Writes out/veto_eval.json."""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "..", "data")

v = pd.read_csv(os.path.join(DATA, "map_vetos.csv"))
dates = json.load(open(os.path.join(DATA, "match_dates.json")))
v["date"] = v.MatchID.astype(str).map(dates)
v = v.dropna(subset=["date"]).sort_values(["date", "MatchID", "step"])
print(f"veto rows: {len(v)}, matches: {v.MatchID.nunique()}")

# active pool at each match = the maps seen in that match's own veto
pool_by_match = v.groupby("MatchID")["map"].apply(list).to_dict()

# build per-team veto history: list of (date, action, map, step, pool)
hist = defaultdict(list)
rows = []
for mid, grp in v.groupby("MatchID", sort=False):
    pool = grp["map"].tolist()
    date = grp["date"].iloc[0]
    rem = list(pool)
    for _, r in grp.sort_values("step").iterrows():
        if r["action"] in ("ban", "pick") and len(rem) > 1:
            rows.append({"mid": mid, "date": date, "team": r["team"],
                         "action": r["action"], "map": r["map"],
                         "step": int(r["step"]), "rem": list(rem)})
        if r["map"] in rem:
            rem.remove(r["map"])
ev = pd.DataFrame(rows)
print(f"decision points: {len(ev)}")

HL = 6.0
LAM = math.log(2) / HL


def team_rates(team, action, asof, lookup):
    """Decayed rate per map: sum w*1[chose m among rem] / sum w*1[m in rem]."""
    num, den = defaultdict(float), defaultdict(float)
    for d, act, mp, rem in lookup.get(team, []):
        if d >= asof or act != action:
            continue
        w = math.exp(-LAM * (pd.Timestamp(asof) - pd.Timestamp(d)).days / 7.0)
        for m in rem:
            den[m] += w
        num[mp] += w
    return num, den


# prebuild per-team chronological decision list
lookup = defaultdict(list)
for r in ev.itertuples(index=False):
    lookup[r.team].append((r.date, r.action, r.map, r.rem))

# global decayed choice rate per map (for shrinkage target)
def global_rates(action, asof, sub):
    num, den = defaultdict(float), defaultdict(float)
    for r in sub:
        d, act, mp, rem = r
        if d >= asof or act != action:
            continue
        w = math.exp(-LAM * (pd.Timestamp(asof) - pd.Timestamp(d)).days / 7.0)
        for m in rem:
            den[m] += w
        num[mp] += w
    return num, den


all_decisions = [(r.date, r.action, r.map, r.rem) for r in ev.itertuples(index=False)]

test = ev[ev.date >= "2025-01-01"]
print(f"test decisions (2025+): {len(test)}")

results = {}
for K in (0.0, 2.0, 4.0, 8.0):
    top1 = defaultdict(int)
    top3 = defaultdict(int)
    cnt = defaultdict(int)
    # cache global rates per (action, month)
    gcache = {}
    for r in test.itertuples(index=False):
        month = r.date[:7]
        gk = (r.action, month)
        if gk not in gcache:
            gcache[gk] = global_rates(r.action, month + "-01", all_decisions)
        gnum, gden = gcache[gk]
        num, den = team_rates(r.team, r.action, r.date, lookup)
        scores = {}
        for m in r.rem:
            n_eff = den.get(m, 0.0)
            raw = num.get(m, 0.0) / n_eff if n_eff > 0 else 0.0
            grate = (gnum.get(m, 0.0) / gden.get(m, 1e-9)) if gden.get(m, 0) > 0 else 1.0 / len(r.rem)
            if K == 0.0:
                scores[m] = raw + 0.02  # current-style smoothing
            else:
                scores[m] = (n_eff * raw + K * grate) / (n_eff + K)
        rank = sorted(scores, key=scores.get, reverse=True)
        key = (r.action, min(r.step, 4))
        cnt[key] += 1
        if rank[0] == r.map:
            top1[key] += 1
        if r.map in rank[:3]:
            top3[key] += 1
    agg = {}
    t1 = sum(top1.values()) / max(sum(cnt.values()), 1)
    t3 = sum(top3.values()) / max(sum(cnt.values()), 1)
    for key in sorted(cnt):
        agg[f"{key[0]}_step{key[1]}"] = {
            "n": cnt[key], "top1": round(top1[key] / cnt[key], 4),
            "top3": round(top3[key] / cnt[key], 4)}
    results[f"K{K}"] = {"top1_all": round(t1, 4), "top3_all": round(t3, 4),
                        "by_step": agg}
    print(f"K={K}: top1={t1:.4f} top3={t3:.4f}")

with open(os.path.join(OUT, "veto_eval.json"), "w") as f:
    json.dump(results, f, indent=1)
print("saved out/veto_eval.json")
