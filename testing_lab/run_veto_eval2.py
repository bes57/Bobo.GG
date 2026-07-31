"""Veto pick-model variants: opponent-awareness + own-strength exponent.
Current: score = (rate+0.02) * (0.3+own_win)^2. Variants add (1.75-opp_win)^q
and vary the own exponent. Walk-forward top-1 on 2025-26 pick steps.
Writes out/veto_eval2.json."""
import json
import math
import os
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

# real map names only
mr = pd.read_csv(os.path.join(DATA, "match_results.csv"), usecols=["MatchID", "WinnerOrg"])
maps_real = set(v["map"].value_counts()[lambda x: x > 50].index)
v = v[v["map"].isin(maps_real)]

# team map win rates (decayed HL6) from match_results + maps csv is heavy;
# approximate with veto-adjacent outcome data: use picks won? Instead reuse
# harness games via engine (already parsed).
import sys
sys.path.insert(0, HERE)
from engine import Engine

eng = Engine()
g_map = np.array([g["map_name"] for g in eng.games])
LAM = math.log(2) / 6.0

# prebuild: for each team, list of (date, map, won)
team_maps = defaultdict(list)
for g in eng.games:
    team_maps[g["winner"]].append((g["date_s"], g["map_name"], 1))
    team_maps[g["loser"]].append((g["date_s"], g["map_name"], 0))
for t in team_maps:
    team_maps[t].sort()


def winpct(team, asof):
    num, den = defaultdict(float), defaultdict(float)
    for d, m, won in team_maps.get(team, []):
        if d >= asof:
            break
        w = math.exp(-LAM * (pd.Timestamp(asof) - pd.Timestamp(d)).days / 7.0)
        den[m] += w
        num[m] += w * won
    return {m: num[m] / den[m] for m in den if den[m] > 0.4}


# pick decision points with both teams known: need opponent org per match
orgs_by_mid = defaultdict(set)
for g in eng.games:
    orgs_by_mid[g["match_id"]].add(g["winner"])
    orgs_by_mid[g["match_id"]].add(g["loser"])

team_hist = defaultdict(list)
points = []
for mid, grp in v.groupby("MatchID", sort=False):
    pool = grp["map"].tolist()
    date = grp["date"].iloc[0]
    rem = list(pool)
    for _, r in grp.sort_values("step").iterrows():
        if r["action"] == "pick" and len(rem) > 1:
            opp = [o for o in orgs_by_mid.get(mid, set()) if o != r["team"]]
            points.append({"date": date, "team": r["team"],
                           "opp": opp[0] if opp else None,
                           "map": r["map"], "rem": list(rem)})
        if r["action"] in ("ban", "pick") and len(rem) > 1:
            team_hist[r["team"]].append((date, r["action"], r["map"], list(rem)))
        if r["map"] in rem:
            rem.remove(r["map"])

test = [p for p in points if p["date"] >= "2025-01-01" and p["opp"]]
print(f"pick decisions 2025+: {len(test)}")


def pick_rates(team, asof):
    num, den = defaultdict(float), defaultdict(float)
    for d, act, mp, rem in team_hist.get(team, []):
        if d >= asof or act != "pick":
            continue
        w = math.exp(-LAM * (pd.Timestamp(asof) - pd.Timestamp(d)).days / 7.0)
        for m in rem:
            den[m] += w
        num[mp] += w
    return {m: (num.get(m, 0.0) / den[m]) if den.get(m) else 0.0 for m in den}


VARIANTS = {
    "current (own^2)": lambda rate, own, opp: (rate + 0.02) * (0.3 + own) ** 2,
    "own^1.5": lambda rate, own, opp: (rate + 0.02) * (0.3 + own) ** 1.5,
    "own^2.5": lambda rate, own, opp: (rate + 0.02) * (0.3 + own) ** 2.5,
    "own^2 x opp^0.5": lambda rate, own, opp: (rate + 0.02) * (0.3 + own) ** 2 * (1.75 - opp) ** 0.5,
    "own^2 x opp^1": lambda rate, own, opp: (rate + 0.02) * (0.3 + own) ** 2 * (1.75 - opp),
    "diff (own-opp)": lambda rate, own, opp: (rate + 0.02) * (0.8 + own - opp) ** 2,
}
results = {}
wp_cache = {}
for vn, fn in VARIANTS.items():
    t1 = t3 = n = 0
    for p in test:
        key = (p["team"], p["date"])
        if key not in wp_cache:
            wp_cache[key] = winpct(p["team"], p["date"])
        keyo = (p["opp"], p["date"])
        if keyo not in wp_cache:
            wp_cache[keyo] = winpct(p["opp"], p["date"])
        own_w = wp_cache[key]
        opp_w = wp_cache[keyo]
        rates = pick_rates(p["team"], p["date"])
        scores = {}
        for m in p["rem"]:
            own = own_w.get(m, 0.5)
            opp = opp_w.get(m, 0.5)
            scores[m] = fn(rates.get(m, 0.0), own, opp)
        rank = sorted(scores, key=scores.get, reverse=True)
        n += 1
        if rank[0] == p["map"]:
            t1 += 1
        if p["map"] in rank[:3]:
            t3 += 1
    results[vn] = {"n": n, "top1": round(t1 / n, 4), "top3": round(t3 / n, 4)}
    print(f"{vn:<20} top1={t1/n:.4f} top3={t3/n:.4f}")

with open(os.path.join(OUT, "veto_eval2.json"), "w") as f:
    json.dump(results, f, indent=1)
print("saved out/veto_eval2.json")
