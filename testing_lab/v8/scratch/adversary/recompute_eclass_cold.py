"""agent:adversary — eclass_on_v6 fade survival + H3 cold-start buckets:
my own masks, overlap audit, fragility, jackknife."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
sys.path.insert(0, TL)

frame = pd.read_csv(os.path.join(V8, "data", "frame_expanded", "series.csv"))
hold = (frame.date > "2024-12-31").values
ev = frame.event_id.values
fmts = frame.fmt.values
EPS = 1e-9


def ll(p):
    return -np.log(np.clip(p, EPS, 1.0))


z6 = np.load(os.path.join(V8, "scratch", "bias_h3", "v6_baseline.npz"))
p0 = z6["p_all"]
valid = z6["valid"]
te = valid & hold

ze = np.load(os.path.join(V8, "scratch", "decay", "probs", "eclass_on_v6_m0.8.npz"))
pe = ze["p"]
d = ll(p0[te]) - ll(pe[te])  # >0 = eclass better
res = {"eclass_overall_milli_mine": round(float(d.mean()) * 1000, 3),
       "published": 0.239}

# my own S3/S4/S5 masks (published: 130 / 1118 / 1189)
dts = pd.to_datetime(frame.date)
last = {}
rest_w = np.full(len(frame), np.nan)
rest_l = np.full(len(frame), np.nan)
for i, (w, l, dt) in enumerate(zip(frame.winner, frame.loser, dts)):
    if w in last:
        rest_w[i] = (dt - last[w]).days
    if l in last:
        rest_l[i] = (dt - last[l]).days
    last[w] = dt
    last[l] = dt
known = ~np.isnan(rest_w) & ~np.isnan(rest_l)
S3 = known & ((rest_w > 45) | (rest_l > 45))
first_day = frame.groupby("event_id").date.transform("min").values
S4 = frame.date.values > first_day
pfav = np.maximum(p0, 1 - p0)
S5 = pfav <= 0.80
res["mask_n_mine_vs_pub"] = {
    "S3": [int((te & S3).sum()), 130],
    "S4": [int((te & S4).sum()), 1118],
    "S5": [int((te & S5).sum()), 1189]}

subs = {"S3": S3, "S4": S4, "S5": S5}
for name, m in subs.items():
    dm = ll(p0[te & m]) - ll(pe[te & m])
    res[f"eclass_{name}_milli_mine"] = round(float(dm.mean()) * 1000, 3)

# overlap: what fraction of holdout rows are in S4∩S5?
res["overlap"] = {
    "S4_and_S5_frac_of_holdout": round(float((te & S4 & S5).sum() / te.sum()), 3),
    "S5_frac": round(float((te & S5).sum() / te.sum()), 3),
    "S4_frac": round(float((te & S4).sum() / te.sum()), 3)}

# fragility of the +0.24m overall
k5 = int(math.ceil(0.05 * d.size))
order = np.argsort(d)[::-1]
kept = np.ones(d.size, bool)
kept[order[:k5]] = False
res["eclass_drop_top5pct_milli"] = round(float(d[kept].mean()) * 1000, 3)
evh = ev[te]
jk = {e: round(float(d[evh != e].mean()) * 1000, 3) for e in np.unique(evh)}
neg = {e: v for e, v in jk.items() if v <= 0}
res["eclass_jackknife"] = {"n_events": len(jk),
                           "n_events_flip_nonpos": len(neg),
                           "flips": neg,
                           "min": min(jk.values()), "max": max(jk.values())}
# eclass gain concentration: delta on EWC-class rows vs elsewhere
evs = frame.event_id.astype(str)
legacy = (evs.str.startswith("2026_ewc") | evs.str.startswith("2026_china_evo")).values
d_in = ll(p0[te & legacy]) - ll(pe[te & legacy])
d_out = ll(p0[te & ~legacy]) - ll(pe[te & ~legacy])
res["eclass_delta_split"] = {
    "on_2026_ewc_class_rows_milli": round(float(d_in.mean()) * 1000, 3),
    "n_in": int((te & legacy).sum()),
    "elsewhere_milli": round(float(d_out.mean()) * 1000, 3),
    "n_out": int((te & ~legacy).sum())}

# ── H3 cold-start buckets ───────────────────────────────────────────────────
from engine import Engine  # noqa: E402

eng = Engine()
prior_maps = {}
cnt = {}
games = sorted(eng.games, key=lambda g: (g["date_s"], str(g["match_id"])))
gi = 0
n = len(frame)
pm_w = np.zeros(n)
pm_l = np.zeros(n)
order_idx = np.argsort(frame.date.values, kind="stable")
# strictly-earlier-day counts
rows_by_date = {}
for i in range(n):
    rows_by_date.setdefault(frame.date.values[i], []).append(i)
for day in sorted(rows_by_date):
    while gi < len(games) and games[gi]["date_s"] < day:
        g = games[gi]
        cnt[g["winner"]] = cnt.get(g["winner"], 0) + 1
        cnt[g["loser"]] = cnt.get(g["loser"], 0) + 1
        gi += 1
    for i in rows_by_date[day]:
        pm_w[i] = cnt.get(frame.winner.values[i], 0)
        pm_l[i] = cnt.get(frame.loser.values[i], 0)

z3 = np.load(os.path.join(V8, "scratch", "bias_h3", "model_probs.npz"))
p5d = z3["p_ss_5d"]
pv6 = z3["p_v6"]
res["cold_buckets_mine"] = {}
for name, m in [("debut(either0)", (np.minimum(pm_w, pm_l) == 0)),
                ("cold(<10)", (np.minimum(pm_w, pm_l) < 10)),
                ("thin(<30)", (np.minimum(pm_w, pm_l) < 30))]:
    mm = te & m
    dd = ll(pv6[mm]) - ll(p5d[mm])
    nn = int(mm.sum())
    entry = {"n": nn, "dll_5d_milli": round(float(dd.mean()) * 1000, 2),
             "bucket_mde_cross": round(5.889 * math.sqrt(1217 / max(nn, 1)), 1)}
    if nn > 2:
        o = np.argsort(dd)[::-1]
        entry["drop_top1_milli"] = round(float(np.delete(dd, o[0]).mean()) * 1000, 2)
        entry["drop_top5pct_milli"] = round(
            float(dd[np.argsort(dd)[:-max(1, int(math.ceil(0.05 * nn)))]].mean()) * 1000, 2)
        evm = ev[mm]
        jkc = {e: round(float(dd[evm != e].mean()) * 1000, 2) for e in np.unique(evm)}
        entry["jackknife_min"] = min(jkc.values())
        entry["jackknife_min_event"] = min(jkc, key=jkc.get)
        entry["n_events"] = len(jkc)
    res["cold_buckets_mine"][name] = entry

with open(os.path.join(HERE, "recompute_eclass_cold.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
