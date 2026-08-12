"""v10 — where does year isolation lose? Preregistered prediction 2 says the
damage concentrates in the opening months of each season, when an isolated
solve has the least history.

Also measures the mechanical consequence: how many series each arm has to
price at or near 50/50 because it has no information yet.

Writes testing_lab/v10/stats/v10_concentration.json.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Engine   # noqa: E402

OUT = os.path.join(HERE, "stats")
V6 = {"decay": {"kind": "games", "consistency": (20.0, 12.0)},
      "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
      "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
      "region_prior_ridge": 1.5}

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts, date = s.fmt.values, s.date.values
FIT1 = date <= "2024-12-31"

rd = {"v6": eng.run(dict(V6))["rdiff"],
      "A_iso_hard": eng.run(dict(V6, year_isolated=True))["rdiff"]}
common = np.isfinite(rd["v6"]) & np.isfinite(rd["A_iso_hard"])


def p_vec(b, r, m):
    pm = 1 / (1 + np.exp(-b * r[m])); fm = fmts[m]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def nll(b, r, m):
    return -np.mean(np.log(np.clip(p_vec(b, r, m), 1e-9, 1)))


beta = {n: float(minimize_scalar(lambda x: nll(x, rd[n], common & FIT1),
                                 bounds=(0.001, 1.0), method="bounded").x)
        for n in rd}

# per-series log-loss, so we can slice it any way we like
ll = {}
for n in rd:
    p = np.full(len(s), np.nan)
    p[common] = p_vec(beta[n], rd[n], common)
    ll[n] = -np.log(np.clip(p, 1e-9, 1))

months = np.array([d[5:7] for d in date])
res = {"beta": {k: round(v, 6) for k, v in beta.items()}, "by_month": [],
       "by_season_phase": [], "coinflip": {}}

MON = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
for m in MON:
    sel = common & (months == m)
    if sel.sum() < 15:
        continue
    d_milli = float((np.nanmean(ll["v6"][sel]) -
                     np.nanmean(ll["A_iso_hard"][sel])) * 1000)
    res["by_month"].append({"month": m, "n": int(sel.sum()),
                            "delta_milli": round(d_milli, 2)})

# season phase = months since that season started (Jan = 0)
phase = np.array([int(d[5:7]) - 1 for d in date])
for lo, hi, lab in ((0, 1, "Jan-Feb"), (2, 3, "Mar-Apr"), (4, 5, "May-Jun"),
                    (6, 7, "Jul-Aug"), (8, 11, "Sep-Dec")):
    sel = common & (phase >= lo) & (phase <= hi)
    if sel.sum() < 15:
        continue
    d_milli = float((np.nanmean(ll["v6"][sel]) -
                     np.nanmean(ll["A_iso_hard"][sel])) * 1000)
    res["by_season_phase"].append({"phase": lab, "n": int(sel.sum()),
                                   "delta_milli": round(d_milli, 2)})

# how often does each arm have nothing to say? |rdiff| ~ 0 => a coin flip
for n in rd:
    r = np.abs(rd[n][common])
    res["coinflip"][n] = {
        "median_abs_rdiff": round(float(np.median(r)), 4),
        "pct_within_0.5": round(float(100 * np.mean(r < 0.5)), 2),
        "pct_within_1.0": round(float(100 * np.mean(r < 1.0)), 2)}

# same, restricted to the opening of each season
early = common & (phase <= 1)
for n in rd:
    r = np.abs(rd[n][early])
    res["coinflip"][n]["janfeb_pct_within_0.5"] = round(
        float(100 * np.mean(r < 0.5)), 2)

with open(os.path.join(OUT, "v10_concentration.json"), "w") as f:
    json.dump(res, f, indent=1)

print("delta by season phase (milli-LL, negative = isolation worse):")
for r_ in res["by_season_phase"]:
    print(f"  {r_['phase']:8} n={r_['n']:4}  {r_['delta_milli']:+8.2f}m")
print("\ncoin-flip rate (|rating gap| < 0.5):")
for n, v in res["coinflip"].items():
    print(f"  {n:12} overall {v['pct_within_0.5']:5.1f}%   "
          f"Jan-Feb {v['janfeb_pct_within_0.5']:5.1f}%   "
          f"median |gap| {v['median_abs_rdiff']:.2f}")
