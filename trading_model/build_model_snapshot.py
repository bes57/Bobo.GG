"""Build model_snapshot.json — the private v5 trading model, packaged.

Run from anywhere:  python3 trading_model/build_model_snapshot.py
Rebuild cadence: after every data refresh (same trigger as the site's
pipeline). The snapshot embeds generated_utc; consumers should apply their
own staleness rules against it.

What it computes (all walk-forward-consistent, fit only on data to date):
  1. Team ratings as of NOW under the v5 config:
       asymmetric games decay (wins HL=20 games, losses HL=12)
       margin^0.75 (RD_SCALE 2.5) · playoff/GF solve weight x1.6
       region-prior ridge 1.5 (teams regress to region trailing mean)
       roster year-boundary continuity 0.3 · ridge 0.5 · champions x2
  2. Region cold-start priors (25th percentile of each region's ratings).
  3. Cross-region additive offsets (fit on all cross-region series to date,
     CN pinned at 0). These REPLACE intl_exp/cn_dog offsets entirely.
  4. Pick-side bonus b_pick (logit of picker winrate, all vetoes to date).
  5. beta = 0.103 — FIXED, scale-bound to this exact config. If any constant
     above changes, beta must be refit before quoting.
"""
import bisect
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "testing_lab"))
sys.path.insert(0, "/Users/benny_es1/VCTMM")

from engine import Engine  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from vctmm.benpom.teams import ORG_REGIONS  # noqa: E402

MODEL_VERSION = "benpom-v6-2026-07-22"
BETA = 0.1299
GF_UPPER_LOGIT = 0.25
REGS = ["Americas", "EMEA", "Pacific", "CN"]

today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)

# ensure a solve exists for TODAY (uses games strictly before today)
if today not in eng.pred_days:
    eng.pred_days = sorted(set(list(eng.pred_days) + [today]))

out = eng.run({"decay": {"kind": "games", "consistency": (20.0, 12.0)},
               "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
               "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
               "region_prior_ridge": 1.5, "w_custom": PO, "daily_out": True})
daily = out["daily_r"]
days_sorted = sorted(daily.keys())
latest_day = days_sorted[-1]
rvec = daily[latest_day]
print(f"ratings solved as of {latest_day} ({len(eng.teams)} teams, "
      f"{len(eng.games)} games)")

# only teams with actual game history get a rating entry
first_game = {org: min(eng.g_date[i] for i in rows_)
              for org, rows_ in eng.team_game_rows.items()}
ratings = {t: round(float(rvec[eng.tidx[t]]), 4)
           for t in eng.teams if t in first_game}

# region cold-start priors (25th pct of rated teams per region)
region_priors = {}
for reg in REGS:
    vals = [ratings[t] for t in ratings if ORG_REGIONS.get(t) == reg]
    if len(vals) >= 6:
        region_priors[reg] = round(float(np.percentile(vals, 25)), 4)

# cross-region offsets: fit on ALL cross-region series to date (CN pinned 0)
rd = out["rdiff"]
valid = ~np.isnan(rd)
cross = (s.reg_w != s.reg_l).values & valid
iw = s.reg_w.map({r: i for i, r in enumerate(REGS)}).fillna(0).astype(int).values
il = s.reg_l.map({r: i for i, r in enumerate(REGS)}).fillna(0).astype(int).values


def series_p(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def nll_off(d3):
    d4 = np.append(d3, 0.0)
    adj = rd[cross] + d4[iw[cross]] - d4[il[cross]]
    p = series_p(1 / (1 + np.exp(-BETA * adj)), fmts[cross])
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))


r_off = minimize(nll_off, np.zeros(3), method="Nelder-Mead")
xregion = {reg: round(float(v), 4)
           for reg, v in zip(REGS, np.append(r_off.x, 0.0))}
print("cross-region offsets:", xregion)

# pick-side bonus: logit of picker winrate over all vetoes to date
v = pd.read_csv(os.path.join(ROOT, "data", "map_vetos.csv"))
picker = {(int(r.MatchID), str(r.map).strip()): r.team
          for r in v.itertuples(index=False) if r.action == "pick"}
n_pick = n_win = 0
for g in eng.games:
    pk = picker.get((g["match_id"], g["map_name"]))
    if pk == g["winner"]:
        n_pick += 1
        n_win += 1
    elif pk == g["loser"]:
        n_pick += 1
pw = n_win / max(n_pick, 1)
b_pick = round(math.log(pw / (1 - pw)), 4)
print(f"b_pick = {b_pick} (picker winrate {pw:.3f}, n={n_pick})")

snapshot = {
    "model_version": MODEL_VERSION,
    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    "ratings_as_of": latest_day,
    "n_games": len(eng.games),
    "beta": BETA,
    "gf_upper_logit": GF_UPPER_LOGIT,
    "b_pick": b_pick,
    "ratings": dict(sorted(ratings.items())),
    "region_priors": region_priors,
    "xregion_offsets": xregion,
    "org_regions": {t: ORG_REGIONS.get(t, "") for t in ratings},
    "config": {"decay": "games, consistency-conditioned: results consistent with team level HL 20, anomalies HL 12 (fixes floor-team inflation)",
               "margin": "|rd|^0.75 * 2.5", "playoff_weight": 1.6,
               "region_prior_ridge": 1.5, "roster": "year-boundary, 0.3",
               "ridge": 0.5, "champions_mult": 2.0},
    "validation": {"holdout_ll": 0.64126, "production_ll": 0.65262,
                   "kalshi_vct86_ll": 0.6441, "kalshi_market_ll": 0.6457,
                   "divergence_trade_roi_5pt": 0.287},
}
path = os.path.join(HERE, "model_snapshot.json")
with open(path, "w") as f:
    json.dump(snapshot, f, indent=1)
print(f"wrote {path} ({len(ratings)} teams)")
