"""Elite-vs-floor fix: test consistency-conditioned decay against the asym
and symmetric baselines, judged on (a) overall holdout LL, (b) the
ELITE-vs-FLOOR bucket (decayed winrate >=0.60 vs <=0.40) where the asym
distortion lives, (c) the current slate's cases. Writes out/floorfix.json."""
import bisect
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine
from harness import paired_bootstrap
from scipy.optimize import minimize_scalar

OUT = os.path.join(HERE, "out")

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
        "region_prior_ridge": 1.5, "w_custom": PO}

# per-team decayed map winrate as of each series date (for bucket definition)
lam_wr = math.log(2) / 16.0
wr_state = defaultdict(lambda: [0.0, 0.0])
wr_at = {}
sdates = sorted(set(s.date))
si = 0
for g in sorted(eng.games, key=lambda g: (g["date_s"], g["match_id"])):
    while si < len(sdates) and sdates[si] <= g["date_s"]:
        for t_, (n_, d_) in wr_state.items():
            if d_ > 3:
                wr_at[(t_, sdates[si])] = n_ / d_
        si += 1
    for team, won in ((g["winner"], 1.0), (g["loser"], 0.0)):
        st = wr_state[team]
        st[0] = st[0] * math.exp(-lam_wr) + won
        st[1] = st[1] * math.exp(-lam_wr) + 1.0
while si < len(sdates):
    for t_, (n_, d_) in wr_state.items():
        if d_ > 3:
            wr_at[(t_, sdates[si])] = n_ / d_
    si += 1

wr_w = np.array([wr_at.get((r.winner, r.date), 0.5) for r in s.itertuples(index=False)])
wr_l = np.array([wr_at.get((r.loser, r.date), 0.5) for r in s.itertuples(index=False)])
hi_w = np.maximum(wr_w, wr_l)
lo_w = np.minimum(wr_w, wr_l)
elite_floor = (hi_w >= 0.60) & (lo_w <= 0.40)
print(f"elite-vs-floor matches: total {int(elite_floor.sum())}, "
      f"holdout {int((elite_floor & test_v).sum())}")


def sp(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


results = {}
probs = {}


def run(name, dcfg):
    out = eng.run({**BASE, "decay": dcfg})
    rd = out["rdiff"]
    v = ~np.isnan(rd)
    b = float(minimize_scalar(
        lambda x: -np.mean(np.log(np.clip(sp(1 / (1 + np.exp(-x * rd[v & train_v])),
                                             fmts[v & train_v]), 1e-9, 1))),
        bounds=(0.02, 0.6), method="bounded").x)
    p = sp(1 / (1 + np.exp(-b * rd)), fmts)
    def ll(m):
        return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))
    ef = v & test_v & elite_floor
    # calibration on elite side within the bucket
    p_elite = np.where(wr_w >= wr_l, p, 1 - p)   # prob assigned to higher-wr team
    elite_won = (wr_w >= wr_l).astype(float)     # did the higher-wr team win?
    pe = p_elite[ef]
    ew = elite_won[ef]
    results[name] = {"beta": round(b, 4),
                     "ll_test": round(ll(v & test_v), 5),
                     "ll_elitefloor": round(ll(ef), 5),
                     "ef_pred": round(float(pe.mean()), 3),
                     "ef_emp": round(float(ew.mean()), 3),
                     "n_ef": int(ef.sum())}
    probs[name] = (p, v)
    r = results[name]
    print(f"{name:<22} ll={r['ll_test']:.5f}  EF-bucket ll={r['ll_elitefloor']:.5f} "
          f"pred {r['ef_pred']:.3f} emp {r['ef_emp']:.3f} (n={r['n_ef']})", flush=True)
    return rd, b


rd_asym, b_asym = run("asym_W20L12 (v5)", {"kind": "games", "hl_games": 20.0,
                                           "hl_games_loss": 12.0})
run("symmetric_16", {"kind": "games", "hl_games": 16.0})
rd_c, b_c = run("consist_20_12", {"kind": "games", "consistency": (20.0, 12.0)})
run("consist_24_10", {"kind": "games", "consistency": (24.0, 10.0)})
run("consist_18_14", {"kind": "games", "consistency": (18.0, 14.0)})

# bootstrap best consistency vs asym overall
pa, va = probs["consist_20_12"]
pb, vb = probs["asym_W20L12 (v5)"]
vv = va & vb & test_v
bt = paired_bootstrap(pa[vv], pb[vv])
results["_boot_consist_vs_asym"] = bt
print("boot consist_20_12 vs asym overall:", bt)
vef = va & vb & test_v & elite_floor
bt2 = paired_bootstrap(pa[vef], pb[vef])
results["_boot_consist_vs_asym_EF"] = bt2
print("boot consist_20_12 vs asym on elite-floor:", bt2)

# slate cases under the consistency model (need today's solve)
today = "2026-07-23"
if today not in eng.pred_days:
    eng.pred_days = sorted(set(list(eng.pred_days) + [today]))
out_c = eng.run({**BASE, "decay": {"kind": "games", "consistency": (20.0, 12.0)},
                 "daily_out": True})
daily = out_c["daily_r"]
latest = daily[sorted(daily.keys())[-1]]
slate = [("LEV", "EG", 0.795), ("PRX", "DFM", 0.885), ("T1", "VL", 0.825),
         ("MIBR", "ENVY", 0.795), ("RRQ", "ZETA", 0.795), ("NRG", "FUR", 0.785),
         ("EDG", "FPX", 0.725)]
print("\nslate under consistency decay (closed form, beta refit "
      f"{results['consist_20_12']['beta']}):")
bC = results["consist_20_12"]["beta"]
slate_out = []
for a, b_, mkt in slate:
    gap = float(latest[eng.tidx[a]] - latest[eng.tidx[b_]])
    pm = 1 / (1 + np.exp(-bC * gap))
    p3 = pm * pm * (3 - 2 * pm)
    slate_out.append({"a": a, "b": b_, "mkt": mkt, "new": round(float(p3), 3)})
    print(f"  {a}-{b_}: market {mkt:.0%}  v5 was ~{'':<2} new model {p3:.1%}  gap {gap:.2f}")
results["slate"] = slate_out
np.save(os.path.join(OUT, "rd_consist.npy"), rd_c)
with open(os.path.join(OUT, "floorfix.json"), "w") as f:
    json.dump(results, f, indent=1, default=str)
print("saved out/floorfix.json")
