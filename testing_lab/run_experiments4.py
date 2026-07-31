"""Round 4: (A) roster-instability prediction shrink, (B) intl offset refit
on the candidate ratings, (C) candidate calibration by year.
Writes out/experiments4.json."""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine
from harness import calib_slope, intl_attendance_asof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BEST = {"decay": {"kind": "exp", "hl": 13.0}, "rd": {"power": 0.75, "scale": 2.5},
        "roster_mode": "year", "roster_persistence": 0.3,
        "ridge": 0.5, "champ_mult": 2.0}
INTL9 = {"2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
         "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
         "2026_masters_santiago", "2026_masters_london", "2026_champions"}

eng = Engine()
s = eng.series.reset_index(drop=True)
out = eng.run(BEST)
rdiff, beta = out["rdiff"], out["beta"]
valid = ~np.isnan(rdiff)
res = {"beta": beta}


def p_series(rd, fmt, b):
    pm = 1 / (1 + np.exp(-b * rd))
    return np.where(np.isin(fmt, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fmt == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


p_cand = p_series(rdiff, s.fmt.values, beta)
test = out["test_mask"]
y25 = test & (s.date <= "2025-12-31").values
y26 = test & (s.date > "2025-12-31").values


def ll(p, m):
    return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))


res["cand_ll_25"], res["cand_ll_26"] = ll(p_cand, y25), ll(p_cand, y26)
res["cand_slope_25"] = calib_slope(p_cand[y25])
res["cand_slope_26"] = calib_slope(p_cand[y26])
print(f"candidate: ll25={res['cand_ll_25']:.5f} ll26={res['cand_ll_26']:.5f}")
print(f"slopes: 25={res['cand_slope_25']} 26={res['cand_slope_26']}")

# ── A) roster-instability shrink ─────────────────────────────────────────────
# instability(team, date) = 5 - |overlap of current lineup with lineup 3 team-
# matches earlier|; flag teams with >=2 changes.
lineup_at = {}
for org, seq in eng.team_match_seq.items():
    for i, (ds, mid) in enumerate(seq):
        prev_i = max(0, i - 3)
        cur = eng.lineups.get((org, mid))
        old = eng.lineups.get((org, seq[prev_i][1]))
        if cur and old:
            lineup_at[(org, ds, mid)] = len(cur & old)

def instability(org, date):
    seq = eng.team_match_seq.get(org, [])
    last = None
    for ds, mid in seq:
        if ds < date:
            v = lineup_at.get((org, ds, mid))
            if v is not None:
                last = v
        else:
            break
    return 5 - last if last is not None else 0


inst_w = np.array([max(instability(r.winner, r.date), instability(r.loser, r.date))
                   for r in s.itertuples(index=False)])
res["n_unstable_test"] = int(((inst_w >= 2) & test).sum())
print(f"unstable (>=2 changes) in test: {res['n_unstable_test']}")

for sf in (0.7, 0.8, 0.9):
    lp = np.log(np.clip(p_cand, 1e-9, 1 - 1e-9) / (1 - np.clip(p_cand, 1e-9, 1 - 1e-9)))
    p_shr = 1 / (1 + np.exp(-np.where(inst_w >= 2, sf * lp, lp)))
    res[f"shrink_{sf}"] = {"ll_test": ll(p_shr, test),
                           "ll_unstable": ll(p_shr, test & (inst_w >= 2)),
                           "cand_ll_unstable": ll(p_cand, test & (inst_w >= 2))}
    print(f"shrink {sf}: {res[f'shrink_{sf}']}")

# ── B) intl offsets refit on candidate ratings ──────────────────────────────
att = intl_attendance_asof(s)
intl_m = s.event_id.isin(INTL9).values & valid
fit_m = intl_m & (s.date <= "2025-12-31").values   # 2024+2025 intl
tst_m = intl_m & (s.date > "2025-12-31").values    # 2026 intl
print(f"\nintl fit n={fit_m.sum()} test n={tst_m.sum()}")


def apply_off(p_a_arr, mask, exp_b, cn_d):
    out_p = p_a_arr.copy()
    for i in np.where(mask)[0]:
        r = s.iloc[i]
        p = out_p[i]
        a_fav = p >= 0.5
        fav, dog = (r.winner, r.loser) if a_fav else (r.loser, r.winner)
        d_att = att.get(fav)
        att_f = bool(d_att and d_att < r.date)
        d_att = att.get(dog)
        att_d = bool(d_att and d_att < r.date)
        delta = exp_b * ((1 if att_f else 0) - (1 if att_d else 0))
        from vctmm.benpom.teams import ORG_REGIONS
        if ORG_REGIONS.get(dog) == "CN" and ORG_REGIONS.get(fav) != "CN":
            delta += cn_d
        if delta:
            pf = p if a_fav else 1 - p
            pf = min(max(pf, 1e-9), 1 - 1e-9)
            pf = 1 / (1 + np.exp(-(np.log(pf / (1 - pf)) + delta)))
            out_p[i] = pf if a_fav else 1 - pf
    return out_p


sys.path.insert(0, "/Users/benny_es1/VCTMM")
grid_res = []
for eb in (0.0, 0.2, 0.4, 0.6):
    for cd in (0.0, 0.2, 0.35, 0.5):
        pfit = apply_off(p_cand, fit_m, eb, cd)
        grid_res.append({"exp": eb, "cn": cd, "ll_fit": ll(pfit, fit_m)})
grid_res.sort(key=lambda g: g["ll_fit"])
res["offset_grid_fit"] = grid_res[:6]
best_off = grid_res[0]
print("offset grid (fit 24-25):", grid_res[:4])
# evaluate best-on-fit vs production offsets vs none on 2026 intl
for lab, (eb, cd) in [("refit", (best_off["exp"], best_off["cn"])),
                      ("prod", (0.4, 0.35)), ("none", (0.0, 0.0))]:
    pt = apply_off(p_cand, tst_m, eb, cd)
    res[f"offset_{lab}_ll26intl"] = ll(pt, tst_m)
    print(f"2026 intl with {lab} offsets ({eb},{cd}): {res[f'offset_{lab}_ll26intl']:.5f}")

with open(os.path.join(OUT, "experiments4.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("saved out/experiments4.json")
