"""Pick-side advantage: after controlling for ratings, does the team that
PICKED the map win it more often? Fit b_pick on train (<=2024) map outcomes,
test on 2025-26. If real, the MC should add b_pick to picked-map logits.
Also: decider-map calibration check (no picker) as control.
Writes out/pickside.json."""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine
from scipy.optimize import minimize, minimize_scalar

OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "..", "data")

eng = Engine()
# daily overall ratings under the candidate config
stage_by_mid = {}
s = eng.series.reset_index(drop=True)
for r in s.itertuples(index=False):
    stage_by_mid[r.match_id] = r.stage
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
out = eng.run({"decay": {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0},
               "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
               "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
               "w_custom": PO, "daily_out": True})
daily = out["daily_r"]
days_sorted = sorted(daily.keys())
import bisect


def rating(org, date):
    j = bisect.bisect_left(days_sorted, date) - 1
    if j < 0 or org not in eng.tidx:
        return None
    return float(daily[days_sorted[j]][eng.tidx[org]])


# picker per (match_id, map): from map_vetos.csv
v = pd.read_csv(os.path.join(DATA, "map_vetos.csv"))
picker = {}
for r in v.itertuples(index=False):
    if r.action == "pick":
        picker[(int(r.MatchID), str(r.map).strip())] = r.team
print(f"pick records: {len(picker)}")

# map outcomes with ratings + pick info
rows = []
for g in eng.games:
    pk = picker.get((g["match_id"], g["map_name"]))
    rw = rating(g["winner"], g["date_s"])
    rl = rating(g["loser"], g["date_s"])
    if rw is None or rl is None:
        continue
    rows.append({"date": g["date_s"], "rdiff_w": rw - rl,
                 "picked_by": ("winner" if pk == g["winner"] else
                               "loser" if pk == g["loser"] else "none")})
df = pd.DataFrame(rows)
train = df.date <= "2024-12-31"
test = df.date > "2024-12-31"
print(f"maps: {len(df)}  with picker: {(df.picked_by!='none').sum()} "
      f"({(df.picked_by!='none').mean():.0%})")

# raw pick-side winrate (unadjusted)
withpick = df[df.picked_by != "none"]
print(f"raw: picker won {(withpick.picked_by=='winner').mean():.3f} of picked maps "
      f"(n={len(withpick)})")

# fit: P(winner side wins) = sigmoid(beta*rdiff + b_pick*pick_sign)
# pick_sign = +1 if winner picked, -1 if loser picked, 0 decider
sign = df.picked_by.map({"winner": 1.0, "loser": -1.0, "none": 0.0}).values
rd = df.rdiff_w.values


def nll(params, mask):
    b, bp = params
    z = np.clip(b * rd[mask] + bp * sign[mask], -30, 30)
    p = 1 / (1 + np.exp(-z))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))


res = {}
r0 = minimize_scalar(lambda b: nll((b, 0.0), train.values), bounds=(0.02, 0.6),
                     method="bounded")
b_only = float(r0.x)
r1 = minimize(lambda pr: nll(pr, train.values), [b_only, 0.0], method="Nelder-Mead")
b_f, bp_f = float(r1.x[0]), float(r1.x[1])
ll_test_nopick = float(nll((b_only, 0.0), test.values))
ll_test_pick = float(nll((b_f, bp_f), test.values))
res["beta_only"] = round(b_only, 4)
res["beta_with_pick"] = round(b_f, 4)
res["b_pick"] = round(bp_f, 4)
res["pick_edge_pct_at_even"] = round(float(1/(1+np.exp(-bp_f)) - 0.5) * 100, 2)
res["ll_map_test_nopick"] = round(ll_test_nopick, 5)
res["ll_map_test_pick"] = round(ll_test_pick, 5)
print(f"\nb_pick = {bp_f:.4f}  (= {res['pick_edge_pct_at_even']:+.1f}% on an even map)")
print(f"map-level test LL: no-pick {ll_test_nopick:.5f} -> with-pick {ll_test_pick:.5f} "
      f"({(ll_test_nopick-ll_test_pick)*1000:+.2f}m)")

# stability: fit per year
for yr in (2023, 2024, 2025, 2026):
    myr = df.date.str.startswith(str(yr)).values
    if myr.sum() < 200:
        continue
    ry = minimize(lambda pr: nll(pr, myr), [b_only, 0.0], method="Nelder-Mead")
    res[f"b_pick_{yr}"] = round(float(ry.x[1]), 4)
    print(f"  {yr}: b_pick = {ry.x[1]:+.4f} (n={int(myr.sum())})")

with open(os.path.join(OUT, "pickside.json"), "w") as f:
    json.dump(res, f, indent=1)
print("saved out/pickside.json")
