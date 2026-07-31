"""Margin-outcome model, step 2 (series level).

Three uses of margins tested walk-forward on the v5 stack:
  A) mixed-loss beta: fit series beta on (1-w)*binary + w*margin-soft labels
  B) margin-residual FORM feature: decayed avg of (map margin - rating-implied
     margin) per team, added to the rating gap with coefficient c (fit train)
  C) the user's evaluation metric: predicted vs realized series avg
     round-margin/map (margin-MSE) for production vs v5 vs variants
Plus: v5 favorite-band pricing vs Kalshi (the 'scared of high probs' check).
Writes out/margin2.json."""
import bisect
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine
from harness import paired_bootstrap
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

OUT = os.path.join(HERE, "out")

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values
link = json.load(open(os.path.join(OUT, "margin_link.json")))
A, SD = link["a_slope"], link["resid_sd"]

rd5 = np.load(os.path.join(OUT, "rd_v5_native.npy"))
valid = ~np.isnan(rd5)

# realized series avg transformed margin (winner-referenced) + per-map margins
mid_maps = defaultdict(list)
for g in eng.games:
    mid_maps[g["match_id"]].append(g)
avg_m = np.full(len(s), np.nan)
for i, row in enumerate(s.itertuples(index=False)):
    maps_ = mid_maps.get(row.match_id, [])
    if not maps_:
        continue
    vals = []
    for g in maps_:
        m_t = np.sign(g["wr"] - g["lr"]) * abs(g["wr"] - g["lr"]) ** 0.75 * 2.5
        vals.append(m_t if g["winner"] == row.winner else -m_t)
    avg_m[i] = float(np.mean(vals))

res = {}


def series_p(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def ll_of(p, m):
    return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))


# baseline v5
b5 = float(minimize_scalar(
    lambda b: -np.mean(np.log(np.clip(series_p(
        1 / (1 + np.exp(-b * rd5[valid & train_v])), fmts[valid & train_v]), 1e-9, 1))),
    bounds=(0.02, 0.6), method="bounded").x)
p5 = series_p(1 / (1 + np.exp(-b5 * rd5)), fmts)
res["v5_ll"] = round(ll_of(p5, valid & test_v), 5)
print(f"v5 baseline: beta={b5:.4f} ll={res['v5_ll']}")

# ── A) mixed-loss beta ───────────────────────────────────────────────────────
y_soft_series = norm.cdf(avg_m / (SD / 1.6))  # soft series outcome from margins


def fit_mixed(w):
    m = valid & train_v & ~np.isnan(avg_m)

    def nll(b):
        p = np.clip(series_p(1 / (1 + np.exp(-b * rd5[m])), fmts[m]), 1e-9, 1 - 1e-9)
        ce_bin = -np.log(p)
        ys = y_soft_series[m]
        ce_soft = -(ys * np.log(p) + (1 - ys) * np.log(1 - p))
        return np.mean((1 - w) * ce_bin + w * ce_soft)
    return float(minimize_scalar(nll, bounds=(0.02, 0.6), method="bounded").x)


for w in (0.25, 0.5, 0.75):
    bw = fit_mixed(w)
    pw = series_p(1 / (1 + np.exp(-bw * rd5)), fmts)
    res[f"mixed_w{w}"] = {"beta": round(bw, 4),
                          "ll_test": round(ll_of(pw, valid & test_v), 5)}
    print(f"mixed-loss w={w}: beta={bw:.4f} ll={res[f'mixed_w{w}']['ll_test']}")

# ── B) margin-residual form feature ─────────────────────────────────────────
# decayed (HL=8 team-maps) average of margin residual per team, walk-forward
out5 = eng.run({"decay": {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0},
                "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
                "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
                "region_prior_ridge": 1.5,
                "w_custom": np.ones(len(eng.games)), "daily_out": True})
daily = out5["daily_r"]
days_sorted = sorted(daily.keys())
lam_f = np.log(2) / 8.0
form = defaultdict(float)   # team -> decayed residual avg (state)
form_w = defaultdict(float)
form_at = {}                # (team, date) snapshots before each series date
game_seq = sorted(eng.games, key=lambda g: (g["date_s"], g["match_id"]))
sdates = sorted(set(s.date))
si = 0
for g in game_seq:
    while si < len(sdates) and sdates[si] <= g["date_s"]:
        for t_ in list(form.keys()):
            form_at[(t_, sdates[si])] = form[t_] / max(form_w[t_], 1e-9)
        si += 1
    j = bisect.bisect_left(days_sorted, g["date_s"]) - 1
    if j < 0:
        continue
    rv = daily[days_sorted[j]]
    gap = float(rv[eng.tidx[g["winner"]]] - rv[eng.tidx[g["loser"]]])
    m_t = np.sign(g["wr"] - g["lr"]) * abs(g["wr"] - g["lr"]) ** 0.75 * 2.5
    resid_w = m_t - A * gap
    for team, r_ in ((g["winner"], resid_w), (g["loser"], -resid_w)):
        form[team] = form[team] * np.exp(-lam_f) + r_
        form_w[team] = form_w[team] * np.exp(-lam_f) + 1.0
while si < len(sdates):
    for t_ in list(form.keys()):
        form_at[(t_, sdates[si])] = form[t_] / max(form_w[t_], 1e-9)
    si += 1

form_diff = np.array([
    form_at.get((r.winner, r.date), 0.0) - form_at.get((r.loser, r.date), 0.0)
    for r in s.itertuples(index=False)])
for c in (0.1, 0.25, 0.5):
    rd_f = rd5 + c * form_diff
    bf = float(minimize_scalar(
        lambda b: -np.mean(np.log(np.clip(series_p(
            1 / (1 + np.exp(-b * rd_f[valid & train_v])),
            fmts[valid & train_v]), 1e-9, 1))),
        bounds=(0.02, 0.6), method="bounded").x)
    pf = series_p(1 / (1 + np.exp(-bf * rd_f)), fmts)
    res[f"form_c{c}"] = {"beta": round(bf, 4),
                         "ll_test": round(ll_of(pf, valid & test_v), 5)}
    print(f"form feature c={c}: ll={res[f'form_c{c}']['ll_test']}")

# ── C) the user's metric: series avg-margin prediction accuracy ─────────────
# predicted avg margin/map = A * rdiff (rating gap IS the margin scale)
mv = valid & test_v & ~np.isnan(avg_m)
e5 = np.load(os.path.join(OUT, "exp5_rdiffs.npz"))
rd_prod = e5["ref_prod_hl6_pow05"]
mid_to_prod = dict(zip(pd.read_csv(os.path.join(OUT, "series2_index.csv")).match_id[:0], []))
# align production rdiff (1578-row original series) to current s by match_id
from harness import load_series as _ls
s0 = _ls()
prod_map = dict(zip(s0.match_id.values, rd_prod))
rd_p = np.array([prod_map.get(m_, np.nan) for m_ in s.match_id.values])
for name, rdv in [("production", rd_p), ("v5", rd5),
                  ("v5+form", rd5 + 0.25 * form_diff)]:
    mm = mv & ~np.isnan(rdv)
    pred_m = A * rdv[mm]
    mse = float(np.mean((pred_m - avg_m[mm]) ** 2))
    corr = float(np.corrcoef(pred_m, avg_m[mm])[0, 1])
    res[f"marginMSE_{name}"] = {"mse": round(mse, 3), "corr": round(corr, 4),
                                "n": int(mm.sum())}
    print(f"margin metric {name:<11} MSE {mse:.2f}  corr {corr:.3f} (n={mm.sum()})")

# bootstraps of best variants vs v5
best_name, best_p = None, None
for k in list(res.keys()):
    if k.startswith(("mixed_", "form_")) and res[k]["ll_test"] < res["v5_ll"]:
        print("candidate better:", k, res[k])
with open(os.path.join(OUT, "margin2.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("saved out/margin2.json")
