"""Margin-outcome model, step 1 (map level).

Instead of fitting the rating->probability link on binary map outcomes
(logistic beta, high variance, shrinks the tails), derive it from ROUND
MARGINS: fit E[margin_t | rating gap] (linear), take the empirical residual
distribution, and set P(win | gap) = 1 - F_resid(-E[margin_t|gap]).
Margins are continuous-ish -> far more information per map -> sharper,
better-tailed link. Also: soft-label beta fit as a middle ground.
Everything walk-forward: fits on <=2024, scored on 2025-26 maps.
Writes out/margin1.json."""
import bisect
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine
from scipy.optimize import minimize_scalar

OUT = os.path.join(HERE, "out")

eng = Engine()
s = eng.series.reset_index(drop=True)
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
V5 = {"decay": {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0},
      "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
      "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
      "region_prior_ridge": 1.5, "w_custom": PO, "daily_out": True}
out = eng.run(V5)
daily = out["daily_r"]
days_sorted = sorted(daily.keys())

# per-map dataset: prior-day rating gap (winner-referenced) + margins
rows = []
for g in eng.games:
    j = bisect.bisect_left(days_sorted, g["date_s"]) - 1
    if j < 0:
        continue
    rv = daily[days_sorted[j]]
    gap = float(rv[eng.tidx[g["winner"]]] - rv[eng.tidx[g["loser"]]])
    raw = g["wr"] - g["lr"]
    rows.append({"date": g["date_s"], "gap": gap, "margin_raw": raw,
                 "margin_t": np.sign(raw) * abs(raw) ** 0.75 * 2.5,
                 "ot": g["wr"] > 13})
df = pd.DataFrame(rows)
train = (df.date <= "2024-12-31").values
test = (df.date > "2024-12-31").values
print(f"maps: {len(df)} (train {train.sum()}, test {test.sum()})")

res = {}

# ── symmetrize: view each map from both sides (gap, margin) and (-gap, -margin)
gap2 = np.concatenate([df.gap.values, -df.gap.values])
mt2 = np.concatenate([df.margin_t.values, -df.margin_t.values])
mraw2 = np.concatenate([df.margin_raw.values, -df.margin_raw.values])
y2 = np.concatenate([np.ones(len(df)), np.zeros(len(df))])
tr2 = np.concatenate([train, train])
te2 = np.concatenate([test, test])

# ── A) margin link: E[m_t | gap] = a*gap; empirical residual CDF ────────────
a_fit = float(np.sum(mt2[tr2] * gap2[tr2]) / np.sum(gap2[tr2] ** 2))
resid = mt2[tr2] - a_fit * gap2[tr2]
res["a_slope"] = round(a_fit, 4)
res["resid_sd"] = round(float(resid.std()), 3)
print(f"margin regression: E[m_t|gap] = {a_fit:.3f}*gap, resid sd {resid.std():.2f}")

# smoothed empirical CDF of residuals (sorted array + interpolation, with
# gaussian tail extrapolation beyond observed range)
rs = np.sort(resid)
n_r = len(rs)
from scipy.stats import norm
sd = resid.std()


def p_win_margin(gaps):
    """P(margin>0 | gap) = 1 - F_resid(-a*gap), empirical CDF + normal tails."""
    x = -a_fit * np.asarray(gaps, dtype=float)
    idx = np.searchsorted(rs, x, side="right")
    F = idx / n_r
    lo_tail = x < rs[0]
    hi_tail = x > rs[-1]
    F = np.where(lo_tail, norm.cdf(x / sd) * (norm.cdf(rs[0] / sd) and
                 (0.5 / n_r) / max(norm.cdf(rs[0] / sd), 1e-12)), F)
    F = np.where(hi_tail, 1 - (0.5 / n_r) * (1 - norm.cdf(x / sd)) /
                 max(1 - norm.cdf(rs[-1] / sd), 1e-12), F)
    return np.clip(1 - F, 1e-6, 1 - 1e-6)


def ll_bin(p, y, m):
    p = np.clip(p[m], 1e-9, 1 - 1e-9)
    yy = y[m]
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


# baseline: logistic beta fit on binary train outcomes
def nll_b(b):
    p = 1 / (1 + np.exp(-b * gap2[tr2]))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y2[tr2] * np.log(p) + (1 - y2[tr2]) * np.log(1 - p))


b_bin = float(minimize_scalar(nll_b, bounds=(0.02, 0.6), method="bounded").x)
p_log = 1 / (1 + np.exp(-b_bin * gap2))
p_mar = p_win_margin(gap2)
res["beta_binary"] = round(b_bin, 4)
res["ll_test_logistic"] = round(ll_bin(p_log, y2, te2), 5)
res["ll_test_marginlink"] = round(ll_bin(p_mar, y2, te2), 5)
print(f"map-level test LL: logistic {res['ll_test_logistic']} vs "
      f"margin-link {res['ll_test_marginlink']}")

# probit-normal variant (parametric)
p_probit = np.clip(norm.cdf(a_fit * gap2 / sd), 1e-6, 1 - 1e-6)
res["ll_test_probit"] = round(ll_bin(p_probit, y2, te2), 5)
print(f"probit-normal: {res['ll_test_probit']}")

# ── B) soft-label beta: fit logistic against margin-derived soft outcomes ───
y_soft = norm.cdf(mt2 / sd)


def nll_soft(b):
    p = np.clip(1 / (1 + np.exp(-b * gap2[tr2])), 1e-9, 1 - 1e-9)
    ys = y_soft[tr2]
    return -np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p))


b_soft = float(minimize_scalar(nll_soft, bounds=(0.02, 0.6), method="bounded").x)
p_soft = 1 / (1 + np.exp(-b_soft * gap2))
res["beta_soft"] = round(b_soft, 4)
res["ll_test_softbeta"] = round(ll_bin(p_soft, y2, te2), 5)
print(f"soft-label beta = {b_soft:.4f} (binary-fit was {b_bin:.4f}) "
      f"-> test LL {res['ll_test_softbeta']}")

# tail reliability on test maps: does any link fix the high bands?
print("\ntail reliability (test maps):")
res["bands"] = {}
for name, parr in [("logistic", p_log), ("margin-link", p_mar),
                   ("soft-beta", p_soft)]:
    bands = []
    for lo, hi in [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        m = te2 & (parr >= lo) & (parr < hi)
        if m.sum() >= 20:
            bands.append({"band": f"[{lo},{hi})", "n": int(m.sum()),
                          "pred": round(float(parr[m].mean()), 3),
                          "emp": round(float(y2[m].mean()), 3)})
    res["bands"][name] = bands
    print(f"  {name:<12} " + "  ".join(
        f"{b['band']}:{b['pred']:.2f}/{b['emp']:.2f}(n{b['n']})" for b in bands))

# persist link parameters for step 2
np.save(os.path.join(OUT, "margin_resid_sorted.npy"), rs)
json.dump({"a_slope": a_fit, "resid_sd": float(sd), "beta_soft": b_soft,
           "beta_binary": b_bin}, open(os.path.join(OUT, "margin_link.json"), "w"))
with open(os.path.join(OUT, "margin1.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nsaved out/margin1.json + margin_link.json + margin_resid_sorted.npy")
