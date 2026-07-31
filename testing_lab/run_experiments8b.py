"""Round 8b — sharpening the 50-50 belt on candidate v2.
Tests (all walk-forward, fit<=2024, score 2025-26):
  power-link alpha (expands small rating gaps), global Platt, per-format
  Platt, rolling-12mo Platt. Then: does the winner close the [0.5,0.7)
  favorite-band gap to Kalshi? Writes out/belt.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine
from harness import paired_bootstrap, reliability

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
from scipy.optimize import minimize, minimize_scalar

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values
rd = np.load(os.path.join(OUT, "cand_v2_rdiff.npy"))
valid = ~np.isnan(rd)
res = {}


def series_p(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def ll_of(p, mask):
    return float(-np.mean(np.log(np.clip(p[mask], 1e-9, 1))))


def belt_table(p, mask, label):
    """favorite-band pred vs empirical on winner-prob array."""
    pf = np.maximum(p[mask], 1 - p[mask])
    fw = (p[mask] >= 0.5).astype(float)
    ties = np.abs(p[mask] - 0.5) < 1e-9
    pf, fw = pf[~ties], fw[~ties]
    out = []
    for lo, hi in [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7),
                   (0.7, 0.8), (0.8, 1.01)]:
        m = (pf >= lo) & (pf < hi)
        if m.sum() >= 10:
            out.append({"band": f"[{lo},{hi})", "n": int(m.sum()),
                        "pred": round(float(pf[m].mean()), 4),
                        "emp": round(float(fw[m].mean()), 4)})
    res[f"belt_{label}"] = out
    print(f"\nBELT {label}:")
    for r in out:
        print(f"  {r['band']:<12} n={r['n']:<4} pred {r['pred']:.3f} emp {r['emp']:.3f} "
              f"gap {r['emp']-r['pred']:+.3f}")
    return out


# 1) v2 baseline
def nll_beta(b, mask):
    pm = 1 / (1 + np.exp(-b * rd[mask]))
    return -np.mean(np.log(np.clip(series_p(pm, fmts[mask]), 1e-9, 1)))


b0 = float(minimize_scalar(lambda x: nll_beta(x, valid & train_v),
                           bounds=(0.02, 0.6), method="bounded").x)
pm0 = 1 / (1 + np.exp(-b0 * rd))
p0 = series_p(pm0, fmts)
res["v2"] = {"beta": round(b0, 4), "ll_test": round(ll_of(p0, valid & test_v), 5)}
print("v2 baseline:", res["v2"])
belt_table(p0, valid & test_v, "v2")

# 2) power-link alpha
for al in (0.65, 0.75, 0.85):
    rd_a = np.sign(rd) * np.abs(rd) ** al

    def nb(b, mask):
        pm = 1 / (1 + np.exp(-b * rd_a[mask]))
        return -np.mean(np.log(np.clip(series_p(pm, fmts[mask]), 1e-9, 1)))
    ba = float(minimize_scalar(lambda x: nb(x, valid & train_v),
                               bounds=(0.02, 1.2), method="bounded").x)
    pa = series_p(1 / (1 + np.exp(-ba * rd_a)), fmts)
    res[f"plink_a{al}"] = {"beta": round(ba, 4),
                           "ll_test": round(ll_of(pa, valid & test_v), 5)}
    print(f"plink a={al}: {res[f'plink_a{al}']}")
    if al == 0.75:
        pa_keep = pa
        belt_table(pa, valid & test_v, f"plink{al}")

# 3) global Platt on v2 final probs (fit train)
lp = np.log(np.clip(p0, 1e-9, 1 - 1e-9) / (1 - np.clip(p0, 1e-9, 1 - 1e-9)))
def platt_nll(ab, mask):
    z = np.clip(ab[0] + ab[1] * lp[mask], -30, 30)
    q = 1 / (1 + np.exp(-z))
    # symmetric two-side view
    return -np.mean(np.log(np.clip(q, 1e-12, 1)))


# fit on both-side representation to keep symmetry: winner probs only is fine
r = minimize(lambda ab: platt_nll(ab, valid & train_v), [0.0, 1.0],
             method="Nelder-Mead")
a_p, b_p = float(r.x[0]), float(r.x[1])
p_platt = 1 / (1 + np.exp(-(a_p + b_p * lp)))
res["platt_global"] = {"a": round(a_p, 4), "b": round(b_p, 4),
                       "ll_test": round(ll_of(p_platt, valid & test_v), 5)}
print("platt global:", res["platt_global"])

# 4) per-format Platt slopes (train)
p_fmt = p0.copy()
for grp, fmset in [("bo3", ("bo3", "bo1")), ("bo5", ("bo5", "bo5_gf"))]:
    mfit = valid & train_v & np.isin(fmts, fmset)
    rr = minimize(lambda ab: platt_nll(ab, mfit), [0.0, 1.0], method="Nelder-Mead")
    mapp = valid & np.isin(fmts, fmset)
    p_fmt[mapp] = 1 / (1 + np.exp(-(rr.x[0] + rr.x[1] * lp[mapp])))
    res[f"platt_{grp}"] = {"a": round(float(rr.x[0]), 4), "b": round(float(rr.x[1]), 4)}
res["platt_perfmt_ll"] = round(ll_of(p_fmt, valid & test_v), 5)
print("platt per-format:", {k: res[k] for k in ("platt_bo3", "platt_bo5")},
      "ll:", res["platt_perfmt_ll"])

# 5) rolling 12-mo Platt slope (deployable check)
p_roll = p0.copy()
months = sorted({d[:7] for d in s.date if d > "2024-12-31"})
for mo in months:
    lo = (pd.Timestamp(mo + "-01") - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    fitm = valid & (s.date >= lo).values & (s.date < mo + "-01").values
    if fitm.sum() < 150:
        continue
    rr = minimize(lambda ab: platt_nll(ab, fitm), [0.0, 1.0], method="Nelder-Mead")
    inmo = valid & s.date.str.startswith(mo).values
    p_roll[inmo] = 1 / (1 + np.exp(-(rr.x[0] + rr.x[1] * lp[inmo])))
res["platt_rolling_ll"] = round(ll_of(p_roll, valid & test_v), 5)
print("platt rolling-12mo ll:", res["platt_rolling_ll"])

# bootstraps vs v2
for name, parr in [("plink_a0.75", pa_keep), ("platt_global", p_platt),
                   ("platt_rolling", p_roll)]:
    vv = valid & test_v
    res[f"boot_{name}"] = paired_bootstrap(parr[vv], p0[vv])
    print(f"boot {name} vs v2: dLL={res[f'boot_{name}']['mean_delta']:+.5f} "
          f"p={res[f'boot_{name}']['p_better']:.3f}")

# 6) Kalshi belt check with the best sharpener
kj = pd.read_csv(os.path.join(OUT, "kalshi_joined3.csv"))
best_name, best_p = min([("v2", p0), ("plink_a0.75", pa_keep),
                         ("platt_global", p_platt)],
                        key=lambda t: ll_of(t[1], valid & test_v))
s2 = s.copy()
s2["p_best"] = best_p
kj = kj.merge(s2[["match_id", "p_best"]], on="match_id", how="left").dropna(
    subset=["p_best"])
pb_fav = np.maximum(kj.p_best, 1 - kj.p_best)
pk_same = np.where(kj.p_best >= 0.5, kj.pk_pre, 1 - kj.pk_pre)
res["kalshi_belt_best"] = {"model": best_name, "bands": []}
for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
    m = (pb_fav >= lo) & (pb_fav < hi)
    if m.sum():
        res["kalshi_belt_best"]["bands"].append(
            {"band": f"[{lo},{hi})", "n": int(m.sum()),
             "model": round(float(pb_fav[m].mean()), 4),
             "kalshi": round(float(pk_same[m].mean()), 4),
             "emp": round(float((kj.p_best[m] >= 0.5).mean()), 4)})
print(f"\nKALSHI BELT ({best_name}):")
for r in res["kalshi_belt_best"]["bands"]:
    print(" ", r)
res["kalshi_ll_best_model"] = round(ll_of(kj.p_best.values, np.ones(len(kj), bool)), 5)
res["kalshi_ll_market"] = round(ll_of(kj.pk_pre.values, np.ones(len(kj), bool)), 5)
print(f"overlap ll: model {res['kalshi_ll_best_model']} vs market {res['kalshi_ll_market']}")

with open(os.path.join(OUT, "belt.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("saved out/belt.json")
