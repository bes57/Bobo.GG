"""Round 8c — symmetric recalibration of the v2 surface (fixing 8b's broken
Platt): slope-only (logit' = b*logit) and odd-cubic (b*logit + c*logit^3),
fit two-sided on train; plus rolling-12mo slope. Writes out/belt2.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine
from harness import paired_bootstrap

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


b0 = float(minimize_scalar(
    lambda b: -np.mean(np.log(np.clip(
        series_p(1 / (1 + np.exp(-b * rd[valid & train_v])),
                 fmts[valid & train_v]), 1e-9, 1))),
    bounds=(0.02, 0.6), method="bounded").x)
p0 = series_p(1 / (1 + np.exp(-b0 * rd)), fmts)
res["v2_ll"] = round(ll_of(p0, valid & test_v), 5)
lp = np.log(np.clip(p0, 1e-9, 1 - 1e-9) / (1 - np.clip(p0, 1e-9, 1 - 1e-9)))


def twoside_nll(transform, mask):
    """Fit on both orientations: (lp, y=1) and (-lp, y=0)."""
    z1 = np.clip(transform(lp[mask]), -30, 30)
    z0 = np.clip(transform(-lp[mask]), -30, 30)
    q1 = 1 / (1 + np.exp(-z1))
    q0 = 1 / (1 + np.exp(-z0))
    return -0.5 * (np.mean(np.log(np.clip(q1, 1e-12, 1))) +
                   np.mean(np.log(np.clip(1 - q0, 1e-12, 1))))


# slope-only
r = minimize_scalar(lambda b: twoside_nll(lambda x: b * x, valid & train_v),
                    bounds=(0.5, 2.0), method="bounded")
b_s = float(r.x)
p_slope = 1 / (1 + np.exp(-np.clip(b_s * lp, -30, 30)))
res["slope_only"] = {"b": round(b_s, 4),
                     "ll_test": round(ll_of(p_slope, valid & test_v), 5)}
print(f"slope-only: b={b_s:.4f} ll={res['slope_only']['ll_test']}")

# odd cubic
rc = minimize(lambda bc: twoside_nll(lambda x: bc[0] * x + bc[1] * x ** 3,
                                     valid & train_v),
              [1.0, 0.0], method="Nelder-Mead")
b_c, c_c = float(rc.x[0]), float(rc.x[1])
p_cub = 1 / (1 + np.exp(-np.clip(b_c * lp + c_c * lp ** 3, -30, 30)))
res["cubic"] = {"b": round(b_c, 4), "c": round(c_c, 5),
                "ll_test": round(ll_of(p_cub, valid & test_v), 5)}
print(f"cubic: b={b_c:.4f} c={c_c:.5f} ll={res['cubic']['ll_test']}")

# rolling 12-mo slope-only (deployable)
p_roll = p0.copy()
slopes = {}
for mo in sorted({d[:7] for d in s.date if d > "2024-12-31"}):
    lo = (pd.Timestamp(mo + "-01") - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    fitm = valid & (s.date >= lo).values & (s.date < mo + "-01").values
    if fitm.sum() < 150:
        continue
    rr = minimize_scalar(lambda b: twoside_nll(lambda x: b * x, fitm),
                         bounds=(0.5, 2.0), method="bounded")
    slopes[mo] = round(float(rr.x), 4)
    inmo = valid & s.date.str.startswith(mo).values
    p_roll[inmo] = 1 / (1 + np.exp(-np.clip(float(rr.x) * lp[inmo], -30, 30)))
res["rolling_slopes"] = slopes
res["rolling_ll"] = round(ll_of(p_roll, valid & test_v), 5)
print(f"rolling slope ll={res['rolling_ll']} slopes={slopes}")

for name, parr in [("slope_only", p_slope), ("cubic", p_cub),
                   ("rolling", p_roll)]:
    vv = valid & test_v
    res[f"boot_{name}"] = paired_bootstrap(parr[vv], p0[vv])
    print(f"boot {name} vs v2: dLL={res[f'boot_{name}']['mean_delta']:+.5f} "
          f"p={res[f'boot_{name}']['p_better']:.3f}")


# belt tables for the best recalibration
def belt(p, mask, label):
    pf = np.maximum(p[mask], 1 - p[mask])
    fw = (p[mask] >= 0.5).astype(float)
    ties = np.abs(p[mask] - 0.5) < 1e-9
    pf, fw = pf[~ties], fw[~ties]
    out = []
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        m = (pf >= lo) & (pf < hi)
        if m.sum() >= 10:
            out.append({"band": f"[{lo},{hi})", "n": int(m.sum()),
                        "pred": round(float(pf[m].mean()), 4),
                        "emp": round(float(fw[m].mean()), 4)})
    res[f"belt_{label}"] = out
    print(f"BELT {label}: " + "  ".join(
        f"{r['band']}n{r['n']} {r['pred']:.3f}/{r['emp']:.3f}" for r in out))


best_name, best_p = min([("v2", p0), ("slope_only", p_slope), ("cubic", p_cub)],
                        key=lambda t: ll_of(t[1], valid & test_v))
belt(p0, valid & test_v, "v2")
belt(best_p, valid & test_v, best_name)
res["best"] = best_name

with open(os.path.join(OUT, "belt2.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("saved out/belt2.json")
