"""agent:adversary — 3e stand-in shrink: full refit with my own optimizer,
bucket-definition audit (n=291 vs preregistered n~278), fragility tests."""
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
sys.path.insert(0, TL)

frame = pd.read_csv(os.path.join(V8, "data", "frame_expanded", "series.csv"))
feat = pd.read_csv(os.path.join(V8, "scratch", "context", "frame_features.csv"))
assert (feat.match_id.values == frame.match_id.values).all()
z6 = np.load(os.path.join(V8, "scratch", "bias_h3", "v6_baseline.npz"))
rdiff = z6["rdiff"]
valid = ~np.isnan(rdiff)
hold = (frame.date > "2024-12-31").values
fmts = frame.fmt.values
ev = frame.event_id.values
EPS = 1e-9


def ll(p):
    return -np.log(np.clip(p, EPS, 1.0))


def closed_form(z, fm):
    pm = 1.0 / (1.0 + np.exp(-z))
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


X1 = (1 - feat.integ_w.values) + (1 - feat.integ_l.values)
eclass = feat.eclass.values
ewc_dummy = (eclass == "ewc_offseason").astype(float)

tr = valid & ~hold
te = valid & hold


def fit(X, x0):
    def nll(theta):
        b, k = theta
        z = b * rdiff[tr] * np.exp(-k * X[tr])
        return float(ll(closed_form(z, fmts[tr])).mean())
    r = minimize(nll, x0, method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    return r.x, r.fun


res = {}
# B0 probs for reference deltas
p0 = closed_form(0.1152 * rdiff, fmts)

(b1, k1), nll_tr = fit(X1, [0.115, 0.3])
zX = b1 * rdiff * np.exp(-k1 * X1)
pX = closed_form(zX, fmts)
res["fit_X1_mine"] = {"beta": round(float(b1), 4), "k": round(float(k1), 4),
                      "ll_train": round(nll_tr, 5),
                      "ll_test": round(float(ll(pX[te]).mean()), 5),
                      "published": {"beta": 0.1251, "k": 0.3466,
                                    "ll_train": 0.6478, "ll_test": 0.64256}}

(bd, kd), nll_trd = fit(ewc_dummy, [0.115, 0.5])
pD = closed_form(bd * rdiff * np.exp(-kd * ewc_dummy), fmts)
res["fit_dummy_mine"] = {"beta": round(float(bd), 4), "k": round(float(kd), 4),
                         "ll_test": round(float(ll(pD[te]).mean()), 5),
                         "published": {"beta": 0.1278, "k": 0.8595,
                                       "ll_test": 0.64166}}

# bucket definitions
evs = frame.event_id.astype(str)
legacy = (evs.str.startswith("2026_ewc") | evs.str.startswith("2026_china_evo")).values
full_ec = eclass == "ewc_offseason"
res["bucket_counts_holdout_valid"] = {
    "legacy2026": int((te & legacy).sum()),
    "eclass_ewc_offseason": int((te & full_ec).sum()),
    "union": int((te & (full_ec | legacy)).sum()),
    "legacy_not_in_eclass": int((te & legacy & ~full_ec).sum())}
# which set of events makes each bucket
res["eclass_events_holdout"] = sorted(set(ev[te & full_ec]))
res["legacy_events_holdout"] = sorted(set(ev[te & legacy]))

for name, mask in [("legacy2026", te & legacy), ("fullclass", te & full_ec),
                   ("union", te & (full_ec | legacy))]:
    dX = ll(p0[mask]) - ll(pX[mask])
    dD = ll(p0[mask]) - ll(pD[mask])
    res[f"bucket_{name}"] = {
        "n": int(mask.sum()),
        "ll_B0": round(float(ll(p0[mask]).mean()), 5),
        "ll_X1": round(float(ll(pX[mask]).mean()), 5),
        "dll_X1_milli": round(float(dX.mean()) * 1000, 2),
        "dll_dummy_milli": round(float(dD.mean()) * 1000, 2)}

# fragility on the fullclass bucket (the +3.46m claim)
mask = te & full_ec
dX = ll(p0[mask]) - ll(pX[mask])
k5 = int(math.ceil(0.05 * dX.size))
order = np.argsort(dX)[::-1]
kept = np.ones(dX.size, bool)
kept[order[:k5]] = False
res["fullclass_drop_top5pct"] = {"k_dropped": k5,
                                 "delta_milli_after": round(float(dX[kept].mean()) * 1000, 2)}
evb = ev[mask]
jk = {e: round(float(dX[evb != e].mean()) * 1000, 2) for e in np.unique(evb)}
res["fullclass_jackknife_by_event"] = jk
res["bucket_mde_within_at_n291"] = round(1.773 * math.sqrt(1217 / 291), 2)

with open(os.path.join(HERE, "recompute_3e.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
