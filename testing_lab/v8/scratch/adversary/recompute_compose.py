"""agent:adversary — independent reconstruction of the three compose stacks."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(V8))

frame = pd.read_csv(os.path.join(V8, "data", "frame_expanded", "series.csv"))
hold = (frame.date > "2024-12-31").values
fmts = frame.fmt.values
ev = frame.event_id.values
EPS = 1e-9
R = 11.2933


def ll(p):
    return -np.log(np.clip(p, EPS, 1.0))


def cf(z, fm):
    pm = 1.0 / (1.0 + np.exp(-z))
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


crn = json.load(open(os.path.join(V8, "crn.json")))


def boot(d, mode, evh=None):
    if mode == "iid":
        cfg = crn["bootstrap"]
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.integers(0, len(d), size=(cfg["n_boot"], len(d)))
        means = d[idx].mean(axis=1)
    else:
        cfg = crn["block_bootstrap"]
        seen, events = set(), []
        for e in evh:
            if e not in seen:
                seen.add(e)
                events.append(e)
        rows_of = {e: np.where(evh == e)[0] for e in events}
        rng = np.random.default_rng(cfg["seed"])
        bidx = rng.integers(0, len(events), size=(cfg["n_boot"], len(events)))
        means = np.array([d[np.concatenate([rows_of[events[j]] for j in bidx[r]])].mean()
                          for r in range(cfg["n_boot"])])
    return {"p_better": float((means > 0).mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5))}


z3 = np.load(os.path.join(V8, "scratch", "bias_h3", "model_probs.npz"))
p_v6, p5d = z3["p_v6"], z3["p_ss_5d"]
vw, vl = z3["vw_5d"], z3["vl_5d"]
neff_min = R / np.maximum(vw, vl)
gate = neff_min < 12
res = {"gate_counts": {"train": int((gate & ~hold).sum()),
                       "holdout": int((gate & hold).sum()),
                       "published": [330, 178]}}

# S1
pS1 = np.where(gate, p5d, p_v6)
d1 = ll(p_v6[hold]) - ll(pS1[hold])
res["S1"] = {"delta_milli_mine": round(float(d1.mean()) * 1000, 3),
             "published": 1.958,
             "iid": boot(d1, "iid"), "block": boot(d1, "block", ev[hold]),
             "gated_rows_delta_milli": round(float((ll(p_v6[hold & gate]) - ll(p5d[hold & gate])).mean()) * 1000, 2),
             "n_gated_hold": int((hold & gate).sum())}
k5 = int(math.ceil(0.05 * d1.size))
o = np.argsort(d1)[::-1]
res["S1"]["drop_top5pct_milli"] = round(float(np.delete(d1, o[:k5]).mean()) * 1000, 3)
evh = ev[hold]
jk = {e: round(float(d1[evh != e].mean()) * 1000, 3) for e in np.unique(evh)}
neg = {e: v for e, v in jk.items() if v <= 0}
res["S1"]["jackknife_flips"] = neg
res["S1"]["jackknife_min"] = min(jk.values())
# overlap between the gate and h3's thin/cold diagnostic rows
from engine import Engine  # noqa: E402
eng = Engine()
cnt = {}
games = sorted(eng.games, key=lambda g: (g["date_s"], str(g["match_id"])))
gi = 0
n = len(frame)
pm_w = np.zeros(n)
pm_l = np.zeros(n)
rows_by_date = {}
for i in range(n):
    rows_by_date.setdefault(frame.date.values[i], []).append(i)
for day in sorted(rows_by_date):
    while gi < len(games) and games[gi]["date_s"] < day:
        g = games[gi]
        cnt[g["winner"]] = cnt.get(g["winner"], 0) + 1
        cnt[g["loser"]] = cnt.get(g["loser"], 0) + 1
        gi += 1
    for i in rows_by_date[day]:
        pm_w[i] = cnt.get(frame.winner.values[i], 0)
        pm_l[i] = cnt.get(frame.loser.values[i], 0)
thin = np.minimum(pm_w, pm_l) < 30
res["S1"]["gate_thin_overlap_holdout"] = {
    "n_gate": int((hold & gate).sum()), "n_thin": int((hold & thin).sum()),
    "n_both": int((hold & gate & thin).sum())}

# S2: joint (beta,k) train fit on fade base
ze = np.load(os.path.join(V8, "scratch", "decay", "probs", "eclass_on_v6_m0.8.npz"))
rd_fade = ze["rdiff"]
feat = pd.read_csv(os.path.join(V8, "scratch", "context", "frame_features.csv"))
X1 = (1 - feat.integ_w.values) + (1 - feat.integ_l.values)
valid = ~np.isnan(rd_fade)
tr = valid & ~hold


def nll_bk(theta, mask, rdv):
    b, k = theta
    return float(ll(cf(b * rdv[mask] * np.exp(-k * X1[mask]), fmts[mask])).mean())


r2 = minimize(nll_bk, [0.13, 0.0], args=(tr, rd_fade), method="Nelder-Mead",
              options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 4000})
b2, k2 = r2.x
pS2 = cf(b2 * rd_fade * np.exp(-k2 * X1), fmts)
te = valid & hold
d2 = ll(p_v6[te]) - ll(pS2[te])
res["S2"] = {"beta": round(float(b2), 4), "k": round(float(k2), 4),
             "delta_milli_mine": round(float(d2.mean()) * 1000, 3),
             "published": -0.14, "n": int(te.sum())}

# S3: gate over S2', (beta,k) refit on non-gated train rows
tr3 = tr & ~gate
r3 = minimize(nll_bk, [0.13, 0.0], args=(tr3, rd_fade), method="Nelder-Mead",
              options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 4000})
b3, k3 = r3.x
pS2p = cf(b3 * rd_fade * np.exp(-k3 * X1), fmts)
pS3 = np.where(gate, p5d, pS2p)
d3 = ll(p_v6[te]) - ll(pS3[te])
res["S3"] = {"beta": round(float(b3), 4), "k": round(float(k3), 4),
             "delta_milli_mine": round(float(d3.mean()) * 1000, 3),
             "published": -7.874,
             "S3_vs_minS1S2_anti_synergy_falsifier_fired":
                 bool(float(d3.mean()) * 1000 < min(res["S1"]["delta_milli_mine"],
                                                    res["S2"]["delta_milli_mine"]) - 0.5)}

with open(os.path.join(HERE, "recompute_compose.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
