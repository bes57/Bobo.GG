"""agent:adversary — independent recompute of Wave-2 scoring paths.

My own code end-to-end: engine solve (raw inputs) -> my own loss math ->
my own CRN bootstrap implementation from crn.json recipe. Compares against
the numbers the agents published. Nothing here imports referee.py.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
sys.path.insert(0, TL)

FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], "FRAME SHA MISMATCH"
frame = pd.read_csv(FRAME)
N = len(frame)
hold = (frame.date > "2024-12-31").values
res = {"frame_sha_ok": True, "n": N, "n_holdout": int(hold.sum())}

EPS = 1e-9


def ll(p):
    return -np.log(np.clip(p, EPS, 1.0))


def closed_form(beta, rdiff, fmts):
    pm = 1.0 / (1.0 + np.exp(-beta * rdiff))
    return np.where(np.isin(fmts, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fmts == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


# my own CRN bootstrap (recipe re-implemented from crn.json text)
def boot_iid(d):
    cfg = crn["bootstrap"]
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.integers(0, len(d), size=(cfg["n_boot"], len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "p_better": float((means > 0).mean())}


def boot_block(d, ev):
    cfg = crn["block_bootstrap"]
    seen, events = set(), []
    for e in ev:
        if e not in seen:
            seen.add(e)
            events.append(e)
    rows_of = {e: np.where(ev == e)[0] for e in events}
    rng = np.random.default_rng(cfg["seed"])
    bidx = rng.integers(0, len(events), size=(cfg["n_boot"], len(events)))
    means = np.empty(cfg["n_boot"])
    for r in range(cfg["n_boot"]):
        rows = np.concatenate([rows_of[events[j]] for j in bidx[r]])
        means[r] = d[rows].mean()
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "p_better": float((means > 0).mean())}


# ── A. independent B0 engine run from raw inputs ────────────────────────────
from engine import Engine  # noqa: E402

eng = Engine()
eng.series = frame.reset_index(drop=True)
eng.pred_days = sorted(frame.date.unique())
stage_by_mid = dict(zip(frame.match_id, frame.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
cfg = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
       "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
       "region_prior_ridge": 1.5, "w_custom": PO,
       "decay": {"kind": "games", "consistency": (20.0, 12.0)}}
out = eng.run(cfg)
rdiff_mine = out["rdiff"]
beta_mine = out["beta"]
fmts = frame.fmt.values
p_mine = closed_form(beta_mine, rdiff_mine, fmts)
valid = ~np.isnan(rdiff_mine)
tr = valid & ~hold
te = valid & hold
res["B0_mine"] = {"beta": beta_mine,
                  "ll_train": round(float(ll(p_mine[tr]).mean()), 5),
                  "ll_test": round(float(ll(p_mine[te]).mean()), 5),
                  "n_train": int(tr.sum()), "n_test": int(te.sum())}

# compare per-series probs to each agent's stored baseline
cmp = {}
for name, path, key in [
        ("bias_h3", "bias_h3/v6_baseline.npz", "p_all"),
        ("bias_h2", "bias_h2/v6_baseline.npz", "p_all"),
        ("bias_h4", "bias_h4/v6_solve.npz", None),
        ("bias_h1", "bias_h1/baseline_v6.npz", None)]:
    z = np.load(os.path.join(V8, "scratch", path))
    if key and key in z.files:
        pa = z[key]
        m = valid & ~np.isnan(pa)
        cmp[name] = {"max_abs_prob_diff": float(np.nanmax(np.abs(pa[m] - p_mine[m]))),
                     "beta_stored": float(np.atleast_1d(z["beta"])[0])}
    else:
        rd = z["rdiff"]
        m = valid & ~np.isnan(rd)
        cmp[name] = {"max_abs_rdiff_diff": float(np.nanmax(np.abs(rd[m] - rdiff_mine[m]))),
                     "beta_stored": float(np.atleast_1d(z["beta"])[0])}
res["stored_baseline_vs_mine"] = cmp

# ── B. H3 numbers from stored model probs, my loss code ─────────────────────
z3 = np.load(os.path.join(V8, "scratch", "bias_h3", "model_probs.npz"))
p_v6, p1a, p5d = z3["p_v6"], z3["p_ss_1a"], z3["p_ss_5d"]
h = hold
ev = frame.event_id.values
d_5d_v6 = (ll(p_v6[h]) - ll(p5d[h]))
d_5d_1a = (ll(p1a[h]) - ll(p5d[h]))
res["h3_check"] = {
    "ll_v6_hold": round(float(ll(p_v6[h]).mean()), 6),
    "ll_1a_hold": round(float(ll(p1a[h]).mean()), 6),
    "ll_5d_hold": round(float(ll(p5d[h]).mean()), 6),
    "delta_5d_vs_v6_milli": round(float(d_5d_v6.mean()) * 1000, 3),
    "delta_5d_vs_1a_milli": round(float(d_5d_1a.mean()) * 1000, 3),
    "boot_iid_5d_vs_1a": boot_iid(d_5d_1a),
    "boot_block_5d_vs_1a": boot_block(d_5d_1a, ev[h]),
}
# fragility of the 5d-vs-1a WIN: drop top 5% contributing series
k = int(math.ceil(0.05 * d_5d_1a.size))
order = np.argsort(d_5d_1a)[::-1]
kept = np.ones(d_5d_1a.size, bool)
kept[order[:k]] = False
res["h3_5d_vs_1a_drop_top5pct"] = {
    "k_dropped": k,
    "delta_milli_after": round(float(d_5d_1a[kept].mean()) * 1000, 3)}
# jackknife by event
jk = {}
for e in np.unique(ev[h]):
    m = ev[h] != e
    jk[e] = round(float(d_5d_1a[m].mean()) * 1000, 3)
res["h3_5d_vs_1a_jackknife_min_max"] = {
    "min_event": min(jk, key=jk.get), "min": min(jk.values()),
    "max_event": max(jk, key=jk.get), "max": max(jk.values())}

with open(os.path.join(HERE, "recompute_core.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
