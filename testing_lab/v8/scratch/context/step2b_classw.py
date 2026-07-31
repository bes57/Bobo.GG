"""agent:context step 2b — 3d learned event-class solve weights.
Nelder-Mead on train NLL (log-space, vct_regular anchored 1.0), then holdout
eval + per-weight 1-D profile grids for CRN bootstrap CIs (computed here from
per-train-row losses; crn iid seed, full-matrix recipe)."""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

V8 = "/Users/benny_es1/PythonTest/testing_lab/v8"
TL = os.path.dirname(V8)
SC = os.path.join(V8, "scratch", "context")
sys.path.insert(0, TL)
from engine import Engine  # noqa: E402

frame = pd.read_csv(os.path.join(SC, "frame_features.csv"))
gdf = pd.read_csv(os.path.join(SC, "game_features.csv"))
crn = json.load(open(os.path.join(V8, "crn.json")))

eng = Engine()
eng.series = frame[["match_id", "date", "event_id", "year", "winner", "loser",
                    "w_maps", "l_maps", "fmt", "stage", "match_name", "reg_w",
                    "reg_l", "intl", "n_maps_played"]].copy()
eng.pred_days = sorted(frame.date.unique())
ECL = gdf.eclass.values
CLS = ["vct_playoffs", "champions", "masters", "ewc_offseason"]
MASKS = {c: ECL == c for c in CLS}

BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 1.0,
        "region_prior_ridge": 1.5,
        "decay": {"kind": "games", "consistency": (20.0, 12.0)}}

fmts = frame.fmt.values
train_m = (frame.date <= "2024-12-31").values


def sp(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def run_w(wts):
    w = np.ones(len(gdf))
    for c in CLS:
        w[MASKS[c]] = wts[c]
    if hasattr(eng, "_prev_rvec"):
        del eng._prev_rvec
    return eng.run({**BASE, "w_custom": w})


hist = []


def obj(logw):
    wts = dict(zip(CLS, np.exp(logw)))
    out = run_w(wts)
    hist.append({"w": {c: round(float(wts[c]), 4) for c in CLS},
                 "ll_train": out["ll_train"], "beta": out["beta"]})
    if len(hist) % 20 == 0:
        print(f"  eval {len(hist)}: {hist[-1]}", flush=True)
    return out["ll_train"]


t0 = time.time()
x0 = np.log([1.6, 2.0, 1.3, 1.0])
fit = minimize(obj, x0, method="Nelder-Mead",
               options={"maxiter": 200, "fatol": 1e-5, "xatol": 0.02})
w_fit = dict(zip(CLS, np.exp(fit.x)))
print(f"fitted ({len(hist)} evals, {time.time()-t0:.0f}s): "
      f"{ {c: round(v,3) for c,v in w_fit.items()} } ll_train={fit.fun:.5f}",
      flush=True)

# holdout eval of fitted config + delta vs B0
b0 = np.load(os.path.join(SC, "b0.npz"))
out_f = run_w(w_fit)
assert (out_f["test_mask"] == b0["test_mask"]).all()
loss_f = -np.log(np.clip(out_f["p_test"], 1e-9, 1))
np.save(os.path.join(SC, "ptest_3d_fitted.npy"), out_f["p_test"])
res = {"fitted": {c: round(float(w_fit[c]), 4) for c in CLS},
       "anchor": "vct_regular=1.0", "n_evals": len(hist),
       "ll_train_fitted": out_f["ll_train"], "beta_fitted": out_f["beta"],
       "ll_test_fitted": out_f["ll_test"],
       "ll_test_B0": float(b0["loss_b0"].mean()),
       "dll_holdout_milli": round(float((b0["loss_b0"] - loss_f).mean()) * 1000, 3),
       "history_tail": hist[-5:]}
print("holdout: fitted %.5f vs B0 %.5f (d=%+.3fm)" % (
    res["ll_test_fitted"], res["ll_test_B0"], res["dll_holdout_milli"]), flush=True)

# ── per-weight profile grids → per-train-row loss matrices ──────────────────
GRID = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.4]
profiles = {}
loss_mats = {}
for c in CLS:
    mat = []
    row_ll = []
    for gv in GRID:
        wts = dict(w_fit)
        wts[c] = gv
        out = run_w(wts)
        rd, beta = out["rdiff"], out["beta"]
        v = ~np.isnan(rd)
        m = v & train_m
        p_tr = sp(1 / (1 + np.exp(-beta * rd[m])), fmts[m])
        lv = -np.log(np.clip(p_tr, 1e-9, 1))
        # engine rounds ll_train to 5 dp — allow round-off
        assert abs(float(lv.mean()) - out["ll_train"]) < 2e-5
        mat.append(lv)
        row_ll.append({"w": gv, "ll_train": out["ll_train"],
                       "ll_test": out["ll_test"]})
        print(f"  profile {c} w={gv} ll_train={out['ll_train']:.5f}", flush=True)
    loss_mats[c] = np.array(mat)          # (10, n_train_valid)
    profiles[c] = row_ll

# CRN bootstrap CI of the 1-D argmin (train rows; crn iid seed, full matrix)
seed = crn["bootstrap"]["seed"]
n_boot = crn["bootstrap"]["n_boot"]
n_tr = loss_mats[CLS[0]].shape[1]
rng = np.random.default_rng(seed)
idx = rng.integers(0, n_tr, size=(n_boot, n_tr))
cis = {}
for c in CLS:
    mat = loss_mats[c]                    # (10, n_tr)
    boot_ll = mat[:, idx].mean(axis=2)    # (10, n_boot)
    pick = np.array(GRID)[np.argmin(boot_ll, axis=0)]
    cis[c] = {"ci_lo": float(np.percentile(pick, 2.5)),
              "ci_hi": float(np.percentile(pick, 97.5)),
              "argmin_dist": {str(g): int((pick == g).sum()) for g in GRID},
              "profile_argmin_w": float(GRID[int(np.argmin([r['ll_train'] for r in profiles[c]]))])}
    print(f"  CI {c}: [{cis[c]['ci_lo']}, {cis[c]['ci_hi']}]", flush=True)

res["profiles"] = profiles
res["ci"] = cis
res["ci_recipe"] = (f"1-D profile argmin over grid {GRID}, others at fitted; "
                    f"CRN iid resample of the {n_tr} valid train rows, seed "
                    f"{seed}, n_boot {n_boot}, full-matrix recipe; declared "
                    "approximation: ignores cross-weight covariance")
with open(os.path.join(SC, "step2b_results.json"), "w") as f:
    json.dump(res, f, indent=1)
np.savez(os.path.join(SC, "profile_losses.npz"),
         **{c: loss_mats[c] for c in CLS})
print(f"step2b DONE in {time.time()-t0:.0f}s", flush=True)
