"""agent:context step 2a — B0 baseline + 3a lineup-conditioned EWC weights +
3c-A elimination solve weights. Selection on TRAIN NLL only; holdout recorded
per grid point for transparency (preregistered), never used to select."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

V8 = "/Users/benny_es1/PythonTest/testing_lab/v8"
TL = os.path.dirname(V8)
SC = os.path.join(V8, "scratch", "context")
sys.path.insert(0, TL)
from engine import Engine  # noqa: E402

frame = pd.read_csv(os.path.join(SC, "frame_features.csv"))
gdf = pd.read_csv(os.path.join(SC, "game_features.csv"))

eng = Engine()
eng.series = frame[["match_id", "date", "event_id", "year", "winner", "loser",
                    "w_maps", "l_maps", "fmt", "stage", "match_name", "reg_w",
                    "reg_l", "intl", "n_maps_played"]].copy()
eng.pred_days = sorted(frame.date.unique())
assert len(gdf) == len(eng.games)
assert all(int(g["match_id"]) == m for g, m in zip(eng.games, gdf.match_id))

PO = gdf.po.values.astype(float)
EWC = gdf.ewc3d.values.astype(bool)
ELIM = gdf.elim.values.astype(bool)
IW = gdf.integ_w.values.astype(float)
IL = gdf.integ_l.values.astype(float)

BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
        "region_prior_ridge": 1.5,
        "decay": {"kind": "games", "consistency": (20.0, 12.0)}}


def run_cfg(w_custom, champ_mult=2.0):
    if hasattr(eng, "_prev_rvec"):
        del eng._prev_rvec           # no cross-config warm-start leak
    cfg = {**BASE, "champ_mult": champ_mult, "w_custom": w_custom}
    t0 = time.time()
    out = eng.run(cfg)
    out["secs"] = round(time.time() - t0, 1)
    return out


results = {"grids": {}}
t_all = time.time()

# ── B0 ──────────────────────────────────────────────────────────────────────
b0 = run_cfg(PO)
test = b0["test_mask"]
loss_b0 = -np.log(np.clip(b0["p_test"], 1e-9, 1))
print(f"B0: beta={b0['beta']} ll_train={b0['ll_train']} ll_test={b0['ll_test']} "
      f"n_test={b0['n_test']} ({b0['secs']}s)", flush=True)
np.savez(os.path.join(SC, "b0.npz"),
         rdiff=b0["rdiff"], p_test=b0["p_test"], test_mask=test,
         beta=b0["beta"], loss_b0=loss_b0,
         event_ids=frame.event_id.values[test].astype(str))
results["B0"] = {"beta": b0["beta"], "ll_train": b0["ll_train"],
                 "ll_test": b0["ll_test"], "n_test": b0["n_test"],
                 "n_train": b0["n_train"]}


def grid_run(tag, weights_of):
    rows = []
    for x, w in weights_of:
        out = run_cfg(w)
        assert (out["test_mask"] == test).all()
        d = loss_b0 - (-np.log(np.clip(out["p_test"], 1e-9, 1)))
        rows.append({"x": x, "beta": out["beta"], "ll_train": out["ll_train"],
                     "ll_test": out["ll_test"],
                     "dll_holdout_milli": round(float(d.mean()) * 1000, 3)})
        np.save(os.path.join(SC, f"ptest_{tag}_{x:.2f}.npy"), out["p_test"])
        print(f"  {tag} x={x:.2f} ll_train={out['ll_train']:.5f} "
              f"ll_test={out['ll_test']:.5f} ({out['secs']}s)", flush=True)
    results["grids"][tag] = rows


# ── 3a integrity-conditioned: f(x) = w0 + (1-w0)x, w = PO * f(iw)f(il) on EWC ─
def w_integrity(w0):
    f = lambda x: w0 + (1 - w0) * x
    return PO * np.where(EWC, f(IW) * f(IL), 1.0)


grid_run("3a_integrity", [(w0, w_integrity(w0))
                          for w0 in np.round(np.arange(0.0, 1.01, 0.1), 2)])

# ── 3a blanket: w = PO * w_e on EWC games ───────────────────────────────────
grid_run("3a_blanket", [(we, PO * np.where(EWC, we, 1.0))
                        for we in np.round(np.arange(0.4, 1.21, 0.1), 2)])

# ── 3c-A elimination solve weight ───────────────────────────────────────────
grid_run("3cA_elim", [(wl, PO * np.where(ELIM, wl, 1.0))
                      for wl in (0.7, 0.85, 1.0, 1.15, 1.3, 1.5)])

with open(os.path.join(SC, "step2a_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print(f"step2a DONE in {time.time()-t_all:.0f}s", flush=True)
