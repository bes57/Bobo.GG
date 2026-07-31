"""agent:context step 4c — labeled SENSITIVITY for the w_ewc CI: same
preregistered profile grid + CRN argmin-bootstrap recipe as 3d, but with the
other weights held at their v6 hand-set values (PO 1.6 / champions x2)
instead of at the fitted point (whose champions collapse makes 'at fitted'
conditioning unrepresentative). Train rows only."""
import json
import os
import sys

import numpy as np
import pandas as pd

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
PO = gdf.po.values.astype(float)
EWC = gdf.ewc3d.values.astype(bool)
fmts = frame.fmt.values
train_m = (frame.date <= "2024-12-31").values

BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
        "region_prior_ridge": 1.5,
        "decay": {"kind": "games", "consistency": (20.0, 12.0)}}


def sp(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


GRID = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.4]
mat, prof = [], []
for gv in GRID:
    if hasattr(eng, "_prev_rvec"):
        del eng._prev_rvec
    out = eng.run({**BASE, "w_custom": PO * np.where(EWC, gv, 1.0)})
    rd, beta = out["rdiff"], out["beta"]
    m = ~np.isnan(rd) & train_m
    lv = -np.log(np.clip(sp(1 / (1 + np.exp(-beta * rd[m])), fmts[m]), 1e-9, 1))
    assert abs(float(lv.mean()) - out["ll_train"]) < 2e-5
    mat.append(lv)
    prof.append({"w": gv, "ll_train": out["ll_train"], "ll_test": out["ll_test"]})
    print(f"  w_ewc={gv} ll_train={out['ll_train']:.5f}", flush=True)
mat = np.array(mat)
rng = np.random.default_rng(crn["bootstrap"]["seed"])
idx = rng.integers(0, mat.shape[1], size=(crn["bootstrap"]["n_boot"], mat.shape[1]))
boot_ll = mat[:, idx].mean(axis=2)
pick = np.array(GRID)[np.argmin(boot_ll, axis=0)]
res = {"grid": prof,
       "argmin_w": float(GRID[int(np.argmin([r["ll_train"] for r in prof]))]),
       "ci_lo": float(np.percentile(pick, 2.5)),
       "ci_hi": float(np.percentile(pick, 97.5)),
       "argmin_dist": {str(g): int((pick == g).sum()) for g in GRID},
       "label": "SENSITIVITY: w_ewc profile with other weights at v6 "
                "hand-set; same preregistered grid + CRN argmin-bootstrap "
                "recipe as 3d"}
json.dump(res, open(os.path.join(SC, "step4c_ewc_profile_v6.json"), "w"), indent=1)
print("step4c DONE:", {k: res[k] for k in ("argmin_w", "ci_lo", "ci_hi")}, flush=True)
