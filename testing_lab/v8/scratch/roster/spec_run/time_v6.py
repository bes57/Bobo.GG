"""Time one full v6 run on the frame + verify reproduction vs stored baseline.
Holdout metrics are popped unseen (reproduction check uses stored p_all only)."""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
sys.path.insert(0, TL)
sys.path.insert(0, V8)
from engine import Engine  # noqa: E402

FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"]
frame = pd.read_csv(FRAME).reset_index(drop=True)

v6 = np.load(os.path.join(V8, "scratch", "bias_h3", "v6_baseline.npz"))
p_v6, beta_v6 = v6["p_all"], float(v6["beta"][0])

t0 = time.time()
eng = Engine()
t1 = time.time()
eng.series = frame.copy().reset_index(drop=True)
eng.pred_days = sorted(frame.date.unique())
stage_by_mid = dict(zip(frame.match_id, frame.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
cfg = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
       "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
       "region_prior_ridge": 1.5,
       "decay": {"kind": "games", "consistency": (20.0, 12.0)},
       "w_custom": np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)}
out = eng.run(cfg)
t2 = time.time()
for k in ("ll_test", "brier_test", "p_test"):
    out.pop(k, None)
rd = out["rdiff"]
ok = ~np.isnan(rd)
fmts = frame.fmt.values


def p_series_closed(beta, rdiff, fm):
    pm = 1.0 / (1.0 + np.exp(-beta * rdiff))
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


p0 = np.full(len(frame), np.nan)
p0[ok] = p_series_closed(out["beta"], rd[ok], fmts[ok])
mx = float(np.nanmax(np.abs(p0 - p_v6)))
print(f"engine load {t1-t0:.1f}s | v6 run {t2-t1:.1f}s | "
      f"beta {out['beta']} (stored {beta_v6}) | max|dp| vs stored {mx:.2e} | "
      f"ll_train {out['ll_train']}")
assert mx <= 1e-12, "v6 reproduction FAILED"
print("v6 reproduction guard OK")
