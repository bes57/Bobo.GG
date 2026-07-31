"""agent:roster step 2 — v6 replay with daily ratings for case trajectories.

IDENTICAL config to scratch/bias_h3/v6_baseline.npz (a replay, not a new look;
compose_looks unit definition excludes v6 baseline replays). Guard: p_all must
match the stored vector to <=1e-12 max abs diff or abort (preregister).
Writes scratch/roster/v6_daily.npz (days x teams rating matrix + p_all).
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
TL = os.path.dirname(V8)
sys.path.insert(0, TL)

OUT = os.path.join(HERE, "v6_daily.npz")
if os.path.exists(OUT):
    print("checkpoint exists, skipping", flush=True)
    sys.exit(0)

FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], f"FRAME SHA MISMATCH {sha}"
frame = pd.read_csv(FRAME)

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
       "decay": {"kind": "games", "consistency": (20.0, 12.0)},
       "daily_out": True}
out = eng.run(cfg)

beta = out["beta"]
rdiff = out["rdiff"]
fmts = frame.fmt.values
pm = 1.0 / (1.0 + np.exp(-beta * rdiff))
p_all = np.where(np.isin(fmts, ("bo5", "bo5_gf")),
                 pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                 np.where(fmts == "bo1", pm, pm ** 2 * (3 - 2 * pm)))

stored = np.load(os.path.join(V8, "scratch", "bias_h3", "v6_baseline.npz"))
mx = float(np.nanmax(np.abs(p_all - stored["p_all"])))
print(f"replay guard: max|p_all - stored| = {mx:.2e}, beta {beta} vs "
      f"{float(stored['beta'][0])}", flush=True)
if not (mx <= 1e-12 and abs(beta - float(stored["beta"][0])) <= 1e-9):
    sys.exit("REPLAY GUARD FAILED — daily replay does not reproduce the stored "
             "v6 baseline. Abort (preregistered).")

days = sorted(out["daily_r"].keys())
R = np.stack([out["daily_r"][d] for d in days])   # (n_days, n_teams)
np.savez_compressed(OUT, R=R, days=np.array(days), teams=np.array(eng.teams),
                    p_all=p_all, rdiff=rdiff, rat_w=out["rat_w"],
                    rat_l=out["rat_l"], beta=np.array([beta]),
                    region_idx=eng.team_region_idx)
print(f"wrote {OUT}: R {R.shape}, {len(days)} solve days", flush=True)
