"""agent:bias-h3 — v6 baseline (consist 20/12) on the canonical expanded frame.

Checkpoint: writes scratch/bias_h3/v6_baseline.npz + v6_baseline_meta.json.
Skips itself if the checkpoint already exists (resumability).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TL = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # testing_lab
V8 = os.path.join(TL, "v8")
sys.path.insert(0, TL)

OUT_NPZ = os.path.join(HERE, "v6_baseline.npz")
OUT_META = os.path.join(HERE, "v6_baseline_meta.json")
FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")

if os.path.exists(OUT_NPZ):
    print("checkpoint exists, skipping", flush=True)
    sys.exit(0)

# frame + sha verify (abort loudly on mismatch — wave2_common law)
crn = json.load(open(os.path.join(V8, "crn.json")))
sha = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
assert sha == crn["frame_expanded"]["series_csv_sha256"], f"FRAME SHA MISMATCH {sha}"
frame = pd.read_csv(FRAME)

t0 = time.time()
from engine import Engine  # noqa: E402

eng = Engine()
eng.series = frame.reset_index(drop=True)
eng.pred_days = sorted(frame.date.unique())

# per-game playoff weight from the frame's stage (run_v7_stage1 semantics:
# stage default 'groups' -> 1.0 for maps of matches outside the frame)
stage_by_mid = dict(zip(frame.match_id, frame.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)

cfg = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
       "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
       "region_prior_ridge": 1.5, "w_custom": PO,
       "decay": {"kind": "games", "consistency": (20.0, 12.0)}}
print(f"engine ready in {time.time()-t0:.1f}s; running v6 walk-forward "
      f"({len(eng.pred_days)} days)...", flush=True)
t1 = time.time()
out = eng.run(cfg)
print(f"v6 run done in {time.time()-t1:.1f}s  beta={out['beta']} "
      f"ll_train={out['ll_train']} ll_test={out['ll_test']} n_test={out['n_test']}",
      flush=True)

# probs for ALL rows (train + holdout) with the engine's own closed form
rdiff = out["rdiff"]
beta = out["beta"]
fmts = frame.fmt.values
pm = 1.0 / (1.0 + np.exp(-beta * rdiff))
p_all = np.where(np.isin(fmts, ("bo5", "bo5_gf")),
                 pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                 np.where(fmts == "bo1", pm, pm ** 2 * (3 - 2 * pm)))

np.savez_compressed(OUT_NPZ, rdiff=rdiff, rat_w=out["rat_w"], rat_l=out["rat_l"],
                    p_all=p_all, beta=np.array([beta]),
                    valid=~np.isnan(rdiff))
meta = {"beta": beta, "ll_train": out["ll_train"], "ll_test": out["ll_test"],
        "brier_test": out["brier_test"], "n_test": out["n_test"],
        "n_train": out["n_train"], "frame_sha256": sha,
        "cfg": {k: (v if k != "w_custom" else "PO_1.6_from_frame_stage")
                for k, v in cfg.items()},
        "runtime_s": round(time.time() - t1, 1)}
with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=1)
print("checkpoint written", flush=True)
