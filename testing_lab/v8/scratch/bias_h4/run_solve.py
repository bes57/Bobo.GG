"""agent:bias-h4 stage 1 — v6 reconstruction on the expanded frame.

One engine solve (v6 champion config, run_v7_stage1.py BASE + consist(20,12)),
series frame replaced by the canonical frame_expanded per wave2_common.md.
Saves rdiff/beta + the engine's game list (for depth features + wr_masks) to
scratch. Aborts loudly if the frame sha256 mismatches crn.json.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
SCRATCH = os.path.join(V8, "scratch", "bias_h4")
LOG = os.path.join(V8, "logs", "bias_h4.log")
FRAME = os.path.join(V8, "data", "frame_expanded", "series.csv")
sys.path.insert(0, TL)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ── frame verification (wave2_common: abort loudly on mismatch) ──────────────
crn = json.load(open(os.path.join(V8, "crn.json")))
want = crn["frame_expanded"]["series_csv_sha256"]
got = hashlib.sha256(open(FRAME, "rb").read()).hexdigest()
if got != want:
    log(f"FATAL frame sha256 mismatch: got {got} want {want}")
    raise SystemExit(1)
frame = pd.read_csv(FRAME)
assert len(frame) == crn["frame_expanded"]["n_total"] == 2058
n_train = int((frame.date <= "2024-12-31").sum())
n_hold = int((frame.date > "2024-12-31").sum())
assert n_train == 841 and n_hold == 1217, (n_train, n_hold)
log(f"stage1: frame verified (sha ok, n=2058/{n_train}/{n_hold})")

from engine import Engine  # noqa: E402

t0 = time.time()
eng = Engine()
log(f"stage1: engine init {time.time()-t0:.1f}s; games={len(eng.games)}, "
    f"teams={len(eng.teams)}")

# replace evaluation frame per wave2_common.md
frame = frame.reset_index(drop=True)
eng.series = frame
eng.pred_days = sorted(frame.date.unique())
log(f"stage1: series frame replaced; pred_days={len(eng.pred_days)}")

# sanity: no aggregate 'all' rows in the game list; log map universe
maps_seen = sorted({g["map_name"] for g in eng.games})
assert "all" not in maps_seen and "All" not in maps_seen, maps_seen
log(f"stage1: game maps ({len(maps_seen)}): {maps_seen}")

# frame teams coverage in the solve
missing = sorted({t for t in set(frame.winner) | set(frame.loser)
                  if t not in eng.tidx})
log(f"stage1: frame teams missing from game index: {missing or 'none'}")

# v6 champion config (run_v7_stage1.py BASE + v6_consist_20_12), stages from
# the canonical frame
stage_by_mid = dict(zip(frame.match_id, frame.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups")
                    for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
V6 = {"decay": {"kind": "games", "consistency": (20.0, 12.0)},
      "rd": {"power": 0.75, "scale": 2.5},
      "roster_mode": "year", "roster_persistence": 0.3,
      "ridge": 0.5, "champ_mult": 2.0, "region_prior_ridge": 1.5,
      "w_custom": PO}
log(f"stage1: v6 solve start (playoff-weighted games: {int((PO>1).sum())})")
t0 = time.time()
out = eng.run(V6)
log(f"stage1: v6 solve done {time.time()-t0:.1f}s — beta={out['beta']} "
    f"ll_train={out['ll_train']} ll_test={out['ll_test']} "
    f"n_train={out['n_train']} n_test={out['n_test']}")

np.savez(os.path.join(SCRATCH, "v6_solve.npz"),
         rdiff=out["rdiff"], rat_w=out["rat_w"], rat_l=out["rat_l"],
         beta=np.array([out["beta"]]))
games_df = pd.DataFrame(
    [{"match_id": g["match_id"], "event_id": g["event_id"],
      "map_name": g["map_name"], "winner": g["winner"], "loser": g["loser"],
      "date_s": g["date_s"]} for g in eng.games])
games_df.to_csv(os.path.join(SCRATCH, "games.csv"), index=False)
meta = {"config": {k: (v if k != "w_custom" else "PO 1.6 playoffs/GF from frame")
                   for k, v in V6.items()},
        "beta": out["beta"], "ll_train": out["ll_train"],
        "ll_test": out["ll_test"], "n_train": out["n_train"],
        "n_test": out["n_test"], "brier_test": out["brier_test"],
        "n_games": len(eng.games), "maps_seen": maps_seen,
        "frame_sha256": got}
json.dump(meta, open(os.path.join(SCRATCH, "v6_meta.json"), "w"), indent=1)
log("stage1: artifacts saved (v6_solve.npz, games.csv, v6_meta.json)")
