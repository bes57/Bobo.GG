"""Engine probe for the v6 site-deploy parity gate (agent:deploy-solve).

Run BY SUBPROCESS from scrapers/BuildRatingTimeline.py --verify-parity.
Points the lab harness at the CANDIDATE timeline JSONs (pre-promotion), runs
testing_lab/engine.py with the exact v6 champion config + daily_out, and dumps
full-precision daily ratings for the requested sample days, plus the engine's
own all-valid-series beta refit for gate-B cross-evidence.

The subprocess boundary is the point: the harness/engine pull the VCTMM
sys.path, which must never enter the site scraper process.
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--timelines", required=True,
                help="comma-separated ABSOLUTE paths to candidate timeline "
                     "JSONs (os.path.join(DATA, abspath) resolves to abspath)")
ap.add_argument("--days", required=True, help="request JSON: {days:[], today:}")
ap.add_argument("--out", required=True)
args = ap.parse_args()

LAB = "/Users/benny_es1/PythonTest/testing_lab"
sys.path.insert(0, LAB)

# Import ORDER is load-bearing: `import engine` must resolve BEFORE harness
# runs, because harness's vctmm import prepends the VCTMM *vendored* dirs to
# sys.path[0] — after which a bare `import engine` silently picks up VCTMM's
# STALE engine copy (ROOT=VCTMM, frozen 4022-game data snapshot). That exact
# failure produced the first gate-A run's 11.9-rating divergence.
import engine  # noqa: E402
assert engine.__file__.startswith(LAB), f"stale engine resolved: {engine.__file__}"
assert engine.ROOT == "/Users/benny_es1/PythonTest", f"engine ROOT {engine.ROOT}"
import harness  # noqa: E402

harness.TIMELINE_FILES = args.timelines.split(",")

Engine = engine.Engine

with open(args.days) as f:
    req = json.load(f)

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values

# playoff/GF solve weight, exactly as trading_model/build_model_snapshot.py
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)

today = req["today"]
if today not in eng.pred_days:
    eng.pred_days = sorted(set(list(eng.pred_days) + [today]))

out = eng.run({"decay": {"kind": "games", "consistency": (20.0, 12.0)},
               "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
               "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
               "region_prior_ridge": 1.5, "w_custom": PO, "daily_out": True})
daily = out["daily_r"]

# all-valid-series beta refit (same closed form / bounds as engine.run)
rdiff = out["rdiff"]
valid = ~np.isnan(out["rat_w"])


def p_series(beta, mask):
    pm = 1 / (1 + np.exp(-beta * rdiff[mask]))
    fm = fmts[mask]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


from scipy.optimize import minimize_scalar  # noqa: E402

beta_all = float(minimize_scalar(
    lambda b: -np.mean(np.log(np.clip(p_series(b, valid), 1e-9, 1))),
    bounds=(0.03, 0.6), method="bounded").x)

res = {
    "teams": eng.teams,
    "engine_file": engine.__file__,
    "pred_days_solved": sorted(daily.keys()),
    "n_games": int(len(eng.games)),
    "n_series": int(len(s)),
    "n_valid": int(valid.sum()),
    "beta_all_refit": beta_all,
    "daily": {d: [float(x) for x in daily[d]]
              for d in req["days"] if d in daily},
}
with open(args.out, "w") as f:
    json.dump(res, f)
print(f"probe: {len(daily)} solved days, {len(res['daily'])}/{len(req['days'])} "
      f"sampled days captured, beta_all={beta_all:.6f}")
