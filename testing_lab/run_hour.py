"""One-hour push: OT margins, rounds-ratio margins, region-prior ridge,
series-counted decay, piecewise beta link, cold-start percentile, and the
closed-form x MC ensemble. All on the candidate base, native data.
Writes out/hour.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine
from harness import paired_bootstrap
from scipy.optimize import minimize, minimize_scalar

OUT = os.path.join(HERE, "out")

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
BASE = {"decay": {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0},
        "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
        "w_custom": PO}
results, keep = {}, {}


def series_pv(b, rdv, mask):
    pm = 1 / (1 + np.exp(-b * rdv[mask]))
    fm = fmts[mask]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def nll(b, rdv, mask):
    return -np.mean(np.log(np.clip(series_pv(b, rdv, mask), 1e-9, 1)))


def score(name, rdv):
    v = ~np.isnan(rdv)
    b = float(minimize_scalar(lambda x: nll(x, rdv, v & train_v),
                              bounds=(0.02, 0.6), method="bounded").x)
    results[name] = {"beta": round(b, 4),
                     "ll_test": round(float(nll(b, rdv, v & test_v)), 5)}
    keep[name] = (rdv, b, v)
    print(f"{name:<28} b={b:.3f} ll={results[name]['ll_test']:.5f}", flush=True)


score("base_asym", eng.run(BASE)["rdiff"])

# A) OT margin handling: OT wins (wr>13) -> reduced effective margin
raw = eng.rd_raw
is_ot = np.array([g["wr"] > 13 for g in eng.games])
for otm in (1.0, 1.5):
    rd_eff = np.where(is_ot, otm, np.abs(raw))
    rd_c = np.copysign(np.abs(rd_eff) ** 0.75 * 2.5, raw)
    score(f"ot_margin_{otm}", eng.run({**BASE, "rd_custom": rd_c})["rdiff"])

# B) rounds-ratio margin: rd / total_rounds, rescaled
tot = np.array([g["wr"] + g["lr"] for g in eng.games], dtype=float)
ratio = raw / np.maximum(tot, 1)
rd_ratio = np.copysign(np.abs(ratio) ** 0.75, ratio) * 2.5 * (24 ** 0.75)
score("rounds_ratio", eng.run({**BASE, "rd_custom": rd_ratio})["rdiff"])

# C) region-prior ridge
for rpr in (0.3, 0.8):
    score(f"region_ridge_{rpr}", eng.run({**BASE, "region_prior_ridge": rpr})["rdiff"])

# D) series-counted decay (HL in series ~ maps/2.3)
for hls, hll in ((8.0, 5.0), (10.0, 6.0)):
    score(f"series_cnt_w{int(hls)}l{int(hll)}", eng.run(
        {**BASE, "decay": {"kind": "games", "hl_games": hls,
                           "hl_games_loss": hll, "count": "series"}})["rdiff"])

# E) piecewise beta link (small vs large gaps, threshold 3), fit on train
rd0, b0, v0 = keep["base_asym"]
def pw_nll(params, mask):
    bs, bl = params
    a = np.abs(rd0[mask])
    b_eff = np.where(a < 3.0, bs, bl)
    z = b_eff * rd0[mask]
    pm = 1 / (1 + np.exp(-z))
    fm = fmts[mask]
    p = np.where(np.isin(fm, ("bo5", "bo5_gf")),
                 pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                 np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))
    return -np.mean(np.log(np.clip(p, 1e-9, 1)))


r = minimize(lambda pr: pw_nll(pr, v0 & train_v), [b0, b0], method="Nelder-Mead")
ll_pw = float(pw_nll(r.x, v0 & test_v))
results["piecewise_beta"] = {"b_small": round(float(r.x[0]), 4),
                             "b_large": round(float(r.x[1]), 4),
                             "ll_test": round(ll_pw, 5)}
print(f"piecewise_beta b_small={r.x[0]:.3f} b_large={r.x[1]:.3f} ll={ll_pw:.5f}")

lb = sorted(((k, v) for k, v in results.items()), key=lambda kv: kv[1]["ll_test"])
print("\n== LEADERBOARD ==")
for name, r_ in lb:
    print(f"  {r_['ll_test']:.5f}  {name}")

# bootstrap best vs base
best = lb[0][0]
if best != "base_asym" and best in keep:
    rd_a, b_a, v_a = keep[best]
    vv = v_a & v0 & test_v
    bt = paired_bootstrap(series_pv(b_a, rd_a, vv), series_pv(b0, rd0, vv))
    results["_boot_best"] = {"best": best, **bt}
    print("boot best vs base:", bt)

with open(os.path.join(OUT, "hour.json"), "w") as f:
    json.dump(results, f, indent=1)
np.savez_compressed(os.path.join(OUT, "hour_rdiffs.npz"),
                    **{k: v[0] for k, v in keep.items()})
print("saved out/hour.json")
