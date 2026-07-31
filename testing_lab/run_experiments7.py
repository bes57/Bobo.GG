"""Round 7: stakes weighting (bo1 down / playoffs up), roster modes under
games decay, and combined final package scoring. Writes out/experiments7.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine
from harness import paired_bootstrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE = {"decay": {"kind": "games", "hl_games": 16.0},
        "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0}

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values
results, rdiffs = {}, {}

# per-game stake class from the series table (match_id -> fmt/stage)
fmt_by_mid = dict(zip(s.match_id, s.fmt))
stage_by_mid = dict(zip(s.match_id, s.stage))
g_fmt = np.array([fmt_by_mid.get(g["match_id"], "bo3") for g in eng.games])
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])


def fit_score(name, rdiff, valid):
    from scipy.optimize import minimize_scalar

    def p_vec(b, mask):
        pm = 1 / (1 + np.exp(-b * rdiff[mask]))
        fm = fmts[mask]
        return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                        pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                        np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))

    def nll(b, mask):
        return -np.mean(np.log(np.clip(p_vec(b, mask), 1e-9, 1)))

    b = float(minimize_scalar(lambda x: nll(x, valid & train_v),
                              bounds=(0.02, 0.6), method="bounded").x)
    results[name] = {"beta": round(b, 4),
                     "ll_test": round(float(nll(b, valid & test_v)), 5)}
    rdiffs[name] = (rdiff, b)
    print(f"{name:<28} b={b:.3f} ll={results[name]['ll_test']:.5f}", flush=True)


def run(name, cfg):
    out = eng.run({**BASE, **cfg})
    fit_score(name, out["rdiff"], ~np.isnan(out["rdiff"]))


run("g16_base", {})

# stakes weighting
for bo1w, pow_ in [(0.5, 1.0), (0.7, 1.0)]:
    wc = np.where(g_fmt == "bo1", bo1w, 1.0)
    run(f"stake_bo1w{bo1w}", {"w_custom": wc})
for pw in (1.3, 1.6):
    wc = np.where(np.isin(g_stage, ("playoffs", "grand_final")), pw, 1.0)
    run(f"stake_po{pw}", {"w_custom": wc})
wc = np.where(g_fmt == "bo1", 0.6, 1.0) * \
     np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.3, 1.0)
run("stake_both", {"w_custom": wc})

# roster modes under games decay
run("g16_roster_none", {"roster_mode": "none"})

# intl-context margin: down-weight intl blowouts? (groups at intls often
# lopsided cross-region); test intl weight 0.8 / 1.2
from engine import Engine as _E
intl_flag = np.array([("masters" in g["event_id"]) or ("champions" in g["event_id"])
                      or ("lock_in" in g["event_id"]) for g in eng.games])
for iw in (0.8, 1.2):
    wc = np.where(intl_flag, iw, 1.0)
    run(f"intl_w{iw}", {"w_custom": wc})

# bootstrap best vs g16_base
lb = sorted(((k, v) for k, v in results.items()), key=lambda kv: kv[1]["ll_test"])
print("\n== LEADERBOARD ==")
for name, r_ in lb:
    print(f"  {r_['ll_test']:.5f}  {name}")
best = lb[0][0]
if best != "g16_base":
    rd_a, b_a = rdiffs[best]
    rd_b, b_b = rdiffs["g16_base"]
    vv = ~np.isnan(rd_a) & ~np.isnan(rd_b) & test_v

    def pv(rd, b, mask):
        pm = 1 / (1 + np.exp(-b * rd[mask]))
        fm = fmts[mask]
        return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                        pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                        np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))
    results["_boot_best_vs_base"] = {"best": best,
                                     **paired_bootstrap(pv(rd_a, b_a, vv), pv(rd_b, b_b, vv))}
    print("boot best vs g16_base:", results["_boot_best_vs_base"])

with open(os.path.join(OUT, "experiments7.json"), "w") as f:
    json.dump(results, f, indent=1)
print("saved out/experiments7.json")
