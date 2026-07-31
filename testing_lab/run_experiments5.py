"""Round 5 — interrogating the half-life result (user: '13w too high').
Mechanism hypotheses: (a) games-played decay (breaks shouldn't burn info),
(b) short calendar HL + soft break boundaries, (c) class-vs-form two-timescale
blend, (d) short HL + heavy ridge (variance story). All rd pow0.75/rp0.3.
Writes out/experiments5.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0}

eng = Engine()
s = eng.series
results, rdiffs = {}, {}
train_m = None
test_m = None
fmts = s.fmt.values
y25 = (s.date > "2024-12-31") & (s.date <= "2025-12-31")
y26 = s.date > "2025-12-31"


def fit_score(name, rdiff, valid):
    global train_m, test_m
    from scipy.optimize import minimize_scalar
    train = valid & (s.date <= "2024-12-31").values
    test = valid & (s.date > "2024-12-31").values

    def p_series(b, mask):
        pm = 1 / (1 + np.exp(-b * rdiff[mask]))
        fm = fmts[mask]
        return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                        pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                        np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))

    def nll(b, mask):
        return -np.mean(np.log(np.clip(p_series(b, mask), 1e-9, 1)))

    b = float(minimize_scalar(lambda x: nll(x, train), bounds=(0.02, 0.6),
                              method="bounded").x)
    ll = float(nll(b, test))
    results[name] = {"beta": round(b, 4), "ll_test": round(ll, 5),
                     "ll_25": round(float(nll(b, test & y25.values)), 5),
                     "ll_26": round(float(nll(b, test & y26.values)), 5)}
    rdiffs[name] = rdiff
    print(f"{name:<28} b={b:.3f} ll={ll:.5f} "
          f"(25 {results[name]['ll_25']:.5f} / 26 {results[name]['ll_26']:.5f})",
          flush=True)


def run(name, cfg):
    t0 = time.time()
    out = eng.run({**BASE, **cfg})
    valid = ~np.isnan(out["rdiff"])
    fit_score(name, out["rdiff"], valid)
    results[name]["secs"] = round(time.time() - t0, 1)
    return out


# references
o_prod = run("ref_prod_hl6_pow05", {"decay": {"kind": "exp", "hl": 6.0},
                                    "rd": {"power": 0.5, "scale": 2.5}})
o_hl6 = run("ref_hl6", {"decay": {"kind": "exp", "hl": 6.0}})
o_hl13 = run("ref_hl13_cand", {"decay": {"kind": "exp", "hl": 13.0}})
o_hl26 = run("ref_hl26", {"decay": {"kind": "exp", "hl": 26.0}})
o_hl4 = run("ref_hl4", {"decay": {"kind": "exp", "hl": 4.0}})

# (a) games-played decay
for hg in (8.0, 12.0, 16.0, 24.0):
    run(f"games_hl{int(hg)}", {"decay": {"kind": "games", "hl_games": hg}})

# (b) short calendar HL + soft break boundaries
for hl in (6.0, 8.0):
    for gmm in (0.5, 0.7, 0.85):
        run(f"break_hl{int(hl)}_g{gmm}",
            {"decay": {"kind": "exp", "hl": hl}, "break_gamma": gmm})

# (d) short HL + heavy ridge
for hl in (6.0, 8.0):
    for rg in (2.0, 4.0, 8.0):
        run(f"ridge_hl{int(hl)}_r{rg}",
            {"decay": {"kind": "exp", "hl": hl}, "ridge": rg})

# (c) two-timescale blends (class vs form) from reference runs
valid_all = ~(np.isnan(o_hl4["rdiff"]) | np.isnan(o_hl26["rdiff"]) |
              np.isnan(o_hl6["rdiff"]) | np.isnan(o_hl13["rdiff"]))
for wname, short_o, long_o in [("hl4xhl26", o_hl4, o_hl26),
                               ("hl6xhl26", o_hl6, o_hl26),
                               ("hl4xhl13", o_hl4, o_hl13)]:
    for w in (0.3, 0.5, 0.7):
        fit_score(f"blend_{wname}_w{w}",
                  w * short_o["rdiff"] + (1 - w) * long_o["rdiff"], valid_all)

lb = sorted(results.items(), key=lambda kv: kv[1]["ll_test"])
print("\n== TOP 14 ==")
for name, r in lb[:14]:
    print(f"  {r['ll_test']:.5f} (25 {r['ll_25']:.5f} / 26 {r['ll_26']:.5f}) {name}")

with open(os.path.join(OUT, "experiments5.json"), "w") as f:
    json.dump(results, f, indent=1)
np.savez_compressed(os.path.join(OUT, "exp5_rdiffs.npz"),
                    **{k: v for k, v in rdiffs.items()})
print("saved out/experiments5.json")
