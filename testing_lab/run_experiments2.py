"""Round 2: interactions of round-1 winners (long memory x roster persistence
x margin power), favorite-margin discount ('top teams don't try'), refined
roster modes, and per-year robustness splits. Writes out/experiments2.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
PROD = {"decay": {"kind": "exp", "hl": 6.0}, "rd": {"power": 0.5, "scale": 2.5},
        "roster_mode": "year", "roster_persistence": 0.3,
        "ridge": 0.5, "champ_mult": 2.0}

eng = Engine()
s = eng.series
y25 = (s.date > "2024-12-31") & (s.date <= "2025-12-31")
y26 = s.date > "2025-12-31"


def score(out):
    ll = {}
    for lab, ym in (("25", y25.values), ("26", y26.values)):
        m = out["test_mask"] & ym
        p = None
        # recompute series probs for the subset from stored rdiff
        import math
        b = out["beta"]
        rd = out["rdiff"][m]
        fm = s.fmt.values[m]
        pm = 1 / (1 + np.exp(-b * rd))
        p = np.where(np.isin(fm, ("bo5", "bo5_gf")),
                     pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                     np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))
        ll[lab] = round(float(-np.mean(np.log(np.clip(p, 1e-9, 1)))), 5)
    return ll


results = {}
probs = {}


def run(name, cfg):
    t0 = time.time()
    out = eng.run(cfg)
    yr = score(out)
    results[name] = {"beta": out["beta"], "ll_test": out["ll_test"],
                     "brier_test": out["brier_test"], "ll_2025": yr["25"],
                     "ll_2026": yr["26"]}
    probs[name] = out["p_test"]
    print(f"{name:<34} b={out['beta']:.3f} ll={out['ll_test']:.5f} "
          f"25={yr['25']:.5f} 26={yr['26']:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return out


run("prod_baseline", dict(PROD))

# interaction grid
for dk, dname in [({"kind": "exp", "hl": 10.0}, "hl10"),
                  ({"kind": "exp", "hl": 13.0}, "hl13"),
                  ({"kind": "exp", "hl": 16.0}, "hl16"),
                  ({"kind": "power", "tau": 8.0, "alpha": 1.2}, "pw8"),
                  ({"kind": "power", "tau": 12.0, "alpha": 1.2}, "pw12"),
                  ({"kind": "boxexp", "c": 6.0, "hl": 10.0}, "bx6_10")]:
    for rp in (0.3, 0.5, 0.7, 0.85):
        for pw in (0.5, 0.75, 1.0):
            run(f"{dname}_rp{rp}_pow{pw}",
                {**PROD, "decay": dk, "roster_persistence": rp,
                 "rd": {"power": pw, "scale": 2.5}})

lb = sorted(results.items(), key=lambda kv: kv[1]["ll_test"])
print("\n== TOP 12 ==")
for name, r in lb[:12]:
    print(f"  {r['ll_test']:.5f} (25 {r['ll_2025']:.5f} / 26 {r['ll_2026']:.5f}) {name}")

# paired bootstrap top vs prod
base = probs["prod_baseline"]
top_name = lb[0][0]
la = -np.log(np.clip(probs[top_name], 1e-9, 1))
lbase = -np.log(np.clip(base, 1e-9, 1))
d = lbase - la
rng = np.random.default_rng(11)
means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
results["_boot_top_vs_prod"] = {
    "top": top_name, "mean_delta": float(d.mean()),
    "ci_lo": float(np.percentile(means, 2.5)),
    "ci_hi": float(np.percentile(means, 97.5)),
    "p_better": float((means > 0).mean())}
print("\nboot top vs prod:", results["_boot_top_vs_prod"])

with open(os.path.join(OUT, "experiments2.json"), "w") as f:
    json.dump(results, f, indent=1)
np.savez_compressed(os.path.join(OUT, "exp2_probs.npz"), **probs)
print("saved out/experiments2.json")
