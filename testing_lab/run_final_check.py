"""Promotion-gate check: candidate (hl13/pow0.75/rp0.3, refit beta) vs
production config on the 2025-26 holdout, per bucket + overall bootstrap.
Writes out/final_check.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine
from harness import paired_bootstrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CAND = {"decay": {"kind": "exp", "hl": 13.0}, "rd": {"power": 0.75, "scale": 2.5},
        "roster_mode": "year", "roster_persistence": 0.3,
        "ridge": 0.5, "champ_mult": 2.0}
PROD = {"decay": {"kind": "exp", "hl": 6.0}, "rd": {"power": 0.5, "scale": 2.5},
        "roster_mode": "year", "roster_persistence": 0.3,
        "ridge": 0.5, "champ_mult": 2.0}

eng = Engine()
s = eng.series.reset_index(drop=True)
oc = eng.run(CAND)
op = eng.run(PROD)
test = oc["test_mask"] & op["test_mask"]
pc, pp = oc["p_test"], op["p_test"]
# p_test arrays are aligned to each run's own test mask — recompute aligned
def probs(out):
    rd, b = out["rdiff"], out["beta"]
    pm = 1 / (1 + np.exp(-b * rd))
    fm = s.fmt.values
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


pc, pp = probs(oc), probs(op)
res = {"beta_cand": oc["beta"], "beta_prod": op["beta"]}


def ll(p, m):
    return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))


res["overall"] = {"n": int(test.sum()), "cand": round(ll(pc, test), 5),
                  "prod": round(ll(pp, test), 5)}
res["boot"] = paired_bootstrap(pc[test], pp[test])
print("overall:", res["overall"], "\nboot:", res["boot"])

buckets = []
def add(mask, name):
    m = test & mask
    if m.sum() < 15:
        return
    buckets.append({"name": name, "n": int(m.sum()),
                    "cand": round(ll(pc, m), 5), "prod": round(ll(pp, m), 5),
                    "delta": round(ll(pp, m) - ll(pc, m), 5)})


for yr in (2025, 2026):
    add((s.year == yr).values, f"year {yr}")
for f in ("bo3", "bo5", "bo5_gf"):
    add((s.fmt == f).values, f"format {f}")
for st in ("groups", "playoffs", "grand_final"):
    add((s.stage == st).values, f"stage {st}")
add(s.intl.values, "international")
add(~s.intl.values, "domestic")
add(((s.reg_w == "CN") | (s.reg_l == "CN")).values, "CN involved")
add((s.reg_w != s.reg_l).values, "cross-region")
same = (s.reg_w == s.reg_l)
for reg in ("Americas", "EMEA", "Pacific", "CN"):
    add((same & (s.reg_w == reg)).values, f"domestic {reg}")
gap = np.abs(oc["rdiff"])
add(gap < 1.5, "close matchups (<1.5)")
add((gap >= 1.5) & (gap < 4), "mid gap [1.5,4)")
add(gap >= 4, "large gap (>=4)")
res["buckets"] = buckets
print(f"\n{'bucket':<24}{'n':>5}  {'cand':>8}{'prod':>9}{'delta':>9}")
worse = 0
for b in buckets:
    flag = " <-- worse" if b["delta"] < -0.004 else ""
    if b["delta"] < -0.004:
        worse += 1
    print(f"{b['name']:<24}{b['n']:>5}  {b['cand']:>8.5f}{b['prod']:>9.5f}"
          f"{b['delta']:>+9.5f}{flag}")
res["n_buckets_worse"] = worse

with open(os.path.join(OUT, "final_check.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nsaved out/final_check.json")
