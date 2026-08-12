"""v10 — unconditional year isolation, era-transfer evaluation.

Preregistered in testing_lab/v10/preregister.v10.md. Governing law:
v9_transfer_protocol.json (T1: beta on FIT1 -> score VAL1; T2: beta on FIT2 ->
score VAL2; advance needs A1-A5, winning BOTH eras).

No hyperparameter is tuned here. Every arm is a re-parameterisation of a
mechanism that already exists, so there is nothing to fit on FIT1 and no grid
to search. Beta is refit per arm per window, paired against v6 refit
identically on the same window.

Writes testing_lab/v10/stats/v10_transfer.json.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import Engine                     # noqa: E402
from harness import paired_bootstrap          # noqa: E402

OUT = os.path.join(HERE, "stats")
os.makedirs(OUT, exist_ok=True)

# v6 champion config, exactly as deployed (engine_probe.py:58-61).
V6 = {"decay": {"kind": "games", "consistency": (20.0, 12.0)},
      "rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
      "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
      "region_prior_ridge": 1.5}

ARMS = {
    "v6":          dict(V6),
    # the operator's hypothesis: no cross-year information at all
    "A_iso_hard":  dict(V6, year_isolated=True),
    # controls that turn the endpoint into a curve
    "A_iso_none":  dict(V6, roster_mode="none"),   # the OTHER endpoint (full carryover)
}

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
date = s.date.values

FIT1 = (date <= "2024-12-31")
VAL1 = (date > "2024-12-31") & (date <= "2025-12-31")
FIT2 = (date <= "2025-12-31")
VAL2 = (date > "2025-12-31") & (date <= "2026-07-28")

print(f"frame: n={len(s)}  FIT1={FIT1.sum()}  VAL1={VAL1.sum()}  "
      f"FIT2={FIT2.sum()}  VAL2={VAL2.sum()}")


def p_vec(b, rdiff, mask):
    pm = 1 / (1 + np.exp(-b * rdiff[mask]))
    fm = fmts[mask]
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


def nll(b, rdiff, mask):
    return -np.mean(np.log(np.clip(p_vec(b, rdiff, mask), 1e-9, 1)))


def fit_beta(rdiff, mask):
    # protocol C1: bounded minimize_scalar, xatol 1e-6, mean series-NLL
    return float(minimize_scalar(lambda x: nll(x, rdiff, mask),
                                 bounds=(0.001, 1.0), method="bounded",
                                 options={"xatol": 1e-6}).x)


rd = {}
for name, cfg in ARMS.items():
    print(f"  solving {name} ...", flush=True)
    out = eng.run(cfg)
    rd[name] = out["rdiff"] if isinstance(out, dict) and "rdiff" in out else out
    if not isinstance(rd[name], np.ndarray):
        rd[name] = np.asarray(rd[name], dtype=float)

valid = {n: np.isfinite(v) for n, v in rd.items()}
common = np.logical_and.reduce(list(valid.values()))
print(f"  scoreable in every arm: {common.sum()} / {len(s)}")

# beta fixture (protocol C1): beta(FIT1, v6) must be 0.1152 +/- 1e-3
b_fix = fit_beta(rd["v6"], common & FIT1)
print(f"  FIXTURE beta(FIT1, v6) = {b_fix:.6f}  (required 0.1152 +/- 1e-3)")
fixture_ok = abs(b_fix - 0.1152) <= 1e-3

res = {"frame_n": int(len(s)),
       "scoreable_common": int(common.sum()),
       "fixture_beta_fit1_v6": round(b_fix, 6),
       "fixture_ok": bool(fixture_ok),
       "windows": {"FIT1": int(FIT1.sum()), "VAL1": int(VAL1.sum()),
                   "FIT2": int(FIT2.sum()), "VAL2": int(VAL2.sum())},
       "arms": {}}

for name in ARMS:
    e = {}
    for tag, fitm, valm in (("T1", FIT1, VAL1), ("T2", FIT2, VAL2)):
        b = fit_beta(rd[name], common & fitm)
        b6 = fit_beta(rd["v6"], common & fitm)
        m = common & valm
        ll_a = nll(b, rd[name], m)
        ll_6 = nll(b6, rd["v6"], m)
        # milli-LL, >0 = arm better than v6
        delta_milli = (ll_6 - ll_a) * 1000.0
        pa = p_vec(b, rd[name], m)
        p6 = p_vec(b6, rd["v6"], m)
        bs = paired_bootstrap(pa, p6)
        e[tag] = {"beta": round(b, 6), "beta_v6": round(b6, 6),
                  "n_scored": int(m.sum()),
                  "ll_arm": round(float(ll_a), 6),
                  "ll_v6": round(float(ll_6), 6),
                  "delta_milli_ll": round(float(delta_milli), 3),
                  "bootstrap": {k: (round(float(v), 6)
                                    if isinstance(v, (int, float)) else v)
                                for k, v in bs.items()}}
        print(f"  {name:12} {tag}: beta {b:.4f} | n {m.sum():4} | "
              f"delta {delta_milli:+8.3f} m")
    res["arms"][name] = e

with open(os.path.join(OUT, "v10_transfer.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\nwrote {OUT}/v10_transfer.json")
