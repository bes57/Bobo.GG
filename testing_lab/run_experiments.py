"""Experiment grid round 1: one-at-a-time screening around the production
config. beta refit on train (<=2024) per config; scored on 2025-26.
Saves out/experiments1.json + out/exp1_probs.npz (for paired bootstraps)."""
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


def variants():
    yield "prod_baseline", dict(PROD)
    # decay families
    for hl in (4, 5, 8, 10, 13):
        yield f"exp_hl{hl}", {**PROD, "decay": {"kind": "exp", "hl": float(hl)}}
    for tau, alpha in ((3, 1.5), (4, 2.0), (6, 2.0), (8, 1.2), (10, 1.0)):
        yield f"power_t{tau}_a{alpha}", {**PROD, "decay": {"kind": "power", "tau": float(tau), "alpha": float(alpha)}}
    for c, hl in ((2, 6), (4, 6), (4, 8), (6, 10)):
        yield f"boxexp_c{c}_hl{hl}", {**PROD, "decay": {"kind": "boxexp", "c": float(c), "hl": float(hl)}}
    for W in (26, 40, 60):
        yield f"linear_W{W}", {**PROD, "decay": {"kind": "linear", "W": float(W)}}
    # roster modes
    yield "roster_none", {**PROD, "roster_mode": "none"}
    for p in (0.0, 0.5, 0.7):
        yield f"roster_year_p{p}", {**PROD, "roster_persistence": p}
    for g in (0.5, 1.0, 1.5):
        yield f"roster_lineup_g{g}", {**PROD, "roster_mode": "lineup", "roster_persistence": g}
    # rd transforms (beta refits absorb scale, so scale stays 2.5 where moot)
    for pw in (0.35, 0.65, 1.0):
        yield f"rd_pow{pw}", {**PROD, "rd": {"power": pw, "scale": 2.5}}
    yield "rd_winonly", {**PROD, "rd": {"mode": "win", "const": 3.0}}
    for w in (0.3, 0.5, 0.7):
        yield f"rd_blend{w}", {**PROD, "rd": {"mode": "blend", "power": 0.5, "scale": 2.5, "win_const": 3.0, "w": w}}
    # ridge / champions multiplier
    for rg in (0.25, 1.0, 2.0):
        yield f"ridge{rg}", {**PROD, "ridge": rg}
    for cm in (1.0, 1.5, 3.0):
        yield f"champ{cm}", {**PROD, "champ_mult": cm}


def main():
    eng = Engine()
    results = {}
    probs = {}
    base = None
    for name, cfg in variants():
        t0 = time.time()
        out = eng.run(cfg)
        results[name] = {k: out[k] for k in
                         ("beta", "ll_train", "ll_test", "brier_test", "n_test")}
        probs[name] = out["p_test"]
        if name == "prod_baseline":
            base = out["p_test"]
        d = ""
        if base is not None and name != "prod_baseline":
            la = -np.log(np.clip(probs[name], 1e-9, 1))
            lb = -np.log(np.clip(base, 1e-9, 1))
            d = f"  dLL={float((lb-la).mean()):+.5f}"
        print(f"{name:<22} beta={out['beta']:.3f} ll_test={out['ll_test']:.5f}{d}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "experiments1.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "exp1_probs.npz"), **probs)
    lb = sorted(results.items(), key=lambda kv: kv[1]["ll_test"])
    print("\n== LEADERBOARD (ll_test) ==")
    for name, r in lb[:15]:
        print(f"  {r['ll_test']:.5f}  {name} (beta {r['beta']})")


if __name__ == "__main__":
    main()
