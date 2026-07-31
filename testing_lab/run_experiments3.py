"""Round 3 on top of the round-2 winner (hl13, pow0.75, rp0.3):
 - favorite-margin discount ("top teams don't try"): discount blowout credit
   when the pre-game favorite wins big; keep underdog upset margins intact
 - residual margin: subtract expected gap, credit only the surprise
 - refined lineup-based roster penalty (only real rebuilds punished)
Writes out/experiments3.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BEST = {"decay": {"kind": "exp", "hl": 13.0}, "rd": {"power": 0.75, "scale": 2.5},
        "roster_mode": "year", "roster_persistence": 0.3,
        "ridge": 0.5, "champ_mult": 2.0}

eng = Engine()
s = eng.series
results, probs = {}, {}


def run(name, cfg):
    t0 = time.time()
    out = eng.run(cfg)
    results[name] = {"beta": out["beta"], "ll_test": out["ll_test"],
                     "brier_test": out["brier_test"]}
    probs[name] = out["p_test"]
    print(f"{name:<30} b={out['beta']:.3f} ll={out['ll_test']:.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


base_out = run("best_r2", {**BEST, "daily_out": True})
run("prod_baseline", {"decay": {"kind": "exp", "hl": 6.0},
                      "rd": {"power": 0.5, "scale": 2.5},
                      "roster_mode": "year", "roster_persistence": 0.3,
                      "ridge": 0.5, "champ_mult": 2.0})

# per-game prior rating gap (winner-referenced), from best_r2 daily ratings
daily = base_out["daily_r"]
days_sorted = sorted(daily.keys())
game_gap = np.zeros(len(eng.games))
for i, g in enumerate(eng.games):
    ds = g["date_s"]
    # latest solve strictly before the game date
    lo, hi, pick = 0, len(days_sorted) - 1, None
    import bisect
    j = bisect.bisect_left(days_sorted, ds) - 1
    if j >= 0:
        rv = daily[days_sorted[j]]
        game_gap[i] = rv[eng.tidx[g["winner"]]] - rv[eng.tidx[g["loser"]]]

rd_raw = eng.rd_raw
base_rd = np.copysign(np.abs(rd_raw) ** 0.75 * 2.5, rd_raw)

# A) favorite blowout discount: when winner was favored by >g0, shrink margin
for g0, c in [(1.0, 0.3), (1.0, 0.5), (2.0, 0.3), (2.0, 0.5), (3.0, 0.5)]:
    fav_win = game_gap > g0
    factor = np.where(fav_win, 1.0 / (1.0 + c * (game_gap - g0) / 3.0), 1.0)
    run(f"favdisc_g{g0}_c{c}", {**BEST, "rd_custom": base_rd * factor})

# B) residual margin: rd' = rd - k*prior_gap (both in rating units)
for k in (0.15, 0.3, 0.5):
    rd_res = base_rd - k * game_gap
    run(f"residual_k{k}", {**BEST, "rd_custom": rd_res})

# C) upset boost: upset margins count extra (mirror of A)
for boost in (1.2, 1.4):
    ups = game_gap < -1.0
    factor = np.where(ups, boost, 1.0)
    run(f"upsboost_{boost}", {**BEST, "rd_custom": base_rd * factor})

# D) refined lineup roster penalty: only real rebuilds (overlap<=3/5) punished,
#    via step function; uses lineup mode with gamma trick replaced by steps
class SteppedEngine(Engine):
    def _continuity_vec(self, ref_date_s, mode, persistence):
        if mode != "lineup":
            return super()._continuity_vec(ref_date_s, mode, persistence)
        n = len(self.games)
        cur = {}
        for org, seq in self.team_match_seq.items():
            latest = None
            for ds, mid in seq:
                if ds < ref_date_s:
                    latest = self.lineups.get((org, mid), None)
                else:
                    break
            cur[org] = latest
        cw = np.ones(n)
        cl = np.ones(n)
        steps = {5: 1.0, 4: 1.0, 3: 0.6, 2: 0.3, 1: 0.15, 0: 0.1}
        for i, g in enumerate(self.games):
            for org, arr in ((g["winner"], cw), (g["loser"], cl)):
                cur_l = cur.get(org)
                then_l = self.lineups.get((org, g["match_id"]))
                if cur_l and then_l:
                    arr[i] = steps.get(min(len(cur_l & then_l), 5), 1.0)
        return cw, cl


seng = SteppedEngine()


def run_s(name, cfg):
    t0 = time.time()
    out = seng.run(cfg)
    results[name] = {"beta": out["beta"], "ll_test": out["ll_test"],
                     "brier_test": out["brier_test"]}
    probs[name] = out["p_test"]
    print(f"{name:<30} b={out['beta']:.3f} ll={out['ll_test']:.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)


run_s("lineup_step", {**BEST, "roster_mode": "lineup", "roster_persistence": 1.0})

# paired bootstraps vs best_r2
base = probs["best_r2"]
rng = np.random.default_rng(3)
lbase = -np.log(np.clip(base, 1e-9, 1))
print("\n== vs best_r2 ==")
for name, p in probs.items():
    if name in ("best_r2",):
        continue
    la = -np.log(np.clip(p, 1e-9, 1))
    d = lbase - la
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    results[name]["vs_best"] = {"mean_delta": float(d.mean()),
                                "p_better": float((means > 0).mean())}
    print(f"  {name:<28} dLL={d.mean():+.5f} p_better={results[name]['vs_best']['p_better']:.3f}")

with open(os.path.join(OUT, "experiments3.json"), "w") as f:
    json.dump(results, f, indent=1)
np.savez_compressed(os.path.join(OUT, "exp3_probs.npz"), **probs)
print("saved out/experiments3.json")
