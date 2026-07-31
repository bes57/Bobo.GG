"""v7 Stage 2 — symmetric recency at the probability layer.

z = beta * rdiff + b_form * (form_a - form_b),  form = wr_short - wr16

(beta, b_form) fit jointly on train (<=2024-12-31), scored on holdout.
Ledger note: the rejected 'winrate additive feature' (v6 session) was a
winrate LEVEL term — redundant with ratings. This is a form DELTA
(short-horizon vs long-horizon winrate), i.e. a regime-change signal the
ratings deliberately smooth over. Different construction, tested on purpose.

Applied on top of: the v6 champion rdiff, the best symmetric rdiff from
stage 1, and the best overall stage-1 candidate if different. Short
horizons: HL3 / HL5 / HL8. Also fits b_form on the champion to answer
'how much recency is the consistency model actually missing?'

Writes out/v7_stage2.json.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine                    # noqa: E402
from harness import paired_bootstrap        # noqa: E402

OUT = os.path.join(HERE, "out")

st1 = json.load(open(os.path.join(OUT, "v7_stage1.json")))
npz = np.load(os.path.join(OUT, "v7_probs.npz"))
test_v = npz["test_v"]
train_v = npz["train_v"]
y26 = npz["y26"]
elite_floor = npz["elite_floor"]
form_shift = npz["form_shift"]

# pick baselines from stage 1
res1 = st1["results"]
best_sym = min((k for k in res1 if k.startswith("sym_")),
               key=lambda k: res1[k]["ll_test"])
best_all = min(res1, key=lambda k: res1[k]["ll_test"])
CHAMP = "v6_consist_20_12"
bases = [CHAMP, best_sym] + ([best_all] if best_all not in (CHAMP, best_sym) else [])
print(f"stage-2 bases: {bases} (best_sym={best_sym}, best_overall={best_all})",
      flush=True)

# form arrays: need wr at HL3/5/8 — HL5/16 already saved; compute 3 & 8 fresh
eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values


def wr_series(hl):
    lam = math.log(2) / hl
    state = defaultdict(lambda: [0.0, 0.0])
    at = {}
    sdates = sorted(set(s.date))
    si = 0
    for g in sorted(eng.games, key=lambda g: (g["date_s"], g["match_id"])):
        while si < len(sdates) and sdates[si] <= g["date_s"]:
            for t_, (n_, d_) in state.items():
                if d_ > 3:
                    at[(t_, sdates[si])] = n_ / d_
            si += 1
        for team, won in ((g["winner"], 1.0), (g["loser"], 0.0)):
            st = state[team]
            st[0] = st[0] * math.exp(-lam) + won
            st[1] = st[1] * math.exp(-lam) + 1.0
    while si < len(sdates):
        for t_, (n_, d_) in state.items():
            if d_ > 3:
                at[(t_, sdates[si])] = n_ / d_
        si += 1
    return at


def arr(at):
    w = np.array([at.get((r.winner, r.date), 0.5) for r in s.itertuples(index=False)])
    l = np.array([at.get((r.loser, r.date), 0.5) for r in s.itertuples(index=False)])
    return w, l


w16_w, w16_l = npz["w16_w"], npz["w16_l"]
horizons = {}
horizons[5] = (npz["w5_w"], npz["w5_l"])
for hl in (3, 8):
    horizons[hl] = arr(wr_series(float(hl)))


def sp(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


results = {}
probs = {}
p_champ = npz[CHAMP]

for base in bases:
    rd = npz[f"rd__{base}"]
    v = ~np.isnan(rd)
    for hl, (ws, wl) in sorted(horizons.items()):
        dform = (ws - w16_w) - (wl - w16_l)   # winner-team form minus loser's

        def nll(params, mask):
            b, bf = params
            z = b * rd[mask] + bf * dform[mask]
            pm = 1 / (1 + np.exp(-z))
            return -np.mean(np.log(np.clip(sp(pm, fmts[mask]), 1e-9, 1)))

        m_tr = v & train_v
        fit = minimize(nll, x0=[0.13, 0.0], args=(m_tr,), method="Nelder-Mead")
        b, bf = fit.x
        z = b * rd + bf * dform
        p = sp(1 / (1 + np.exp(-z)), fmts)

        def ll(m):
            return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))

        name = f"{base}+form{hl}"
        m_te = v & test_v
        bt = paired_bootstrap(p[m_te], p_champ[m_te])
        results[name] = {
            "beta": round(float(b), 4), "b_form": round(float(bf), 4),
            "ll_test": round(ll(m_te), 5),
            "ll_2026": round(ll(v & y26), 5),
            "ll_formshift": round(ll(m_te & form_shift), 5),
            "ll_elitefloor": round(ll(m_te & elite_floor), 5),
            "boot_vs_v6": bt,
        }
        probs[name] = p
        r = results[name]
        print(f"{name:<28} b_form={r['b_form']:+.3f} ll={r['ll_test']:.5f} "
              f"FS={r['ll_formshift']:.5f} EF={r['ll_elitefloor']:.5f} "
              f"boot d={bt['mean_delta']*1000:+.2f}m p={bt['p_better']:.3f}",
              flush=True)

with open(os.path.join(OUT, "v7_stage2.json"), "w") as f:
    json.dump(results, f, indent=1)
np.savez(os.path.join(OUT, "v7_probs2.npz"), **probs)
print("saved out/v7_stage2.json", flush=True)
