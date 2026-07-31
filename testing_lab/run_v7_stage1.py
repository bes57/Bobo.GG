"""v7 Stage 1 — recency & symmetry grid.

User hypotheses under test (2026-07-23):
  H1: the model has too little recency bias.
  H2: asymmetric treatment of results (consistency-conditioned decay) is
      undesirable — a symmetric scheme should do the job.

Key framing: consistency decay is ANTI-recency by construction — a form
change is 'anomalous' vs the team's trailing level, so it fades FASTER
(HL12). H1 and H2 are the same complaint seen from two sides.

Grid: symmetric HL sweep, shorter-HL consistency variants, box/power
shapes, references (v6 champion, v5 asym). Buckets: elite-vs-floor (the
EG guard that motivated v6) and a NEW form-shift bucket (|wr5-wr16| >=
0.15 for either team). Bootstrap every config vs the v6 champion.

Writes out/v7_stage1.json + out/v7_probs.npz; prints progress.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine                      # noqa: E402
from harness import paired_bootstrap          # noqa: E402
from scipy.optimize import minimize_scalar    # noqa: E402

OUT = os.path.join(HERE, "out")

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
train_v = (s.date <= "2024-12-31").values
test_v = (s.date > "2024-12-31").values
y26 = (s.date >= "2026-01-01").values
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
        "region_prior_ridge": 1.5, "w_custom": PO}


# ── per-team decayed winrates at two horizons, as-of each series date ────────
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

wr16 = wr_series(16.0)
wr5 = wr_series(5.0)
w16_w = np.array([wr16.get((r.winner, r.date), 0.5) for r in s.itertuples(index=False)])
w16_l = np.array([wr16.get((r.loser, r.date), 0.5) for r in s.itertuples(index=False)])
w5_w = np.array([wr5.get((r.winner, r.date), 0.5) for r in s.itertuples(index=False)])
w5_l = np.array([wr5.get((r.loser, r.date), 0.5) for r in s.itertuples(index=False)])

hi_w = np.maximum(w16_w, w16_l)
lo_w = np.minimum(w16_w, w16_l)
elite_floor = (hi_w >= 0.60) & (lo_w <= 0.40)
form_gap_w = w5_w - w16_w
form_gap_l = w5_l - w16_l
form_shift = (np.abs(form_gap_w) >= 0.15) | (np.abs(form_gap_l) >= 0.15)
print(f"buckets: elite-floor holdout n={int((elite_floor & test_v).sum())}, "
      f"form-shift holdout n={int((form_shift & test_v).sum())}", flush=True)


def sp(pm, fm):
    return np.where(np.isin(fm, ("bo5", "bo5_gf")),
                    pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2),
                    np.where(fm == "bo1", pm, pm ** 2 * (3 - 2 * pm)))


results = {}
probs = {}
rdiffs = {}


def run(name, dcfg, extra=None):
    cfg = {**BASE, "decay": dcfg}
    if extra:
        cfg.update(extra)
    out = eng.run(cfg)
    rd = out["rdiff"]
    v = ~np.isnan(rd)
    b = float(minimize_scalar(
        lambda x: -np.mean(np.log(np.clip(sp(1 / (1 + np.exp(-x * rd[v & train_v])),
                                             fmts[v & train_v]), 1e-9, 1))),
        bounds=(0.02, 0.6), method="bounded").x)
    p = sp(1 / (1 + np.exp(-b * rd)), fmts)

    def ll(m):
        return float(-np.mean(np.log(np.clip(p[m], 1e-9, 1))))

    # form-shift bucket: does the model catch the mover? score prob assigned
    # to the team whose form improved more
    fs = v & test_v & form_shift
    mover_is_w = (form_gap_w >= form_gap_l)
    p_mover = np.where(mover_is_w, p, 1 - p)
    mover_won = mover_is_w.astype(float)
    results[name] = {
        "beta": round(b, 4),
        "ll_test": round(ll(v & test_v), 5),
        "ll_2026": round(ll(v & y26), 5),
        "ll_elitefloor": round(ll(v & test_v & elite_floor), 5),
        "ll_formshift": round(ll(fs), 5),
        "formshift_pred_mover": round(float(p_mover[fs].mean()), 4),
        "formshift_emp_mover": round(float(mover_won[fs].mean()), 4),
        "n_test": int((v & test_v).sum()),
    }
    probs[name] = p
    rdiffs[name] = rd
    r = results[name]
    print(f"{name:<20} ll={r['ll_test']:.5f} 26={r['ll_2026']:.5f} "
          f"EF={r['ll_elitefloor']:.5f} FS={r['ll_formshift']:.5f} "
          f"mover pred/emp {r['formshift_pred_mover']:.3f}/{r['formshift_emp_mover']:.3f} "
          f"b={r['beta']}", flush=True)


# ── references ───────────────────────────────────────────────────────────────
run("v6_consist_20_12", {"kind": "games", "consistency": (20.0, 12.0)})
run("v5_asym_W20L12", {"kind": "games", "hl_games": 20.0, "hl_games_loss": 12.0})

# ── H2: symmetric HL sweep (also the H1 recency axis) ────────────────────────
for hl in (6, 8, 10, 12, 14, 16, 20, 24):
    run(f"sym_{hl}", {"kind": "games", "hl_games": float(hl)})

# ── H1 within the consistency family: shorter HLs across the board ──────────
run("consist_16_10", {"kind": "games", "consistency": (16.0, 10.0)})
run("consist_14_8", {"kind": "games", "consistency": (14.0, 8.0)})
run("consist_12_8", {"kind": "games", "consistency": (12.0, 8.0)})
# REVERSED conditioning: anomalies PERSIST, consistent results fade — the
# pro-recency mirror image (a surprise is news, confirmation is redundant)
run("surprise_12_20", {"kind": "games", "consistency": (12.0, 20.0)})
run("surprise_16_24", {"kind": "games", "consistency": (16.0, 24.0)})

# ── other symmetric shapes ───────────────────────────────────────────────────
run("boxexp_c3_hl8", {"kind": "games", "form": "boxexp", "c": 3.0, "hl_games": 8.0})
run("boxexp_c5_hl10", {"kind": "games", "form": "boxexp", "c": 5.0, "hl_games": 10.0})
run("power_t6_a15", {"kind": "games", "form": "power", "tau": 6.0, "alpha": 1.5})

# ── bootstrap everything vs the champion ─────────────────────────────────────
champ = "v6_consist_20_12"
vv = ~np.isnan(rdiffs[champ])
boots = {}
for name in results:
    if name == champ:
        continue
    m = vv & ~np.isnan(rdiffs[name]) & test_v
    boots[name] = paired_bootstrap(probs[name][m], probs[champ][m])
    bt = boots[name]
    print(f"boot {name:<20} vs v6: delta={bt['mean_delta']*1000:+.2f}m "
          f"p_better={bt['p_better']:.3f}", flush=True)

with open(os.path.join(OUT, "v7_stage1.json"), "w") as f:
    json.dump({"results": results, "boots": boots,
               "buckets": {"elite_floor_n": int((elite_floor & test_v).sum()),
                           "form_shift_n": int((form_shift & test_v).sum())}},
              f, indent=1)
np.savez(os.path.join(OUT, "v7_probs.npz"),
         **{k: v for k, v in probs.items()},
         **{f"rd__{k}": v for k, v in rdiffs.items()},
         test_v=test_v, train_v=train_v, y26=y26,
         elite_floor=elite_floor, form_shift=form_shift,
         w5_w=w5_w, w5_l=w5_l, w16_w=w16_w, w16_l=w16_l)
print("saved out/v7_stage1.json + out/v7_probs.npz", flush=True)
