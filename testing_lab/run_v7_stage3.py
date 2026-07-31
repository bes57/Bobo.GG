"""v7 Stage 3 — case evidence + market benchmark.

1. SLUMPING-FAVORITE analog set (the NS-GE shape): holdout series where the
   rating favorite (gap >= 2.0) enters on a 2+ series losing streak. Did
   they underperform the model's price? Scored under v6, sym_20, v6+form5.
2. Today's slate (NS-GE, SEN-LOUD + top upcoming) under the finalists.
3. Kalshi 2026 overlap: v6 vs sym_20 vs market (t2h), sanctioned benchmark.

Writes out/v7_stage3.json.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine                    # noqa: E402
from harness import paired_bootstrap        # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

OUT = os.path.join(HERE, "out")
npz = np.load(os.path.join(OUT, "v7_probs.npz"))
npz2 = np.load(os.path.join(OUT, "v7_probs2.npz"))
test_v = npz["test_v"]

eng = Engine()
s = eng.series.reset_index(drop=True)
fmts = s.fmt.values
out3 = {}

# ── 1. slumping-favorite analog set ─────────────────────────────────────────
# consecutive series losses entering each match, per team
streak = defaultdict(int)
loss_streak_w = np.zeros(len(s), dtype=int)
loss_streak_l = np.zeros(len(s), dtype=int)
for i, r in enumerate(s.itertuples(index=False)):
    loss_streak_w[i] = streak[r.winner]
    loss_streak_l[i] = streak[r.loser]
    streak[r.winner] = 0
    streak[r.loser] += 1

rd_v6 = npz["rd__v6_consist_20_12"]
v = ~np.isnan(rd_v6)
fav_is_winner = rd_v6 >= 0
gap = np.abs(rd_v6)
fav_streak = np.where(fav_is_winner, loss_streak_w, loss_streak_l)

models = {"v6": npz["v6_consist_20_12"], "sym_20": npz["sym_20"],
          "v6+form5": npz2["v6_consist_20_12+form5"]}
for gmin, smin, tag in ((2.0, 2, "gap2_streak2"), (2.5, 2, "gap2.5_streak2"),
                        (2.0, 1, "gap2_streak1")):
    m = v & test_v & (gap >= gmin) & (fav_streak >= smin)
    res = {"n": int(m.sum())}
    fav_won = fav_is_winner[m]
    res["fav_emp_winrate"] = round(float(fav_won.mean()), 4)
    for name, p in models.items():
        p_fav = np.where(fav_is_winner, p, 1 - p)[m]
        ll = float(-np.mean(np.log(np.clip(np.where(fav_won, p_fav, 1 - p_fav),
                                           1e-9, 1))))
        res[name] = {"fav_pred_mean": round(float(p_fav.mean()), 4),
                     "ll": round(ll, 5)}
    out3[f"slump_fav_{tag}"] = res
    print(f"slumping favorite [{tag}] n={res['n']}: emp {res['fav_emp_winrate']:.3f} | "
          + " | ".join(f"{k} pred {res[k]['fav_pred_mean']:.3f} ll {res[k]['ll']:.4f}"
                       for k in models), flush=True)

# ── 2. today's slate under the finalists ─────────────────────────────────────
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
        "region_prior_ridge": 1.5, "w_custom": PO}
today = "2026-07-24"
if today not in eng.pred_days:
    eng.pred_days = sorted(set(list(eng.pred_days) + [today]))

st1 = json.load(open(os.path.join(OUT, "v7_stage1.json")))["results"]
st2 = json.load(open(os.path.join(OUT, "v7_stage2.json")))
finalists = {
    "v6": ({"kind": "games", "consistency": (20.0, 12.0)}, st1["v6_consist_20_12"]["beta"], None),
    "sym_20": ({"kind": "games", "hl_games": 20.0}, st1["sym_20"]["beta"], None),
    "v6+form5": ({"kind": "games", "consistency": (20.0, 12.0)},
                 st2["v6_consist_20_12+form5"]["beta"],
                 st2["v6_consist_20_12+form5"]["b_form"]),
}

# final decayed winrate states (for the form term, as of now)
def wr_now(hl):
    lam = math.log(2) / hl
    state = defaultdict(lambda: [0.0, 0.0])
    for g in sorted(eng.games, key=lambda g: (g["date_s"], g["match_id"])):
        for team, won in ((g["winner"], 1.0), (g["loser"], 0.0)):
            st = state[team]
            st[0] = st[0] * math.exp(-lam) + won
            st[1] = st[1] * math.exp(-lam) + 1.0
    return {t: (n / d if d > 3 else 0.5) for t, (n, d) in state.items()}

wr5_now, wr16_now = wr_now(5.0), wr_now(16.0)

up = json.load(open(os.path.join(os.path.dirname(HERE), "data",
                                 "upcoming_matches.json")))
up_rows = up if isinstance(up, list) else list(up.values())[0]
slate_matches = [(r["org_a"], r["org_b"], r.get("format", "bo3"), r.get("date"))
                 for r in up_rows][:14]

slate = {}
for name, (dcfg, beta, b_form) in finalists.items():
    o = eng.run({**BASE, "decay": dcfg, "daily_out": True})
    daily = o["daily_r"]
    latest = daily[sorted(daily.keys())[-1]]
    rows = []
    for a, b_, fmt, dt_ in slate_matches:
        if a not in eng.tidx or b_ not in eng.tidx:
            continue
        gp = float(latest[eng.tidx[a]] - latest[eng.tidx[b_]])
        z = beta * gp
        if b_form is not None:
            df = (wr5_now.get(a, .5) - wr16_now.get(a, .5)) - \
                 (wr5_now.get(b_, .5) - wr16_now.get(b_, .5))
            z += b_form * df
        pm = 1 / (1 + np.exp(-z))
        p3 = pm ** 3 * (1 + 3 * (1 - pm) + 6 * (1 - pm) ** 2) \
            if fmt in ("bo5", "bo5_gf") else \
            (pm if fmt == "bo1" else pm * pm * (3 - 2 * pm))
        rows.append({"a": a, "b": b_, "date": dt_, "fmt": fmt,
                     "p_a": round(float(p3), 4), "gap": round(gp, 3)})
    slate[name] = rows
    print(f"slate[{name}]: " + ", ".join(
        f"{r['a']}-{r['b']} {r['p_a']:.1%}" for r in rows[:6]), flush=True)
out3["slate"] = slate
out3["form_states"] = {t: {"wr5": round(wr5_now.get(t, .5), 3),
                           "wr16": round(wr16_now.get(t, .5), 3)}
                       for t in ("NS", "GE", "SEN", "LOUD", "FS", "GEN")}

# ── 3. kalshi 2026 overlap benchmark ────────────────────────────────────────
k = pd.read_csv(os.path.join(HERE, "kalshi", "kalshi_matches.csv"))
k = k[(k.excluded != True) & k.winner_org.notna()].copy()  # noqa: E712
k["pair"] = [frozenset((a, b)) for a, b in zip(k.org_a, k.org_b)]
k["d"] = pd.to_datetime(k.date_utc.str[:10])
s["pair"] = [frozenset((a, b)) for a, b in zip(s.winner, s.loser)]
s["d"] = pd.to_datetime(s.date)
s["pos"] = np.arange(len(s))
s26 = s[s.date >= "2026-01-01"]
join = []
for _, kr in k.iterrows():
    cand = s26[(s26.pair == kr["pair"]) & (abs((s26.d - kr.d).dt.days) <= 1)]
    if len(cand) == 0:
        continue
    sr = cand.iloc[0]
    pa = kr.prob_a_t2h
    if pd.isna(pa):
        continue
    pk = pa if kr.winner_org == kr.org_a else 1.0 - pa
    join.append((int(sr.pos), float(pk)))
pos = np.array([j[0] for j in join])
pk = np.array([j[1] for j in join])
kal = {"n": len(join)}
def _ll(p):
    return round(float(-np.mean(np.log(np.clip(p, 1e-9, 1)))), 5)
kal["market_t2h"] = _ll(pk)
for name, p in models.items():
    ok = ~np.isnan(p[pos])
    kal[name] = _ll(p[pos][ok])
    bt = paired_bootstrap(p[pos][ok], pk[ok])
    kal[f"{name}_vs_market"] = {kk: round(vv, 5) for kk, vv in bt.items()}
out3["kalshi"] = kal
print("kalshi overlap:", kal, flush=True)

with open(os.path.join(OUT, "v7_stage3.json"), "w") as f:
    json.dump(out3, f, indent=1)
print("saved out/v7_stage3.json", flush=True)
