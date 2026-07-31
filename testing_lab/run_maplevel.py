"""Map-level surface check: does the candidate (HL13/pow0.75) also win when
predictions use per-map ratings with James-Stein shrinkage (the traded MC's
input), and does SHRINK_K want retuning? Scored on actually-played maps
2025-26 (walk-forward, map-level log-loss). Also: divergence taxonomy for the
Kalshi report. Writes out/maplevel.json."""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine, decay_weight

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

CONFIGS = {
    "prod": {"hl": 6.0, "pow": 0.5, "beta_overall": None},
    "cand": {"hl": 13.0, "pow": 0.75, "beta_overall": None},
}

eng = Engine()
games = eng.games
g_map = np.array([g["map_name"] for g in games])
maps_all = sorted(set(g_map))
map_idx = {m: np.where(g_map == m)[0] for m in maps_all}
n_t = len(eng.teams)

# map-level outcomes to score: maps of series in 2025-26
score_games = [(i, g) for i, g in enumerate(games) if g["date_s"] >= "2025-01-01"]
train_games = [(i, g) for i, g in enumerate(games)
               if "2023-06-01" <= g["date_s"] <= "2024-12-31"]
print(f"map-level: score n={len(score_games)}, beta-train n={len(train_games)}")

results = {}


def massey(mask_idx, weights, rd_vec):
    M = np.zeros((n_t, n_t))
    p = np.zeros(n_t)
    wi, li = eng.wi[mask_idx], eng.li[mask_idx]
    np.add.at(M, (wi, wi), weights)
    np.add.at(M, (li, li), weights)
    np.add.at(M, (wi, li), -weights)
    np.add.at(M, (li, wi), -weights)
    np.add.at(p, wi, weights * rd_vec)
    np.add.at(p, li, -weights * rd_vec)
    M[np.diag_indices(n_t)] += 0.5
    M[-1, :] = 1.0
    p[-1] = 0.0
    try:
        return np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, p, rcond=None)[0]


for cname, cfg in CONFIGS.items():
    rd_t = np.copysign(np.abs(eng.rd_raw) ** cfg["pow"] * 2.5, eng.rd_raw)
    # per-day caches
    days = sorted({g["date_s"] for _, g in score_games} |
                  {g["date_s"] for _, g in train_games})
    overall_by_day = {}
    permap_by_day = {}
    neff_by_day = {}
    for day in days:
        day_num = int(np.datetime64(day, "D").astype(int))
        hist = np.where(eng.g_dnum < day_num)[0]
        if len(hist) < 30:
            continue
        weeks = (day_num - eng.g_dnum[hist]) / 7.0
        w = decay_weight(weeks, "exp", hl=cfg["hl"])
        w = w * np.where(eng.champ[hist], 2.0, 1.0)
        overall_by_day[day] = massey(hist, w, rd_t[hist])
        pm = {}
        ne = {}
        for m in maps_all:
            mi = np.intersect1d(hist, map_idx[m], assume_unique=False)
            if len(mi) < 8:
                continue
            weeks_m = (day_num - eng.g_dnum[mi]) / 7.0
            wm = decay_weight(weeks_m, "exp", hl=cfg["hl"])
            pm[m] = massey(mi, wm, rd_t[mi])
            cnt = np.zeros(n_t)
            np.add.at(cnt, eng.wi[mi], wm)
            np.add.at(cnt, eng.li[mi], wm)
            ne[m] = cnt
        permap_by_day[day] = pm
        neff_by_day[day] = ne

    def rating(day, team_i, mp, k):
        ov = overall_by_day[day][team_i]
        if k is None:   # overall only
            return ov
        pm = permap_by_day[day].get(mp)
        if pm is None:
            return ov
        ne = neff_by_day[day][mp][team_i]
        a = ne / (ne + k)
        return a * pm[team_i] + (1 - a) * ov

    for k in (None, 3.0, 5.0, 10.0, 20.0):
        # fit beta on train, score on test
        def diffs(pairs):
            out = []
            for _, g in pairs:
                d = g["date_s"]
                if d not in overall_by_day:
                    continue
                wi_, li_ = eng.tidx[g["winner"]], eng.tidx[g["loser"]]
                out.append(rating(d, wi_, g["map_name"], k) -
                           rating(d, li_, g["map_name"], k))
            return np.array(out)

        dtr = diffs(train_games)
        dte = diffs(score_games)
        from scipy.optimize import minimize_scalar
        def nll(b, dd):
            return -np.mean(np.log(np.clip(1 / (1 + np.exp(-b * dd)), 1e-9, 1)))
        b = float(minimize_scalar(lambda x: nll(x, dtr), bounds=(0.03, 0.6),
                                  method="bounded").x)
        ll = float(nll(b, dte))
        key = f"{cname}_k{'ov' if k is None else int(k)}"
        results[key] = {"beta_map": round(b, 4), "ll_map_test": round(ll, 5),
                        "n": int(len(dte))}
        print(f"{key:<14} beta={b:.3f} map-ll={ll:.5f} (n={len(dte)})")

with open(os.path.join(OUT, "maplevel.json"), "w") as f:
    json.dump(results, f, indent=1)

# ── divergence taxonomy ─────────────────────────────────────────────────────
kj = pd.read_csv(os.path.join(OUT, "kalshi_joined3.csv"))
kj = kj.sort_values("div", ascending=False).head(12)
tax = []
for r in kj.itertuples(index=False):
    ctx = {}
    for org in (r.winner, r.loser):
        seq = eng.team_match_seq.get(org, [])
        prior = [(d, mid) for d, mid in seq if d < r.date]
        if len(prior) >= 4:
            cur = eng.lineups.get((org, prior[-1][1]))
            old = eng.lineups.get((org, prior[-4][1]))
            overlap = len(cur & old) if cur and old else None
        else:
            overlap = None
        wins = 0
        tot = 0
        for d, mid in prior[-5:]:
            for g in games:
                if g["match_id"] == mid:
                    tot += 1
                    if g["winner"] == org:
                        wins += 1
                    break
        gap_days = (pd.Timestamp(r.date) - pd.Timestamp(prior[-1][0])).days if prior else None
        ctx[org] = {"lineup_overlap_last4": overlap,
                    "recent_form_maps": f"{wins}/{tot}",
                    "days_since_last": gap_days}
    tax.append({"date": r.date, "match": f"{r.winner} bt {r.loser}",
                "benpom": r.p_benpom, "kalshi": r.pk_pre,
                "ctx": ctx})
with open(os.path.join(OUT, "divergence_taxonomy.json"), "w") as f:
    json.dump(tax, f, indent=1, default=str)
print("\nTAXONOMY (winner first):")
for t in tax:
    print(f"  {t['date']} {t['match']} bp={t['benpom']:.2f} k={t['kalshi']:.2f}")
    for org, c in t["ctx"].items():
        print(f"    {org}: overlap4={c['lineup_overlap_last4']} form={c['recent_form_maps']} "
              f"rest={c['days_since_last']}d")
print("saved out/maplevel.json + divergence_taxonomy.json")
