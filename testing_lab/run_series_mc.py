"""Series-level veto-MC check: does raising per-map SHRINK_K help the surface
the bot actually quotes? Simulates the production-style veto (ban/pick rates
decayed HL=6w, own-strength factor on picks) over per-map ratings built at
several shrinkage levels, walk-forward, 2025-26 series.
Configs: prod constants + k5 (as deployed), candidate + k in {5, 20, overall}.
Writes out/series_mc.json."""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine, decay_weight

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
N_SIMS = 2000
VETO_HL = 6.0
VLAM = math.log(2) / VETO_HL

eng = Engine()
s = eng.series.reset_index(drop=True)
games = eng.games
g_map = np.array([g["map_name"] for g in games])
maps_all = sorted(set(g_map))
mIdx = {m: i for i, m in enumerate(maps_all)}
map_rows = {m: np.where(g_map == m)[0] for m in maps_all}
n_t = len(eng.teams)

# veto history
v = pd.read_csv(os.path.join(DATA, "map_vetos.csv"))
dates_j = json.load(open(os.path.join(DATA, "match_dates.json")))
v["date"] = v.MatchID.astype(str).map(dates_j)
v = v.dropna(subset=["date"]).sort_values(["date", "MatchID", "step"])
v = v[v["map"].isin(maps_all)]  # drop junk rows ('stats unavailable...' etc.)
team_hist = defaultdict(list)
for mid, grp in v.groupby("MatchID", sort=False):
    pool = grp["map"].tolist()
    date = grp["date"].iloc[0]
    rem = list(pool)
    for _, r in grp.sort_values("step").iterrows():
        if r["action"] in ("ban", "pick") and len(rem) > 1:
            team_hist[r["team"]].append((date, r["action"], r["map"], list(rem)))
        if r["map"] in rem:
            rem.remove(r["map"])

# pool per match date: maps seen in vetos within trailing 45 days
v_by_date = v.groupby("date")["map"].apply(set).sort_index()
v_dates = v_by_date.index.tolist()


def pool_asof(date):
    seen = set()
    for d in reversed(v_dates):
        if d < date:
            seen |= v_by_date[d]
            if len(seen) >= 7 and (pd.Timestamp(date) - pd.Timestamp(d)).days > 45:
                break
    return sorted(seen)[:9] if len(seen) >= 7 else maps_all


def veto_rates(team, action, asof):
    num, den = defaultdict(float), defaultdict(float)
    for d, act, mp, rem in team_hist.get(team, []):
        if d >= asof or act != action:
            continue
        w = math.exp(-VLAM * (pd.Timestamp(asof) - pd.Timestamp(d)).days / 7.0)
        for m in rem:
            den[m] += w
        num[mp] += w
    return {m: (num.get(m, 0.0) / den[m]) if den.get(m) else 0.0 for m in den} \
        if den else {}


# decayed map win pct per team (for own/opp factors), asof
def map_winpct(day_num, hl):
    wnum = np.zeros((n_t, len(maps_all)))
    wden = np.zeros((n_t, len(maps_all)))
    hist = np.where(eng.g_dnum < day_num)[0]
    weeks = (day_num - eng.g_dnum[hist]) / 7.0
    w = decay_weight(weeks, "exp", hl=hl)
    for j, gi in enumerate(hist):
        mi = mIdx[g_map[gi]]
        wnum[eng.wi[gi], mi] += w[j]
        wden[eng.wi[gi], mi] += w[j]
        wden[eng.li[gi], mi] += w[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        wp = wnum / wden
    return wp, wden


def massey(rows, weights, rd_vec):
    M = np.zeros((n_t, n_t))
    p = np.zeros(n_t)
    wi, li = eng.wi[rows], eng.li[rows]
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


CONFIGS = {
    "prod_k5": {"hl": 6.0, "pow": 0.5, "k": 5.0, "beta": 0.170},
    "cand_k5": {"hl": 13.0, "pow": 0.75, "k": 5.0, "beta": 0.116},
    "cand_k20": {"hl": 13.0, "pow": 0.75, "k": 20.0, "beta": 0.116},
    "cand_overall": {"hl": 13.0, "pow": 0.75, "k": None, "beta": 0.116},
}

test_rows = s[(s.date > "2024-12-31")].index.values
print(f"series to simulate: {len(test_rows)} x {N_SIMS} sims x {len(CONFIGS)} configs")
rng = np.random.default_rng(42)
results = {}

for cname, cfg in CONFIGS.items():
    rd_t = np.copysign(np.abs(eng.rd_raw) ** cfg["pow"] * 2.5, eng.rd_raw)
    cache_day = {}
    lls = []
    for si in test_rows:
        row = s.iloc[si]
        day = row.date
        if day not in cache_day:
            day_num = int(np.datetime64(day, "D").astype(int))
            hist = np.where(eng.g_dnum < day_num)[0]
            if len(hist) < 30:
                cache_day[day] = None
            else:
                weeks = (day_num - eng.g_dnum[hist]) / 7.0
                w = decay_weight(weeks, "exp", hl=cfg["hl"]) * \
                    np.where(eng.champ[hist], 2.0, 1.0)
                overall = massey(hist, w, rd_t[hist])
                permap, neff = {}, {}
                if cfg["k"] is not None:
                    for m in maps_all:
                        rowsm = np.intersect1d(hist, map_rows[m])
                        if len(rowsm) < 8:
                            continue
                        weeks_m = (day_num - eng.g_dnum[rowsm]) / 7.0
                        wm = decay_weight(weeks_m, "exp", hl=cfg["hl"])
                        permap[m] = massey(rowsm, wm, rd_t[rowsm])
                        cnt = np.zeros(n_t)
                        np.add.at(cnt, eng.wi[rowsm], wm)
                        np.add.at(cnt, eng.li[rowsm], wm)
                        neff[m] = cnt
                wp, wden = map_winpct(day_num, cfg["hl"])
                cache_day[day] = (overall, permap, neff, wp, wden)
        cd = cache_day[day]
        if cd is None:
            continue
        overall, permap, neff, wp, wden = cd
        ia, ib = eng.tidx[row.winner], eng.tidx[row.loser]

        pool = pool_asof(day)
        nm = len(pool)

        def rating_vec(ti):
            out = np.empty(nm)
            for j, m in enumerate(pool):
                ov = overall[ti]
                if cfg["k"] is None or m not in permap:
                    out[j] = ov
                else:
                    ne = neff[m][ti]
                    a = ne / (ne + cfg["k"])
                    out[j] = a * permap[m][ti] + (1 - a) * ov
            return out

        ra, rb = rating_vec(ia), rating_vec(ib)
        p_map = 1.0 / (1.0 + np.exp(-cfg["beta"] * (ra - rb)))

        def winp(ti):
            out = np.empty(nm)
            for j, m in enumerate(pool):
                mi = mIdx[m]
                out[j] = wp[ti, mi] if wden[ti, mi] > 0.4 else 0.5
                if not np.isfinite(out[j]):
                    out[j] = 0.5
            return out

        wa, wb = winp(ia), winp(ib)
        bra = veto_rates(row.winner, "ban", day)
        brb = veto_rates(row.loser, "ban", day)
        pra = veto_rates(row.winner, "pick", day)
        prb = veto_rates(row.loser, "pick", day)

        def score_vec(rates, base):
            return np.array([(rates.get(m, 0.0) + 0.02) for m in pool]) * base

        ban_a = score_vec(bra, 0.75 + wb)   # ban what opp is good on
        ban_b = score_vec(brb, 0.75 + wa)
        pick_a = score_vec(pra, (0.3 + wa) ** 2)
        pick_b = score_vec(prb, (0.3 + wb) ** 2)

        fmt = row.fmt
        steps = ([("A", ban_a), ("B", ban_b), ("A", pick_a), ("B", pick_b),
                  ("A", ban_a), ("B", ban_b)] if fmt in ("bo3", "bo1") else
                 [("A", ban_a), ("A", ban_a), ("A", pick_a), ("B", pick_b),
                  ("A", pick_a), ("B", pick_b)] if fmt == "bo5_gf" else
                 [("A", ban_a), ("B", ban_b), ("A", pick_a), ("B", pick_b),
                  ("A", pick_a), ("B", pick_b)])
        thresh = 3 if fmt in ("bo5", "bo5_gf") else (1 if fmt == "bo1" else 2)

        alive = np.ones((N_SIMS, nm), dtype=bool)
        played = np.zeros((N_SIMS, nm), dtype=bool)
        for side, sc in steps:
            scm = np.where(alive, sc[None, :], 0.0)
            tot = scm.sum(axis=1, keepdims=True)
            tot[tot == 0] = 1.0
            cum = np.cumsum(scm / tot, axis=1)
            r = rng.random((N_SIMS, 1))
            choice = (r <= cum).argmax(axis=1)
            is_pick = sc is pick_a or sc is pick_b
            rowsidx = np.arange(N_SIMS)
            if is_pick:
                played[rowsidx, choice] = True
            alive[rowsidx, choice] = False
        # decider = last alive map
        rem_counts = alive.sum(axis=1)
        has_dec = rem_counts > 0
        dec_choice = alive.argmax(axis=1)
        played[np.arange(N_SIMS)[has_dec], dec_choice[has_dec]] = True

        wins = rng.random((N_SIMS, nm)) < p_map[None, :]
        sw = (wins & played).sum(axis=1)
        p_series = float(((sw >= thresh)).mean())
        p_series = min(max(p_series, 1e-4), 1 - 1e-4)
        lls.append(-math.log(p_series))
    results[cname] = {"ll_series_mc": round(float(np.mean(lls)), 5),
                      "n": len(lls)}
    print(f"{cname:<14} series-MC ll={results[cname]['ll_series_mc']:.5f} "
          f"(n={results[cname]['n']})", flush=True)

with open(os.path.join(OUT, "series_mc.json"), "w") as f:
    json.dump(results, f, indent=1)
print("saved out/series_mc.json")
