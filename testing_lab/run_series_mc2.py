"""Series-MC v2: the games-decay package on the traded surface.
Configs: games16 + k in {5, 20, overall}, plus k20 with intra-series
correlation sigma=0.7 (momentum test). Compare to prod_k5 from series_mc.json.
Writes out/series_mc2.json."""
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
VLAM = math.log(2) / 6.0
HLG = 16.0
LAMG = math.log(2) / HLG

eng = Engine()
s = eng.series.reset_index(drop=True)
games = eng.games
g_map = np.array([g["map_name"] for g in games])
maps_all = sorted(set(g_map))
mIdx = {m: i for i, m in enumerate(maps_all)}
map_rows = {m: np.where(g_map == m)[0] for m in maps_all}
n_t = len(eng.teams)

v = pd.read_csv(os.path.join(DATA, "map_vetos.csv"))
dates_j = json.load(open(os.path.join(DATA, "match_dates.json")))
v["date"] = v.MatchID.astype(str).map(dates_j)
v = v.dropna(subset=["date"]).sort_values(["date", "MatchID", "step"])
v = v[v["map"].isin(maps_all)]
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


def games_weights(rows, day_num):
    """Per-row sqrt(w_winner*w_loser) with games-ago decay within `rows`."""
    pos = {gi: k for k, gi in enumerate(rows)}
    w_w = np.ones(len(rows))
    w_l = np.ones(len(rows))
    per_team = defaultdict(list)
    for gi in rows:
        g = games[gi]
        per_team[g["winner"]].append(gi)
        per_team[g["loser"]].append(gi)
    for org, lst in per_team.items():
        n_p = len(lst)
        for k, gi in enumerate(lst):
            wv = math.exp(-LAMG * (n_p - 1 - k))
            if games[gi]["winner"] == org:
                w_w[pos[gi]] = wv
            else:
                w_l[pos[gi]] = wv
    return np.sqrt(w_w * w_l)


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


def map_winpct(day_num):
    wnum = np.zeros((n_t, len(maps_all)))
    wden = np.zeros((n_t, len(maps_all)))
    hist = np.where(eng.g_dnum < day_num)[0]
    weeks = (day_num - eng.g_dnum[hist]) / 7.0
    w = decay_weight(weeks, "exp", hl=6.0)
    for j, gi in enumerate(hist):
        mi = mIdx[g_map[gi]]
        wnum[eng.wi[gi], mi] += w[j]
        wden[eng.wi[gi], mi] += w[j]
        wden[eng.li[gi], mi] += w[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        wp = wnum / wden
    return wp, wden


CONFIGS = {
    "g16_k5": {"k": 5.0, "sigma": 0.0},
    "g16_k20": {"k": 20.0, "sigma": 0.0},
    "g16_overall": {"k": None, "sigma": 0.0},
    "g16_k20_corr07": {"k": 20.0, "sigma": 0.7},
}
BETA = 0.111
POW = 0.75

rd_t = np.copysign(np.abs(eng.rd_raw) ** POW * 2.5, eng.rd_raw)
test_rows = s[(s.date > "2024-12-31")].index.values
rng = np.random.default_rng(7)
results = {}

# shared per-day cache (ratings identical across configs; only k/sigma differ
# at prediction time, so cache overall+permap+neff once)
cache_day = {}


def day_cache(day):
    if day in cache_day:
        return cache_day[day]
    day_num = int(np.datetime64(day, "D").astype(int))
    hist = np.where(eng.g_dnum < day_num)[0]
    if len(hist) < 30:
        cache_day[day] = None
        return None
    w = games_weights(hist, day_num) * np.where(eng.champ[hist], 2.0, 1.0)
    overall = massey(hist, w, rd_t[hist])
    permap, neff = {}, {}
    for m in maps_all:
        rowsm = np.intersect1d(hist, map_rows[m])
        if len(rowsm) < 8:
            continue
        wm = games_weights(rowsm, day_num)
        permap[m] = massey(rowsm, wm, rd_t[rowsm])
        cnt = np.zeros(n_t)
        np.add.at(cnt, eng.wi[rowsm], wm)
        np.add.at(cnt, eng.li[rowsm], wm)
        neff[m] = cnt
    wp, wden = map_winpct(day_num)
    cache_day[day] = (overall, permap, neff, wp, wden)
    return cache_day[day]


for cname, cfg in CONFIGS.items():
    lls = []
    for si in test_rows:
        row = s.iloc[si]
        cd = day_cache(row.date)
        if cd is None:
            continue
        overall, permap, neff, wp, wden = cd
        ia, ib = eng.tidx[row.winner], eng.tidx[row.loser]
        pool = pool_asof(row.date)
        nm = len(pool)

        def rvec(ti):
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

        ra, rb = rvec(ia), rvec(ib)
        logits = BETA * (ra - rb)

        def winp(ti):
            out = np.empty(nm)
            for j, m in enumerate(pool):
                mi = mIdx[m]
                out[j] = wp[ti, mi] if wden[ti, mi] > 0.4 else 0.5
                if not np.isfinite(out[j]):
                    out[j] = 0.5
            return out

        wa, wb = winp(ia), winp(ib)
        bra = veto_rates(row.winner, "ban", row.date)
        brb = veto_rates(row.loser, "ban", row.date)
        pra = veto_rates(row.winner, "pick", row.date)
        prb = veto_rates(row.loser, "pick", row.date)

        def sv(rates, base):
            return np.array([(rates.get(m, 0.0) + 0.02) for m in pool]) * base

        ban_a, ban_b = sv(bra, 0.75 + wb), sv(brb, 0.75 + wa)
        pick_a, pick_b = sv(pra, (0.3 + wa) ** 2), sv(prb, (0.3 + wb) ** 2)
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
            rr = rng.random((N_SIMS, 1))
            choice = (rr <= cum).argmax(axis=1)
            is_pick = sc is pick_a or sc is pick_b
            ridx = np.arange(N_SIMS)
            if is_pick:
                played[ridx, choice] = True
            alive[ridx, choice] = False
        rem_counts = alive.sum(axis=1)
        has_dec = rem_counts > 0
        dec_choice = alive.argmax(axis=1)
        played[np.arange(N_SIMS)[has_dec], dec_choice[has_dec]] = True

        lg = logits[None, :]
        if cfg["sigma"] > 0:
            z = rng.normal(0.0, cfg["sigma"], size=(N_SIMS, 1))
            lg = lg + z
        pmap_sim = 1.0 / (1.0 + np.exp(-lg))
        wins = rng.random((N_SIMS, nm)) < pmap_sim
        sw = (wins & played).sum(axis=1)
        p_series = float((sw >= thresh).mean())
        p_series = min(max(p_series, 1e-4), 1 - 1e-4)
        lls.append(-math.log(p_series))
    results[cname] = {"ll_series_mc": round(float(np.mean(lls)), 5), "n": len(lls)}
    print(f"{cname:<18} ll={results[cname]['ll_series_mc']:.5f} (n={len(lls)})",
          flush=True)

with open(os.path.join(OUT, "series_mc2.json"), "w") as f:
    json.dump(results, f, indent=1)
print("saved out/series_mc2.json")
