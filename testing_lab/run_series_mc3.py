"""Traded-surface package v3 (native data): series veto-MC with
  - asym games decay (W20/L12) overall + per-map layers
  - within-map games-decay HL sweep x SHRINK_K sweep
  - opponent-aware pick scores in the veto sim
  - pick-side bonus on picked maps (walk-forward b_pick, refit yearly)
2,000 sims/match, scored on 2025-26. Writes out/series_mc3.json."""
import bisect
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine

OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "..", "data")
N_SIMS = 2000
VLAM = math.log(2) / 6.0
HLW, HLL = 20.0, 12.0
LAMW, LAML = math.log(2) / HLW, math.log(2) / HLL
BETA = 0.103

eng = Engine()
s = eng.series.reset_index(drop=True)
games = eng.games
g_map = np.array([g["map_name"] for g in games])
maps_all = sorted(set(g_map))
mIdx = {m: i for i, m in enumerate(maps_all)}
map_rows = {m: np.where(g_map == m)[0] for m in maps_all}
n_t = len(eng.teams)
stage_by_mid = dict(zip(s.match_id, s.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in games])
PO_W = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)

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


def asym_games_weights(rows, w_extra=None):
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
            ago = n_p - 1 - k
            if games[gi]["winner"] == org:
                w_w[pos[gi]] = math.exp(-LAMW * ago)
            else:
                w_l[pos[gi]] = math.exp(-LAML * ago)
    w = np.sqrt(w_w * w_l)
    if w_extra is not None:
        w = w * w_extra
    return w


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


rd_t = np.copysign(np.abs(eng.rd_raw) ** 0.75 * 2.5, eng.rd_raw)

# yearly walk-forward b_pick (fit on all data before Jan 1 of each test year)
picker = {}
for r in v.itertuples(index=False):
    if r.action == "pick":
        picker[(int(r.MatchID), str(r.map).strip())] = r.team
B_PICK = {}
from scipy.optimize import minimize
for yr in (2025, 2026):
    cutoff = f"{yr}-01-01"
    rows_fit = []
    for g in games:
        if g["date_s"] >= cutoff:
            continue
        pk = picker.get((g["match_id"], g["map_name"]))
        sign = 1.0 if pk == g["winner"] else (-1.0 if pk == g["loser"] else 0.0)
        rows_fit.append(sign)
    # cheap: reuse global fit shape — b_pick from logistic on sign alone with
    # rating control folded into intercept-free symmetric design; approximate
    # with frequency: picker winrate -> logit
    signs = np.array(rows_fit)
    pw = (signs == 1).sum() / max((signs != 0).sum(), 1)
    B_PICK[yr] = math.log(pw / (1 - pw))
print("b_pick by test year (freq-based, walk-forward):",
      {k: round(v_, 4) for k, v_ in B_PICK.items()})

CONFIGS = {
    "overall_only": {"k": None, "hl_map": None, "opp_pick": False, "pick_bonus": False},
    "k20_hl16": {"k": 20.0, "hl_map": 16.0, "opp_pick": False, "pick_bonus": False},
    "k20_hl8": {"k": 20.0, "hl_map": 8.0, "opp_pick": False, "pick_bonus": False},
    "k40_hl8": {"k": 40.0, "hl_map": 8.0, "opp_pick": False, "pick_bonus": False},
    "k20_hl8_opp": {"k": 20.0, "hl_map": 8.0, "opp_pick": True, "pick_bonus": False},
    "k20_hl8_opp_pb": {"k": 20.0, "hl_map": 8.0, "opp_pick": True, "pick_bonus": True},
    "overall_opp_pb": {"k": None, "hl_map": None, "opp_pick": True, "pick_bonus": True},
}

if os.environ.get("MC_ONLY"):
    CONFIGS = {"overall_opp_pb": {"k": None, "hl_map": None,
                                  "opp_pick": True, "pick_bonus": True}}
test_rows = s[(s.date > "2024-12-31")].index.values
rng = np.random.default_rng(23)
results = {}
probs_store = {}
day_cache = {}


def get_day(day, hl_map, k):
    key = (day, hl_map)
    if key in day_cache:
        return day_cache[key]
    day_num = int(np.datetime64(day, "D").astype(int))
    hist = np.where(eng.g_dnum < day_num)[0]
    if len(hist) < 30:
        day_cache[key] = None
        return None
    w = asym_games_weights(hist, PO_W[hist])
    overall = massey(hist, w, rd_t[hist])
    permap, neff = {}, {}
    if hl_map is not None:
        lam_m = math.log(2) / hl_map
        for m in maps_all:
            rowsm = np.intersect1d(hist, map_rows[m])
            if len(rowsm) < 8:
                continue
            pos = {gi: kk for kk, gi in enumerate(rowsm)}
            wm = np.ones(len(rowsm))
            per_team = defaultdict(list)
            for gi in rowsm:
                g = games[gi]
                per_team[g["winner"]].append(gi)
                per_team[g["loser"]].append(gi)
            w_w = np.ones(len(rowsm))
            w_l = np.ones(len(rowsm))
            for org, lst in per_team.items():
                n_p = len(lst)
                for kk, gi in enumerate(lst):
                    ago = n_p - 1 - kk
                    if games[gi]["winner"] == org:
                        w_w[pos[gi]] = math.exp(-lam_m * ago)
                    else:
                        w_l[pos[gi]] = math.exp(-lam_m * ago)
            wm = np.sqrt(w_w * w_l)
            permap[m] = massey(rowsm, wm, rd_t[rowsm])
            cnt = np.zeros(n_t)
            np.add.at(cnt, eng.wi[rowsm], wm)
            np.add.at(cnt, eng.li[rowsm], wm)
            neff[m] = cnt
    # decayed map winpct for veto factors (calendar 6w as production)
    wnum = np.zeros((n_t, len(maps_all)))
    wden = np.zeros((n_t, len(maps_all)))
    weeks = (day_num - eng.g_dnum[hist]) / 7.0
    wv = np.exp(-VLAM * weeks)
    for j, gi in enumerate(hist):
        mi = mIdx[g_map[gi]]
        wnum[eng.wi[gi], mi] += wv[j]
        wden[eng.wi[gi], mi] += wv[j]
        wden[eng.li[gi], mi] += wv[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        wp = wnum / wden
    day_cache[key] = (overall, permap, neff, wp, wden)
    return day_cache[key]


for cname, cfg in CONFIGS.items():
    lls = []
    for si in test_rows:
        row = s.iloc[si]
        cd = get_day(row.date, cfg["hl_map"], cfg["k"])
        if cd is None:
            continue
        overall, permap, neff, wp, wden = cd
        if row.winner not in eng.tidx or row.loser not in eng.tidx:
            continue
        ia, ib = eng.tidx[row.winner], eng.tidx[row.loser]
        pool = pool_asof(row.date)
        nm = len(pool)

        def rvec(ti):
            outv = np.empty(nm)
            for j, m in enumerate(pool):
                ov = overall[ti]
                if cfg["k"] is None or m not in permap:
                    outv[j] = ov
                else:
                    ne = neff[m][ti]
                    a = ne / (ne + cfg["k"])
                    outv[j] = a * permap[m][ti] + (1 - a) * ov
            return outv

        ra, rb = rvec(ia), rvec(ib)
        logits = BETA * (ra - rb)

        def winp(ti):
            outv = np.empty(nm)
            for j, m in enumerate(pool):
                mi = mIdx[m]
                outv[j] = wp[ti, mi] if wden[ti, mi] > 0.4 else 0.5
                if not np.isfinite(outv[j]):
                    outv[j] = 0.5
            return outv

        wa, wb = winp(ia), winp(ib)
        bra = veto_rates(row.winner, "ban", row.date)
        brb = veto_rates(row.loser, "ban", row.date)
        pra = veto_rates(row.winner, "pick", row.date)
        prb = veto_rates(row.loser, "pick", row.date)

        def sv(rates, base):
            return np.array([(rates.get(m, 0.0) + 0.02) for m in pool]) * base

        ban_a, ban_b = sv(bra, 0.75 + wb), sv(brb, 0.75 + wa)
        if cfg["opp_pick"]:
            pick_a = sv(pra, (0.3 + wa) ** 2 * (1.75 - wb))
            pick_b = sv(prb, (0.3 + wb) ** 2 * (1.75 - wa))
        else:
            pick_a = sv(pra, (0.3 + wa) ** 2)
            pick_b = sv(prb, (0.3 + wb) ** 2)
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
        picked_by_a = np.zeros((N_SIMS, nm), dtype=bool)
        picked_by_b = np.zeros((N_SIMS, nm), dtype=bool)
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
                if side == "A":
                    picked_by_a[ridx, choice] = True
                else:
                    picked_by_b[ridx, choice] = True
            alive[ridx, choice] = False
        rem_counts = alive.sum(axis=1)
        has_dec = rem_counts > 0
        dec_choice = alive.argmax(axis=1)
        played[np.arange(N_SIMS)[has_dec], dec_choice[has_dec]] = True

        lg = np.broadcast_to(logits[None, :], (N_SIMS, nm)).copy()
        if cfg["pick_bonus"]:
            bp = B_PICK[2026] if row.date >= "2026-01-01" else B_PICK[2025]
            lg = lg + bp * picked_by_a.astype(float) - bp * picked_by_b.astype(float)
        pmap_sim = 1.0 / (1.0 + np.exp(-lg))
        wins = rng.random((N_SIMS, nm)) < pmap_sim
        sw = (wins & played).sum(axis=1)
        p_series = float((sw >= thresh).mean())
        p_series = min(max(p_series, 1e-4), 1 - 1e-4)
        lls.append(-math.log(p_series))
        probs_store[(cname, int(row.match_id))] = p_series
    results[cname] = {"ll_series_mc": round(float(np.mean(lls)), 5), "n": len(lls)}
    print(f"{cname:<18} ll={results[cname]['ll_series_mc']:.5f} (n={len(lls)})",
          flush=True)

with open(os.path.join(OUT, "series_mc3.json"), "w") as f:
    json.dump(results, f, indent=1)
if os.environ.get("MC_ONLY"):
    with open(os.path.join(OUT, "mc_probs.json"), "w") as f:
        json.dump({f"{k[1]}": v_ for k, v_ in probs_store.items()}, f)
    print("saved out/mc_probs.json")
print("saved out/series_mc3.json")
