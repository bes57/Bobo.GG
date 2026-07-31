"""E2: round-level Bradley-Terry, joint fit with preregistered map-level
fallback. Parity-gated weight replication; lambda train-only; CRN-judged.
Writes stats/h1_roundbt.json + scratch npz."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from h1_lib import (SCRATCH, STATS, TRAIN_END, load_frame, log, make_engine,
                    series_prob, fit_beta, v6_cfg)
from e2_lib import bt_fit, daily_weights, massey_parity, race, sig, drace

sys.path.insert(0, "/Users/benny_es1/PythonTest/testing_lab/v8")
import referee

frame = load_frame()
eng = make_engine(frame)
stage_by_mid = dict(zip(frame.match_id, frame.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
W_CUSTOM = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
n_t = len(eng.teams)
base = np.load(os.path.join(SCRATCH, "baseline_v6.npz"), allow_pickle=True)

# ── parity gate ─────────────────────────────────────────────────────────────
t0 = time.time()
gap, same_valid = massey_parity(eng, frame, W_CUSTOM, base["rdiff"])
log(f"E2 parity gate: max|rdiff gap| = {gap:.2e}, same_valid={same_valid} "
    f"({time.time()-t0:.1f}s)")
if gap > 1e-8 or not same_valid:
    raise RuntimeError("PARITY GATE FAILED — weight replication drifted")

# ── build round cells + fallback rows ───────────────────────────────────────
ro = pd.read_csv("/Users/benny_es1/PythonTest/data/enriched/round_outcomes.csv")
agg = (ro.groupby(["match_id", "map_num", "winner_org", "winner_side"])
       .size().unstack("winner_side", fill_value=0).reset_index())
for c in ("attack", "defense"):
    if c not in agg.columns:
        agg[c] = 0
# per (match, map): rows per org with attack/defense round wins
by_map = {}
for r in agg.itertuples(index=False):
    by_map.setdefault((r.match_id, r.map_num), {})[r.winner_org] = (
        int(r.attack), int(r.defense))
maps_sorted = {}
for (mid, mnum) in by_map:
    maps_sorted.setdefault(mid, []).append(mnum)
for mid in maps_sorted:
    maps_sorted[mid].sort()

games_by_match = {}
for gi, g in enumerate(eng.games):
    games_by_match.setdefault(g["match_id"], []).append(gi)

cells = {"gi": [], "att": [], "dfn": [], "k": [], "n": []}
fb = {"gi": [], "i": [], "j": []}
n_cov = n_fb = n_mismatch = 0
for mid, gis in games_by_match.items():
    mnums = maps_sorted.get(mid, [])
    for k_map, gi in enumerate(gis):
        g = eng.games[gi]
        ok = False
        if k_map < len(mnums):
            d = by_map[(mid, mnums[k_map])]
            worg = g["winner"]; lorg = g["loser"]
            if worg in d or lorg in d:
                wa, wd = d.get(worg, (0, 0))
                la, ld = d.get(lorg, (0, 0))
                # validation: round counts must reproduce the map score
                if wa + wd == g["wr"] and la + ld == g["lr"] and \
                        worg in eng.tidx and lorg in eng.tidx:
                    wi_, li_ = eng.tidx[worg], eng.tidx[lorg]
                    # cell 1: winner attacking vs loser defending
                    cells["gi"] += [gi, gi]
                    cells["att"] += [wi_, li_]
                    cells["dfn"] += [li_, wi_]
                    cells["k"] += [wa, la]
                    cells["n"] += [wa + ld, la + wd]
                    ok = True
        if ok:
            n_cov += 1
        else:
            if k_map < len(mnums):
                n_mismatch += 1
            fb["gi"].append(gi)
            fb["i"].append(eng.tidx[g["winner"]])
            fb["j"].append(eng.tidx[g["loser"]])
            n_fb += 1
cells = {k: np.array(v) for k, v in cells.items()}
fb = {k: np.array(v) for k, v in fb.items()}
log(f"E2 rows: covered maps {n_cov}, fallback maps {n_fb} "
    f"(score-mismatch/junk demoted to fallback: {n_mismatch}), "
    f"cells {len(cells['gi'])}")
if n_mismatch > 0.02 * (n_cov + n_mismatch):
    raise RuntimeError("round/score mismatch rate > 2% — fail loudly")

# per-event coverage table (frame matches)
cov_gi = set(cells["gi"].tolist())
gcov = np.zeros(len(eng.games), dtype=bool)
gcov[list(cov_gi)] = True
mid_cov = {}
for gi, g in enumerate(eng.games):
    mid_cov.setdefault(g["match_id"], []).append(gcov[gi])
frame_cov = frame.match_id.map(lambda m: float(np.mean(mid_cov.get(m, [0.0]))))
ev_cov = (frame.assign(cov=frame_cov).groupby("event_id")["cov"]
          .agg(["mean", "count"]))

# ── day-row preparation ─────────────────────────────────────────────────────
def day_rows(w_g):
    cm = w_g[cells["gi"]] > 0
    fm = w_g[fb["gi"]] > 0
    return {"ca": cells["att"][cm], "cd": cells["dfn"][cm],
            "ck": cells["k"][cm].astype(float),
            "cn": cells["n"][cm].astype(float),
            "cw": w_g[cells["gi"]][cm],
            "fi": fb["i"][fm], "fj": fb["j"][fm], "fw": w_g[fb["gi"]][fm]}


def bt_day(rows, lam, prior_vec, s0, h0, max_iter=60):
    """Fisher-scoring Newton for one day. Model B (map-only) = rows with
    empty 'ca' and every map in fi/fj/fw."""
    s = s0.copy(); h = h0
    lam_reg = 3.0 * lam
    it_used = 0
    for it in range(max_iter):
        grad = np.zeros(n_t + 1)
        H = np.zeros((n_t + 1, n_t + 1))
        if len(rows["ca"]):
            eta = s[rows["ca"]] - s[rows["cd"]] + h
            p = sig(eta)
            g_row = rows["cw"] * (rows["ck"] - rows["cn"] * p)
            f_row = rows["cw"] * rows["cn"] * p * (1 - p) + 1e-12
            np.add.at(grad, rows["ca"], g_row)
            np.add.at(grad, rows["cd"], -g_row)
            np.add.at(H, (rows["ca"], rows["ca"]), f_row)
            np.add.at(H, (rows["cd"], rows["cd"]), f_row)
            np.add.at(H, (rows["ca"], rows["cd"]), -f_row)
            np.add.at(H, (rows["cd"], rows["ca"]), -f_row)
            grad[n_t] += g_row.sum()
            H[n_t, n_t] += f_row.sum()
            hcol = np.full(len(rows["ca"]), n_t)
            np.add.at(H, (rows["ca"], hcol), f_row)
            np.add.at(H, (hcol, rows["ca"]), f_row)
            np.add.at(H, (rows["cd"], hcol), -f_row)
            np.add.at(H, (hcol, rows["cd"]), -f_row)
        fi, fj, fw = rows["fi"], rows["fj"], rows["fw"]
        if len(fi):
            q = s[fi] - s[fj]
            sp_, sm_ = sig(q + h), sig(q - h)
            pb = 0.5 * (sp_ + sm_)
            dpb = 0.5 * (sp_ * (1 - sp_) + sm_ * (1 - sm_))
            P = np.clip(race(pb), 1e-9, 1 - 1e-9)
            dP = drace(pb) * dpb
            u = fw * dP / P
            f2 = fw * dP * dP / (P * (1 - P)) + 1e-12
            np.add.at(grad, fi, u)
            np.add.at(grad, fj, -u)
            np.add.at(H, (fi, fi), f2)
            np.add.at(H, (fj, fj), f2)
            np.add.at(H, (fi, fj), -f2)
            np.add.at(H, (fj, fi), -f2)
        grad[:n_t] += -lam * s - lam_reg * (s - prior_vec)
        H[np.arange(n_t), np.arange(n_t)] += lam + lam_reg
        grad[n_t] += -1e-4 * h
        H[n_t, n_t] += 1e-4 + 1e-9
        step = np.linalg.solve(H, grad)
        nrm = np.max(np.abs(step))
        if nrm > 5.0:
            step *= 5.0 / nrm
        s += step[:n_t]; h += step[n_t]
        it_used = it + 1
        if nrm < 1e-8:
            break
    return s, h, it_used


def region_prior(s_prev):
    prior = np.zeros(n_t)
    if s_prev is not None:
        for ri_ in range(4):
            m = eng.team_region_idx == ri_
            if m.sum() >= 4:
                prior[m] = s_prev[m].mean()
    return prior


def run_walkforward(lams, days, tag):
    """Per-lambda rdiff vectors over frame rows (fit on games < day)."""
    from collections import defaultdict
    s_by_day = defaultdict(list)
    for i, r in enumerate(frame.itertuples(index=False)):
        s_by_day[r.date].append(i)
    state = {l: {"s": np.zeros(n_t), "h": 0.0, "prev": None} for l in lams}
    rd = {l: np.full(len(frame), np.nan) for l in lams}
    h_last = {}
    t0 = time.time()
    for di, day in enumerate(days):
        m_hist, w = daily_weights(eng, day, W_CUSTOM)
        if m_hist is None:
            continue
        w_g = np.zeros(len(eng.games))
        w_g[np.where(m_hist)[0]] = w
        rows = day_rows(w_g)
        for l in lams:
            st = state[l]
            s_fit, h_fit, _ = bt_day(rows, l, region_prior(st["prev"]),
                                     st["s"], st["h"])
            st["s"], st["h"], st["prev"] = s_fit, h_fit, s_fit
            h_last[l] = h_fit
            for i in s_by_day[day]:
                rd[l][i] = (s_fit[eng.tidx[frame.winner.iloc[i]]]
                            - s_fit[eng.tidx[frame.loser.iloc[i]]])
    log(f"E2 walkforward[{tag}] {len(days)} days x {len(lams)} lambdas: "
        f"{time.time()-t0:.0f}s (h_last={ {l: round(h_last.get(l, np.nan), 4) for l in lams} })")
    return rd, h_last


fmts = frame.fmt.values
train_m = (frame.date <= TRAIN_END).values
hold_m = (frame.date > TRAIN_END).values

# ── lambda selection: walk-forward WITHIN TRAIN only ────────────────────────
LAMS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
train_days = [d for d in eng.pred_days if d <= TRAIN_END]
rd_tr, _ = run_walkforward(LAMS, train_days, "lambda-grid/train")
lam_tab = []
for l in LAMS:
    v = ~np.isnan(rd_tr[l]) & train_m
    b = fit_beta(rd_tr[l], fmts, v, bounds=(0.03, 40.0))
    p = series_prob(b, rd_tr[l][v], fmts[v])
    ll = float(-np.mean(np.log(np.clip(p, 1e-9, 1))))
    lam_tab.append({"lam": l, "beta": round(b, 4), "ll_train_wf": round(ll, 5),
                    "n": int(v.sum())})
    log(f"E2 lambda {l}: train wf LL {ll:.5f} beta {b:.4f} n {int(v.sum())}")
LAM = min(lam_tab, key=lambda r: r["ll_train_wf"])["lam"]
log(f"E2 chosen lambda = {LAM} (train-only)")

# ── final full run ──────────────────────────────────────────────────────────
rd_all, h_last = run_walkforward([LAM], eng.pred_days, "final")
rd_bt = rd_all[LAM]
valid = ~np.isnan(rd_bt)
beta_bt = fit_beta(rd_bt, fmts, valid & train_m, bounds=(0.03, 40.0))
p_bt = np.full(len(frame), np.nan)
m = valid
p_bt[m] = series_prob(beta_bt, rd_bt[m], fmts[m])
ll_hold = float(-np.mean(np.log(np.clip(p_bt[valid & hold_m], 1e-9, 1))))
log(f"E2 final: beta {beta_bt:.4f} holdout LL {ll_hold:.5f} "
    f"n {int((valid & hold_m).sum())}")

np.savez_compressed(os.path.join(SCRATCH, "roundbt.npz"), rdiff=rd_bt,
                    p=p_bt, beta=beta_bt, lam=LAM)
json.dump({"lam_tab": lam_tab, "h_last": h_last[LAM]},
          open(os.path.join(SCRATCH, "roundbt_lam.json"), "w"), indent=1)
log("E2 fit artifacts saved; judging + effective-sample in next step")
