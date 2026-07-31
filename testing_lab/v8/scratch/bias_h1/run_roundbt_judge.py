"""E2 step 2: judge round-BT vs v6 (CRN referee) + MEASURED effective-sample
multiplier (Fisher + cluster bootstrap, preregistered). Writes
stats/h1_roundbt.json and stats/h1_bias_caterpillar.json."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from h1_lib import SCRATCH, STATS, TRAIN_END, load_frame, log, make_engine
from e2_lib import daily_weights, race, sig, drace

sys.path.insert(0, "/Users/benny_es1/PythonTest/testing_lab/v8")
import referee

frame = load_frame()
eng = make_engine(frame)
n_t = len(eng.teams)
stage_by_mid = dict(zip(frame.match_id, frame.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in eng.games])
W_CUSTOM = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)

base = np.load(os.path.join(SCRATCH, "baseline_v6.npz"), allow_pickle=True)
bt = np.load(os.path.join(SCRATCH, "roundbt.npz"), allow_pickle=True)
lamj = json.load(open(os.path.join(SCRATCH, "roundbt_lam.json")))
LAM = float(bt["lam"])

# ── rebuild cells/fallback (same deterministic code as run_roundbt) ────────
ro = pd.read_csv("/Users/benny_es1/PythonTest/data/enriched/round_outcomes.csv")
agg = (ro.groupby(["match_id", "map_num", "winner_org", "winner_side"])
       .size().unstack("winner_side", fill_value=0).reset_index())
for c in ("attack", "defense"):
    if c not in agg.columns:
        agg[c] = 0
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
            worg, lorg = g["winner"], g["loser"]
            if worg in d or lorg in d:
                wa, wd = d.get(worg, (0, 0))
                la, ld = d.get(lorg, (0, 0))
                if wa + wd == g["wr"] and la + ld == g["lr"] and \
                        worg in eng.tidx and lorg in eng.tidx:
                    wi_, li_ = eng.tidx[worg], eng.tidx[lorg]
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
gcov = np.zeros(len(eng.games), dtype=bool)
gcov[np.unique(cells["gi"])] = True
log(f"E2-judge: rebuilt cells (cov {n_cov} fb {n_fb} mismatch {n_mismatch})")

g_winner_idx = np.array([eng.tidx[g["winner"]] for g in eng.games])
g_loser_idx = np.array([eng.tidx[g["loser"]] for g in eng.games])

# per-event coverage of frame matches (map share with round cells)
mid_cov = {}
for gi, g in enumerate(eng.games):
    mid_cov.setdefault(g["match_id"], []).append(bool(gcov[gi]))
frame_cov = frame.match_id.map(lambda m: float(np.mean(mid_cov.get(m, [0.0]))))
ev_tab = (frame.assign(cov=frame_cov).groupby("event_id")["cov"]
          .agg(["mean", "count"]).reset_index())
coverage = {
    "n_maps_covered": n_cov, "n_maps_fallback": n_fb,
    "score_mismatch_demoted": n_mismatch,
    "map_coverage_share": round(n_cov / (n_cov + n_fb), 4),
    "holdout_series_full_cov": round(float(
        (frame_cov[(frame.date > TRAIN_END).values] == 1.0).mean()), 4),
    "per_event": [{"event_id": r["event_id"], "map_cov": round(r["mean"], 3),
                   "n": int(r["count"])} for r in ev_tab.to_dict("records")],
}

# ── judging vs v6 ───────────────────────────────────────────────────────────
s = frame
fmts = s.fmt.values
hold = (s.date > TRAIN_END).values
p6 = np.full(len(s), np.nan); p6[base["test_mask"]] = base["p_test"]
p_bt = bt["p"]
joint = ~np.isnan(p6) & ~np.isnan(p_bt) & hold
d = referee.per_series_ll(p6[joint]) - referee.per_series_ll(p_bt[joint])
bt_iid = referee.paired_bootstrap_crn(d, mode="iid")
bt_blk = referee.paired_bootstrap_crn(d, mode="block_event",
                                      event_ids=s.event_id.values[joint])
roi = referee.expected_roi_of_dll(float(np.mean(d)), p6[joint])
bias6 = referee.per_team_bias(p6, s.winner.values, s.loser.values,
                              holdout=hold, valid=~np.isnan(p6))
bias_bt = referee.per_team_bias(p_bt, s.winner.values, s.loser.values,
                                holdout=hold, valid=~np.isnan(p_bt))
buckets = referee.bucketed(s, p_bt, p_ref=p6, rdiff=bt["rdiff"],
                           holdout=hold, valid=~np.isnan(p_bt),
                           games=eng.games)
log(f"E2 judge: dLL={np.mean(d)*1000:+.3f}m iid[{bt_iid['ci_lo']*1000:+.2f},"
    f"{bt_iid['ci_hi']*1000:+.2f}] blk[{bt_blk['ci_lo']*1000:+.2f},"
    f"{bt_blk['ci_hi']*1000:+.2f}] maxbias {bias_bt['max_abs_bias']} "
    f"(v6 {bias6['max_abs_bias']})")

# ── effective-sample measurement at 2025-01-01 ─────────────────────────────
REF_DAY = "2025-01-01"
m_hist, w = daily_weights(eng, REF_DAY, W_CUSTOM)
w_g = np.zeros(len(eng.games)); w_g[np.where(m_hist)[0]] = w

def rows_A(w_g_):
    cm = w_g_[cells["gi"]] > 0
    fm = w_g_[fb["gi"]] > 0
    return {"ca": cells["att"][cm], "cd": cells["dfn"][cm],
            "ck": cells["k"][cm].astype(float),
            "cn": cells["n"][cm].astype(float), "cw": w_g_[cells["gi"]][cm],
            "fi": fb["i"][fm], "fj": fb["j"][fm], "fw": w_g_[fb["gi"]][fm]}

def rows_B(w_g_):
    am = w_g_ > 0
    return {"ca": np.array([], dtype=int), "cd": np.array([], dtype=int),
            "ck": np.array([]), "cn": np.array([]), "cw": np.array([]),
            "fi": g_winner_idx[am], "fj": g_loser_idx[am], "fw": w_g_[am]}

def bt_fit_local(rows, s0, h0, max_iter=60):
    s_ = s0.copy(); h = h0
    lam, lam_reg = LAM, 3.0 * LAM
    for it in range(max_iter):
        grad = np.zeros(n_t + 1); H = np.zeros((n_t + 1, n_t + 1))
        if len(rows["ca"]):
            eta = s_[rows["ca"]] - s_[rows["cd"]] + h
            p = sig(eta)
            g_row = rows["cw"] * (rows["ck"] - rows["cn"] * p)
            f_row = rows["cw"] * rows["cn"] * p * (1 - p) + 1e-12
            np.add.at(grad, rows["ca"], g_row)
            np.add.at(grad, rows["cd"], -g_row)
            np.add.at(H, (rows["ca"], rows["ca"]), f_row)
            np.add.at(H, (rows["cd"], rows["cd"]), f_row)
            np.add.at(H, (rows["ca"], rows["cd"]), -f_row)
            np.add.at(H, (rows["cd"], rows["ca"]), -f_row)
            grad[n_t] += g_row.sum(); H[n_t, n_t] += f_row.sum()
            hcol = np.full(len(rows["ca"]), n_t)
            np.add.at(H, (rows["ca"], hcol), f_row)
            np.add.at(H, (hcol, rows["ca"]), f_row)
            np.add.at(H, (rows["cd"], hcol), -f_row)
            np.add.at(H, (hcol, rows["cd"]), -f_row)
        if len(rows["fi"]):
            q = s_[rows["fi"]] - s_[rows["fj"]]
            sp_, sm_ = sig(q + h), sig(q - h)
            pb = 0.5 * (sp_ + sm_)
            dpb = 0.5 * (sp_ * (1 - sp_) + sm_ * (1 - sm_))
            P = np.clip(race(pb), 1e-9, 1 - 1e-9)
            dP = drace(pb) * dpb
            u = rows["fw"] * dP / P
            f2 = rows["fw"] * dP * dP / (P * (1 - P)) + 1e-12
            np.add.at(grad, rows["fi"], u)
            np.add.at(grad, rows["fj"], -u)
            np.add.at(H, (rows["fi"], rows["fi"]), f2)
            np.add.at(H, (rows["fj"], rows["fj"]), f2)
            np.add.at(H, (rows["fi"], rows["fj"]), -f2)
            np.add.at(H, (rows["fj"], rows["fi"]), -f2)
        grad[:n_t] += -lam * s_ - lam_reg * s_
        H[np.arange(n_t), np.arange(n_t)] += lam + lam_reg
        grad[n_t] += -1e-4 * h
        H[n_t, n_t] += 1e-4 + 1e-9
        step = np.linalg.solve(H, grad)
        nrm = np.max(np.abs(step))
        if nrm > 5.0:
            step *= 5.0 / nrm
        s_ += step[:n_t]; h += step[n_t]
        if nrm < 1e-8:
            break
    return s_, h

def fisher_lik(rows, s_, h):
    """Likelihood-only Fisher over s (no priors, h fixed)."""
    I = np.zeros((n_t, n_t))
    if len(rows["ca"]):
        p = sig(s_[rows["ca"]] - s_[rows["cd"]] + h)
        f = rows["cw"] * rows["cn"] * p * (1 - p)
        np.add.at(I, (rows["ca"], rows["ca"]), f)
        np.add.at(I, (rows["cd"], rows["cd"]), f)
        np.add.at(I, (rows["ca"], rows["cd"]), -f)
        np.add.at(I, (rows["cd"], rows["ca"]), -f)
    if len(rows["fi"]):
        q = s_[rows["fi"]] - s_[rows["fj"]]
        sp_, sm_ = sig(q + h), sig(q - h)
        pb = 0.5 * (sp_ + sm_)
        dpb = 0.5 * (sp_ * (1 - sp_) + sm_ * (1 - sm_))
        P = np.clip(race(pb), 1e-9, 1 - 1e-9)
        dP = drace(pb) * dpb
        f = rows["fw"] * dP * dP / (P * (1 - P))
        np.add.at(I, (rows["fi"], rows["fi"]), f)
        np.add.at(I, (rows["fj"], rows["fj"]), f)
        np.add.at(I, (rows["fi"], rows["fj"]), -f)
        np.add.at(I, (rows["fj"], rows["fi"]), -f)
    return I

rA = rows_A(w_g); rB = rows_B(w_g)
sA, hA = bt_fit_local(rA, np.zeros(n_t), 0.0)
sB, hB = bt_fit_local(rB, np.zeros(n_t), 0.0)

# qualifying teams + pin (most train series)
tr_counts = pd.concat([frame[frame.date <= TRAIN_END].winner,
                       frame[frame.date <= TRAIN_END].loser]).value_counts()
qual = [t for t in eng.teams if tr_counts.get(t, 0) >= 25]
pin = eng.tidx[tr_counts.index[0]]
keep = np.array([i for i in range(n_t) if i != pin])

def se_from(I):
    Ik = I[np.ix_(keep, keep)] + 1e-6 * np.eye(len(keep))
    C = np.linalg.inv(Ik)
    se = np.full(n_t, np.nan)
    se[keep] = np.sqrt(np.clip(np.diag(C), 0, None))
    return se

seA = se_from(fisher_lik(rA, sA, hA))
seB = se_from(fisher_lik(rB, sB, hB))
ratios = []
for t in qual:
    i = eng.tidx[t]
    if i != pin and np.isfinite(seA[i]) and np.isfinite(seB[i]) and seA[i] > 0:
        ratios.append((t, float((seB[i] / seA[i]) ** 2)))
k_fisher = float(np.median([r for _, r in ratios]))
log(f"E2 k_eff (Fisher, median over {len(ratios)} teams) = {k_fisher:.2f}")

# per-map info ratio (unweighted counts, at joint fit params)
cov_m = gcov & (w_g > 0)
info_r = []
p1 = sig(sA[cells["att"]] - sA[cells["dfn"]] + hA)
cell_info = cells["n"] * p1 * (1 - p1)
cell_by_gi = {}
for idx, gi in enumerate(cells["gi"]):
    cell_by_gi.setdefault(gi, 0.0)
    cell_by_gi[gi] += float(cell_info[idx])
for gi in np.where(cov_m)[0]:
    q = sA[g_winner_idx[gi]] - sA[g_loser_idx[gi]]
    sp_, sm_ = sig(q + hA), sig(q - hA)
    pb = 0.5 * (sp_ + sm_)
    dpb = 0.5 * (sp_ * (1 - sp_) + sm_ * (1 - sm_))
    P = float(np.clip(race(pb), 1e-9, 1 - 1e-9))
    dP = float(drace(pb) * dpb)
    mi_ = dP * dP / (P * (1 - P))
    if gi in cell_by_gi and mi_ > 0:
        info_r.append(cell_by_gi[gi] / mi_)
info_ratio_mean = float(np.mean(info_r)); info_ratio_med = float(np.median(info_r))
log(f"E2 per-map info ratio: mean {info_ratio_mean:.2f} med {info_ratio_med:.2f}")

# cluster bootstrap by match (derived CRN stream, preregistered)
B = 200
rng = np.random.default_rng([20260728, 774411])
g_mid = np.array([g["match_id"] for g in eng.games])
hist_mids = np.unique(g_mid[w_g > 0])
rows_of_mid = {m_: np.where(g_mid == m_)[0] for m_ in hist_mids}
bootA = np.zeros((B, n_t)); bootB = np.zeros((B, n_t))
t0 = time.time()
for b in range(B):
    draw = rng.integers(0, len(hist_mids), size=len(hist_mids))
    mult = np.zeros(len(eng.games))
    for j in draw:
        mult[rows_of_mid[hist_mids[j]]] += 1.0
    wb = w_g * mult
    sa, ha = bt_fit_local(rows_A(wb), sA, hA, max_iter=30)
    sb, hb = bt_fit_local(rows_B(wb), sB, hB, max_iter=30)
    bootA[b] = sa - sa[pin]
    bootB[b] = sb - sb[pin]
seA_b = bootA.std(axis=0, ddof=1); seB_b = bootB.std(axis=0, ddof=1)
ratios_b = [float((seB_b[eng.tidx[t]] / seA_b[eng.tidx[t]]) ** 2)
            for t in qual if eng.tidx[t] != pin and seA_b[eng.tidx[t]] > 0]
k_boot = float(np.median(ratios_b))
log(f"E2 k_eff (cluster bootstrap B={B}, {time.time()-t0:.0f}s) = {k_boot:.2f}")

mde_within, mde_cross = 1.773, 5.889
ELITE = ["T1", "PRX", "100T", "NRG", "TL"]; WEAK = ["TS", "JDG", "TE", "C9"]
def team_rows(bias, names):
    t = {r["team"]: r for r in bias["teams"]}
    return {nm: t.get(nm) for nm in names}

res = {
    "written_by": "agent:bias_h1",
    "preregistered": "preregister.bias_h1.md E2",
    "model": "round BT: P(round)=sig(s_i-s_j+h*att); covered maps 2 binomial "
             "cells; uncovered maps race(Bin24) map-Bernoulli in same joint "
             "fit; v6 per-day weights (parity-gated 6.1e-12); lambda + 3lambda "
             "region ridge; beta train-only",
    "coverage": coverage,
    "lambda_selection": {"grid": lamj["lam_tab"], "chosen": LAM,
                         "rule": "train-only walk-forward series LL"},
    "attack_adv_h_final": round(float(lamj["h_last"]), 4),
    "beta": round(float(bt["beta"]), 4),
    "holdout": {"n_joint": int(joint.sum()),
                "ll_v6": round(float(referee.logloss(p6[joint])), 5),
                "ll_roundbt": round(float(referee.logloss(p_bt[joint])), 5),
                "dll_milli_vs_v6": round(float(np.mean(d)) * 1000, 3),
                "boot_iid": bt_iid, "boot_block": bt_blk,
                "expected_roi": roi,
                "mde_quote": {"within_milli": mde_within,
                              "cross_milli": mde_cross,
                              "applicable": "cross-family (different "
                              "likelihood family): 5.889"}},
    "bias": {"v6": {"max_abs": bias6["max_abs_bias"],
                    "mean_abs": bias6["mean_abs_bias"]},
             "roundbt": {"max_abs": bias_bt["max_abs_bias"],
                         "mean_abs": bias_bt["mean_abs_bias"]},
             "elite_five_v6": team_rows(bias6, ELITE),
             "elite_five_roundbt": team_rows(bias_bt, ELITE),
             "weak_quartet_v6": team_rows(bias6, WEAK),
             "weak_quartet_roundbt": team_rows(bias_bt, WEAK)},
    "bias_tables": {"v6": bias6, "roundbt": bias_bt},
    "buckets": buckets,
    "effective_sample": {
        "ref_day": REF_DAY,
        "k_eff_fisher_median": round(k_fisher, 2),
        "k_eff_fisher_teams": [{"team": t, "ratio": round(r, 2)}
                               for t, r in sorted(ratios, key=lambda x: -x[1])],
        "k_eff_cluster_boot_median": round(k_boot, 2),
        "boot_B": B, "boot_rng": "default_rng([20260728, 774411]) — derived "
                                 "from crn bootstrap seed (preregistered)",
        "per_map_info_ratio_mean": round(info_ratio_mean, 2),
        "per_map_info_ratio_median": round(info_ratio_med, 2),
        "order_of_magnitude_claim": ("FALSIFIED (k_eff < 7)"
                                     if max(k_fisher, k_boot) < 7 else
                                     "SUPPORTED"),
        "round_referee_mde_first_order": {
            "within_milli": round(mde_within / np.sqrt(max(k_boot, 1e-9)), 2),
            "cross_milli": round(mde_cross / np.sqrt(max(k_boot, 1e-9)), 2),
            "caveat": "first-order 1/sqrt(k) scaling of the series MDE using "
                      "the cluster-bootstrap k; fit-side information gain, "
                      "assumed to transfer to evaluation variance; covered "
                      "subset only (83.7% of holdout series)"}},
}
with open(os.path.join(STATS, "h1_roundbt.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
log("E2 written stats/h1_roundbt.json")

# ── paired caterpillar (v6 vs tobit vs roundbt), chart-ready ───────────────
tob = json.load(open(os.path.join(STATS, "h1_tobit.json")))
tb = {r["team"]: r for r in tob["primary_s1.0"]["bias"]["teams"]}
v6b = {r["team"]: r for r in bias6["teams"]}
btb = {r["team"]: r for r in bias_bt["teams"]}
teams_all = sorted(set(v6b) | set(tb) | set(btb),
                   key=lambda t: v6b.get(t, {"bias": 0})["bias"])
cat = {"written_by": "agent:bias_h1",
       "frame": "frame_expanded holdout n=1217, min_n=25, prob-pts x100 in chart",
       "note": "bias = mean predicted P(win) - actual win rate; negative = "
               "under-rated. v6 = expanded-frame baseline caterpillar.",
       "teams": [{"team": t,
                  "n": v6b.get(t, btb.get(t, tb.get(t, {}))).get("n"),
                  "v6": v6b[t]["bias"] if t in v6b else None,
                  "tobit": tb[t]["bias"] if t in tb else None,
                  "roundbt": btb[t]["bias"] if t in btb else None}
                 for t in teams_all],
       "summary": {"v6": {"max_abs": bias6["max_abs_bias"],
                          "mean_abs": bias6["mean_abs_bias"]},
                   "tobit": {"max_abs": tob["primary_s1.0"]["bias"]["max_abs_bias"],
                             "mean_abs": tob["primary_s1.0"]["bias"]["mean_abs_bias"]},
                   "roundbt": {"max_abs": bias_bt["max_abs_bias"],
                               "mean_abs": bias_bt["mean_abs_bias"]}}}
with open(os.path.join(STATS, "h1_bias_caterpillar.json"), "w") as f:
    json.dump(cat, f, indent=1, default=float)
log("caterpillar written stats/h1_bias_caterpillar.json")
