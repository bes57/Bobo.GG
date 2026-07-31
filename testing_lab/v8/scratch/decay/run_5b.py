"""agent:decay — 5b: five new decay conditioning axes, train-grid select,
one holdout verdict per axis (preregister.decay.md)."""
import json
import os
import sys
import time

import numpy as np

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
from decay_lib import (Runner, jlog, build_lineup_tables,  # noqa: E402
                       rotation_dates, V8)

sys.path.insert(0, V8)
import referee  # noqa: E402

STATS = os.path.join(V8, "stats")
FAMILY_MDE = {"within": 1.773, "cross": 5.889}

t0 = time.time()
rn = Runner()
jlog(f"5b runner ready ({time.time()-t0:.0f}s)")

# baselines from 5a cache
v6 = rn.run_cfg("v6_consist_20_12", rn.lam_arrays("consist"))
p6, rd6 = v6["p"], v6["rdiff"]
sym = {hl: rn.run_cfg(f"sym_{hl}", rn.lam_arrays("sym", hl=float(hl)))
       for hl in (16, 20, 24)}
hold = rn.test_v
ev_hold = rn.frame.event_id.values
assert not v6.get("cached") is None


def ll_tr(r):
    v = ~np.isnan(r["rdiff"])
    return rn.ll(r["p"], v & rn.train_v)


def ll_te(r):
    v = ~np.isnan(r["rdiff"])
    return rn.ll(r["p"], v & rn.test_v)


def judge(p_c, regime, p_ref=p6, ref_rd=rd6):
    m = hold & ~np.isnan(p_ref) & ~np.isnan(p_c)
    d = referee.delta_vector(p_c[m], p_ref[m])
    n = int(m.sum())
    l6 = referee.per_series_ll(p_ref[m])
    Xc = (l6 - l6.mean()).reshape(-1, 1)
    th = np.linalg.lstsq(Xc, d, rcond=None)[0]
    d_cv = d - Xc @ th
    b_iid = referee.paired_bootstrap_crn(d, mode="iid")
    b_blk = referee.paired_bootstrap_crn(d, mode="block_event", event_ids=ev_hold[m])
    c_iid = referee.paired_bootstrap_crn(d_cv, mode="iid")
    mde_raw = 2.8016 * float(np.std(d, ddof=1)) / np.sqrt(n) * 1000
    mde_cv = 2.8016 * float(np.std(d_cv, ddof=1)) / np.sqrt(n) * 1000
    dm = float(np.mean(d)) * 1000
    if (abs(dm) >= mde_raw and
            ((b_iid["p_better"] >= 0.95 and b_blk["p_better"] >= 0.95) or
             (b_iid["p_better"] <= 0.05 and b_blk["p_better"] <= 0.05))):
        verdict = "WIN" if dm > 0 else "KILL"
    else:
        verdict = "INSIDE NOISE FLOOR"
    roi = referee.expected_roi_of_dll(float(np.mean(d)), p_ref[m])
    return {"n": n, "delta_milli": round(dm, 3), "regime": regime,
            "family_mde_milli": FAMILY_MDE[regime],
            "pair_mde_raw_milli": round(mde_raw, 3),
            "pair_mde_cv_milli": round(mde_cv, 3),
            "boot_iid": {k: b_iid[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
            "boot_block": {k: b_blk[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better", "n_events")},
            "boot_iid_cv": {k: c_iid[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
            "verdict": verdict, "expected_roi_delta": roi["expected_roi_delta"],
            "delta_logit_equiv": roi["delta_logit_equiv"]}


axes = {}

# ── axis a: lineup continuity (symmetric) ───────────────────────────────────
jlog("axis a: building lineup tables")
org_matches, lineup_ov = build_lineup_tables(rn)
n_lu = sum(1 for org, seq in org_matches.items() for t in seq if t[2])
jlog(f"axis a: lineup sets on {n_lu} org-matches "
     f"({sum(len(s) for s in org_matches.values())} total)")
grid_a = {}
for hl in (16, 20, 24):
    lam = rn.lam_arrays("sym", hl=float(hl))
    for gm in (0.5, 1.0, 2.0, 4.0):
        name = f"lineup_h{hl}_g{gm}"
        r = rn.run_cfg(name, lam, year_cont=False, lineup_ov=lineup_ov,
                       lineup_gamma=gm)
        grid_a[name] = {"hl": hl, "gamma": gm, "beta": round(r["beta"], 4),
                        "ll_train": round(ll_tr(r), 5), "ll_test": round(ll_te(r), 5)}
        jlog(f"  {name}: train={grid_a[name]['ll_train']} test={grid_a[name]['ll_test']}")
# exact gamma=0 controls (sym, no year continuity)
ctrl_nc = {}
for hl in (16, 20, 24):
    r = rn.run_cfg(f"sym_{hl}_nc", rn.lam_arrays("sym", hl=float(hl)),
                   year_cont=False)
    ctrl_nc[hl] = r
    grid_a[f"sym_{hl}_nc"] = {"hl": hl, "gamma": 0.0, "beta": round(r["beta"], 4),
                              "ll_train": round(ll_tr(r), 5),
                              "ll_test": round(ll_te(r), 5)}
    jlog(f"  sym_{hl}_nc: train={grid_a[f'sym_{hl}_nc']['ll_train']} "
         f"test={grid_a[f'sym_{hl}_nc']['ll_test']}")
best_a = min((k for k in grid_a if k.startswith("lineup")),
             key=lambda k: grid_a[k]["ll_train"])
ra = rn.run_cfg(best_a, None)  # cached
hl_a, gm_a = grid_a[best_a]["hl"], grid_a[best_a]["gamma"]
axes["a_lineup_continuity"] = {
    "outcome_symmetric": True,
    "definition": "w_side = exp(-ln2/HL*games_ago)*max(|Lcur∩Lthen|/max(|Lcur|,|Lthen|,5),0.04)^gamma; "
                  "year continuity replaced by lineup continuity",
    "grid": grid_a, "selected": best_a,
    "selected_params": {"hl": hl_a, "gamma": gm_a},
    "vs_v6": judge(ra["p"], "cross"),
    "vs_own_sym_control_nc": judge(ra["p"], "within", p_ref=ctrl_nc[hl_a]["p"],
                                   ref_rd=ctrl_nc[hl_a]["rdiff"]),
    "vs_own_sym_control_yearcont": judge(ra["p"], "within", p_ref=sym[hl_a]["p"],
                                         ref_rd=sym[hl_a]["rdiff"]),
}
jlog(f"axis a selected {best_a}: vs v6 {axes['a_lineup_continuity']['vs_v6']['delta_milli']:+.2f}m "
     f"({axes['a_lineup_continuity']['vs_v6']['verdict']})")

# ── axis b: opponent quality of the anomaly ─────────────────────────────────
z = np.load(os.path.join(SCR, "v6_daily.npz"))
days, R = list(z["days"]), z["R"]
day_dnum = np.array([int(np.datetime64(d, "D").astype(int)) for d in days])
# active mask per day: team played >=1 game strictly before day
n_played_by_day = np.zeros((len(days), rn.n_t), dtype=bool)
for org in rn.orgs:
    ti = rn.tidx[org]
    od = rn.org_dnum[org]
    n_played_by_day[:, ti] = np.searchsorted(od, day_dnum, side="left") > 0
opp_class = {}  # (org, ri) -> -1/0/+1 for org's ANOMALOUS games
n_cls = {-1: 0, 0: 0, 1: 0}
for org, rows in rn.org_rows.items():
    for ri in rows:
        if rn.consist[(org, ri)]:
            continue
        g = rn.games[ri]
        opp = g["loser"] if g["winner"] == org else g["winner"]
        di = int(np.searchsorted(day_dnum, rn.g_dnum[ri], side="right")) - 1
        c = 0
        if di >= 0:
            act = n_played_by_day[di]
            if act.sum() >= 8 and act[rn.tidx[opp]]:
                rv = R[di][act]
                q25, q75 = np.percentile(rv, [25, 75])
                ro = R[di][rn.tidx[opp]]
                c = 1 if ro >= q75 else (-1 if ro <= q25 else 0)
        opp_class[(org, ri)] = c
        n_cls[c] += 1
jlog(f"axis b: anomalous sides elite/mid/floor = {n_cls[1]}/{n_cls[0]}/{n_cls[-1]}")
grid_b = {}
for m_ in (1.33, 1.67, 2.0):
    ov = {k: (12.0 * m_ if c > 0 else (12.0 / m_ if c < 0 else 12.0))
          for k, c in opp_class.items()}
    name = f"oppq_m{m_}"
    r = rn.run_cfg(name, rn.lam_arrays("consist", hl_anom_override=ov))
    grid_b[name] = {"m": m_, "beta": round(r["beta"], 4),
                    "ll_train": round(ll_tr(r), 5), "ll_test": round(ll_te(r), 5)}
    jlog(f"  {name}: train={grid_b[name]['ll_train']} test={grid_b[name]['ll_test']}")
best_b = min(grid_b, key=lambda k: grid_b[k]["ll_train"])
rb = rn.run_cfg(best_b, None)
axes["b_opponent_quality_of_anomaly"] = {
    "outcome_symmetric": False,
    "definition": "v6 consist(20,12); anomalous HL 12*m vs elite (top-quartile "
                  "trailing v6 rating that day), 12/m vs floor (bottom quartile), 12 mid; "
                  "walk-forward daily v6 solve, latest pred-day <= game date",
    "n_anom_sides": n_cls, "grid": grid_b, "selected": best_b,
    "vs_v6": judge(rb["p"], "within"),
}
jlog(f"axis b selected {best_b}: {axes['b_opponent_quality_of_anomaly']['vs_v6']['delta_milli']:+.2f}m "
     f"({axes['b_opponent_quality_of_anomaly']['vs_v6']['verdict']})")

# ── axis c: anomaly margin ──────────────────────────────────────────────────
grid_c = {}
for k_ in (0.25, 0.5, 1.0):
    ov = {}
    for org, rows in rn.org_rows.items():
        for ri in rows:
            if not rn.consist[(org, ri)]:
                ov[(org, ri)] = 12.0 * float(np.clip(
                    (abs(rn.rd_raw[ri]) / 5.0) ** k_, 0.5, 2.0))
    name = f"amargin_k{k_}"
    r = rn.run_cfg(name, rn.lam_arrays("consist", hl_anom_override=ov))
    grid_c[name] = {"k": k_, "beta": round(r["beta"], 4),
                    "ll_train": round(ll_tr(r), 5), "ll_test": round(ll_te(r), 5)}
    jlog(f"  {name}: train={grid_c[name]['ll_train']} test={grid_c[name]['ll_test']}")
best_c = min(grid_c, key=lambda k: grid_c[k]["ll_train"])
rc = rn.run_cfg(best_c, None)
axes["c_anomaly_margin"] = {
    "outcome_symmetric": False,
    "definition": "v6 consist(20,12); anomalous HL = 12*clip((|rd|/5)^k, .5, 2)",
    "grid": grid_c, "selected": best_c, "vs_v6": judge(rc["p"], "within"),
}
jlog(f"axis c selected {best_c}: {axes['c_anomaly_margin']['vs_v6']['delta_milli']:+.2f}m "
     f"({axes['c_anomaly_margin']['vs_v6']['verdict']})")

# ── axis d: event class of the result ───────────────────────────────────────
pmx = json.load(open(os.path.join(STATS, "power_mde_expanded.json")))
ewc_set = {e for e, c in pmx["new_events"].items() if c == "ewc_offseason"}
ewc_pref = ("2026_ewc", "2026_china_evo")
g_ewc = np.array([g["event_id"] in ewc_set or g["event_id"].startswith(ewc_pref)
                  for g in rn.games])
jlog(f"axis d: ewc_offseason games = {int(g_ewc.sum())}/{rn.n_g}")
grid_d = {}
for hl in (16, 20, 24):
    for me in (0.4, 0.6, 0.8):
        cm = np.where(g_ewc, me, 1.0)
        name = f"eclass_h{hl}_m{me}"
        r = rn.run_cfg(name, rn.lam_arrays("sym", hl=float(hl), class_mult=cm))
        grid_d[name] = {"hl": hl, "m_ewc": me, "beta": round(r["beta"], 4),
                        "ll_train": round(ll_tr(r), 5), "ll_test": round(ll_te(r), 5)}
        jlog(f"  {name}: train={grid_d[name]['ll_train']} test={grid_d[name]['ll_test']}")
best_d = min(grid_d, key=lambda k: grid_d[k]["ll_train"])
rd_ = rn.run_cfg(best_d, None)
hl_d, me_d = grid_d[best_d]["hl"], grid_d[best_d]["m_ewc"]
cm_d = np.where(g_ewc, me_d, 1.0)
rd_v6 = rn.run_cfg(f"eclass_on_v6_m{me_d}",
                   rn.lam_arrays("consist", class_mult=cm_d))
axes["d_event_class"] = {
    "outcome_symmetric": True,
    "definition": "HL_eff = HL*m_ewc for ewc_offseason-class games (set = "
                  "2026_ewc*/2026_china_evo* + power new_events ewc_offseason); sym base",
    "ewc_event_set": sorted(ewc_set) + ["2026_ewc*", "2026_china_evo* (prefixes)"],
    "n_ewc_games": int(g_ewc.sum()), "grid": grid_d, "selected": best_d,
    "vs_v6": judge(rd_["p"], "cross"),
    "vs_own_sym_control": judge(rd_["p"], "within", p_ref=sym[hl_d]["p"],
                                ref_rd=sym[hl_d]["rdiff"]),
    "on_top_of_v6": {"name": f"eclass_on_v6_m{me_d}",
                     "beta": round(rd_v6["beta"], 4),
                     "ll_train": round(ll_tr(rd_v6), 5),
                     "ll_test": round(ll_te(rd_v6), 5),
                     "vs_v6": judge(rd_v6["p"], "within"),
                     "outcome_symmetric_of_addon": True},
}
jlog(f"axis d selected {best_d}: vs v6 {axes['d_event_class']['vs_v6']['delta_milli']:+.2f}m "
     f"({axes['d_event_class']['vs_v6']['verdict']}); on-v6 "
     f"{axes['d_event_class']['on_top_of_v6']['vs_v6']['delta_milli']:+.2f}m")

# ── axis e: patch / map-pool boundary ───────────────────────────────────────
rots, clusters = rotation_dates(rn)
rot_dnum = np.array([int(np.datetime64(r, "D").astype(int)) for r in rots])
jlog(f"axis e: {len(rots)} rotation dates: {rots}")
grid_e = {}
for hl in (16, 20, 24):
    for gp in (0.85, 0.7, 0.55):
        name = f"rot_h{hl}_g{gp}"
        r = rn.run_cfg(name, rn.lam_arrays("sym", hl=float(hl)),
                       rot_dates=rot_dnum, rot_gamma=gp)
        grid_e[name] = {"hl": hl, "gamma_p": gp, "beta": round(r["beta"], 4),
                        "ll_train": round(ll_tr(r), 5), "ll_test": round(ll_te(r), 5)}
        jlog(f"  {name}: train={grid_e[name]['ll_train']} test={grid_e[name]['ll_test']}")
best_e = min(grid_e, key=lambda k: grid_e[k]["ll_train"])
re_ = rn.run_cfg(best_e, None)
hl_e, gp_e = grid_e[best_e]["hl"], grid_e[best_e]["gamma_p"]
re_v6 = rn.run_cfg(f"rot_on_v6_g{gp_e}", rn.lam_arrays("consist"),
                   rot_dates=rot_dnum, rot_gamma=gp_e)
axes["e_patch_boundary"] = {
    "outcome_symmetric": True,
    "definition": "weight *= gamma_p^(# derived pool rotations in (game, day]); "
                  "sym base; rotations from map appearance windows (>=60d gaps, "
                  "14d clustering, preregistered)",
    "rotation_dates": rots, "rotation_clusters": clusters,
    "grid": grid_e, "selected": best_e,
    "vs_v6": judge(re_["p"], "cross"),
    "vs_own_sym_control": judge(re_["p"], "within", p_ref=sym[hl_e]["p"],
                                ref_rd=sym[hl_e]["rdiff"]),
    "on_top_of_v6": {"name": f"rot_on_v6_g{gp_e}",
                     "beta": round(re_v6["beta"], 4),
                     "ll_train": round(ll_tr(re_v6), 5),
                     "ll_test": round(ll_te(re_v6), 5),
                     "vs_v6": judge(re_v6["p"], "within"),
                     "outcome_symmetric_of_addon": True},
}
jlog(f"axis e selected {best_e}: vs v6 {axes['e_patch_boundary']['vs_v6']['delta_milli']:+.2f}m "
     f"({axes['e_patch_boundary']['vs_v6']['verdict']}); on-v6 "
     f"{axes['e_patch_boundary']['on_top_of_v6']['vs_v6']['delta_milli']:+.2f}m")

res = {"preregistered": "testing_lab/v8/preregister.decay.md",
       "selection_rule": "argmin ll_train (walk-forward preds inside train, "
                         "beta refit per config); holdout verdict only for the selected config",
       "v6_ll_test": round(rn.ll(p6, ~np.isnan(rd6) & hold), 5),
       "axes": axes}
json.dump(res, open(os.path.join(STATS, "decay_axes.json"), "w"), indent=1)
jlog("wrote stats/decay_axes.json")
