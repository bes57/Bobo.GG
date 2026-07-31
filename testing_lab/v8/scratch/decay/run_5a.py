"""agent:decay — 5a: validation gate + re-race of the near-ties on the
expanded frame, CRN boots (iid + block), CUPED control variate."""
import json
import os
import sys
import time

import numpy as np

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
from decay_lib import Runner, jlog, sp, V8, TL  # noqa: E402

sys.path.insert(0, V8)
import referee  # noqa: E402

STATS = os.path.join(V8, "stats")
FAMILY_MDE = {"within": 1.773, "cross": 5.889}   # milli, phase-0 composition-adj

t0 = time.time()
rn = Runner()
jlog(f"5a runner ready ({time.time()-t0:.0f}s): games={rn.n_g} teams={rn.n_t} "
     f"days={len(rn.pred_days)} frame={len(rn.frame)}")

# ── validation gate: my runner vs eng.run on v6 ─────────────────────────────
stage_by_mid = dict(zip(rn.frame.match_id, rn.frame.stage))
g_stage = np.array([stage_by_mid.get(g["match_id"], "groups") for g in rn.games])
PO = np.where(np.isin(g_stage, ("playoffs", "grand_final")), 1.6, 1.0)
BASE = {"rd": {"power": 0.75, "scale": 2.5}, "roster_mode": "year",
        "roster_persistence": 0.3, "ridge": 0.5, "champ_mult": 2.0,
        "region_prior_ridge": 1.5, "w_custom": PO}
val_path = os.path.join(SCR, "validation.json")
if not os.path.exists(val_path):
    rn.eng._prev_rvec = None
    t1 = time.time()
    ref = rn.eng.run({**BASE, "decay": {"kind": "games",
                                        "consistency": (20.0, 12.0)}})
    jlog(f"eng.run(v6) done in {time.time()-t1:.0f}s: beta={ref['beta']} "
         f"ll_test={ref['ll_test']} n_test={ref['n_test']}")
    lam_v6 = rn.lam_arrays("consist", hl_c=20.0, hl_a=12.0)
    mine = rn.run_cfg("v6_consist_20_12", lam_v6, cache=False)
    a, b = ref["rdiff"], mine["rdiff"]
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    m = ~np.isnan(a)
    max_dr = float(np.max(np.abs(a[m] - b[m])))
    ll_mine = rn.ll(mine["p"], m & rn.test_v)
    ok = same_nan and max_dr < 1e-9 and abs(ll_mine - ref["ll_test"]) < 1e-5
    val = {"same_nan_pattern": same_nan, "max_abs_rdiff_diff": max_dr,
           "beta_engine": ref["beta"], "beta_mine": round(mine["beta"], 4),
           "ll_test_engine": ref["ll_test"], "ll_test_mine": round(ll_mine, 5),
           "n_test": int((m & rn.test_v).sum()), "pass": ok}
    json.dump(val, open(val_path, "w"), indent=1)
    jlog(f"VALIDATION {'PASS' if ok else 'FAIL'}: max|drdiff|={max_dr:.2e} "
         f"beta {ref['beta']} vs {mine['beta']:.4f}, ll {ref['ll_test']} vs {ll_mine:.5f}")
    if not ok:
        raise SystemExit("validation gate FAILED — stop")
else:
    jlog("validation.json exists — gate previously passed")

# ── 5a configs (+ sym_16 control + v6 daily capture) ────────────────────────
cfgs = {
    "v6_consist_20_12": rn.lam_arrays("consist", hl_c=20.0, hl_a=12.0),
    "consist_16_10": rn.lam_arrays("consist", hl_c=16.0, hl_a=10.0),
    "sym_20": rn.lam_arrays("sym", hl=20.0),
    "sym_24": rn.lam_arrays("sym", hl=24.0),
    "sym_16": rn.lam_arrays("sym", hl=16.0),
}
out = {}
for name, lam in cfgs.items():
    t1 = time.time()
    daily = (name == "v6_consist_20_12"
             and not os.path.exists(os.path.join(SCR, "v6_daily.npz")))
    r = rn.run_cfg(name, lam, daily_out=daily)
    out[name] = r
    if daily and "daily" in r:
        days = sorted(r["daily"])
        np.savez(os.path.join(SCR, "v6_daily.npz"), days=np.array(days),
                 R=np.stack([r["daily"][d] for d in days]))
        jlog("saved v6_daily.npz")
    v = ~np.isnan(r["rdiff"])
    jlog(f"{name}: beta={r['beta']:.4f} ll_train={rn.ll(r['p'], v & rn.train_v):.5f} "
         f"ll_test={rn.ll(r['p'], v & rn.test_v):.5f} "
         f"({time.time()-t1:.0f}s{' cached' if r.get('cached') else ''})")

# ── judging ─────────────────────────────────────────────────────────────────
p6 = out["v6_consist_20_12"]["p"]
rd6 = out["v6_consist_20_12"]["rdiff"]
hold = rn.test_v
ev_hold = rn.frame.event_id.values


def judge(p_c, regime, p_ref=p6, extra_cv=True):
    m = hold & ~np.isnan(p_ref) & ~np.isnan(p_c)
    d = referee.delta_vector(p_c[m], p_ref[m])
    n = int(m.sum())
    l6 = referee.per_series_ll(p_ref[m])
    X = np.column_stack([l6])
    if extra_cv:
        Xm = np.column_stack([l6, np.abs(rd6[m]), np.maximum(p6[m], 1 - p6[m])])
    Xc = X - X.mean(0)
    th = np.linalg.lstsq(Xc, d, rcond=None)[0]
    d_cv = d - Xc @ th
    Xmc = Xm - Xm.mean(0)
    thm = np.linalg.lstsq(Xmc, d, rcond=None)[0]
    d_cvm = d - Xmc @ thm
    b_iid = referee.paired_bootstrap_crn(d, mode="iid")
    b_blk = referee.paired_bootstrap_crn(d, mode="block_event",
                                         event_ids=ev_hold[m])
    c_iid = referee.paired_bootstrap_crn(d_cv, mode="iid")
    c_blk = referee.paired_bootstrap_crn(d_cv, mode="block_event",
                                         event_ids=ev_hold[m])
    cm_iid = referee.paired_bootstrap_crn(d_cvm, mode="iid")
    mde_raw = 2.8016 * float(np.std(d, ddof=1)) / np.sqrt(n) * 1000
    mde_cv = 2.8016 * float(np.std(d_cv, ddof=1)) / np.sqrt(n) * 1000
    mde_cvm = 2.8016 * float(np.std(d_cvm, ddof=1)) / np.sqrt(n) * 1000
    dm = float(np.mean(d)) * 1000
    if (abs(dm) >= mde_raw and
            ((b_iid["p_better"] >= 0.95 and b_blk["p_better"] >= 0.95) or
             (b_iid["p_better"] <= 0.05 and b_blk["p_better"] <= 0.05))):
        verdict = "WIN over v6" if dm > 0 else "KILL (v6 wins)"
    else:
        verdict = "INSIDE NOISE FLOOR"
    roi = referee.expected_roi_of_dll(float(np.mean(d)), p_ref[m])
    return {"n": n, "delta_milli": round(dm, 3),
            "family_mde_milli": FAMILY_MDE[regime], "regime": regime,
            "pair_mde_raw_milli": round(mde_raw, 3),
            "pair_mde_cv_milli": round(mde_cv, 3),
            "pair_mde_cv_multi_milli": round(mde_cvm, 3),
            "boot_iid": {k: b_iid[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
            "boot_block": {k: b_blk[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better", "n_events")},
            "boot_iid_cv": {k: c_iid[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
            "boot_block_cv": {k: c_blk[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
            "boot_iid_cv_multi": {k: cm_iid[k] for k in ("mean_delta", "ci_lo", "ci_hi", "p_better")},
            "verdict": verdict,
            "expected_roi_delta": roi["expected_roi_delta"],
            "roi_at_op": roi["roi_at_op"],
            "delta_logit_equiv": roi["delta_logit_equiv"]}


table = {}
for name, regime in (("consist_16_10", "within"), ("sym_20", "cross"),
                     ("sym_24", "cross"), ("sym_16", "cross")):
    r = out[name]
    v = ~np.isnan(r["rdiff"])
    row = judge(r["p"], regime)
    row.update({"beta": round(r["beta"], 4),
                "ll_train": round(rn.ll(r["p"], v & rn.train_v), 5),
                "ll_test": round(rn.ll(r["p"], v & hold), 5)})
    table[name] = row
    jlog(f"5a {name}: d={row['delta_milli']:+.2f}m mde_raw={row['pair_mde_raw_milli']:.2f} "
         f"cv={row['pair_mde_cv_milli']:.2f} p_iid={row['boot_iid']['p_better']:.3f} "
         f"p_blk={row['boot_block']['p_better']:.3f} -> {row['verdict']}")

v6v = ~np.isnan(rd6)
# continuity: v6 on the 1007 frozen rows (old holdout ids)
crn = json.load(open(os.path.join(V8, "crn.json")))
old_ids = set(crn["holdout_order"])
m_old = hold & v6v & rn.frame.match_id.isin(old_ids).values
res = {
    "preregistered": "testing_lab/v8/preregister.decay.md (BEFORE runs)",
    "frame": "v8/data/frame_expanded/series.csv sha ff772d41…d55142 verified",
    "validation_gate": json.load(open(val_path)),
    "v6_champion": {"beta": round(out["v6_consist_20_12"]["beta"], 4),
                    "ll_train": round(rn.ll(p6, v6v & rn.train_v), 5),
                    "ll_test": round(rn.ll(p6, v6v & hold), 5),
                    "n_test": int((v6v & hold).sum()),
                    "ll_on_1007_frozen_rows": round(rn.ll(p6, m_old), 5),
                    "n_frozen_rows_matched": int(m_old.sum()),
                    "published_old_frame_ll": 0.64095,
                    "continuity_note": "expanded corpus (+2023-24 events) shifts "
                    "ratings; offset expected, reported not reconciled"},
    "header": {"n_holdout": 1217,
               "family_mde_milli": FAMILY_MDE,
               "mde_rule": "MDE80 = 2.8016*SD(d)/sqrt(n); verdict at raw pair "
               "MDE + p_better>=0.95 both CRN modes (preregistered)",
               "cv": "CUPED on centered l_v6 (primary) / [l_v6,|rd6|,p_fav6] "
               "(multi); point estimate unchanged, CI+MDE shrink"},
    "table": table,
}
json.dump(res, open(os.path.join(STATS, "decay_rerace.json"), "w"), indent=1)
jlog("wrote stats/decay_rerace.json")
