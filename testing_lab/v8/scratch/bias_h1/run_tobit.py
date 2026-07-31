"""E1: Tobit/censored-margin EM via iteratively-reweighted rd_custom.
Prereg gate said SKIP (premise weak); executed anyway as a labeled
confirmatory kill (see log 22:0x deviation note). Writes stats/h1_tobit.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_lib import (SCRATCH, STATS, TRAIN_END, asof_lookup, game_class,
                    impute_targets, load_frame, log, make_engine, t_of_m,
                    v6_cfg)

sys.path.insert(0, "/Users/benny_es1/PythonTest/testing_lab/v8")
import referee

frame = load_frame()
eng = make_engine(frame)
cfg0 = v6_cfg(eng, frame, daily_out=True)
base = np.load(os.path.join(SCRATCH, "baseline_v6.npz"), allow_pickle=True)
teams = list(base["teams"])
assert teams == eng.teams

base_target = t_of_m(np.array([g["wr"] - g["lr"] for g in eng.games],
                              dtype=float))
cls = np.array([game_class(g) for g in eng.games])
gm_train = np.array([g["date_s"] <= TRAIN_END for g in eng.games])
frame_mids = set(frame.match_id)
in_frame = np.array([g["match_id"] in frame_mids for g in eng.games])

def daily_from(npz_days, npz_R):
    return {d: npz_R[i] for i, d in enumerate(npz_days)}

def game_mu(daily):
    lu = asof_lookup(daily, eng.pred_days)
    mu = np.full(len(eng.games), np.nan)
    for i, g in enumerate(eng.games):
        r = lu(g["date_s"])
        if r is not None:
            mu[i] = r[eng.tidx[g["winner"]]] - r[eng.tidx[g["loser"]]]
    return mu

# σ from v6 baseline: train REG frame maps, resid of signed target vs mu
daily0 = daily_from(base["days"], base["R"])
mu0 = game_mu(daily0)
m_sig = gm_train & in_frame & (cls == "REG") & ~np.isnan(mu0)
sigma_hat = float(np.std(base_target[m_sig] - mu0[m_sig], ddof=1))
log(f"E1 sigma_hat = {sigma_hat:.4f} (train REG maps n={int(m_sig.sum())})")

def run_em(sigma, tag):
    mu = mu0.copy()
    hist = []
    prev_t = base_target.copy()
    out = None
    for k in range(1, 5):
        tgt = impute_targets(eng.games, mu, sigma, base_target)
        dmax = float(np.max(np.abs(tgt - prev_t)))
        prev_t = tgt
        t0 = time.time()
        out = eng.run({**cfg0, "rd_custom": tgt})
        mu = game_mu(out["daily_r"])
        hist.append({"iter": k, "max_abs_dtarget": round(dmax, 4),
                     "beta": out["beta"], "ll_test": out["ll_test"],
                     "secs": round(time.time() - t0, 1)})
        log(f"E1[{tag}] iter {k}: max|dT|={dmax:.4f} beta={out['beta']} "
            f"ll_test={out['ll_test']}")
        if k > 1 and dmax < 0.05:
            break
    return out, hist, tgt

out_c, hist_c, tgt_c = run_em(sigma_hat, "s1.0")
out_lo, hist_lo, _ = run_em(sigma_hat * 0.8, "s0.8")
out_hi, hist_hi, _ = run_em(sigma_hat * 1.25, "s1.25")

# ── judging vs v6 on the expanded holdout ──────────────────────────────────
s = frame
fmts = s.fmt.values
holdout = (s.date > TRAIN_END).values
p6 = np.full(len(s), np.nan); p6[base["test_mask"]] = base["p_test"]

def judge(out, tag):
    pc = np.full(len(s), np.nan); pc[out["test_mask"]] = out["p_test"]
    joint = ~np.isnan(p6) & ~np.isnan(pc) & holdout
    d = referee.per_series_ll(p6[joint]) - referee.per_series_ll(pc[joint])
    bt_iid = referee.paired_bootstrap_crn(d, mode="iid")
    bt_blk = referee.paired_bootstrap_crn(d, mode="block_event",
                                          event_ids=s.event_id.values[joint])
    roi = referee.expected_roi_of_dll(float(np.mean(d)), p6[joint])
    bias = referee.per_team_bias(pc, s.winner.values, s.loser.values,
                                 holdout=holdout, valid=~np.isnan(pc))
    log(f"E1[{tag}] dLL={np.mean(d)*1000:+.3f}m iid CI "
        f"[{bt_iid['ci_lo']*1000:+.2f},{bt_iid['ci_hi']*1000:+.2f}] "
        f"maxbias={bias['max_abs_bias']}")
    return {"n_joint": int(joint.sum()), "beta": out["beta"],
            "ll_test": out["ll_test"],
            "dll_milli_vs_v6": round(float(np.mean(d)) * 1000, 3),
            "boot_iid": bt_iid, "boot_block": bt_blk,
            "expected_roi": roi, "bias": bias}

bias6 = referee.per_team_bias(p6, s.winner.values, s.loser.values,
                              holdout=holdout, valid=~np.isnan(p6))
res_c = judge(out_c, "s1.0")
res_lo = judge(out_lo, "s0.8")
res_hi = judge(out_hi, "s1.25")

# buckets for the primary candidate
pc = np.full(len(s), np.nan); pc[out_c["test_mask"]] = out_c["p_test"]
buckets = referee.bucketed(s, pc, p_ref=p6, rdiff=out_c["rdiff"],
                           holdout=holdout, valid=~np.isnan(pc),
                           games=eng.games)

ELITE = ["T1", "PRX", "100T", "NRG", "TL"]
WEAK = ["TS", "JDG", "TE", "C9"]
def team_rows(b, names):
    t = {r["team"]: r for r in b["teams"]}
    return {nm: t.get(nm) for nm in names}

n_cap = int((cls == "CAP").sum()); n_ot = int((cls == "OT").sum())
res = {
    "written_by": "agent:bias_h1",
    "preregistered": "preregister.bias_h1.md E1 (gate said SKIP; run as labeled confirmatory kill — deviation documented in log + outcomes)",
    "mechanism": "CAP right-censored at 17.113; OT interval-censored (0,4.204]; EM K<=4, sigma train-only",
    "sigma_hat": round(sigma_hat, 4),
    "censoring_mass": {"n_games": len(eng.games), "CAP": n_cap, "OT": n_ot,
                       "CAP_share": round(n_cap / len(eng.games), 5),
                       "OT_share": round(n_ot / len(eng.games), 5)},
    "em_history": {"s1.0": hist_c, "s0.8": hist_lo, "s1.25": hist_hi},
    "v6_ref": {"beta": float(base["beta"]), "ll_test": 0.64216,
               "bias": bias6},
    "primary_s1.0": res_c,
    "sensitivity": {"s0.8": {k: res_lo[k] for k in
                             ("dll_milli_vs_v6", "ll_test", "beta")},
                    "s1.25": {k: res_hi[k] for k in
                              ("dll_milli_vs_v6", "ll_test", "beta")}},
    "elite_five_bias_v6": team_rows(bias6, ELITE),
    "elite_five_bias_tobit": team_rows(res_c["bias"], ELITE),
    "weak_quartet_bias_v6": team_rows(bias6, WEAK),
    "weak_quartet_bias_tobit": team_rows(res_c["bias"], WEAK),
    "buckets": buckets,
    "mde_quote": {"within_family_milli": 1.773, "cross_family_milli": 5.889,
                  "source": "stats/power_mde_expanded.json"},
}
with open(os.path.join(STATS, "h1_tobit.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
np.savez_compressed(os.path.join(SCRATCH, "tobit_primary.npz"),
                    rdiff=out_c["rdiff"], p_test=out_c["p_test"],
                    test_mask=out_c["test_mask"], beta=out_c["beta"])
log("E1 written stats/h1_tobit.json")
