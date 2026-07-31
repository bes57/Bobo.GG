"""Step 1: v6 baseline on the expanded frame (daily_out) + E3 censoring
diagnostic. Writes scratch/bias_h1/baseline_v6.npz + stats/h1_censor_diag.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_lib import (C13, SCRATCH, STATS, TRAIN_END, asof_lookup, game_class,
                    load_frame, log, make_engine, t_of_m, v6_cfg)

frame = load_frame()
log("frame loaded + sha verified (n=2058/841/1217)")
eng = make_engine(frame)
log(f"engine: {len(eng.games)} games, {len(eng.teams)} teams, "
    f"{len(eng.pred_days)} pred days")

cfg = v6_cfg(eng, frame, daily_out=True)
t0 = time.time()
out = eng.run(cfg)
log(f"v6 baseline run: {time.time()-t0:.1f}s  beta={out['beta']} "
    f"ll_test={out['ll_test']} n_test={out['n_test']} n_train={out['n_train']}")

daily = out["daily_r"]
days = sorted(daily.keys())
R = np.array([daily[d] for d in days])
np.savez_compressed(
    os.path.join(SCRATCH, "baseline_v6.npz"),
    rdiff=out["rdiff"], rat_w=out["rat_w"], rat_l=out["rat_l"],
    p_test=out["p_test"], test_mask=out["test_mask"], beta=out["beta"],
    days=np.array(days), R=R, teams=np.array(eng.teams))
log("baseline_v6.npz saved (incl. daily ratings)")

# ── E3 diagnostic (train frame maps only) ───────────────────────────────────
frame_train_mids = set(frame[frame.date <= TRAIN_END].match_id)
lookup = asof_lookup(daily, eng.pred_days)
tidx = eng.tidx

rows = []           # (mu_w, mu_l, margin, lr, cls)
n_no_rating = 0
cls_counts = {}
for g in eng.games:
    if g["match_id"] not in frame_train_mids:
        continue
    c = game_class(g)
    cls_counts[c] = cls_counts.get(c, 0) + 1
    r = lookup(g["date_s"])
    if r is None:
        n_no_rating += 1
        continue
    rows.append((r[tidx[g["winner"]]], r[tidx[g["loser"]]],
                 g["wr"] - g["lr"], g["lr"], c))
rw = np.array([x[0] for x in rows])
rl = np.array([x[1] for x in rows])
marg = np.array([x[2] for x in rows], dtype=float)
lr_ = np.array([x[3] for x in rows])
cls = np.array([x[4] for x in rows])
keep = cls != "JUNK"
n_junk = int((~keep).sum())
rw, rl, marg, lr_, cls = rw[keep], rl[keep], marg[keep], lr_[keep], cls[keep]
log(f"E3 sample: {len(rw)} train maps with as-of ratings "
    f"(no-rating skipped {n_no_rating}, junk {n_junk}, classes {cls_counts})")

# 1. cap share by winner trailing-rating quartile
q = np.quantile(rw, [0.25, 0.5, 0.75])
quart = np.digitize(rw, q)          # 0..3
by_q = []
for k in range(4):
    m = quart == k
    d = {"quartile": f"Q{k+1}", "n": int(m.sum()),
         "r_w_mean": round(float(rw[m].mean()), 3),
         "cap_share": round(float((cls[m] == "CAP").mean()), 5),
         "nearcap_share_m11": round(float((marg[m] >= 11).mean()), 5),
         "ot_share": round(float((cls[m] == "OT").mean()), 5),
         "loser_rounds_mean": round(float(lr_[m].mean()), 3)}
    d["loser_rounds_dist"] = {str(v): int((lr_[m] == v).sum())
                              for v in range(0, 13)}
    by_q.append(d)
mid = (quart == 1) | (quart == 2)
cap_q4 = float((cls[quart == 3] == "CAP").mean())
cap_mid = float((cls[mid] == "CAP").mean())
ratio = cap_q4 / max(cap_mid, 1e-9)
near_q4 = float((marg[quart == 3] >= 11).mean())
near_mid = float((marg[mid] >= 11).mean())

# 2. residuals vs predicted margin
y = t_of_m(marg)                      # winner-referenced realized target
x = rw - rl
a = float((x @ y) / (x @ x))          # OLS through origin
yhat = a * x
resid = y - yhat
dec_edges = np.quantile(yhat, np.linspace(0, 1, 11))
dec = np.clip(np.digitize(yhat, dec_edges[1:-1]), 0, 9)
dec_rows = []
for k in range(10):
    m = dec == k
    rmean = float(resid[m].mean())
    se = float(resid[m].std(ddof=1) / np.sqrt(m.sum()))
    dec_rows.append({"decile": k + 1, "n": int(m.sum()),
                     "yhat_mean": round(float(yhat[m].mean()), 3),
                     "y_mean": round(float(y[m].mean()), 3),
                     "resid_mean": round(rmean, 3),
                     "ci95": [round(rmean - 1.96 * se, 3),
                              round(rmean + 1.96 * se, 3)],
                     "cap_share": round(float((cls[m] == "CAP").mean()), 4)})
top = dec_rows[-1]
p1 = ratio >= 2.0
p2 = top["ci95"][1] < 0
gate = "RUN_E1_AND_E2" if (p1 or p2) else "SKIP_E1_PREMISE_WEAK"

diag = {
    "written_by": "agent:bias_h1", "preregistered": "preregister.bias_h1.md E3",
    "sample": {"n_train_maps": len(rw), "skipped_no_asof_rating": n_no_rating,
               "junk_margin_excluded": n_junk, "class_counts": cls_counts},
    "cap_share_by_winner_rating_quartile": by_q,
    "cap_ratio_Q4_over_mid": round(ratio, 3),
    "cap_Q4": round(cap_q4, 5), "cap_mid_Q2Q3": round(cap_mid, 5),
    "nearcap_ratio_Q4_over_mid": round(near_q4 / max(near_mid, 1e-9), 3),
    "margin_link_slope_a_train": round(a, 5),
    "resid_by_predicted_decile": dec_rows,
    "top_decile_resid": top,
    "prong_P1_capratio_ge_2": bool(p1),
    "prong_P2_top_resid_neg": bool(p2),
    "gate_decision": gate,
    "cap_target_const": round(C13, 4),
}
with open(os.path.join(STATS, "h1_censor_diag.json"), "w") as f:
    json.dump(diag, f, indent=1)
log(f"E3 done: cap Q4 {cap_q4:.4f} vs mid {cap_mid:.4f} ratio {ratio:.2f} "
    f"(P1={p1}); top-decile resid {top['resid_mean']} CI {top['ci95']} "
    f"(P2={p2}) -> {gate}")
print(json.dumps({k: diag[k] for k in
                  ("cap_ratio_Q4_over_mid", "top_decile_resid",
                   "gate_decision")}, indent=1))
