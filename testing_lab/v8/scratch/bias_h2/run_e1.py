"""E1 connectivity diagnostic (preregister.bias_h2.md). Gates E2/E3.

1. v6 baseline on the expanded frame (the one baseline holdout scoring, shared
   by all H2 experiments) -> per-team bias (referee.per_team_bias, min_n=25).
2. Walk-forward graph features per team at each holdout month boundary,
   aggregated with the team's holdout-series month weights.
3. Pearson/Spearman correlations (signed bias and |bias| vs each feature) with
   4000 team-resample CIs (PCG64, crn bootstrap seed 20260728, one matrix).
4. Partial Spearman bias~centrality | mean v6 rating (secondary, non-gating).
5. Corpus-expansion centrality shift at 2026-07-01 for named teams.
Writes stats/h2_centrality.json + scratch npz for later phases.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ELITE, FLOOR, HERE, STATS, V8, build_engine, load_frame,
                    log, masks, probs_full, run_cfg, v6_cfg)

sys.path.insert(0, os.path.dirname(V8))
import referee  # noqa: E402

HL_W = 26.0          # calendar half-life (weeks) for graph edge decay
LAM = math.log(2) / HL_W

frame = load_frame()
log("[T1-e1] frame loaded n=%d; building engine (expanded corpus)" % len(frame))
eng, PO = build_engine(frame)
log("[T1-e1] engine: %d games, %d teams, %d pred days"
    % (len(eng.games), len(eng.teams), len(eng.pred_days)))

# ── v6 baseline (the single baseline holdout scoring for all of H2) ──────────
out = run_cfg(eng, {**v6_cfg(PO), "daily_out": True})
rdiff = out["rdiff"]
beta = out["beta"]
valid, train_m, hold_m = masks(frame, rdiff)
fmts = frame.fmt.values
p_all = probs_full(rdiff, fmts, beta)
log("[T1-e1] v6 baseline: beta=%.4f ll_train=%.5f ll_test=%.5f n_test=%d"
    % (beta, out["ll_train"], out["ll_test"], int(hold_m.sum())))

bias_tbl = referee.per_team_bias(p_all, frame.winner.values, frame.loser.values,
                                 holdout=hold_m, valid=valid, min_n=25)
log("[T1-e1] per_team_bias: %d teams (min_n=25), max|bias|=%.4f mean|bias|=%.4f"
    % (bias_tbl["n_teams"], bias_tbl["max_abs_bias"], bias_tbl["mean_abs_bias"]))

np.savez(os.path.join(HERE, "v6_baseline.npz"), rdiff=rdiff, p_all=p_all,
         beta=beta, valid=valid, train_m=train_m, hold_m=hold_m,
         rat_w=out["rat_w"], rat_l=out["rat_l"])
daily = out["daily_r"]
daily_days = sorted(daily.keys())

# ── graph features, walk-forward at each holdout month boundary ──────────────
g_dnum = eng.g_dnum
wi, li = eng.wi, eng.li
n_t = len(eng.teams)
tidx = eng.tidx
sys.path.insert(0, "/Users/benny_es1/VCTMM")
from vctmm.benpom.teams import ORG_REGIONS  # noqa: E402
region = np.array([ORG_REGIONS.get(t) or "?" for t in eng.teams])

hold_rows = frame[(frame.date > "2024-12-31")]
months = sorted({d[:7] for d in hold_rows.date})
bounds = [m + "-01" for m in months]
log("[T2-e1] holdout months: %s" % ", ".join(months))


def graph_features(bound_s, excl_events=None):
    """Per-team features from games dated < bound_s. Returns dict of arrays."""
    bnum = int(np.datetime64(bound_s, "D").astype(int))
    m = g_dnum < bnum
    if excl_events is not None:
        ev = np.array([g["event_id"] for g in eng.games])
        m = m & ~np.isin(ev, list(excl_events))
    idx = np.where(m)[0]
    w_dec = np.exp(-LAM * (bnum - g_dnum[idx]) / 7.0)
    W = np.zeros((n_t, n_t))
    np.add.at(W, (wi[idx], li[idx]), w_dec)
    np.add.at(W, (li[idx], wi[idx]), w_dec)
    # eigenvector centrality (symmetric, Perron): principal eigvec, abs, max=1
    vals, vecs = np.linalg.eigh(W)
    v = np.abs(vecs[:, -1])
    if v.max() > 0:
        v = v / v.max()
    # distinct opponents / games (raw, cumulative)
    opp_sets = defaultdict(set)
    n_games = np.zeros(n_t)
    for j in idx:
        a, b = wi[j], li[j]
        opp_sets[a].add(b)
        opp_sets[b].add(a)
        n_games[a] += 1
        n_games[b] += 1
    opp_count = np.array([len(opp_sets[i]) for i in range(n_t)], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        diversity = np.where(n_games > 0, opp_count / n_games, np.nan)
    # cross-region decayed share among known-region opponents
    xr_num = np.zeros(n_t)
    xr_den = np.zeros(n_t)
    for j, wd in zip(idx, w_dec):
        a, b = wi[j], li[j]
        for me, opp in ((a, b), (b, a)):
            if region[opp] != "?":
                xr_den[me] += wd
                if region[me] != "?" and region[me] != region[opp]:
                    xr_num[me] += wd
    with np.errstate(invalid="ignore", divide="ignore"):
        xshare = np.where(xr_den > 0, xr_num / xr_den, np.nan)
    return {"eig": v, "opp_count": opp_count, "diversity": diversity,
            "xshare": xshare, "n_games": n_games}


feat_by_month = {}
for b in bounds:
    feat_by_month[b] = graph_features(b)
    log("  [T2-e1] %s: graph teams w/ games=%d" %
        (b, int((feat_by_month[b]["n_games"] > 0).sum())))

# month weights = team's holdout series count in that month
wt = defaultdict(lambda: defaultdict(float))
for r in hold_rows.itertuples(index=False):
    mk = r.date[:7] + "-01"
    wt[r.winner][mk] += 1.0
    wt[r.loser][mk] += 1.0

# mean v6 rating per team, same month weights (rating at last solved day < bound)
def rating_at(bound_s):
    prior_days = [d for d in daily_days if d < bound_s]
    return daily[prior_days[-1]] if prior_days else np.zeros(n_t)

rat_by_month = {b: rating_at(b) for b in bounds}


def agg(team, key):
    num = den = 0.0
    for b, w in wt[team].items():
        f = rat_by_month[b][tidx[team]] if key == "rating" else \
            feat_by_month[b][key][tidx[team]]
        if f is None or (isinstance(f, float) and np.isnan(f)):
            continue
        num += w * f
        den += w
    return num / den if den > 0 else np.nan


teams = [r["team"] for r in bias_tbl["teams"]]
missing = [t for t in teams if t not in tidx]
if missing:
    raise RuntimeError(f"bias-table teams missing from engine index: {missing}")
rows = []
for r in bias_tbl["teams"]:
    t = r["team"]
    rows.append({"team": t, "region": ORG_REGIONS.get(t, "?"),
                 "n": r["n"], "ll": r["ll"], "bias": r["bias"],
                 "abs_bias": abs(r["bias"]),
                 "eig_centrality": agg(t, "eig"),
                 "opp_count": agg(t, "opp_count"),
                 "opp_diversity": agg(t, "diversity"),
                 "xregion_share": agg(t, "xshare"),
                 "mean_rating": agg(t, "rating")})
sc = pd.DataFrame(rows)
log("[T3-e1] scatter table built: %d teams" % len(sc))

# ── correlations with team-bootstrap CIs (crn seed) ─────────────────────────
crn = json.load(open(os.path.join(V8, "crn.json")))
seed = crn["bootstrap"]["seed"]
n_boot = int(crn["bootstrap"]["n_boot"])
nT = len(sc)
rng = np.random.default_rng(seed)
bidx = rng.integers(0, nT, size=(n_boot, nT))   # full matrix, one call


def pearson(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def boot_ci(x, y, stat):
    vals = np.array([stat(x[bidx[r]], y[bidx[r]]) for r in range(n_boot)])
    ok = ~np.isnan(vals)
    return (float(np.percentile(vals[ok], 2.5)),
            float(np.percentile(vals[ok], 97.5)), int((~ok).sum()))


features = ["eig_centrality", "opp_count", "opp_diversity", "xregion_share"]
corr = []
for target in ("bias", "abs_bias"):
    yv = sc[target].values.astype(float)
    for f in features:
        xv = sc[f].values.astype(float)
        pe, spv = pearson(xv, yv), spearman(xv, yv)
        plo, phi, pd_ = boot_ci(xv, yv, pearson)
        slo, shi, sd_ = boot_ci(xv, yv, spearman)
        corr.append({"target": target, "feature": f,
                     "pearson": round(pe, 4), "pearson_ci": [round(plo, 4), round(phi, 4)],
                     "spearman": round(spv, 4), "spearman_ci": [round(slo, 4), round(shi, 4)],
                     "n_teams": nT, "degenerate_resamples": pd_ + sd_})
        log("  [T4-e1] %s ~ %s: pearson %.3f [%.3f,%.3f]  spearman %.3f [%.3f,%.3f]"
            % (target, f, pe, plo, phi, spv, slo, shi))

# partial Spearman bias ~ centrality | mean_rating (secondary, non-gating)
def partial_spear(b_, c_, r_):
    rb, rc, rr = rankdata(b_), rankdata(c_), rankdata(r_)
    X = np.column_stack([np.ones(len(rr)), rr])
    res_b = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    res_c = rc - X @ np.linalg.lstsq(X, rc, rcond=None)[0]
    return pearson(res_b, res_c)


bv, cv, rv = (sc["bias"].values.astype(float),
              sc["eig_centrality"].values.astype(float),
              sc["mean_rating"].values.astype(float))
pp = partial_spear(bv, cv, rv)
pvals = np.array([partial_spear(bv[bidx[r]], cv[bidx[r]], rv[bidx[r]])
                  for r in range(n_boot)])
ok = ~np.isnan(pvals)
partial = {"stat": "partial spearman bias ~ eig_centrality | mean_rating",
           "value": round(float(pp), 4),
           "ci": [round(float(np.percentile(pvals[ok], 2.5)), 4),
                  round(float(np.percentile(pvals[ok], 97.5)), 4)]}
log("  [T4-e1] partial spearman bias~eig | rating: %.3f [%.3f,%.3f]"
    % (pp, partial["ci"][0], partial["ci"][1]))

# ── GATE (preregistered) ────────────────────────────────────────────────────
def get(t, f):
    return next(c for c in corr if c["target"] == t and c["feature"] == f)

g1 = get("abs_bias", "eig_centrality")
g2 = get("bias", "eig_centrality")
alive1 = g1["spearman_ci"][1] < 0
alive2 = g2["spearman_ci"][1] < 0
gate_pass = bool(alive1 or alive2)
gate = {"rule": "H2 alive iff >=1 of spearman(|bias|,eig) / spearman(bias,eig) "
                "has 95% CI excluding 0 with negative sign (preregistered)",
        "abs_bias_vs_eig": {"spearman": g1["spearman"], "ci": g1["spearman_ci"],
                            "passes": bool(alive1)},
        "bias_vs_eig": {"spearman": g2["spearman"], "ci": g2["spearman_ci"],
                        "passes": bool(alive2)},
        "verdict": "ALIVE — proceed to E2/E3" if gate_pass
                   else "DEAD — no bias-centrality relationship; stop"}
log("[T5-e1] GATE: %s" % gate["verdict"])

# ── corpus-expansion centrality shift at 2026-07-01 ─────────────────────────
new_events = list(json.load(open(os.path.join(STATS, "power_mde_expanded.json")))
                  ["new_events"].keys())
T_REF = "2026-07-01"
post = graph_features(T_REF)
pre = graph_features(T_REF, excl_events=new_events)
rank_post = rankdata(-post["eig"], method="min")
rank_pre = rankdata(-pre["eig"], method="min")
shift_rows = []
for t in ELITE + FLOOR:
    i = tidx[t]
    shift_rows.append({"team": t, "group": "elite" if t in ELITE else "floor",
                       "eig_pre": round(float(pre["eig"][i]), 4),
                       "eig_post": round(float(post["eig"][i]), 4),
                       "delta": round(float(post["eig"][i] - pre["eig"][i]), 4),
                       "rank_pre": int(rank_pre[i]), "rank_post": int(rank_post[i])})
bt_idx = [tidx[t] for t in teams]
mean_shift = float(np.mean(np.abs(post["eig"][bt_idx] - pre["eig"][bt_idx])))
log("[T6-e1] centrality shift @%s: mean|delta| over bias-table teams = %.4f"
    % (T_REF, mean_shift))
for r in shift_rows:
    log("    %(team)s (%(group)s): %(eig_pre).3f -> %(eig_post).3f "
        "(rank %(rank_pre)d -> %(rank_post)d)" % r)

out_json = {
    "generated_by": "agent:bias-h2 E1, 2026-07-28",
    "preregistered": "preregister.bias_h2.md E1 (written before running)",
    "frame": {"path": "testing_lab/v8/data/frame_expanded/series.csv",
              "sha256_verified": True, "n_holdout": int(hold_m.sum())},
    "v6_baseline": {"beta": beta, "ll_holdout": out["ll_test"],
                    "n_teams_bias": bias_tbl["n_teams"],
                    "max_abs_bias": bias_tbl["max_abs_bias"],
                    "mean_abs_bias": bias_tbl["mean_abs_bias"]},
    "feature_spec": {"graph": "expanded games list, walk-forward at each "
                              "holdout month boundary, edge weight exp decay "
                              "HL 26 weeks; per-team aggregate weighted by "
                              "holdout series per month",
                     "eig": "principal eigvec of symmetric W (eigh), abs, max=1"},
    "scatter": rows,
    "correlations": corr,
    "partial": partial,
    "bootstrap": {"seed": seed, "n_boot": n_boot, "unit": "teams",
                  "note": "crn seed/generator/n_boot reused for team "
                          "resampling (crn stores series-level matrices); "
                          "full index matrix drawn in one call"},
    "gate": gate,
    "expansion_shift": {"t_ref": T_REF, "excluded_events": new_events,
                        "named_teams": shift_rows,
                        "mean_abs_delta_bias_table_teams": round(mean_shift, 4)},
}
with open(os.path.join(STATS, "h2_centrality.json"), "w") as f:
    json.dump(out_json, f, indent=1)
log("[T7-e1] wrote stats/h2_centrality.json — gate %s"
    % ("PASS" if gate_pass else "FAIL (H2 dead)"))
