"""agent:bias-h4 stage 2 — E1 dispersion diagnostic, E2 dispersion links,
E3 interaction guard. Protocol: testing_lab/v8/preregister.bias_h4.md
(written before this file ran). Inputs: scratch v6_solve.npz + games.csv
(stage 1), canonical frame, data/map_vetos.csv + data/match_dates.json.
Outputs: stats/h4_dispersion_diag.json, stats/h4_series_link.json,
stats/h4_bias_caterpillar.json. Randomness: crn.json mc_seeds[0] (cell
D_sweep bootstraps, cells in documented order), mc_seeds[1] (parameter
refit bootstraps); holdout judging via referee.paired_bootstrap_crn.
"""
import json
import math
import os
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

TL = "/Users/benny_es1/PythonTest/testing_lab"
V8 = os.path.join(TL, "v8")
SCRATCH = os.path.join(V8, "scratch", "bias_h4")
STATS = os.path.join(V8, "stats")
LOG = os.path.join(V8, "logs", "bias_h4.log")
DATA = "/Users/benny_es1/PythonTest/data"
sys.path.insert(0, TL)
sys.path.insert(0, V8)
import referee  # noqa: E402

EPS = 1e-12
SQRT2 = math.sqrt(2.0)
GHX, GHW_RAW = np.polynomial.hermite.hermgauss(31)
GHW = GHW_RAW / math.sqrt(math.pi)
NOW = time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ── load inputs ──────────────────────────────────────────────────────────────
crn = json.load(open(os.path.join(V8, "crn.json")))
SEED_CELLS = int(crn["mc_seeds"][0])   # design 1: D_sweep cell bootstraps
SEED_PARAMS = int(crn["mc_seeds"][1])  # design 2: dispersion-param refits
frame = pd.read_csv(os.path.join(V8, "data", "frame_expanded",
                                 "series.csv")).reset_index(drop=True)
sv = np.load(os.path.join(SCRATCH, "v6_solve.npz"))
rdiff = sv["rdiff"]
BETA6 = float(sv["beta"][0])
meta6 = json.load(open(os.path.join(SCRATCH, "v6_meta.json")))
games_df = pd.read_csv(os.path.join(SCRATCH, "games.csv"))

n = len(frame)
fmts = frame.fmt.values
is5 = np.isin(fmts, ("bo5", "bo5_gf"))
is1 = fmts == "bo1"
valid = ~np.isnan(rdiff)
train = valid & (frame.date <= "2024-12-31").values
hold = valid & (frame.date > "2024-12-31").values
assert int(hold.sum()) == 1217, int(hold.sum())
l_maps = frame.l_maps.values.astype(int)

# winner-referenced per-map prob under v6 and the iid series closed form
lq6 = BETA6 * rdiff
pmap6 = sigmoid(lq6)


def sp_iid(q, is5_, is1_):
    om = 1 - q
    return np.where(is1_, q,
                    np.where(is5_, q ** 3 * (1 + 3 * om + 6 * om * om),
                             q ** 2 * (3 - 2 * q)))


pv6 = sp_iid(pmap6, is5, is1)

# ── depth feature (walk-forward; preregister E1) ─────────────────────────────
t0 = time.time()
REAL_MAPS = sorted(set(games_df.map_name) - {"TBD"})  # Summit is a real 2026
# Stage-2 map (23 games, 2026_stage2); the single 'TBD' row is junk.
g_real = games_df[games_df.map_name.isin(REAL_MAPS)]
day = lambda s: pd.to_datetime(s).values.astype("datetime64[D]").astype(int)
g_day = day(g_real.date_s)
midx = {m: i for i, m in enumerate(REAL_MAPS)}
g_midx = g_real.map_name.map(midx).values

team_days, team_maps = {}, {}
for side in ("winner", "loser"):
    for t, grp_idx in g_real.groupby(side).groups.items():
        loc = g_real.index.get_indexer(grp_idx)
        team_days.setdefault(t, []).append(g_day[loc])
        team_maps.setdefault(t, []).append(g_midx[loc])
for t in team_days:
    d_ = np.concatenate(team_days[t])
    m_ = np.concatenate(team_maps[t])
    o = np.argsort(d_, kind="stable")
    team_days[t], team_maps[t] = d_[o], m_[o]

vet = pd.read_csv(os.path.join(DATA, "map_vetos.csv"))
vet = vet[vet["map"].isin(REAL_MAPS)]
mdates = json.load(open(os.path.join(DATA, "match_dates.json")))
vet["date"] = vet.MatchID.astype(str).map(mdates)
vet = vet.dropna(subset=["date"])
v_day = day(vet.date)
v_midx = vet["map"].map(midx).values
o = np.argsort(v_day, kind="stable")
v_day, v_midx = v_day[o], v_midx[o]


def pool_at(D):
    for lo_off in (60, 120):
        a, b = np.searchsorted(v_day, D - lo_off), np.searchsorted(v_day, D)
        maps = set(v_midx[a:b])
        if len(maps) >= 5:
            return maps, lo_off
    a = np.searchsorted(v_day, D)
    maps = set(v_midx[:a])
    return (maps, "all-history") if maps else (None, "none")


def played_90(team, D):
    dd = team_days.get(team)
    if dd is None:
        return set()
    a, b = np.searchsorted(dd, D - 90), np.searchsorted(dd, D)
    return set(team_maps[team][a:b])


f_day = day(frame.date)
depth_w = np.full(n, np.nan)
depth_l = np.full(n, np.nan)
pool_fallbacks = {60: 0, 120: 0, "all-history": 0, "none": 0}
pool_cache = {}
for i in range(n):
    D = f_day[i]
    if D not in pool_cache:
        pool_cache[D] = pool_at(D)
        pool_fallbacks[pool_cache[D][1]] += 1
    pool, _ = pool_cache[D]
    if pool is None:
        continue
    depth_w[i] = len(played_90(frame.winner.iloc[i], D) & pool) / len(pool)
    depth_l[i] = len(played_90(frame.loser.iloc[i], D) & pool) / len(pool)

fav_is_w = pmap6 >= 0.5
tie = np.abs(pmap6 - 0.5) < 1e-9
p_fav = np.maximum(pmap6, 1 - pmap6)
depth_fav = np.where(fav_is_w, depth_w, depth_l)
log(f"depth: built in {time.time()-t0:.1f}s; real maps={len(REAL_MAPS)}; "
    f"pool windows used {pool_fallbacks}; depth_fav defined on "
    f"{int((~np.isnan(depth_fav)).sum())}/{n} rows; ties={int(tie.sum())}")

# ── E1 dispersion diagnostic (train, fmt != bo1) ─────────────────────────────
diag = train & ~is1
dq = pmap6.copy()          # winner-referenced map prob
d_lq = lq6.copy()
imp_sweep = np.where(is5, 1.0 / (1 + 3 * (1 - dq) + 6 * (1 - dq) ** 2),
                     1.0 / (3 - 2 * dq))
imp_lmean = np.where(
    is5, (3 * (1 - dq) + 12 * (1 - dq) ** 2) /
         (1 + 3 * (1 - dq) + 6 * (1 - dq) ** 2),
    2 * (1 - dq) / (3 - 2 * dq))
obs_sweep = (l_maps == 0).astype(float)

# depth terciles + z scaling, frozen on TRAIN diag rows with depth defined
dmask = diag & ~np.isnan(depth_fav)
tcuts = np.quantile(depth_fav[dmask], [1 / 3, 2 / 3])
z_mu, z_sd = float(np.mean(depth_fav[dmask])), float(np.std(depth_fav[dmask]))
z_all = np.where(np.isnan(depth_fav), 0.0, (depth_fav - z_mu) / max(z_sd, EPS))
terc = np.full(n, -1)
terc[~np.isnan(depth_fav) & (depth_fav <= tcuts[0])] = 1
terc[~np.isnan(depth_fav) & (depth_fav > tcuts[0]) & (depth_fav <= tcuts[1])] = 2
terc[~np.isnan(depth_fav) & (depth_fav > tcuts[1])] = 3
log(f"depth terciles (train diag): cuts={np.round(tcuts,4).tolist()} "
    f"z mu/sd={z_mu:.4f}/{z_sd:.4f}")


# dispersion model machinery ---------------------------------------------------
def sigma_score(lq, is5_, sig):
    """Unconditional winner-ref score probs under shared u~N(0,sig)."""
    Z = lq[:, None] + SQRT2 * np.asarray(sig)[:, None] * GHX[None, :]
    Q = sigmoid(Z)
    OM = 1 - Q
    q2 = Q * Q
    q3 = q2 * Q
    s0 = np.where(is5_[:, None], q3, q2)
    s1 = np.where(is5_[:, None], 3 * q3 * OM, 2 * q2 * OM)
    s2 = np.where(is5_[:, None], 6 * q3 * OM * OM, 0.0)
    return ((s0 * GHW).sum(1), (s1 * GHW).sum(1), (s2 * GHW).sum(1),
            (Q * GHW).sum(1))  # last: E[q_u] for bo1


def h_score(lq, is5_, h):
    """Unconditional winner-ref score probs under alternating pick-spread h
    (independent maps), averaged over the two start assignments."""
    h = np.asarray(h)
    P0 = np.zeros_like(lq)
    P1 = np.zeros_like(lq)
    P2 = np.zeros_like(lq)
    i3 = np.where(~is5_)[0]
    i5 = np.where(is5_)[0]
    for s in (1.0, -1.0):
        if len(i3):
            offs = [s, -s, 0.0]
            p = [sigmoid(lq[i3] + o * h[i3]) for o in offs]
            P0[i3] += 0.5 * p[0] * p[1]
            P1[i3] += 0.5 * (p[0] * (1 - p[1]) + (1 - p[0]) * p[1]) * p[2]
        if len(i5):
            offs = [s, -s, s, -s, 0.0]
            p = [sigmoid(lq[i5] + o * h[i5]) for o in offs]
            for j, tgt in ((0, P0), (1, P1), (2, P2)):
                m = 3 + j
                tot = np.zeros(len(i5))
                for L in combinations(range(m - 1), j):
                    term = np.ones(len(i5))
                    for t_ in range(m):
                        term = term * ((1 - p[t_]) if t_ in L else p[t_])
                    tot += term
                tgt[i5] += 0.5 * tot
    return P0, P1, P2


def cond_nll_sigma(rows, sig_val_or_arr):
    sig = np.broadcast_to(np.asarray(sig_val_or_arr, dtype=float),
                          d_lq[rows].shape)
    P0, P1, P2, _ = sigma_score(d_lq[rows], is5[rows], sig)
    P = np.stack([P0, P1, P2], 1)
    Pl = P[np.arange(len(rows)), l_maps[rows]]
    return -np.mean(np.log(np.clip(Pl, EPS, None) /
                           np.clip(P0 + P1 + P2, EPS, None)))


def cond_nll_h(rows, h_val_or_arr):
    h = np.broadcast_to(np.asarray(h_val_or_arr, dtype=float),
                        d_lq[rows].shape)
    P0, P1, P2 = h_score(d_lq[rows], is5[rows], h)
    P = np.stack([P0, P1, P2], 1)
    Pl = P[np.arange(len(rows)), l_maps[rows]]
    return -np.mean(np.log(np.clip(Pl, EPS, None) /
                           np.clip(P0 + P1 + P2, EPS, None)))


def fit_sigma(rows):
    r = minimize_scalar(lambda s: cond_nll_sigma(rows, s), bounds=(0.0, 2.5),
                        method="bounded", options={"xatol": 1e-4})
    return float(r.x)


def fit_h(rows):
    r = minimize_scalar(lambda h: cond_nll_h(rows, h), bounds=(0.0, 3.0),
                        method="bounded", options={"xatol": 1e-4})
    return float(r.x)


# cells in FIXED documented order (CRN stream order) ---------------------------
diag_idx = np.where(diag)[0]
cells = [
    ("overall", diag),
    ("fmt_bo3", diag & ~is5),
    ("fmt_bo5", diag & is5),
    ("fav_050_070", diag & ~tie & (p_fav < 0.7)),
    ("fav_070_100", diag & ~tie & (p_fav >= 0.7)),
    ("depth_T1", diag & (terc == 1)),
    ("depth_T2", diag & (terc == 2)),
    ("depth_T3", diag & (terc == 3)),
    ("depth_T1_fav070", diag & (terc == 1) & ~tie & (p_fav >= 0.7)),
    ("depth_T2_fav070", diag & (terc == 2) & ~tie & (p_fav >= 0.7)),
    ("depth_T3_fav070", diag & (terc == 3) & ~tie & (p_fav >= 0.7)),  # C*
]
PARAM_CI_CELLS = {"overall", "fmt_bo3", "fmt_bo5",
                  "depth_T1_fav070", "depth_T3_fav070"}
N_BOOT_CELL, N_BOOT_PARAM = 4000, 2000

rng_cells = np.random.default_rng(SEED_CELLS)
rng_params = np.random.default_rng(SEED_PARAMS)
cell_rows_map, cell_boot_D = {}, {}
cell_out = []
t0 = time.time()
for name, m in cells:
    rows = np.where(m)[0]
    nc = len(rows)
    rec = {"name": name, "n": int(nc)}
    if nc == 0:
        cell_out.append(rec)
        continue
    obs, imp = obs_sweep[rows], imp_sweep[rows]
    D = float(obs.mean() - imp.mean())
    idx = rng_cells.integers(0, nc, size=(N_BOOT_CELL, nc))  # one call/cell
    Db = obs[idx].mean(1) - imp[idx].mean(1)
    rec.update({
        "obs_sweep_share": round(float(obs.mean()), 4),
        "implied_sweep_share": round(float(imp.mean()), 4),
        "D_sweep_pp": round(100 * D, 2),
        "D_sweep_ci_pp": [round(100 * float(np.percentile(Db, 2.5)), 2),
                          round(100 * float(np.percentile(Db, 97.5)), 2)],
        "obs_mean_lmaps": round(float(l_maps[rows].mean()), 4),
        "implied_mean_lmaps": round(float(imp_lmean[rows].mean()), 4),
        "sigma_hat": round(fit_sigma(rows), 4),
        "h_hat": round(fit_h(rows), 4),
    })
    cell_rows_map[name] = rows
    cell_boot_D[name] = Db
    cell_out.append(rec)
log(f"E1 cells + D_sweep boots done {time.time()-t0:.1f}s")

# parameter bootstrap CIs (mc_seeds[1] stream, designated cells in cell order)
t0 = time.time()
for rec in cell_out:
    if rec["name"] not in PARAM_CI_CELLS or rec["n"] == 0:
        continue
    rows = cell_rows_map[rec["name"]]
    nc = len(rows)
    idx = rng_params.integers(0, nc, size=(N_BOOT_PARAM, nc))
    sb = np.empty(N_BOOT_PARAM)
    hb = np.empty(N_BOOT_PARAM)
    for r in range(N_BOOT_PARAM):
        rr = rows[idx[r]]
        sb[r] = fit_sigma(rr)
        hb[r] = fit_h(rr)
    rec["sigma_ci"] = [round(float(np.percentile(sb, 2.5)), 4),
                       round(float(np.percentile(sb, 97.5)), 4)]
    rec["h_ci"] = [round(float(np.percentile(hb, 2.5)), 4),
                   round(float(np.percentile(hb, 97.5)), 4)]
    log(f"E1 param CI {rec['name']}: sigma={rec['sigma_hat']} "
        f"{rec['sigma_ci']} h={rec['h_hat']} {rec['h_ci']}")
log(f"E1 param boots done {time.time()-t0:.1f}s")

# joint (sigma, h) overall — descriptive
rows_all = cell_rows_map["overall"]


def _joint_nll(rows, s_, h_):
    """Shared-u AND pick-spread jointly: quadrature over u, spread inside."""
    lqr, is5r = d_lq[rows], is5[rows]
    P0 = np.zeros(len(rows)); P1 = np.zeros(len(rows)); P2 = np.zeros(len(rows))
    for k in range(len(GHX)):
        u = SQRT2 * s_ * GHX[k]
        a0, a1, a2 = h_score(lqr + u, is5r, np.full(len(rows), h_))
        P0 += GHW[k] * a0; P1 += GHW[k] * a1; P2 += GHW[k] * a2
    P = np.stack([P0, P1, P2], 1)
    Pl = P[np.arange(len(rows)), l_maps[rows]]
    return -np.mean(np.log(np.clip(Pl, EPS, None) /
                           np.clip(P0 + P1 + P2, EPS, None)))


rj = minimize(lambda x: _joint_nll(rows_all, abs(x[0]), abs(x[1])),
              x0=[0.3, 0.3], method="Nelder-Mead",
              options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-7})
joint_fit = {"sigma": round(abs(float(rj.x[0])), 4),
             "h": round(abs(float(rj.x[1])), 4),
             "nll": round(float(rj.fun), 6)}
log(f"E1 joint fit (descriptive): {joint_fit}")

# gates ------------------------------------------------------------------------
def _get(name_):
    return next(c for c in cell_out if c["name"] == name_)


def _ci_excl0(ci):
    return ci[0] > 0 or ci[1] < 0


cstar = _get("depth_T3_fav070")
G1 = bool(cstar.get("D_sweep_ci_pp") and cstar["D_sweep_ci_pp"][1] < 0)
grad = None
if "depth_T1_fav070" in cell_boot_D and "depth_T3_fav070" in cell_boot_D:
    gdiff = cell_boot_D["depth_T1_fav070"] - cell_boot_D["depth_T3_fav070"]
    grad = {"point_pp": round(_get("depth_T1_fav070")["D_sweep_pp"] -
                              cstar["D_sweep_pp"], 2),
            "ci_pp": [round(float(np.percentile(gdiff, 2.5)) * 100, 2),
                      round(float(np.percentile(gdiff, 97.5)) * 100, 2)]}
g2_terms = {
    "overall_D": _ci_excl0(_get("overall")["D_sweep_ci_pp"]),
    "bo3_D": _ci_excl0(_get("fmt_bo3")["D_sweep_ci_pp"]),
    "bo5_D": _ci_excl0(_get("fmt_bo5")["D_sweep_ci_pp"]),
    "sigma_overall": _get("overall").get("sigma_ci", [0, 0])[0] > 1e-3,
    "h_overall": _get("overall").get("h_ci", [0, 0])[0] > 1e-3,
    "depth_gradient": bool(grad and _ci_excl0(grad["ci_pp"])),
}
G2 = any(g2_terms.values())
gates = {"G1_underdispersed_deep_pool_favorites": G1,
         "G2_any_exploitable_dispersion": G2, "G2_terms": g2_terms,
         "gradient_T1fav_minus_T3fav": grad,
         "rule": "preregister.bias_h4.md E1/E2: L1 always; L2 if G1 or G2; "
                 "L3+L4 only if G1"}
log(f"gates: G1={G1} G2={G2} terms={g2_terms}")

diag_json = {
    "generated_by": "agent:bias-h4", "generated": NOW,
    "preregistered": "testing_lab/v8/preregister.bias_h4.md (before runs)",
    "frame_sha256": meta6["frame_sha256"],
    "v6_reconstruction": {"config": meta6["config"], "beta": BETA6,
                          "ll_train": meta6["ll_train"],
                          "ll_holdout": meta6["ll_test"],
                          "n_train_valid": meta6["n_train"],
                          "n_holdout": meta6["n_test"]},
    "definitions": {
        "sample": "train rows (date<=2024-12-31), valid rdiff, fmt!=bo1",
        "D_sweep_pp": "100*(obs share l_maps==0  -  mean iid P(sweep|win) at "
                      "v6 map prob); >0 over-dispersed, <0 under-dispersed",
        "conditional": "score distribution GIVEN series winner — insensitive "
                       "to first-order rating miscalibration",
        "sigma_hat": "ML shared series effect u~N(0,sigma), logit-additive, "
                     "GH-31 quadrature, conditional score likelihood",
        "h_hat": "ML alternating pick-spread (bo3 [+h,-h,0]; bo5 "
                 "[+h,-h,+h,-h,0]), averaged over both starts, independent "
                 "maps — the under-dispersion arm",
        "depth": "favorite's distinct real maps played (officials, trailing "
                 "90d) ∩ veto-era pool (map_vetos.csv, trailing 60d, "
                 "fallback 120d) / pool size; MapNum=='all' never enters "
                 "(engine loader filters it); 'TBD' junk map dropped",
        "terciles_train_cuts": [round(float(c), 4) for c in tcuts],
        "z_scaling_train": {"mu": round(z_mu, 4), "sd": round(z_sd, 4)},
    },
    "n_diag": int(diag.sum()),
    "depth_coverage": {"defined": int((~np.isnan(depth_fav)).sum()),
                       "of": n, "pool_windows": {str(k): v for k, v in
                                                 pool_fallbacks.items()}},
    "crn": {"cell_boot": {"seed": SEED_CELLS, "n_boot": N_BOOT_CELL,
                          "order": [c[0] for c in cells]},
            "param_boot": {"seed": SEED_PARAMS, "n_boot": N_BOOT_PARAM,
                           "cells": sorted(PARAM_CI_CELLS)}},
    "cells": cell_out,
    "joint_fit_overall": joint_fit,
    "gates": gates,
}
json.dump(diag_json, open(os.path.join(STATS, "h4_dispersion_diag.json"),
                          "w"), indent=1)
log("E1 written: stats/h4_dispersion_diag.json")

# ── E2 series links (gated) ──────────────────────────────────────────────────
def pwin_sigma(beta, sig_arr, rows):
    lq = beta * rdiff[rows]
    P0, P1, P2, Eq = sigma_score(lq, is5[rows], sig_arr[rows])
    return np.where(is1[rows], Eq, P0 + P1 + P2)


def pwin_h(beta, h_arr, rows):
    lq = beta * rdiff[rows]
    P0, P1, P2 = h_score(lq, is5[rows], h_arr[rows])
    return np.where(is1[rows], sigmoid(lq), P0 + P1 + P2)


tr_rows = np.where(train)[0]
all_rows = np.arange(n)


def refit_beta(pwin_fn, disp_arr):
    r = minimize_scalar(
        lambda b: -np.mean(np.log(np.clip(pwin_fn(b, disp_arr, tr_rows),
                                          EPS, 1))),
        bounds=(0.02, 0.8), method="bounded", options={"xatol": 1e-5})
    return float(r.x)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


links = {}
sig_glob = _get("overall")["sigma_hat"]
h_glob = _get("overall")["h_hat"]

# L1 — global sigma_u (always runs; Done-when deliverable)
t0 = time.time()
arr = np.full(n, sig_glob)
b1 = refit_beta(pwin_sigma, arr)
links["L1_sigma_global"] = {"params": {"sigma": sig_glob,
                                       "sigma_ci": _get("overall").get("sigma_ci")},
                            "beta_refit": b1,
                            "p": pwin_sigma(b1, arr, all_rows),
                            "disp_desc": "sigma constant"}
log(f"L1 fit: sigma={sig_glob} beta={b1:.4f} ({time.time()-t0:.1f}s)")

# L2 — sigma(depth) (if G1 or G2)
if G1 or G2:
    t0 = time.time()
    a0 = math.log(math.expm1(max(sig_glob, 0.05)))
    best = None
    for b_start in (0.0, 0.3, -0.3):
        rr = minimize(lambda x: cond_nll_sigma(
            rows_all, softplus(x[0] + x[1] * z_all[rows_all])),
            x0=[a0, b_start], method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-7})
        if best is None or rr.fun < best.fun:
            best = rr
    a_fit, bcoef = float(best.x[0]), float(best.x[1])
    # bootstrap CI on the depth coefficient (continues mc_seeds[1] stream)
    nb = len(rows_all)
    idxb = rng_params.integers(0, nb, size=(N_BOOT_PARAM, nb))
    bcs = np.empty(N_BOOT_PARAM)
    for r_ in range(N_BOOT_PARAM):
        rrows = rows_all[idxb[r_]]
        rr = minimize(lambda x: cond_nll_sigma(
            rrows, softplus(x[0] + x[1] * z_all[rrows])),
            x0=[a_fit, bcoef], method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 5e-4, "fatol": 1e-6})
        bcs[r_] = rr.x[1]
    b_ci = [round(float(np.percentile(bcs, 2.5)), 4),
            round(float(np.percentile(bcs, 97.5)), 4)]
    sig_arr2 = softplus(a_fit + bcoef * z_all)
    b2 = refit_beta(pwin_sigma, sig_arr2)
    links["L2_sigma_depth"] = {
        "params": {"a": round(a_fit, 4), "b_depth": round(bcoef, 4),
                   "b_depth_ci": b_ci,
                   "sigma_at_z(-1,0,1)": [round(float(softplus(a_fit + bcoef * z)), 4)
                                          for z in (-1, 0, 1)]},
        "beta_refit": b2, "p": pwin_sigma(b2, sig_arr2, all_rows),
        "disp_desc": "sigma=softplus(a+b*z_depth_fav), z train-frozen, "
                     "undefined depth -> z=0"}
    log(f"L2 fit: a={a_fit:.4f} b={bcoef:.4f} ci={b_ci} beta={b2:.4f} "
        f"({time.time()-t0:.1f}s)")
else:
    links["L2_sigma_depth"] = {"skipped": "gates G1 and G2 both false"}

# L3/L4 — pick-spread links (only if G1)
if G1:
    arr3 = np.full(n, h_glob)
    b3 = refit_beta(pwin_h, arr3)
    links["L3_h_global"] = {"params": {"h": h_glob,
                                       "h_ci": _get("overall").get("h_ci")},
                            "beta_refit": b3,
                            "p": pwin_h(b3, arr3, all_rows),
                            "disp_desc": "h constant"}
    a0 = math.log(math.expm1(max(h_glob, 0.05)))
    best = None
    for b_start in (0.0, 0.3, -0.3):
        rr = minimize(lambda x: cond_nll_h(
            rows_all, softplus(x[0] + x[1] * z_all[rows_all])),
            x0=[a0, b_start], method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-7})
        if best is None or rr.fun < best.fun:
            best = rr
    h_arr4 = softplus(float(best.x[0]) + float(best.x[1]) * z_all)
    b4 = refit_beta(pwin_h, h_arr4)
    links["L4_h_depth"] = {"params": {"a": round(float(best.x[0]), 4),
                                      "b_depth": round(float(best.x[1]), 4)},
                           "beta_refit": b4,
                           "p": pwin_h(b4, h_arr4, all_rows),
                           "disp_desc": "h=softplus(a+b*z_depth_fav)"}
else:
    links["L3_h_global"] = {"skipped": "gate G1 false (no under-dispersion "
                                       "for deep-pool favorites)"}
    links["L4_h_depth"] = {"skipped": "gate G1 false"}

# ── holdout judging (referee) ────────────────────────────────────────────────
hv_rows = np.where(hold)[0]
ev_hold = frame.event_id.values[hv_rows]
games_list = games_df.to_dict("records")
mde_within = json.load(open(os.path.join(STATS, "power_mde_expanded.json"))
                       )["checkpoint_quote"]["within_milli"]
p_v6_full = np.where(valid, pv6, np.nan)
bias_v6 = referee.per_team_bias(p_v6_full, frame.winner.values,
                                frame.loser.values, holdout=hold, valid=valid)
cat = {"generated_by": "agent:bias-h4", "generated": NOW,
       "definition": "referee.per_team_bias — probability points, holdout "
                     "n=1217, min_n=25; negative = model under-rates team",
       "v6": bias_v6, "links": {}}
link_results = {}
strong = ~np.isnan(p_v6_full) & hold & (np.maximum(pv6, 1 - pv6) >= 0.7)
b5 = strong & is5
b3m = strong & ~is5 & ~is1

for name, L in links.items():
    if "p" not in L:
        link_results[name] = {k: v for k, v in L.items()}
        continue
    p = L["p"]
    p_full = np.where(valid, p, np.nan)
    d = referee.delta_vector(p[hv_rows], pv6[hv_rows])
    bi = referee.paired_bootstrap_crn(d, mode="iid")
    bb = referee.paired_bootstrap_crn(d, mode="block_event",
                                      event_ids=ev_hold)
    roi = referee.expected_roi_of_dll(float(d.mean()), pv6[hv_rows])
    buck = referee.bucketed(frame, p_full, p_ref=p_v6_full, rdiff=rdiff,
                            games=games_list)
    bias = referee.per_team_bias(p_full, frame.winner.values,
                                 frame.loser.values, holdout=hold,
                                 valid=valid)
    cat["links"][name] = bias
    pf6 = np.maximum(pv6, 1 - pv6)
    pfL = np.where(pv6 >= 0.5, p, 1 - p)
    dmilli = float(d.mean()) * 1000
    inside = abs(dmilli) < mde_within
    link_results[name] = {
        "params": L["params"], "beta_refit": L["beta_refit"],
        "disp_desc": L["disp_desc"],
        "ll_holdout": round(float(np.mean(-np.log(np.clip(p[hv_rows], EPS, 1)))), 5),
        "ll_holdout_v6": round(float(np.mean(-np.log(np.clip(pv6[hv_rows], EPS, 1)))), 5),
        "delta_milli_vs_v6": round(dmilli, 3),
        "mde_within_milli": mde_within,
        "inside_noise_floor": inside,
        "boot_iid": bi, "boot_block_event": bb,
        "expected_roi": roi,
        "strong_fav_deltaP": {
            "bo5_pfav_ge_070": {"n": int(b5.sum()),
                                "mean_deltaP_favside": round(float(np.mean(
                                    pfL[b5] - pf6[b5])), 4) if b5.sum() else None},
            "bo3_pfav_ge_070": {"n": int(b3m.sum()),
                                "mean_deltaP_favside": round(float(np.mean(
                                    pfL[b3m] - pf6[b3m])), 4) if b3m.sum() else None}},
        "buckets": buck,
    }
    log(f"{name}: dLL={dmilli:+.3f}m (MDE {mde_within}m, inside={inside}) "
        f"iid CI=[{bi['ci_lo']*1000:.2f},{bi['ci_hi']*1000:.2f}]m "
        f"block CI=[{bb['ci_lo']*1000:.2f},{bb['ci_hi']*1000:.2f}]m "
        f"dP_bo5fav={link_results[name]['strong_fav_deltaP']['bo5_pfav_ge_070']}")

# ── E3 interaction guard: exact reduction at zero dispersion ─────────────────
qg = np.linspace(0.02, 0.98, 49)
lg = np.log(qg / (1 - qg))
red = []
for fmt_name, i5g, i1g in (("bo1", False, True), ("bo3", False, False),
                           ("bo5", True, False)):
    i5v = np.full(49, i5g)
    i1v = np.full(49, i1g)
    base = sp_iid(qg, i5v, i1v)
    P0, P1, P2, Eq = sigma_score(lg, i5v, np.zeros(49))
    ps = np.where(i1v, Eq, P0 + P1 + P2)
    H0, H1, H2 = h_score(lg, i5v, np.zeros(49))
    ph = np.where(i1v, qg, H0 + H1 + H2)
    red.append({"fmt": fmt_name,
                "max_abs_diff_sigma0": float(np.max(np.abs(ps - base))),
                "max_abs_diff_h0": float(np.max(np.abs(ph - base)))})
guard = {
    "reduction_check": red,
    "statement": "Dispersion acts ONLY at the series-aggregation layer over "
                 "the single p: link inputs are (beta*rdiff, fmt, "
                 "z_depth_of_favorite as the sigma/h covariate). No per-map "
                 "ratings, no pick bonus, no map identities anywhere in the "
                 "link — the ledger id 25 per-map+pick kill stays dead; no "
                 "map-level double counting is possible by construction.",
    "momentum_distinction": "NOT intra-series momentum (ledger id 27): the "
                            "series effect u is exchangeable, drawn before "
                            "map 1; no map outcome feeds another map's "
                            "probability.",
}
ok = all(r["max_abs_diff_sigma0"] < 1e-9 and r["max_abs_diff_h0"] < 1e-9
         for r in red)
log(f"E3 reduction check pass={ok}: {red}")

link_json = {
    "generated_by": "agent:bias-h4", "generated": NOW,
    "preregistered": "testing_lab/v8/preregister.bias_h4.md",
    "frame_sha256": meta6["frame_sha256"],
    "v6": {"beta": BETA6, "ll_holdout": meta6["ll_test"], "n_holdout": 1217},
    "fitting_protocol": "dispersion params: ML on TRAIN conditional score "
                        "likelihood (stage A); beta: refit per link on TRAIN "
                        "series win/loss with dispersion frozen (stage B); "
                        "holdout untouched by any fit",
    "gates": gates,
    "links": link_results,
    "interaction_guard": guard,
    "units_note": "delta_milli = milli-LL vs v6 on holdout n=1217; "
                  "expected_roi via referee.expected_roi_of_dll (reporting "
                  "unit only, never selection)",
}
json.dump(link_json, open(os.path.join(STATS, "h4_series_link.json"), "w"),
          indent=1)
json.dump(cat, open(os.path.join(STATS, "h4_bias_caterpillar.json"), "w"),
          indent=1)
log("E2/E3 written: stats/h4_series_link.json, stats/h4_bias_caterpillar.json")
log("stage2 complete")
