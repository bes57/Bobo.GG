"""agent:bias-h3 — Experiment 2 (5d): heterogeneous half-lives via pooled process noise.

Base = plain SS-core 1a (train-selected within the plain family): V0 frozen at
1a's fitted value; the K=3 per-cell q's are the only new free parameters, fit
train-only (beta refit at every likelihood evaluation). Axes (walk-forward):
  A roster stability : matches_since_change at the game's match — <=3 / 4-10 / >10
  B org age          : days since org's first corpus game — <180 / 180-540 / >540
  C rating volatility: trailing std of last 12 standardized innovations from the
                       FITTED 1a filter — train-terciles; <6 prior z -> middle cell
Per cell: MLE q_k (coordinate descent, Brent on log q), profile-likelihood 95% CI
(Delta total train NLL = 1.92), DerSimonian-Laird partial pooling on log q.
Checkpoints to sweep_5d.json. Publishes NOTHING itself — stats emission is a
separate step so the fit can be inspected first.
"""
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib_h3 import GameData, implied_half_life, steady_state_neff  # noqa: E402

CKPT = os.path.join(HERE, "sweep_5d.json")
CORE_KEY = "1a|q/R=0.00564622|V0/R=1.27156"


def load_ckpt():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {}


def save_ckpt(ck):
    tmp = CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f, indent=1)
    os.replace(tmp, CKPT)


def build_cells(gd):
    """Per-game per-side cell ids for axes A and B (+ diagnostics)."""
    tp = pd.read_csv(os.path.join(HERE, "lineup_topup.csv"))
    msc = {(r.org, r.match_id): r.matches_since_change
           for r in tp.itertuples(index=False)}
    first_date = dict(tp.groupby("org").date.min())
    n = gd.n_games
    cA = np.full((n, 2), 2, dtype=int)   # default stable
    cB = np.full((n, 2), 2, dtype=int)
    missA = 0
    g_dnum = pd.to_datetime(gd.g_date).values.astype("datetime64[D]").astype(int)
    first_dnum = {o: int(np.datetime64(d, "D").astype(int))
                  for o, d in first_date.items()}
    teams = gd.teams
    for j in range(n):
        for side, ti in ((0, gd.wi[j]), (1, gd.li[j])):
            org = teams[ti]
            m = msc.get((org, gd.g_mid[j]))
            if m is None:
                missA += 1
            else:
                cA[j, side] = 0 if m <= 3 else (1 if m <= 10 else 2)
            fd = first_dnum.get(org)
            if fd is not None:
                age = g_dnum[j] - fd
                cB[j, side] = 0 if age < 180 else (1 if age <= 540 else 2)
    return cA, cB, missA


def build_cells_vol(gd, q_core, V0):
    """Axis C: trailing std of last 12 z-innovations (fitted 1a filter),
    strictly earlier in processing order; <6 prior z -> cell 1 (middle).
    Tercile thresholds from TRAIN games only."""
    f = gd.run_filter(q_core, V0, collect_z=True)
    gidx, tidx_, zs = f["z"]
    n = gd.n_games
    hist = defaultdict(list)
    vol = np.full((n, 2), np.nan)
    # z records are in processing order (two per game)
    side_of = {}
    for j in range(n):
        side_of[(j, gd.wi[j])] = 0
        side_of[(j, gd.li[j])] = 1
    for g, t, z in zip(gidx, tidx_, zs):
        h = hist[t]
        if len(h) >= 6:
            vol[g, side_of[(g, t)]] = float(np.std(h[-12:]))
        h.append(z)
    tr = gd.g_date <= "2024-12-31"
    pool = vol[tr].ravel()
    pool = pool[~np.isnan(pool)]
    t1, t2 = np.percentile(pool, [33.333, 66.667])
    cC = np.full((n, 2), 1, dtype=int)          # undefined -> middle
    cC[vol <= t1] = 0                            # low vol
    cC[vol > t2] = 2                             # high vol
    return cC, float(t1), float(t2), int(np.isnan(vol).sum())


def axis_nll(gd, cells, logqs, V0):
    q_vec = np.exp(logqs)[cells]                # (n_games, 2)
    f = gd.run_filter(0.0, V0, q_vec=q_vec)
    beta, tr_nll = gd.fit_beta(f["mu"], f["s2"])
    return tr_nll, beta, f


def fit_axis(gd, cells, V0, q0, tag, ck, n_sweeps=3):
    """Coordinate descent on log q_k; every accepted state checkpointed."""
    if tag in ck and ck[tag].get("done"):
        return ck[tag]
    logqs = np.array(ck.get(tag, {}).get("logqs", [math.log(q0)] * 3))
    start_sweep = ck.get(tag, {}).get("sweep", 0)
    nll, beta, _ = axis_nll(gd, cells, logqs, V0)
    for sweep in range(start_sweep, n_sweeps):
        for k in range(3):
            def f1(lq):
                t = logqs.copy()
                t[k] = lq
                return axis_nll(gd, cells, t, V0)[0]
            res = minimize_scalar(f1, bounds=(logqs[k] - 3.0, logqs[k] + 3.0),
                                  method="bounded", options={"xatol": 1e-3})
            if res.fun < nll - 1e-9:
                logqs[k] = float(res.x)
                nll = float(res.fun)
        ck[tag] = {"logqs": list(logqs), "sweep": sweep + 1,
                   "nll_train": nll, "done": False}
        save_ckpt(ck)
        print(f"  [{tag}] sweep {sweep+1}: q={np.exp(logqs).round(5).tolist()} "
              f"nll={nll:.6f}", flush=True)
    # profile 95% CIs (Delta total train NLL = 1.92 => Delta mean = 1.92/n_train)
    n_train_rows = int(gd.train_mask.sum())
    dmean = 1.92 / n_train_rows
    cis = []
    for k in range(3):
        def f1(lq):
            t = logqs.copy()
            t[k] = lq
            return axis_nll(gd, cells, t, V0)[0]
        lo = hi = logqs[k]
        step = 0.15
        while f1(lo - step) - nll < dmean and lo - step > logqs[k] - 6:
            lo -= step
        while f1(hi + step) - nll < dmean and hi + step < logqs[k] + 6:
            hi += step
        # bisect the crossing on each side
        def cross(a, b):
            for _ in range(25):
                m = 0.5 * (a + b)
                if f1(m) - nll < dmean:
                    a = m
                else:
                    b = m
                if abs(b - a) < 1e-3:
                    break
            return 0.5 * (a + b)
        lo_c = cross(logqs[k], lo - step) if lo - step > logqs[k] - 6 else lo - step
        hi_c = cross(logqs[k], hi + step) if hi + step < logqs[k] + 6 else hi + step
        cis.append([float(lo_c), float(hi_c)])
        print(f"  [{tag}] cell {k} logq CI: [{lo_c:.3f},{hi_c:.3f}]", flush=True)
    # curvature SE on log q (for DL pooling)
    ses = []
    for k in range(3):
        h = 0.10
        def f1(lq):
            t = logqs.copy()
            t[k] = lq
            return axis_nll(gd, cells, t, V0)[0]
        d2 = (f1(logqs[k] + h) - 2 * nll + f1(logqs[k] - h)) / h ** 2 * n_train_rows
        ses.append(float(1.0 / math.sqrt(max(d2, 1e-9))))
    nll_f, beta_f, filt = axis_nll(gd, cells, logqs, V0)
    ck[tag] = {"logqs": list(map(float, logqs)), "nll_train": float(nll_f),
               "beta": float(beta_f), "ci_logq": cis, "se_logq": ses,
               "done": True}
    save_ckpt(ck)
    return ck[tag]


def dl_pool(logqs, ses):
    """DerSimonian-Laird random-effects pooling on log q (K=3)."""
    y = np.array(logqs)
    v = np.array(ses) ** 2
    w = 1 / v
    ybar = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - ybar) ** 2))
    K = len(y)
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (K - 1)) / c)
    wstar = 1 / (v + tau2)
    mu = float(np.sum(wstar * y) / np.sum(wstar))
    shrunk = [(float((yi / vi + mu / tau2) / (1 / vi + 1 / tau2)) if tau2 > 0 else mu)
              for yi, vi in zip(y, v)]
    return {"tau2": tau2, "Q": Q, "pooled_mu_logq": mu, "shrunk_logq": shrunk,
            "shrink_fraction": [float(tau2 / (tau2 + vi)) if tau2 > 0 else 0.0
                                for vi in v]}


def main():
    t0 = time.time()
    gd = GameData()
    ck = load_ckpt()
    core = json.load(open(os.path.join(HERE, "sweep_core.json")))["points"][CORE_KEY]
    q0 = core["q_over_R"] * gd.R
    V0 = core["V0_over_R"] * gd.R
    print(f"base 1a: q={q0:.5f} V0={V0:.3f} R={gd.R:.4f}", flush=True)

    cA, cB, missA = build_cells(gd)
    print(f"axis A cells (per-side games): {[int((cA==k).sum()) for k in range(3)]}"
          f" missing->stable {missA}", flush=True)
    cC, t1, t2, nanC = build_cells_vol(gd, q0, V0)
    print(f"axis C terciles at z-std {t1:.3f}/{t2:.3f}; undefined sides {nanC}",
          flush=True)
    ck.setdefault("meta", {})
    ck["meta"].update({
        "base_core_key": CORE_KEY, "q0": q0, "V0": V0, "R": gd.R,
        "axisA_cells_games": [int((cA == k).sum()) for k in range(3)],
        "axisA_missing_sides_defaulted_stable": missA,
        "axisB_cells_games": [int((cB == k).sum()) for k in range(3)],
        "axisC_cells_games": [int((cC == k).sum()) for k in range(3)],
        "axisC_tercile_thresholds": [t1, t2],
        "axisC_undefined_sides_to_mid": nanC})
    save_ckpt(ck)

    axes = {"A_roster": cA, "B_orgage": cB, "C_volatility": cC}
    for tag, cells in axes.items():
        print(f"fitting axis {tag} ...", flush=True)
        res = fit_axis(gd, cells, V0, q0, tag, ck)
        qs = np.exp(res["logqs"])
        pool = dl_pool(res["logqs"], res["se_logq"])
        res["pooling"] = pool
        res["q"] = [float(x) for x in qs]
        res["q_pooled"] = [float(np.exp(x)) for x in pool["shrunk_logq"]]
        res["hl_games"] = [implied_half_life(x, gd.R) for x in qs]
        res["hl_games_pooled"] = [implied_half_life(np.exp(x), gd.R)
                                  for x in pool["shrunk_logq"]]
        res["hl_ci"] = [[implied_half_life(math.exp(hi), gd.R),
                         implied_half_life(math.exp(lo), gd.R)]
                        for lo, hi in res["ci_logq"]]   # q lo -> HL hi
        res["neff_ss"] = [steady_state_neff(x, gd.R) for x in qs]
        # holdout score of the fitted axis model (record only)
        nll_tr, beta, f = axis_nll(gd, cells, np.array(res["logqs"]), V0)
        sc = gd.score(beta, f["mu"], f["s2"])
        res["ll_holdout"] = round(sc["ll_holdout"], 6)
        res["ll_train_final"] = round(nll_tr, 6)
        ck[tag] = res
        save_ckpt(ck)
        print(f"  [{tag}] q={np.round(qs,5).tolist()} HL={np.round(res['hl_games'],1).tolist()} "
              f"ll_train={res['nll_train']:.6f} ll_holdout={res['ll_holdout']}", flush=True)
    print(f"5d fits done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
