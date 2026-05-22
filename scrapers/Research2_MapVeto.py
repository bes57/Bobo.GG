#!/usr/bin/env python3
"""
Research2_MapVeto.py
====================

Phase-2 research: test whether map-level features and veto-pool information
lower series Brier below the current optimized BenPom baseline for VCT
2024 → 2026-stage-1 (n=1217, non-tied series).

Baseline (the model we want to beat):
    p_series = sigmoid(0.154 * (winner_before - loser_before)
                       + 0.36  * intl_exp_diff)

Methodology
-----------
* Time-based holdout: train on first 75 % by date, test on last 25 %.
* For each candidate feature build:
        Logit(y) = alpha + b1 * logit(p_baseline) + b2 * feature
  on the TRAIN slice, then evaluate Brier / Platt-slope on TEST.
* Compare to a baseline-only model (alpha + b * logit(p_base)) on the
  SAME train/test split — both fits are calibrated, so the comparison is fair.
* LR test on train log-likelihoods (chi2 with df = # added features).
* Bootstrap CI on Δ-Brier over the held-out test set (200 resamples).

Pass criteria for "ship-worthy"
-------------------------------
* Held-out Brier improvement   ≥ 0.0010
* Platt slope on test set      in [0.85, 1.15]
* LR p                         < 0.0083  (Bonferroni 0.05 / 6)
* Bootstrap 95 % CI on Δ-Brier excludes zero

Features tested
---------------
1. mappool_versatility_diff   - distinct maps played in past 60 days (fav-dog)
2. strong_map_overlap         - #maps in fav top-3 also in dog top-3
3. fav_top1_vs_dog_top1       - fav best-map rating - dog best-map rating
4. best_case_gap              - MAX over pool of (fav-dog) per-map rating diff
5. worst_case_gap             - MIN over pool of (fav-dog) per-map rating diff
6. veto_pickrate_top1_diff    - fav top-pick map prob - dog top-pick map prob

Plus a separate "veto-weighted" series model:
    p_series_pred = sigmoid( SUM_m pi_m * b_per_map_diff * Δ_m )
where pi_m is the per-map play probability inferred from each side's
historical pick/ban behaviour at the current snapshot.

Output: data/projection_research_phase2_mapveto.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MAPS_DIR = os.path.join(DATA, "maps")

# ── constants ─────────────────────────────────────────────────────────────
BETA_BASELINE = 0.154         # per-rating-point logit coefficient
INTL_COEF = 0.36              # baseline intl_exp_diff coefficient
TRAIN_FRAC = 0.75
N_FEATURES_TESTED = 6
ALPHA_BONF = 0.05 / N_FEATURES_TESTED
N_BOOTSTRAP = 200
VERSATILITY_WINDOW_DAYS = 60
SHIP_BRIER_THRESHOLD = 0.0010

INTL_EVENTS = {
    '2024_masters_madrid', '2024_masters_shanghai', '2024_champions',
    '2025_masters_bangkok', '2025_masters_toronto', '2025_champions',
    '2026_masters_santiago',
}

# Map of EVENT_ID -> snap_id used to attach a snapshot to each match.  We pick
# the snapshot that ends BEFORE the match was played: usually the "before_*"
# snapshot for the same event, or the "after_*" snapshot of the previous one.
# We compute this dynamically from ref_dates instead of hard-coding.


# ── small helpers ────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def safe_logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def brier(probs, outcomes):
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def fit_logit_newton(X, y, max_iter=200, tol=1e-8, ridge=1e-6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        z = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        grad = X.T @ (y - p)
        H = -(X.T * W) @ X - ridge * np.eye(k)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    z = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) +
                      (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
    W = p * (1 - p)
    H = (X.T * W) @ X + ridge * np.eye(k)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    return beta, cov, ll


def predict_logit(X, beta):
    z = np.clip(np.asarray(X) @ np.asarray(beta), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def platt_slope_on(test_logits, y_test):
    """Single-parameter Platt slope: fit Logit(y) = b * logit(p_pred)
    (no intercept) on the test set itself — this is the 'isotonic-by-slope'
    diagnostic the task asks for."""
    X = np.asarray(test_logits, dtype=float).reshape(-1, 1)
    beta, _, _ = fit_logit_newton(X, np.asarray(y_test, dtype=float))
    return float(beta[0])


# ── data loading ─────────────────────────────────────────────────────────
def load_all_matches():
    files = [
        ("rating_timeline_2024.json", 2024),
        ("rating_timeline_2025.json", 2025),
        ("rating_timeline.json", 2026),
    ]
    out = []
    for fname, season in files:
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            print(f"[warn] missing {path}", file=sys.stderr)
            continue
        with open(path) as f:
            d = json.load(f)
        for m in d.get("match_events", []):
            mm = dict(m)
            mm["season"] = season
            out.append(mm)
    out.sort(key=lambda r: (r["date"], r["match_id"]))
    return out


def load_map_ratings():
    with open(os.path.join(DATA, "map_ratings.json")) as f:
        return json.load(f)


def load_veto_model():
    with open(os.path.join(DATA, "veto_model.json")) as f:
        return json.load(f)


# Build a sorted (ref_date, year, snap_id, teams_dict) table for fast lookup.
def build_snap_index(map_ratings):
    rows = []
    for y, yd in map_ratings.get("ratings", {}).items():
        for sid, snap in yd.get("snapshots", {}).items():
            ref = snap.get("ref_date")
            if not ref:
                continue
            rows.append({
                "year": y,
                "snap_id": sid,
                "key": f"{y}_{sid}",
                "ref_date": datetime.strptime(ref, "%Y-%m-%d").date(),
                "teams": snap.get("teams", {}),
            })
    rows.sort(key=lambda r: r["ref_date"])
    return rows


def snap_for_match(snap_index, match_date):
    """Return the most recent snapshot whose ref_date <= match_date.
    If no such snapshot exists (very-early matches), return the earliest one."""
    candidates = [s for s in snap_index if s["ref_date"] <= match_date]
    if not candidates:
        return snap_index[0]
    return candidates[-1]


# ── feature engineering ──────────────────────────────────────────────────
def compute_map_pool(snap, veto_model):
    """Return list of maps in the active map pool at this snapshot.
    Prefer the veto_model snap_pools key (which mirrors official pool) and
    fall back to whatever maps both teams have rating entries for."""
    pool = veto_model.get("snap_pools", {}).get(snap["key"])
    if pool:
        return list(pool)
    # fallback: union of maps appearing in any team's map-ratings at snap
    s = set()
    for org, t in snap["teams"].items():
        s.update(t.get("maps", {}).keys())
    return sorted(s)


def team_map_rating(snap, org, mapname):
    """Returns float rating or None."""
    t = snap["teams"].get(org)
    if not t:
        return None
    m = t.get("maps", {}).get(mapname)
    if not m:
        return None
    r = m.get("rating")
    return float(r) if r is not None else None


def top_n_maps(snap, org, pool, n=3):
    """Return list of (map_name, rating) for org's top-n maps within pool."""
    t = snap["teams"].get(org)
    if not t:
        return []
    out = []
    for mname in pool:
        r = team_map_rating(snap, org, mname)
        if r is not None:
            out.append((mname, r))
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out[:n]


def per_map_diff_pool(snap, fav, dog, pool):
    """Return dict mapname -> (fav_rating - dog_rating) only when BOTH defined."""
    out = {}
    for m in pool:
        rf = team_map_rating(snap, fav, m)
        rd = team_map_rating(snap, dog, m)
        if rf is not None and rd is not None:
            out[m] = rf - rd
    return out


def veto_play_distribution(veto_model, snap_key, fav, dog, pool):
    """Approximate per-map play probability pi_m for the upcoming series.

    Heuristic:
      pi_m  ∝  (1 − ban_fav_m) * (1 − ban_dog_m)
               * (pick_fav_m + pick_dog_m + epsilon)

    The product of (1−ban) reflects "neither team banned this map";
    the sum of picks reflects "at least one side likes to pick it".
    Normalises over the pool.  Returns None if a side has no data."""
    teams = veto_model.get("teams", {}).get(snap_key, {})
    fav_v = teams.get(fav)
    dog_v = teams.get(dog)
    if not fav_v or not dog_v:
        return None
    bans_f = fav_v.get("bans", {})
    bans_d = dog_v.get("bans", {})
    picks_f = fav_v.get("picks", {})
    picks_d = dog_v.get("picks", {})
    scores = {}
    for m in pool:
        bf = float(bans_f.get(m, 0.0))
        bd = float(bans_d.get(m, 0.0))
        pf = float(picks_f.get(m, 0.0))
        pd_ = float(picks_d.get(m, 0.0))
        s = max(1e-6, (1.0 - bf)) * max(1e-6, (1.0 - bd)) * (pf + pd_ + 0.05)
        scores[m] = s
    tot = sum(scores.values())
    if tot <= 0:
        return None
    return {m: scores[m] / tot for m in scores}


# ── main feature table builder ───────────────────────────────────────────
def build_feature_table(matches, snap_index, veto_model):
    """Walk through matches in date order, building per-series features.
    Versatility uses a sliding 60-day window of past maps per team."""

    # Per-team chronological list of (date, set_of_maps_in_series).
    team_map_log = defaultdict(list)  # org -> list[(date, [mapnames])]

    # Track intl_this_year for the baseline intl_exp_diff feature
    intl_this_year = defaultdict(set)

    rows = []
    for m in matches:
        try:
            dt = datetime.strptime(m["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        wb = float(m["winner_before"])
        lb = float(m["loser_before"])
        if wb == lb:
            continue  # exclude ties
        if wb > lb:
            fav, dog, fav_b, dog_b, fav_won = m["winner"], m["loser"], wb, lb, 1
        else:
            fav, dog, fav_b, dog_b, fav_won = m["loser"], m["winner"], lb, wb, 0
        rating_diff = fav_b - dog_b

        # baseline intl_exp_diff (BEFORE this match)
        year = dt.year
        fav_intl = 1 if fav in intl_this_year[year] else 0
        dog_intl = 1 if dog in intl_this_year[year] else 0
        intl_exp_diff = fav_intl - dog_intl

        # baseline series p
        logit_base = BETA_BASELINE * rating_diff + INTL_COEF * intl_exp_diff
        p_base = float(sigmoid(logit_base))

        # snapshot for map/veto features
        snap = snap_for_match(snap_index, dt)
        pool = compute_map_pool(snap, veto_model)

        # ---- feature 1: map-pool versatility (past 60 days) ----
        cutoff = dt - timedelta(days=VERSATILITY_WINDOW_DAYS)

        def versatility(org):
            log = team_map_log.get(org, [])
            seen = set()
            for d_, maps in log:
                if d_ >= cutoff and d_ < dt:
                    seen.update(maps)
            return len(seen) if log else None

        fav_vers = versatility(fav)
        dog_vers = versatility(dog)
        vers_diff = (fav_vers - dog_vers) if (fav_vers is not None and dog_vers is not None) else None

        # ---- feature 2/3: strong-map overlap + fav_top1 vs dog_top1 ----
        fav_top3 = top_n_maps(snap, fav, pool, n=3)
        dog_top3 = top_n_maps(snap, dog, pool, n=3)
        if fav_top3 and dog_top3:
            fav_names = {x[0] for x in fav_top3}
            dog_names = {x[0] for x in dog_top3}
            overlap = len(fav_names & dog_names)
            fav_top1 = fav_top3[0][1]
            dog_top1 = dog_top3[0][1]
            top1_diff = fav_top1 - dog_top1
        else:
            overlap = None
            top1_diff = None

        # ---- features 4/5: best/worst case per-map gap over pool ----
        gaps = per_map_diff_pool(snap, fav, dog, pool)
        if gaps:
            best_case = max(gaps.values())
            worst_case = min(gaps.values())
        else:
            best_case = None
            worst_case = None

        # ---- feature 6: top-pick map rating advantage (veto-informed) ----
        veto_dist = veto_play_distribution(veto_model, snap["key"], fav, dog, pool)
        if veto_dist:
            # Each side's "most-picked" map under the joint distribution: take
            # fav's own pick-rate top map and dog's own pick-rate top map;
            # report (fav rating on fav-top-pick) − (dog rating on dog-top-pick)
            teams_v = veto_model.get("teams", {}).get(snap["key"], {})
            fav_picks = teams_v.get(fav, {}).get("picks", {})
            dog_picks = teams_v.get(dog, {}).get("picks", {})
            fav_pick_top = max(pool, key=lambda mm: fav_picks.get(mm, 0.0), default=None) if fav_picks else None
            dog_pick_top = max(pool, key=lambda mm: dog_picks.get(mm, 0.0), default=None) if dog_picks else None
            r_fav_on_top = team_map_rating(snap, fav, fav_pick_top) if fav_pick_top else None
            r_dog_on_top = team_map_rating(snap, dog, dog_pick_top) if dog_pick_top else None
            if r_fav_on_top is not None and r_dog_on_top is not None:
                pickrate_top1_diff = r_fav_on_top - r_dog_on_top
            else:
                pickrate_top1_diff = None
        else:
            pickrate_top1_diff = None

        # ---- veto-weighted series prediction (for separate model) ----
        if veto_dist and gaps:
            # Build per-map win prob on each map then mix with veto_dist.
            # Per-map win prob uses the per-map rating diff (Δ_m) with the
            # SAME beta the baseline uses (0.154) — this isolates the lift
            # from going "uniform p_map -> map-aware p_map".
            mix = 0.0
            tot_pi = 0.0
            for mp, pi in veto_dist.items():
                if mp not in gaps:
                    continue
                # Using Pythagorean-like map win prob: sigmoid(BETA * Δ_m)
                p_map = float(sigmoid(BETA_BASELINE * gaps[mp]))
                mix += pi * p_map
                tot_pi += pi
            if tot_pi > 0:
                p_map_avg = mix / tot_pi
                # bo3-equivalent series win prob from per-map mean
                # P(series win) = p^2 + 2*p^2*(1-p) = p^2*(3-2p)
                p_veto_series = p_map_avg ** 2 * (3 - 2 * p_map_avg)
            else:
                p_veto_series = None
        else:
            p_veto_series = None

        rows.append({
            "match_id": m["match_id"],
            "date": m["date"],
            "event_id": m["event_id"],
            "season": m["season"],
            "fav": fav, "dog": dog,
            "fav_before": fav_b, "dog_before": dog_b,
            "rating_diff": rating_diff,
            "intl_exp_diff": intl_exp_diff,
            "p_base": p_base,
            "y": fav_won,
            "snap_key": snap["key"],
            "pool_size": len(pool),
            # features
            "mappool_versatility_diff": vers_diff,
            "strong_map_overlap": overlap,
            "fav_top1_vs_dog_top1": top1_diff,
            "best_case_gap": best_case,
            "worst_case_gap": worst_case,
            "veto_pickrate_top1_diff": pickrate_top1_diff,
            "p_veto_series": p_veto_series,
        })

        # update team_map_log AFTER computing features (pre-match snapshot)
        match_maps = [mm.get("map") for mm in (m.get("maps") or []) if mm.get("map")]
        if match_maps:
            team_map_log[fav].append((dt, list(match_maps)))
            team_map_log[dog].append((dt, list(match_maps)))
        # intl_this_year update (this match contributes if intl event)
        if m["event_id"] in INTL_EVENTS:
            intl_this_year[year].add(m["winner"])
            intl_this_year[year].add(m["loser"])

    return rows


# ── evaluation ───────────────────────────────────────────────────────────
def split_train_test(rows, train_frac=TRAIN_FRAC):
    rows_sorted = sorted(rows, key=lambda r: (r["date"], r["match_id"]))
    cut = int(round(len(rows_sorted) * train_frac))
    return rows_sorted[:cut], rows_sorted[cut:]


def baseline_eval(train, test):
    """Fit alpha + b * logit(p_base) on TRAIN, evaluate Brier + Platt on TEST."""
    yt = np.array([r["y"] for r in train], dtype=float)
    yte = np.array([r["y"] for r in test], dtype=float)
    lt = np.array([safe_logit(r["p_base"]) for r in train])
    lte = np.array([safe_logit(r["p_base"]) for r in test])
    X_tr = np.column_stack([np.ones(len(yt)), lt])
    b, _, ll = fit_logit_newton(X_tr, yt)
    X_te = np.column_stack([np.ones(len(yte)), lte])
    p_test = predict_logit(X_te, b)
    br = brier(p_test, yte)
    slope = platt_slope_on(safe_logit(p_test), yte)
    return {
        "beta": b.tolist(), "train_ll": ll,
        "test_brier": float(br), "test_platt_b": float(slope),
        "n_train": len(train), "n_test": len(test),
        "p_test": p_test,
    }


def feature_eval(train, test, feat_name, baseline_train_ll):
    """Fit baseline + feature on TRAIN, evaluate on TEST.  Returns metrics."""
    # Restrict to rows where feature is defined in BOTH train and test
    tr_sub = [r for r in train if r.get(feat_name) is not None]
    te_sub = [r for r in test if r.get(feat_name) is not None]
    n_used_train = len(tr_sub)
    n_used_test = len(te_sub)
    if n_used_train < 30 or n_used_test < 30:
        return {"name": feat_name, "n_used_train": n_used_train,
                "n_used_test": n_used_test, "error": "too few samples"}

    y_tr = np.array([r["y"] for r in tr_sub], dtype=float)
    y_te = np.array([r["y"] for r in te_sub], dtype=float)
    lt_tr = np.array([safe_logit(r["p_base"]) for r in tr_sub])
    lt_te = np.array([safe_logit(r["p_base"]) for r in te_sub])
    ft_tr = np.array([float(r[feat_name]) for r in tr_sub])
    ft_te = np.array([float(r[feat_name]) for r in te_sub])

    # baseline-only on same train subset (for LR test and same-row brier ref)
    X0_tr = np.column_stack([np.ones(len(y_tr)), lt_tr])
    b0, _, ll0 = fit_logit_newton(X0_tr, y_tr)
    X0_te = np.column_stack([np.ones(len(y_te)), lt_te])
    p0_te = predict_logit(X0_te, b0)
    brier0_te = brier(p0_te, y_te)

    X1_tr = np.column_stack([np.ones(len(y_tr)), lt_tr, ft_tr])
    b1, cov1, ll1 = fit_logit_newton(X1_tr, y_tr)
    X1_te = np.column_stack([np.ones(len(y_te)), lt_te, ft_te])
    p1_te = predict_logit(X1_te, b1)
    brier1_te = brier(p1_te, y_te)

    coef = float(b1[2])
    se = float(np.sqrt(max(cov1[2, 2], 0.0)))
    z = coef / se if se > 0 else 0.0
    p_wald = float(2 * (1 - stats.norm.cdf(abs(z))))

    lr_stat = 2 * (ll1 - ll0)
    lr_p = float(1 - stats.chi2.cdf(max(lr_stat, 0), df=1))

    platt = platt_slope_on(safe_logit(p1_te), y_te)
    delta_brier = brier0_te - brier1_te

    # Bootstrap CI on Δ-Brier over TEST set
    rng = np.random.default_rng(42)
    boot = []
    n_te = len(y_te)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_te, n_te)
        d = brier(p0_te[idx], y_te[idx]) - brier(p1_te[idx], y_te[idx])
        boot.append(d)
    boot_arr = np.array(boot)
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    pct_pos = float((boot_arr > 0).mean())

    ship = bool(
        delta_brier >= SHIP_BRIER_THRESHOLD
        and 0.85 <= platt <= 1.15
        and lr_p < ALPHA_BONF
        and ci_lo > 0.0
    )

    notes_bits = []
    if delta_brier < SHIP_BRIER_THRESHOLD:
        notes_bits.append(f"ΔBrier {delta_brier:+.5f} < {SHIP_BRIER_THRESHOLD:+.4f}")
    if not (0.85 <= platt <= 1.15):
        notes_bits.append(f"slope {platt:.3f} outside [0.85,1.15]")
    if lr_p >= ALPHA_BONF:
        notes_bits.append(f"LR p={lr_p:.4f} ≥ {ALPHA_BONF:.4f}")
    if ci_lo <= 0:
        notes_bits.append(f"boot CI lo={ci_lo:+.5f} ≤ 0")
    notes = "; ".join(notes_bits) if notes_bits else "passes all gates"

    return {
        "name": feat_name,
        "n_used_train": n_used_train,
        "n_used_test": n_used_test,
        "coef": coef,
        "se": se,
        "p": p_wald,
        "test_brier": float(brier1_te),
        "test_brier_baseline_on_subset": float(brier0_te),
        "delta_test_brier": float(delta_brier),
        "test_platt_b": float(platt),
        "lr_stat": float(lr_stat),
        "lr_p": lr_p,
        "boot_ci_lo": ci_lo,
        "boot_ci_hi": ci_hi,
        "boot_pct_positive": pct_pos,
        "ship_worthy": ship,
        "notes": notes,
    }


def veto_weighted_eval(train, test, baseline_train_ll):
    """Evaluate the standalone veto-weighted series predictor:
    Fit alpha + b * logit(p_veto_series) on train, evaluate on test."""
    tr_sub = [r for r in train if r.get("p_veto_series") is not None]
    te_sub = [r for r in test if r.get("p_veto_series") is not None]
    if len(tr_sub) < 30 or len(te_sub) < 30:
        return {"error": "too few samples",
                "n_used_train": len(tr_sub), "n_used_test": len(te_sub)}
    y_tr = np.array([r["y"] for r in tr_sub], dtype=float)
    y_te = np.array([r["y"] for r in te_sub], dtype=float)
    lt_tr = np.array([safe_logit(r["p_veto_series"]) for r in tr_sub])
    lt_te = np.array([safe_logit(r["p_veto_series"]) for r in te_sub])

    X_tr = np.column_stack([np.ones(len(y_tr)), lt_tr])
    b, _, _ = fit_logit_newton(X_tr, y_tr)
    X_te = np.column_stack([np.ones(len(y_te)), lt_te])
    p_te = predict_logit(X_te, b)
    br = brier(p_te, y_te)
    slope = platt_slope_on(safe_logit(p_te), y_te)

    # baseline-only on the same test subset for fair comparison
    lt_te_base = np.array([safe_logit(r["p_base"]) for r in te_sub])
    lt_tr_base = np.array([safe_logit(r["p_base"]) for r in tr_sub])
    Xb_tr = np.column_stack([np.ones(len(y_tr)), lt_tr_base])
    bb, _, _ = fit_logit_newton(Xb_tr, y_tr)
    Xb_te = np.column_stack([np.ones(len(y_te)), lt_te_base])
    pb_te = predict_logit(Xb_te, bb)
    br_base = brier(pb_te, y_te)

    return {
        "n_used_train": len(tr_sub), "n_used_test": len(te_sub),
        "test_brier": float(br),
        "test_brier_baseline_on_subset": float(br_base),
        "delta_test_brier": float(br_base - br),
        "test_platt_b": float(slope),
        "notes": ("Mixes per-map Δ ratings by veto-implied play probability, "
                  "then maps to series via P=p^2*(3-2p). "
                  "Compared to baseline fit on same subset."),
    }


# ── main ─────────────────────────────────────────────────────────────────
def main():
    matches = load_all_matches()
    print(f"Loaded {len(matches)} matches.")
    snap_index = build_snap_index(load_map_ratings())
    print(f"Loaded {len(snap_index)} map-rating snapshots.")
    veto_model = load_veto_model()
    print(f"Loaded veto_model with {len(veto_model.get('teams', {}))} team snapshots.")

    rows = build_feature_table(matches, snap_index, veto_model)
    print(f"Built feature table: {len(rows)} non-tied series.")

    # Verify baseline brier matches the spec (~0.2301)
    y_all = np.array([r["y"] for r in rows], dtype=float)
    p_all = np.array([r["p_base"] for r in rows], dtype=float)
    full_baseline_brier = brier(p_all, y_all)
    print(f"Full-sample raw baseline Brier (uncalibrated): {full_baseline_brier:.4f}")

    train, test = split_train_test(rows)
    base = baseline_eval(train, test)
    print(f"Time split: n_train={base['n_train']}, n_test={base['n_test']}")
    print(f"Baseline on test: Brier={base['test_brier']:.4f}  Platt slope={base['test_platt_b']:.3f}")

    feat_names = [
        "mappool_versatility_diff",
        "strong_map_overlap",
        "fav_top1_vs_dog_top1",
        "best_case_gap",
        "worst_case_gap",
        "veto_pickrate_top1_diff",
    ]

    results = []
    for fn in feat_names:
        res = feature_eval(train, test, fn, base["train_ll"])
        results.append(res)
        if "error" in res:
            print(f"  {fn:<26} n_train={res['n_used_train']} ERROR {res['error']}")
        else:
            mark = "*SHIP*" if res["ship_worthy"] else "      "
            print(f"  {fn:<26} {mark} ΔBrier={res['delta_test_brier']:+.5f}  "
                  f"slope={res['test_platt_b']:.3f}  LRp={res['lr_p']:.4f}  "
                  f"bootCI=[{res['boot_ci_lo']:+.5f},{res['boot_ci_hi']:+.5f}]")

    veto_w = veto_weighted_eval(train, test, base["train_ll"])
    if "error" in veto_w:
        print(f"Veto-weighted model: ERROR {veto_w['error']}")
    else:
        print(f"Veto-weighted model: Brier={veto_w['test_brier']:.4f}  "
              f"vs baseline-on-subset {veto_w['test_brier_baseline_on_subset']:.4f}  "
              f"(Δ {veto_w['delta_test_brier']:+.5f})  slope={veto_w['test_platt_b']:.3f}")

    # ----- verdict / headline -----
    ship_features = [r for r in results if r.get("ship_worthy")]
    interesting = [r for r in results
                   if "error" not in r and r["delta_test_brier"] >= SHIP_BRIER_THRESHOLD]

    if ship_features:
        best = max(ship_features, key=lambda r: r["delta_test_brier"])
        verdict = f"ship: {best['name']}"
        headline = (f"Ship-worthy: {best['name']} ΔBrier={best['delta_test_brier']:+.5f} "
                    f"(LRp={best['lr_p']:.4f}, slope={best['test_platt_b']:.3f}, "
                    f"bootCI [{best['boot_ci_lo']:+.5f},{best['boot_ci_hi']:+.5f}])")
    elif interesting:
        best = max(interesting, key=lambda r: r["delta_test_brier"])
        verdict = "marginal"
        headline = (f"Marginal: best is {best['name']} ΔBrier={best['delta_test_brier']:+.5f} "
                    f"but fails gates ({best['notes']}).")
    else:
        verdict = "no signal"
        # report best ΔBrier even when below threshold for context
        valid = [r for r in results if "error" not in r]
        if valid:
            best = max(valid, key=lambda r: r["delta_test_brier"])
            headline = (f"No signal: best feature {best['name']} only "
                        f"ΔBrier={best['delta_test_brier']:+.5f} on held-out test.")
        else:
            headline = "No signal: no feature could be evaluated."

    out = {
        "n_series": len(rows),
        "n_train": base["n_train"],
        "n_test": base["n_test"],
        "train_frac": TRAIN_FRAC,
        "alpha_bonferroni": ALPHA_BONF,
        "ship_brier_threshold": SHIP_BRIER_THRESHOLD,
        "raw_full_baseline_brier_unfit": float(full_baseline_brier),
        "baseline": {
            "test_brier": float(base["test_brier"]),
            "test_platt_b": float(base["test_platt_b"]),
        },
        "features": [
            {k: v for k, v in r.items()}
            for r in results
        ],
        "veto_weighted_model": veto_w,
        "headline": headline,
        "verdict": verdict,
        "generated": datetime.utcnow().isoformat() + "Z",
    }

    out_path = os.path.join(DATA, "projection_research_phase2_mapveto.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print(f"HEADLINE: {headline}")
    print(f"VERDICT:  {verdict}")


if __name__ == "__main__":
    main()
