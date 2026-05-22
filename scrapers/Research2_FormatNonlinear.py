"""
Research2_FormatNonlinear.py
============================

Tests whether non-linear transforms of the rating differential OR
format-aware (bo3 vs bo5) modeling can lower the series Brier below the
current baseline:

    p_series = sigmoid(0.154 * Δrating + 0.36 * intl_exp_diff_term)

with bo3/bo5 closed-form mapping from per-map probability.

Baseline reported:
  Series Brier: 0.2301  (n=1217, 2024 → 2026 stage 1)
  Platt slope (series): 0.914

Methodology:
  - Time-based split: first 75% (train) / last 25% (test).
  - 5-fold time-based CV for likelihood ratio tests where applicable.
  - Cluster bootstrap (series resample) 200 reps for CIs on format-specific β.
  - Bonferroni-corrected α = 0.05 / 6.

Outputs:
  data/projection_research_phase2_format.json
  Console summary.

Run:
  .venv/bin/python scrapers/Research2_FormatNonlinear.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

SEED = 0
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# ── Constants (mirror AnalyzeProjectionCalibration.py) ─────────────────────
BETA_BASELINE = 0.154
INTL_BONUS_BASELINE = 0.36

INTL_EVENTS = {
    "2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
    "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
    "2026_masters_santiago",
}

TRAIN_FRAC = 0.75
N_BOOT = 200
N_CV_FOLDS = 5
BONFERRONI_ALPHA = 0.05 / 6.0
SHIP_DELTA_BRIER = 0.0010


# ── Helpers ────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))


def safe_logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def series_prob_from_map(p, bo):
    """Closed-form prob favorite wins series given per-map p."""
    if bo == 3:
        return (p ** 2) * (3 - 2 * p)
    if bo == 5:
        return (p ** 3) * (10 - 15 * p + 6 * p * p)
    return p


def series_prob_from_map_vec(p, bo):
    """Vectorized variant."""
    p = np.asarray(p, dtype=float)
    bo = np.asarray(bo)
    out = np.empty_like(p)
    bo3 = bo == 3
    bo5 = bo == 5
    out[bo3] = (p[bo3] ** 2) * (3 - 2 * p[bo3])
    pp = p[bo5]
    out[bo5] = (pp ** 3) * (10 - 15 * pp + 6 * pp * pp)
    other = ~(bo3 | bo5)
    out[other] = p[other]
    return out


def infer_bo(series_score, n_maps):
    if not series_score:
        return 3 if n_maps <= 3 else 5
    try:
        a, b = series_score.split("-")
        a, b = int(a), int(b)
        m = max(a, b)
        if m >= 3:
            return 5
        if m == 2:
            return 3
    except Exception:
        pass
    return 3 if n_maps <= 3 else 5


def brier(probs, outcomes):
    if len(probs) == 0:
        return float("nan")
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def platt_fit(probs, outcomes):
    """Logistic regression of outcomes ~ a + b * logit(p) by MLE."""
    if len(probs) < 30:
        return {"a": float("nan"), "b": float("nan"), "n": len(probs)}
    x = safe_logit(np.asarray(probs))
    y = np.asarray(outcomes, dtype=float)

    def nll(params):
        a, b = params
        z = a + b * x
        z = np.clip(z, -25, 25)
        # binary cross entropy
        return float(np.sum(np.logaddexp(0, -z) + (1 - y) * z))

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
    a, b = res.x
    return {"a": float(a), "b": float(b), "n": int(len(x))}


# ── Load data ──────────────────────────────────────────────────────────────
def load_matches():
    files = [
        ("rating_timeline_2024.json", 2024),
        ("rating_timeline_2025.json", 2025),
        ("rating_timeline.json",      2026),
    ]
    out = []
    for fname, season in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[warn] missing {path}", file=sys.stderr)
            continue
        with open(path) as f:
            d = json.load(f)
        for m in d.get("match_events", []):
            wb = float(m.get("winner_before", 0.0))
            lb = float(m.get("loser_before", 0.0))
            if wb == lb:
                continue
            m = dict(m)
            m["season"] = season
            out.append(m)
    out.sort(key=lambda r: (r["date"], r["match_id"]))
    return out


def _build_intl_attendance(matches):
    attendance = {}
    for m in sorted(matches, key=lambda r: (r["date"], r["match_id"])):
        if m["event_id"] not in INTL_EVENTS:
            continue
        season = m["season"]
        for org in (m["winner"], m["loser"]):
            attendance.setdefault((org, season), []).append((m["date"], m["event_id"]))
    return attendance


def _intl_exp_diff(fav_org, dog_org, season, match_date, attendance):
    def attended_before(org):
        for d, _ in attendance.get((org, season), []):
            if d < match_date:
                return True
        return False
    f = attended_before(fav_org)
    d = attended_before(dog_org)
    return (1 if f else 0) - (1 if d else 0)


def build_series_rows(matches):
    """Build series-level rows; signed delta = fav_rating - dog_rating (>0)."""
    attendance = _build_intl_attendance(matches)
    rows = []
    for m in matches:
        wb = float(m["winner_before"])
        lb = float(m["loser_before"])
        abs_delta = abs(wb - lb)
        fav_won_series = 1 if wb > lb else 0
        if wb > lb:
            fav_org, dog_org = m["winner"], m["loser"]
        else:
            fav_org, dog_org = m["loser"], m["winner"]
        n_maps = len(m.get("maps", []))
        bo = infer_bo(m.get("series_score", ""), n_maps)
        intl_exp_diff = _intl_exp_diff(
            fav_org, dog_org, m["season"], m["date"], attendance
        )
        rows.append({
            "match_id":      m["match_id"],
            "date":          m["date"],
            "event_id":      m["event_id"],
            "season":        m["season"],
            "intl":          m["event_id"] in INTL_EVENTS,
            "bo":            bo,
            "is_bo5":        1 if bo == 5 else 0,
            "abs_delta":     abs_delta,
            "y":             fav_won_series,
            "intl_exp_diff": intl_exp_diff,
            "fav_org":       fav_org,
            "dog_org":       dog_org,
        })
    rows.sort(key=lambda r: (r["date"], r["match_id"]))
    return rows


# ── Train/test split (time-based) ──────────────────────────────────────────
def time_split(rows, train_frac=TRAIN_FRAC):
    n = len(rows)
    cut = int(round(n * train_frac))
    return rows[:cut], rows[cut:]


# ── Baseline series probability ────────────────────────────────────────────
def baseline_series_p(rows, beta=BETA_BASELINE, intl_bonus=INTL_BONUS_BASELINE):
    """Returns array of baseline p_series for each row."""
    probs = []
    for r in rows:
        p_map = float(sigmoid(beta * r["abs_delta"]))
        p_ser = series_prob_from_map(p_map, r["bo"])
        if intl_bonus != 0 and r["intl_exp_diff"] != 0:
            ps = max(min(p_ser, 1 - 1e-9), 1e-9)
            z = math.log(ps / (1 - ps)) + intl_bonus * r["intl_exp_diff"]
            p_ser = float(sigmoid(z))
        probs.append(p_ser)
    return np.array(probs)


# ── Generic MLE model on series outcomes ───────────────────────────────────
def fit_logistic(X, y):
    """Fit logistic regression by MLE (no intercept added automatically)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    k = X.shape[1]

    def nll(params):
        z = X @ params
        z = np.clip(z, -25, 25)
        return float(np.sum(np.logaddexp(0, -z) + (1 - y) * z))

    res = minimize(nll, x0=np.zeros(k), method="L-BFGS-B")
    return res.x, float(res.fun)


def neg_log_lik_series(probs, y):
    """Series-level cross-entropy in nats."""
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


# ── Experiment 1: Format-specific β ────────────────────────────────────────
def fit_format_betas(rows, init=(BETA_BASELINE, BETA_BASELINE),
                     intl_bonus=INTL_BONUS_BASELINE):
    """Fit β_bo3, β_bo5 such that
        p_map = sigmoid(β_bo * |Δ|)
        p_ser = closed_form(p_map, bo)  (+ intl logit offset)
    minimizing series NLL."""
    abs_d = np.array([r["abs_delta"] for r in rows])
    bo    = np.array([r["bo"] for r in rows])
    y     = np.array([r["y"] for r in rows], dtype=float)
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)

    def nll(params):
        b3, b5 = params
        p_map = np.where(bo == 5, sigmoid(b5 * abs_d), sigmoid(b3 * abs_d))
        p_ser = series_prob_from_map_vec(p_map, bo)
        if intl_bonus != 0:
            z = safe_logit(p_ser) + intl_bonus * intl_diff
            p_ser = sigmoid(z)
        p_ser = np.clip(p_ser, 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p_ser) + (1 - y) * np.log(1 - p_ser)))

    res = minimize(nll, x0=np.array(init), method="L-BFGS-B",
                   bounds=[(0.001, 1.0), (0.001, 1.0)])
    return res.x, float(res.fun)


def fit_single_beta(rows, init=BETA_BASELINE,
                    intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    bo    = np.array([r["bo"] for r in rows])
    y     = np.array([r["y"] for r in rows], dtype=float)
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)

    def nll(params):
        b = params[0]
        p_map = sigmoid(b * abs_d)
        p_ser = series_prob_from_map_vec(p_map, bo)
        if intl_bonus != 0:
            z = safe_logit(p_ser) + intl_bonus * intl_diff
            p_ser = sigmoid(z)
        p_ser = np.clip(p_ser, 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p_ser) + (1 - y) * np.log(1 - p_ser)))

    res = minimize(nll, x0=np.array([init]), method="L-BFGS-B",
                   bounds=[(0.001, 1.0)])
    return float(res.x[0]), float(res.fun)


def predict_format_betas(rows, b3, b5, intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    bo    = np.array([r["bo"] for r in rows])
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    p_map = np.where(bo == 5, sigmoid(b5 * abs_d), sigmoid(b3 * abs_d))
    p_ser = series_prob_from_map_vec(p_map, bo)
    if intl_bonus != 0:
        z = safe_logit(p_ser) + intl_bonus * intl_diff
        p_ser = sigmoid(z)
    return np.clip(p_ser, 1e-12, 1 - 1e-12)


def predict_single_beta(rows, b, intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    bo    = np.array([r["bo"] for r in rows])
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    p_map = sigmoid(b * abs_d)
    p_ser = series_prob_from_map_vec(p_map, bo)
    if intl_bonus != 0:
        z = safe_logit(p_ser) + intl_bonus * intl_diff
        p_ser = sigmoid(z)
    return np.clip(p_ser, 1e-12, 1 - 1e-12)


# ── Experiment 2: Non-linear in rating diff (logit-space, direct on series) ─
# We fit a series-level logistic regression on signed features of |Δ| plus
# the intl_exp_diff covariate. The "linear-Δ" reference here is direct-fit
# logistic on (1, |Δ|, intl_exp_diff). We compare against richer feature sets
# (Δ², Δ³, splines) by LR test.

def make_features_linear(rows):
    abs_d = np.array([r["abs_delta"] for r in rows])
    intl  = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    X = np.column_stack([np.ones(len(rows)), abs_d, intl])
    return X


def make_features_poly(rows, degree=2):
    abs_d = np.array([r["abs_delta"] for r in rows])
    intl  = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    cols = [np.ones(len(rows)), abs_d]
    for d in range(2, degree + 1):
        cols.append(abs_d ** d)
    cols.append(intl)
    return np.column_stack(cols)


def make_features_spline(rows, knots=(2.0, 4.0)):
    """Piecewise-linear: knots are at |Δ| = 2 and 4."""
    abs_d = np.array([r["abs_delta"] for r in rows])
    intl  = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    cols = [np.ones(len(rows)), abs_d]
    for k in knots:
        cols.append(np.maximum(abs_d - k, 0))
    cols.append(intl)
    return np.column_stack(cols)


# Cauchy link: p = 0.5 + (1/π) arctan(β · Δ + intl_bonus * intl_diff)
def fit_cauchy(rows, intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    y     = np.array([r["y"] for r in rows], dtype=float)
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)

    def nll(params):
        b = params[0]
        z = b * abs_d + intl_bonus * intl_diff
        p = 0.5 + (1.0 / np.pi) * np.arctan(z)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize(nll, x0=np.array([0.5]), method="L-BFGS-B",
                   bounds=[(0.001, 5.0)])
    return float(res.x[0]), float(res.fun)


def predict_cauchy(rows, b, intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    z = b * abs_d + intl_bonus * intl_diff
    p = 0.5 + (1.0 / np.pi) * np.arctan(z)
    return np.clip(p, 1e-12, 1 - 1e-12)


# Probit link: p = Φ(β · Δ + intl_bonus * intl_diff)
def fit_probit(rows, intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    y     = np.array([r["y"] for r in rows], dtype=float)
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)

    def nll(params):
        b = params[0]
        z = b * abs_d + intl_bonus * intl_diff
        p = norm.cdf(z)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize(nll, x0=np.array([0.3]), method="L-BFGS-B",
                   bounds=[(0.001, 5.0)])
    return float(res.x[0]), float(res.fun)


def predict_probit(rows, b, intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    z = b * abs_d + intl_bonus * intl_diff
    p = norm.cdf(z)
    return np.clip(p, 1e-12, 1 - 1e-12)


# ── Experiment 3: correlated-maps closed form ──────────────────────────────
# p_bo_corr = α + (1-α) * f_bo(p), where f_bo is standard binomial closed form.
# Here α is a probability-space inflation of the favorite's series prob.
# Equivalent re-param: convex combination toward 1 (so "ties favor favorite").

def fit_correlated_closed_form(rows, init=(BETA_BASELINE, 0.0, 0.0),
                               intl_bonus=INTL_BONUS_BASELINE):
    """Fit (β, α3, α5) such that
        p_map = sigmoid(β |Δ|)
        f_bo = standard closed form
        p_ser = α_bo + (1 - α_bo) * f_bo
    Then apply intl_bonus in logit space.
    α can be ∈ [-0.5, 0.5] (negative = favorites less inflated than independent)."""
    abs_d = np.array([r["abs_delta"] for r in rows])
    bo    = np.array([r["bo"] for r in rows])
    y     = np.array([r["y"] for r in rows], dtype=float)
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)

    def nll(params):
        b, a3, a5 = params
        p_map = sigmoid(b * abs_d)
        f = series_prob_from_map_vec(p_map, bo)
        alpha = np.where(bo == 5, a5, a3)
        # convex combination toward 1: p = alpha + (1-alpha) * f  for α >=0
        # for α <0, this becomes negative inflation - so allow but clip later
        p_ser = alpha + (1 - alpha) * f
        p_ser = np.clip(p_ser, 1e-12, 1 - 1e-12)
        if intl_bonus != 0:
            z = safe_logit(p_ser) + intl_bonus * intl_diff
            p_ser = sigmoid(z)
        p_ser = np.clip(p_ser, 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p_ser) + (1 - y) * np.log(1 - p_ser)))

    res = minimize(nll, x0=np.array(init), method="L-BFGS-B",
                   bounds=[(0.001, 1.0), (-0.5, 0.5), (-0.5, 0.5)])
    return res.x, float(res.fun)


def predict_correlated_closed_form(rows, b, a3, a5,
                                   intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    bo    = np.array([r["bo"] for r in rows])
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    p_map = sigmoid(b * abs_d)
    f = series_prob_from_map_vec(p_map, bo)
    alpha = np.where(bo == 5, a5, a3)
    p_ser = alpha + (1 - alpha) * f
    p_ser = np.clip(p_ser, 1e-12, 1 - 1e-12)
    if intl_bonus != 0:
        z = safe_logit(p_ser) + intl_bonus * intl_diff
        p_ser = sigmoid(z)
    return np.clip(p_ser, 1e-12, 1 - 1e-12)


# ── Experiment 4: bo5 intercept shift on series logit ──────────────────────
def fit_bo5_intercept(rows, intl_bonus=INTL_BONUS_BASELINE):
    """Start from closed-form baseline p_ser, then add (a + c*is_bo5) shift
    in logit space (plus intl)."""
    abs_d = np.array([r["abs_delta"] for r in rows])
    bo    = np.array([r["bo"] for r in rows])
    y     = np.array([r["y"] for r in rows], dtype=float)
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    is_bo5 = (bo == 5).astype(float)

    def nll(params):
        b, c = params
        p_map = sigmoid(b * abs_d)
        p_ser = series_prob_from_map_vec(p_map, bo)
        z = safe_logit(p_ser) + intl_bonus * intl_diff + c * is_bo5
        p_ser = sigmoid(z)
        p_ser = np.clip(p_ser, 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p_ser) + (1 - y) * np.log(1 - p_ser)))

    res = minimize(nll, x0=np.array([BETA_BASELINE, 0.0]), method="L-BFGS-B",
                   bounds=[(0.001, 1.0), (-1.5, 1.5)])
    return res.x, float(res.fun)


def predict_bo5_intercept(rows, b, c, intl_bonus=INTL_BONUS_BASELINE):
    abs_d = np.array([r["abs_delta"] for r in rows])
    bo    = np.array([r["bo"] for r in rows])
    intl_diff = np.array([r["intl_exp_diff"] for r in rows], dtype=float)
    is_bo5 = (bo == 5).astype(float)
    p_map = sigmoid(b * abs_d)
    p_ser = series_prob_from_map_vec(p_map, bo)
    z = safe_logit(p_ser) + intl_bonus * intl_diff + c * is_bo5
    return np.clip(sigmoid(z), 1e-12, 1 - 1e-12)


# ── Bootstrap (cluster = series; each row is already a series) ──────────────
def cluster_bootstrap_format_betas(rows, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(rows)
    boots = []
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [rows[j] for j in idx]
        try:
            (b3, b5), _ = fit_format_betas(sample)
            boots.append((b3, b5))
        except Exception:
            continue
    arr = np.array(boots)
    return arr


# ── Main pipeline ──────────────────────────────────────────────────────────
def main():
    matches = load_matches()
    rows = build_series_rows(matches)
    print(f"Loaded {len(matches)} matches → {len(rows)} series rows")

    train, test = time_split(rows, TRAIN_FRAC)
    n_train, n_test = len(train), len(test)
    print(f"Time split: train={n_train}, test={n_test}  "
          f"(cut date={train[-1]['date']} → {test[0]['date']})")

    # ── Baseline ───────────────────────────────────────────────────────────
    p_base_train = baseline_series_p(train)
    p_base_test  = baseline_series_p(test)
    y_train = np.array([r["y"] for r in train], dtype=float)
    y_test  = np.array([r["y"] for r in test],  dtype=float)

    baseline_train_brier = brier(p_base_train, y_train)
    baseline_test_brier  = brier(p_base_test,  y_test)
    baseline_pooled_brier = brier(baseline_series_p(rows),
                                  np.array([r["y"] for r in rows], dtype=float))
    baseline_platt_test  = platt_fit(p_base_test, y_test)
    baseline_nll_train   = neg_log_lik_series(p_base_train, y_train)
    baseline_nll_test    = neg_log_lik_series(p_base_test,  y_test)

    print(f"\nBaseline (β=0.154, intl_bonus=0.36):")
    print(f"  Pooled Brier   : {baseline_pooled_brier:.4f} (n={len(rows)})")
    print(f"  Train Brier    : {baseline_train_brier:.4f}")
    print(f"  Test  Brier    : {baseline_test_brier:.4f}")
    print(f"  Test  Platt b  : {baseline_platt_test['b']:.4f}")

    # Refit single β on training set as the "null" model for LR tests.
    b_null_train, nll_null_train = fit_single_beta(train)
    p_null_test = predict_single_beta(test, b_null_train)
    null_test_brier = brier(p_null_test, y_test)
    print(f"\nNull (single β fit on train): β={b_null_train:.4f}  "
          f"train_nll={nll_null_train:.3f}  test_Brier={null_test_brier:.4f}")

    experiments = []

    def lr_p(nll_null, nll_alt, df):
        stat = 2 * (nll_null - nll_alt)
        if stat <= 0:
            return 1.0
        return float(chi2.sf(stat, df=df))

    def package(name, n_params, p_train, p_test, lr_p_val, notes,
                extra=None):
        tb = brier(p_train, y_train)
        teb = brier(p_test,  y_test)
        platt = platt_fit(p_test, y_test)
        delta_brier = baseline_test_brier - teb
        delta_platt = abs(baseline_platt_test["b"] - 1.0) - abs(platt["b"] - 1.0)
        ship = (delta_brier >= SHIP_DELTA_BRIER and
                delta_platt > 0 and
                lr_p_val < BONFERRONI_ALPHA)
        out = {
            "name":              name,
            "n_params":          int(n_params),
            "train_brier":       float(tb),
            "test_brier":        float(teb),
            "test_platt_b":      float(platt["b"]),
            "test_platt_a":      float(platt["a"]),
            "lr_p":              float(lr_p_val),
            "delta_test_brier":  float(delta_brier),
            "delta_platt_to_1":  float(delta_platt),
            "ship_worthy":       bool(ship),
            "notes":             notes,
        }
        if extra:
            out.update(extra)
        return out

    # ── Experiment 1: format-specific β ────────────────────────────────────
    (b3, b5), nll_alt_train = fit_format_betas(train)
    p_train_1 = predict_format_betas(train, b3, b5)
    p_test_1  = predict_format_betas(test,  b3, b5)
    lr_p_1 = lr_p(nll_null_train, nll_alt_train, df=1)
    # Cluster bootstrap (resample series with replacement) on TRAIN
    boot = cluster_bootstrap_format_betas(train, n_boot=N_BOOT, seed=SEED)
    if len(boot) > 0:
        b3_ci = (float(np.percentile(boot[:, 0], 2.5)),
                 float(np.percentile(boot[:, 0], 97.5)))
        b5_ci = (float(np.percentile(boot[:, 1], 2.5)),
                 float(np.percentile(boot[:, 1], 97.5)))
        # sign-stability: fraction of bootstraps where b5 > b3 (or vice versa)
        agree_bo5_bigger = float(np.mean(boot[:, 1] > boot[:, 0]))
        agree_bo3_bigger = float(np.mean(boot[:, 0] > boot[:, 1]))
        sign_stable = max(agree_bo5_bigger, agree_bo3_bigger)
    else:
        b3_ci = b5_ci = (float("nan"), float("nan"))
        sign_stable = float("nan")
    notes_1 = (f"β_bo3={b3:.4f} (CI {b3_ci[0]:.3f},{b3_ci[1]:.3f}), "
               f"β_bo5={b5:.4f} (CI {b5_ci[0]:.3f},{b5_ci[1]:.3f}), "
               f"sign-stable={sign_stable:.2%}")
    experiments.append(package(
        "format_specific_beta", n_params=2, p_train=p_train_1,
        p_test=p_test_1, lr_p_val=lr_p_1, notes=notes_1,
        extra={
            "beta_bo3": float(b3), "beta_bo5": float(b5),
            "beta_bo3_ci": list(b3_ci), "beta_bo5_ci": list(b5_ci),
            "sign_stable_frac": float(sign_stable),
        },
    ))

    # ── Experiment 2: non-linear in rating diff ────────────────────────────
    # Reference for these is a direct logistic on (1, |Δ|, intl_exp_diff)
    # so LR test compares to same family without higher-order terms.
    X_lin_train = make_features_linear(train)
    X_lin_test  = make_features_linear(test)
    beta_lin, nll_lin = fit_logistic(X_lin_train, y_train)
    p_lin_train = sigmoid(X_lin_train @ beta_lin)
    p_lin_test  = sigmoid(X_lin_test  @ beta_lin)

    # Polynomial (degree 2)
    X_p2_train = make_features_poly(train, degree=2)
    X_p2_test  = make_features_poly(test,  degree=2)
    beta_p2, nll_p2 = fit_logistic(X_p2_train, y_train)
    p_p2_train = sigmoid(X_p2_train @ beta_p2)
    p_p2_test  = sigmoid(X_p2_test  @ beta_p2)
    lr_p_p2 = lr_p(nll_lin, nll_p2, df=1)
    experiments.append(package(
        "poly_deg2", n_params=4, p_train=p_p2_train, p_test=p_p2_test,
        lr_p_val=lr_p_p2,
        notes=f"coef(Δ²)={beta_p2[2]:.4f}; LR vs linear-direct on train"))

    # Polynomial (degree 3)
    X_p3_train = make_features_poly(train, degree=3)
    X_p3_test  = make_features_poly(test,  degree=3)
    beta_p3, nll_p3 = fit_logistic(X_p3_train, y_train)
    p_p3_train = sigmoid(X_p3_train @ beta_p3)
    p_p3_test  = sigmoid(X_p3_test  @ beta_p3)
    lr_p_p3 = lr_p(nll_lin, nll_p3, df=2)
    experiments.append(package(
        "poly_deg3", n_params=5, p_train=p_p3_train, p_test=p_p3_test,
        lr_p_val=lr_p_p3,
        notes=f"coef(Δ²)={beta_p3[2]:.4f}, coef(Δ³)={beta_p3[3]:.4f}"))

    # Spline (knots at 2, 4)
    X_sp_train = make_features_spline(train, knots=(2.0, 4.0))
    X_sp_test  = make_features_spline(test,  knots=(2.0, 4.0))
    beta_sp, nll_sp = fit_logistic(X_sp_train, y_train)
    p_sp_train = sigmoid(X_sp_train @ beta_sp)
    p_sp_test  = sigmoid(X_sp_test  @ beta_sp)
    lr_p_sp = lr_p(nll_lin, nll_sp, df=2)
    experiments.append(package(
        "spline_knots_2_4", n_params=5, p_train=p_sp_train,
        p_test=p_sp_test, lr_p_val=lr_p_sp,
        notes=("piecewise linear in |Δ|; "
               f"slope changes Δ@2={beta_sp[2]:+.4f}, Δ@4={beta_sp[3]:+.4f}")))

    # Cauchy link (heavy-tailed)
    b_cauchy, nll_cauchy = fit_cauchy(train)
    p_cauchy_train = predict_cauchy(train, b_cauchy)
    p_cauchy_test  = predict_cauchy(test,  b_cauchy)
    # Non-nested with closed-form, so LR test isn't strict; use vs single-β
    # closed-form null (same df = 1). Will report but interpret cautiously.
    lr_p_cauchy = lr_p(nll_null_train, nll_cauchy, df=0) if nll_cauchy < nll_null_train else 1.0
    experiments.append(package(
        "cauchy_link", n_params=1, p_train=p_cauchy_train,
        p_test=p_cauchy_test, lr_p_val=lr_p_cauchy,
        notes=(f"β_cauchy={b_cauchy:.4f}; non-nested vs closed-form baseline "
               f"(LR p shown vs single-β NLL)")))

    # Probit link
    b_probit, nll_probit = fit_probit(train)
    p_probit_train = predict_probit(train, b_probit)
    p_probit_test  = predict_probit(test,  b_probit)
    lr_p_probit = lr_p(nll_null_train, nll_probit, df=0) if nll_probit < nll_null_train else 1.0
    experiments.append(package(
        "probit_link", n_params=1, p_train=p_probit_train,
        p_test=p_probit_test, lr_p_val=lr_p_probit,
        notes=(f"β_probit={b_probit:.4f}; non-nested vs closed-form baseline")))

    # ── Experiment 3: correlated maps closed form ──────────────────────────
    (b_c, a3, a5), nll_corr = fit_correlated_closed_form(train)
    p_corr_train = predict_correlated_closed_form(train, b_c, a3, a5)
    p_corr_test  = predict_correlated_closed_form(test,  b_c, a3, a5)
    lr_p_corr = lr_p(nll_null_train, nll_corr, df=2)
    experiments.append(package(
        "correlated_maps", n_params=3, p_train=p_corr_train,
        p_test=p_corr_test, lr_p_val=lr_p_corr,
        notes=(f"β={b_c:.4f}, α_bo3={a3:+.4f}, α_bo5={a5:+.4f}; "
               "α>0 inflates fav prob above independent-maps closed form"),
        extra={"corr_beta": float(b_c), "corr_alpha_bo3": float(a3),
               "corr_alpha_bo5": float(a5)}))

    # ── Experiment 4: bo5 intercept shift ──────────────────────────────────
    (b_5shift, c_5shift), nll_5shift = fit_bo5_intercept(train)
    p_5shift_train = predict_bo5_intercept(train, b_5shift, c_5shift)
    p_5shift_test  = predict_bo5_intercept(test,  b_5shift, c_5shift)
    lr_p_5shift = lr_p(nll_null_train, nll_5shift, df=1)
    experiments.append(package(
        "bo5_intercept_shift", n_params=2, p_train=p_5shift_train,
        p_test=p_5shift_test, lr_p_val=lr_p_5shift,
        notes=(f"β={b_5shift:.4f}, bo5_logit_shift={c_5shift:+.4f}; "
               "positive=bo5 favors favorite more than closed-form predicts"),
        extra={"bo5_shift": float(c_5shift)}))

    # ── Verdict ────────────────────────────────────────────────────────────
    shipworthy = [e for e in experiments if e["ship_worthy"]]
    if shipworthy:
        # Pick the largest test-Brier improvement
        best = max(shipworthy, key=lambda e: e["delta_test_brier"])
        verdict = f"ship: {best['name']}"
    else:
        # Any with delta_test_brier > 0 but not meeting bar = marginal
        improvers = [e for e in experiments if e["delta_test_brier"] > 0]
        if improvers:
            best = max(improvers, key=lambda e: e["delta_test_brier"])
            if (best["delta_test_brier"] >= SHIP_DELTA_BRIER / 2 or
                best["lr_p"] < 0.05):
                verdict = "marginal"
            else:
                verdict = "no signal"
        else:
            verdict = "no signal"

    # 1-line headline
    best_test = min(experiments, key=lambda e: e["test_brier"])
    headline = (f"Best test Brier {best_test['test_brier']:.4f} from "
                f"'{best_test['name']}' vs baseline {baseline_test_brier:.4f} "
                f"(Δ={baseline_test_brier - best_test['test_brier']:+.4f}); "
                f"verdict={verdict}")

    out = {
        "n_series":              len(rows),
        "n_train":               n_train,
        "n_test":                n_test,
        "train_cut_date":        train[-1]["date"],
        "test_start_date":       test[0]["date"],
        "seed":                  SEED,
        "bonferroni_alpha":      BONFERRONI_ALPHA,
        "ship_delta_brier":      SHIP_DELTA_BRIER,
        "baseline_pooled_brier": float(baseline_pooled_brier),
        "baseline_train_brier":  float(baseline_train_brier),
        "baseline_test_brier":   float(baseline_test_brier),
        "baseline_platt_test":   float(baseline_platt_test["b"]),
        "baseline_nll_train":    float(baseline_nll_train),
        "baseline_nll_test":     float(baseline_nll_test),
        "null_single_beta_train":     {"beta": float(b_null_train),
                                       "train_nll": float(nll_null_train),
                                       "test_brier": float(null_test_brier)},
        "experiments":           experiments,
        "headline":              headline,
        "verdict":               verdict,
        "generated":             datetime.utcnow().isoformat() + "Z",
    }

    out_path = os.path.join(DATA_DIR, "projection_research_phase2_format.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"n_series={len(rows)}  train={n_train}  test={n_test}")
    print(f"Baseline   : train Brier={baseline_train_brier:.4f}  "
          f"test Brier={baseline_test_brier:.4f}  "
          f"test Platt b={baseline_platt_test['b']:.4f}")
    print(f"Null fit β : {b_null_train:.4f}  test Brier={null_test_brier:.4f}")
    print(f"\n{'Experiment':<24} {'k':>2}  "
          f"{'TrainB':>7}  {'TestB':>7}  {'dTest':>7}  "
          f"{'PlattB':>7}  {'LRp':>10}  {'Ship?':>6}")
    print("-" * 90)
    for e in experiments:
        print(f"{e['name']:<24} {e['n_params']:>2}  "
              f"{e['train_brier']:>7.4f}  {e['test_brier']:>7.4f}  "
              f"{e['delta_test_brier']:>+7.4f}  "
              f"{e['test_platt_b']:>7.4f}  {e['lr_p']:>10.3g}  "
              f"{'YES' if e['ship_worthy'] else 'no':>6}")
    print(f"\nHeadline: {headline}")
    print(f"Verdict : {verdict}")

    return out


if __name__ == "__main__":
    main()
