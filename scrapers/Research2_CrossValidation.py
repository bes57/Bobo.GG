"""
Research2_CrossValidation.py
============================

Stress-test the "optimized" BenPom series projection model
    p_series = sigmoid(0.154 * Δrating + 0.36 * intl_exp_diff)
against the baseline live model
    p_series = sigmoid(0.136 * Δrating)
under proper TIME-BASED cross-validation.

The pooled-MLE fit (β=0.154, intl_bonus=0.36, pooled Brier 0.2301 vs baseline
0.2317) used 100% of the data for fitting. This script answers: is the gain
real or partly overfit?

Tests:
 1. Time-based 75/25 holdout, with MLE refit on train.
 2. Expanding-window walk-forward CV (5 folds).
 3. Leave-one-event-out CV (per-event held-out Brier).
 4. Bootstrap 95% CI on intl_bonus coefficient.
 5. Sensitivity grid of intl_bonus on test set.
 6. Oracle-Brier floor estimate via Δ-bucket empirical win-rate.

Outputs:
  data/projection_research_phase2_cv.json
  static/projection_test/phase2_cv_per_event.png

Conventions:
  - Strictly time-based splits everywhere (no random shuffles that leak).
  - Bonferroni α = 0.05/5 = 0.01 reference for shipping.
  - β is fit on per-MAP outcomes (consistent with how BenPom is calibrated),
    intl_bonus is fit on per-SERIES outcomes (it is a series-level offset).
  - All MLE uses scipy.optimize.minimize on negative log-likelihood with
    deterministic init.  numpy seed = 0 for bootstrap.
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar, minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "static", "projection_test")
os.makedirs(OUT_DIR, exist_ok=True)

LIVE_BETA = 0.136
OPTIMIZED_BETA = 0.154
OPTIMIZED_INTL_BONUS = 0.36

INTL_EVENTS = {
    "2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
    "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
    "2026_masters_santiago",
}

RNG = np.random.default_rng(0)


# ── Helpers ──────────────────────────────────────────────────────────────────
def sigmoid(x):
    x = np.clip(x, -25, 25)
    return 1.0 / (1.0 + np.exp(-x))


def series_prob_from_map(p, bo):
    if bo == 3:
        return (p ** 2) * (3 - 2 * p)
    if bo == 5:
        return (p ** 3) * (10 - 15 * p + 6 * p * p)
    return p


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
        return None
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def safe_logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def platt_fit_slope(probs, outcomes):
    """Fit Platt slope b via MLE.  Returns just b (a is allowed to drift).
    b<1 = over-confident, b>1 = under-confident."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if len(probs) < 20:
        return None
    x = safe_logit(probs)
    y = outcomes

    def nll(params):
        a, b = params
        z = np.clip(a + b * x, -25, 25)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(nll, x0=[0.0, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    return float(res.x[1])


# ── Data load (mirrors AnalyzeProjectionCalibration.load_matches) ─────────────
def load_matches():
    files = [
        ("rating_timeline_2024.json", 2024),
        ("rating_timeline_2025.json", 2025),
        ("rating_timeline.json", 2026),
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
    attendance = defaultdict(list)
    for m in matches:
        if m["event_id"] not in INTL_EVENTS:
            continue
        season = m["season"]
        for org in (m["winner"], m["loser"]):
            attendance[(org, season)].append((m["date"], m["event_id"]))
    return attendance


def _intl_exp_diff(fav_org, dog_org, season, match_date, attendance):
    def attended_before(org):
        for d, _ in attendance.get((org, season), []):
            if d < match_date:
                return True
        return False
    f = 1 if attended_before(fav_org) else 0
    d = 1 if attended_before(dog_org) else 0
    return f - d


def build_rows(matches):
    """Return (map_rows, series_rows).  Each row has the raw features and
    metadata needed to score under any (β, intl_bonus)."""
    attendance = _build_intl_attendance(matches)
    map_rows = []
    series_rows = []
    for m in matches:
        wb = float(m["winner_before"])
        lb = float(m["loser_before"])
        delta = wb - lb
        abs_delta = abs(delta)
        fav_won = 1 if wb > lb else 0
        if wb > lb:
            fav_org, dog_org = m["winner"], m["loser"]
        else:
            fav_org, dog_org = m["loser"], m["winner"]
        n_maps = len(m.get("maps", []))
        bo = infer_bo(m.get("series_score", ""), n_maps)
        intl_diff = _intl_exp_diff(fav_org, dog_org, m["season"],
                                   m["date"], attendance)

        series_rows.append({
            "match_id": m["match_id"],
            "date": m["date"],
            "event_id": m["event_id"],
            "season": m["season"],
            "abs_delta": abs_delta,
            "bo": bo,
            "y": fav_won,
            "intl_diff": intl_diff,
            "fav": fav_org,
            "dog": dog_org,
        })

        for mp in m.get("maps", []):
            mw = mp.get("winner")
            if mw is None:
                continue
            map_rows.append({
                "match_id": m["match_id"],
                "date": m["date"],
                "event_id": m["event_id"],
                "season": m["season"],
                "abs_delta": abs_delta,
                "y": 1 if mw == fav_org else 0,
            })
    return map_rows, series_rows


# ── Scoring under a model ────────────────────────────────────────────────────
def score_series(series_rows, beta, intl_bonus):
    """Return p_series array under given (β, intl_bonus)."""
    probs = np.empty(len(series_rows))
    for i, r in enumerate(series_rows):
        p_map = sigmoid(beta * r["abs_delta"])
        p_ser = series_prob_from_map(p_map, r["bo"])
        if intl_bonus != 0.0 and r["intl_diff"] != 0:
            ps = max(min(p_ser, 1 - 1e-9), 1e-9)
            logit_ps = math.log(ps / (1 - ps)) + intl_bonus * r["intl_diff"]
            p_ser = 1.0 / (1.0 + math.exp(-logit_ps))
        probs[i] = p_ser
    return probs


# ── MLE fit of (β, intl_bonus) jointly on a training set ─────────────────────
def fit_beta_and_bonus(series_rows, init_beta=0.14, init_bonus=0.0,
                       fit_bonus=True):
    """Maximize series-level log-likelihood over (β, intl_bonus).

    β controls the map sigmoid (then lifted to series via bo3/bo5 formula);
    intl_bonus is an additive series-logit offset times intl_diff ∈ {-1,0,+1}.
    """
    abs_deltas = np.array([r["abs_delta"] for r in series_rows])
    bos = np.array([r["bo"] for r in series_rows])
    ys = np.array([r["y"] for r in series_rows], dtype=float)
    intl = np.array([r["intl_diff"] for r in series_rows], dtype=float)

    def nll(params):
        if fit_bonus:
            b, ib = params
        else:
            b = params[0]
            ib = 0.0
        if b <= 0 or b > 1.0:
            return 1e9
        p_map = 1.0 / (1.0 + np.exp(-np.clip(b * abs_deltas, -25, 25)))
        # series formula vectorized
        p_ser = np.where(
            bos == 5,
            (p_map ** 3) * (10 - 15 * p_map + 6 * p_map * p_map),
            (p_map ** 2) * (3 - 2 * p_map),
        )
        if fit_bonus:
            mask = intl != 0
            if mask.any():
                ps = np.clip(p_ser[mask], 1e-9, 1 - 1e-9)
                logit_ps = np.log(ps / (1 - ps)) + ib * intl[mask]
                p_ser[mask] = 1.0 / (1.0 + np.exp(-np.clip(logit_ps, -25, 25)))
        p_ser = np.clip(p_ser, 1e-12, 1 - 1e-12)
        return -np.sum(ys * np.log(p_ser) + (1 - ys) * np.log(1 - p_ser))

    if fit_bonus:
        res = minimize(nll, x0=[init_beta, init_bonus], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
        return float(res.x[0]), float(res.x[1])
    else:
        res = minimize(nll, x0=[init_beta], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
        return float(res.x[0]), 0.0


# ── Oracle Brier estimate via Δ-bucket empirical winrate ─────────────────────
def oracle_brier(series_rows):
    """Estimate the lowest achievable Brier given only |Δrating| info.
    For each Δ-bucket, the oracle predicts the empirical win-rate inside that
    bucket.  Use ~20 quantile bins to limit per-bin variance.
    Caveat: this is a slightly optimistic floor (in-sample empirical rates),
    but it bounds how much room realistically remains."""
    abs_deltas = np.array([r["abs_delta"] for r in series_rows])
    ys = np.array([r["y"] for r in series_rows], dtype=float)
    # Quantile bin edges
    edges = np.quantile(abs_deltas, np.linspace(0, 1, 21))
    edges[0] = -1e-9
    edges[-1] = edges[-1] + 1e-6
    preds = np.empty(len(series_rows))
    for i in range(20):
        lo, hi = edges[i], edges[i + 1]
        mask = (abs_deltas > lo) & (abs_deltas <= hi)
        if mask.sum() == 0:
            continue
        preds[mask] = ys[mask].mean()
    return brier(preds.tolist(), ys.tolist())


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    matches = load_matches()
    _, series_rows = build_rows(matches)
    n = len(series_rows)
    print(f"Loaded {n} series (model has opinion).")

    # Series are already chronologically sorted via load_matches sort key.
    # Sanity: ensure they're sorted by date.
    series_rows.sort(key=lambda r: (r["date"], r["match_id"]))

    # ── Sanity: pooled Brier under the two stated configs ──────────────────
    p_live_full = score_series(series_rows, LIVE_BETA, 0.0)
    p_opt_full = score_series(series_rows, OPTIMIZED_BETA, OPTIMIZED_INTL_BONUS)
    ys_all = np.array([r["y"] for r in series_rows], dtype=float)
    print(f"Sanity pooled Brier - live β=0.136 no bonus:    {brier(p_live_full.tolist(), ys_all.tolist()):.4f}")
    print(f"Sanity pooled Brier - optimized β=0.154 + 0.36: {brier(p_opt_full.tolist(), ys_all.tolist()):.4f}")

    out = {
        "meta": {
            "generated": datetime.utcnow().isoformat() + "Z",
            "n_series_total": n,
            "live": {"beta": LIVE_BETA, "intl_bonus": 0.0},
            "optimized": {"beta": OPTIMIZED_BETA, "intl_bonus": OPTIMIZED_INTL_BONUS},
            "bonferroni_alpha": 0.01,
            "seed": 0,
        }
    }

    # ── 1. Time-based 75/25 holdout ────────────────────────────────────────
    split_idx = int(round(n * 0.75))
    train = series_rows[:split_idx]
    test = series_rows[split_idx:]
    ys_train = np.array([r["y"] for r in train], dtype=float)
    ys_test = np.array([r["y"] for r in test], dtype=float)

    p_live_test = score_series(test, LIVE_BETA, 0.0)
    p_opt_test = score_series(test, OPTIMIZED_BETA, OPTIMIZED_INTL_BONUS)

    refit_beta, refit_bonus = fit_beta_and_bonus(train)
    p_refit_test = score_series(test, refit_beta, refit_bonus)

    live_test_brier = brier(p_live_test.tolist(), ys_test.tolist())
    opt_test_brier = brier(p_opt_test.tolist(), ys_test.tolist())
    refit_test_brier = brier(p_refit_test.tolist(), ys_test.tolist())

    out["holdout_75_25"] = {
        "n_train": len(train),
        "n_test": len(test),
        "train_date_range": [train[0]["date"], train[-1]["date"]],
        "test_date_range": [test[0]["date"], test[-1]["date"]],
        "live_test_brier": live_test_brier,
        "live_test_platt": platt_fit_slope(p_live_test.tolist(), ys_test.tolist()),
        "optimized_test_brier": opt_test_brier,
        "optimized_test_platt": platt_fit_slope(p_opt_test.tolist(), ys_test.tolist()),
        "refit_on_train_brier": refit_test_brier,
        "refit_test_platt": platt_fit_slope(p_refit_test.tolist(), ys_test.tolist()),
        "refit_beta": refit_beta,
        "refit_intl_bonus": refit_bonus,
        "optimized_beats_live_on_test": opt_test_brier < live_test_brier,
        "refit_beats_live_on_test": refit_test_brier < live_test_brier,
    }
    print(f"\n[1] Holdout 75/25: n_train={len(train)} n_test={len(test)}")
    print(f"    live Brier = {live_test_brier:.4f}")
    print(f"    optimized Brier = {opt_test_brier:.4f}")
    print(f"    refit (β={refit_beta:.3f}, bonus={refit_bonus:.3f}) Brier = {refit_test_brier:.4f}")

    # ── 2. Expanding-window walk-forward CV (5 folds) ──────────────────────
    n_folds = 5
    chunk = n // n_folds
    fold_briers_live = []
    fold_briers_opt = []
    fold_briers_refit = []
    fold_betas = []
    fold_bonuses = []
    fold_details = []
    for k in range(1, n_folds):
        tr = series_rows[: k * chunk]
        te = series_rows[k * chunk: (k + 1) * chunk if k < n_folds - 1 else n]
        ys_te = np.array([r["y"] for r in te], dtype=float)
        b_fit, ib_fit = fit_beta_and_bonus(tr)
        pl = score_series(te, LIVE_BETA, 0.0)
        po = score_series(te, OPTIMIZED_BETA, OPTIMIZED_INTL_BONUS)
        pr = score_series(te, b_fit, ib_fit)
        bl = brier(pl.tolist(), ys_te.tolist())
        bo = brier(po.tolist(), ys_te.tolist())
        br = brier(pr.tolist(), ys_te.tolist())
        fold_briers_live.append(bl)
        fold_briers_opt.append(bo)
        fold_briers_refit.append(br)
        fold_betas.append(b_fit)
        fold_bonuses.append(ib_fit)
        fold_details.append({
            "fold": k,
            "n_train": len(tr), "n_test": len(te),
            "train_end": tr[-1]["date"],
            "test_start": te[0]["date"], "test_end": te[-1]["date"],
            "refit_beta": b_fit, "refit_intl_bonus": ib_fit,
            "live_brier": bl, "optimized_brier": bo, "refit_brier": br,
        })
        print(f"[2] fold {k}: n_tr={len(tr)} n_te={len(te)} "
              f"live={bl:.4f} opt={bo:.4f} refit={br:.4f} "
              f"(β={b_fit:.3f}, bonus={ib_fit:.3f})")

    out["expanding_window_cv"] = {
        "n_folds": len(fold_details),
        "live_avg_brier": float(np.mean(fold_briers_live)),
        "optimized_avg_brier": float(np.mean(fold_briers_opt)),
        "refit_avg_brier": float(np.mean(fold_briers_refit)),
        "per_fold_brier_live": fold_briers_live,
        "per_fold_brier_optimized": fold_briers_opt,
        "per_fold_brier_refit": fold_briers_refit,
        "per_fold_refit_beta": fold_betas,
        "per_fold_refit_intl_bonus": fold_bonuses,
        "fold_details": fold_details,
        "optimized_beats_live_folds": int(sum(1 for o, l in zip(fold_briers_opt, fold_briers_live) if o < l)),
    }

    # ── 3. Leave-one-event-out CV ──────────────────────────────────────────
    events = sorted({r["event_id"] for r in series_rows})
    leave_event_out = []
    per_event_diffs = []
    for ev in events:
        te = [r for r in series_rows if r["event_id"] == ev]
        tr = [r for r in series_rows if r["event_id"] != ev]
        if len(te) < 4:
            continue
        ys_te = np.array([r["y"] for r in te], dtype=float)
        pl = score_series(te, LIVE_BETA, 0.0)
        po = score_series(te, OPTIMIZED_BETA, OPTIMIZED_INTL_BONUS)
        bl = brier(pl.tolist(), ys_te.tolist())
        bo = brier(po.tolist(), ys_te.tolist())
        # Bootstrap CI on (opt - live) per-event diff (within-event resampling)
        n_ev = len(te)
        diffs = []
        rng = np.random.default_rng(0)
        for _ in range(500):
            idx = rng.integers(0, n_ev, size=n_ev)
            d = brier(po[idx].tolist(), ys_te[idx].tolist()) - brier(pl[idx].tolist(), ys_te[idx].tolist())
            diffs.append(d)
        ci_lo, ci_hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
        leave_event_out.append({
            "event_id": ev,
            "n_test": n_ev,
            "live_test_brier": bl,
            "optimized_test_brier": bo,
            "diff_opt_minus_live": bo - bl,
            "diff_ci_lo": ci_lo,
            "diff_ci_hi": ci_hi,
            "optimized_better": bo < bl,
        })
        per_event_diffs.append(bo - bl)
        print(f"[3] {ev:<28} n={n_ev:>3} live={bl:.4f} opt={bo:.4f} Δ={bo-bl:+.4f}")

    out["leave_one_event_out"] = leave_event_out
    out["leave_one_event_out_summary"] = {
        "n_events": len(leave_event_out),
        "n_events_optimized_better": int(sum(1 for e in leave_event_out if e["optimized_better"])),
        "pct_events_optimized_better": (
            sum(1 for e in leave_event_out if e["optimized_better"]) / len(leave_event_out)
            if leave_event_out else 0.0
        ),
        "mean_diff_opt_minus_live": float(np.mean(per_event_diffs)) if per_event_diffs else None,
    }

    # ── 4. Bootstrap stability of intl_bonus ───────────────────────────────
    n_reps = 500
    bonus_samples = []
    rng = np.random.default_rng(0)
    n_train_full = len(series_rows)
    for _ in range(n_reps):
        idx = rng.integers(0, n_train_full, size=n_train_full)
        boot_rows = [series_rows[i] for i in idx]
        try:
            _, ib = fit_beta_and_bonus(boot_rows)
            bonus_samples.append(ib)
        except Exception:
            continue
    bonus_samples = np.array(bonus_samples)
    out["intl_bonus_bootstrap"] = {
        "n_reps": int(len(bonus_samples)),
        "median": float(np.median(bonus_samples)),
        "mean": float(np.mean(bonus_samples)),
        "ci_95": [float(np.quantile(bonus_samples, 0.025)),
                  float(np.quantile(bonus_samples, 0.975))],
        "pct_positive": float(np.mean(bonus_samples > 0)),
        "pct_above_0.36": float(np.mean(bonus_samples > 0.36)),
    }
    print(f"\n[4] Bootstrap intl_bonus: median={np.median(bonus_samples):.3f} "
          f"CI95=[{np.quantile(bonus_samples,0.025):.3f}, {np.quantile(bonus_samples,0.975):.3f}] "
          f"%>0={np.mean(bonus_samples>0)*100:.1f}%")

    # ── 5. Sensitivity grid of intl_bonus on test set ──────────────────────
    # Use the 75/25 split. For each candidate bonus, also fit β on train at
    # that fixed bonus (so β and bonus aren't double-counting the offset).
    bonus_grid = [0.0, 0.1, 0.2, 0.3, 0.36, 0.5, 0.7]
    grid_rows = []
    for ib in bonus_grid:
        # Fit β on train holding intl_bonus = ib
        def nll_b(b):
            if b <= 0 or b > 1.0:
                return 1e9
            p = score_series(train, b, ib)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.sum(ys_train * np.log(p) + (1 - ys_train) * np.log(1 - p))
        res = minimize_scalar(nll_b, bounds=(0.05, 0.5), method="bounded",
                              options={"xatol": 1e-6})
        b_star = float(res.x)
        p_train = score_series(train, b_star, ib)
        p_test = score_series(test, b_star, ib)
        grid_rows.append({
            "value": ib,
            "best_beta_at_bonus": b_star,
            "train_brier": brier(p_train.tolist(), ys_train.tolist()),
            "test_brier": brier(p_test.tolist(), ys_test.tolist()),
        })
    best_bonus = min(grid_rows, key=lambda r: r["test_brier"])
    out["intl_bonus_grid"] = grid_rows
    out["best_intl_bonus_on_test"] = best_bonus["value"]
    print(f"\n[5] Sensitivity grid:")
    for r in grid_rows:
        print(f"    bonus={r['value']:.2f}  β*={r['best_beta_at_bonus']:.3f}  "
              f"train={r['train_brier']:.4f}  test={r['test_brier']:.4f}")
    print(f"    best on test: bonus={best_bonus['value']}")

    # ── 6. Oracle Brier estimate ───────────────────────────────────────────
    oracle_b = oracle_brier(series_rows)
    pooled_live = brier(p_live_full.tolist(), ys_all.tolist())
    pooled_opt = brier(p_opt_full.tolist(), ys_all.tolist())
    out["oracle_brier_estimate"] = oracle_b
    out["headroom"] = {
        "pooled_live_brier": pooled_live,
        "pooled_optimized_brier": pooled_opt,
        "oracle_brier_estimate": oracle_b,
        "live_minus_oracle": pooled_live - oracle_b,
        "optimized_minus_oracle": pooled_opt - oracle_b,
        "improvement_optimized_over_live": pooled_live - pooled_opt,
        "fraction_of_headroom_captured": (
            (pooled_live - pooled_opt) / (pooled_live - oracle_b)
            if pooled_live > oracle_b else None
        ),
    }
    print(f"\n[6] Oracle Brier (Δ-bucket empirical floor): {oracle_b:.4f}")
    print(f"    pooled live Brier:     {pooled_live:.4f} (gap to oracle = {pooled_live-oracle_b:+.4f})")
    print(f"    pooled optimized Brier: {pooled_opt:.4f} (gap to oracle = {pooled_opt-oracle_b:+.4f})")
    if pooled_live > oracle_b:
        frac = (pooled_live - pooled_opt) / (pooled_live - oracle_b) * 100
        print(f"    optimized captures {frac:.1f}% of available headroom over live")

    # ── Verdict ────────────────────────────────────────────────────────────
    opt_beats_holdout = out["holdout_75_25"]["optimized_beats_live_on_test"]
    pct_events_better = out["leave_one_event_out_summary"]["pct_events_optimized_better"]
    bonus_ci = out["intl_bonus_bootstrap"]["ci_95"]
    ci_excludes_zero = bonus_ci[0] > 0 or bonus_ci[1] < 0

    if not opt_beats_holdout:
        verdict = "fully overfit"
    elif opt_beats_holdout and pct_events_better > 0.5 and ci_excludes_zero:
        verdict = "robust"
    else:
        verdict = "partly overfit"

    # Headline
    delta_holdout = out["holdout_75_25"]["live_test_brier"] - out["holdout_75_25"]["optimized_test_brier"]
    headline = (
        f"Optimized {('beats' if opt_beats_holdout else 'fails vs')} live on 75/25 holdout "
        f"(Δ={delta_holdout:+.4f}); intl_bonus bootstrap 95% CI={bonus_ci}; "
        f"better in {out['leave_one_event_out_summary']['n_events_optimized_better']}/"
        f"{out['leave_one_event_out_summary']['n_events']} events  →  {verdict}"
    )
    out["headline"] = headline
    out["verdict"] = verdict
    print(f"\n[VERDICT] {verdict}")
    print(f"[HEADLINE] {headline}")

    # ── Write JSON ─────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "projection_research_phase2_cv.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # ── Chart: per-event Brier improvement (optimized − live), with CI ─────
    leo = sorted(leave_event_out, key=lambda r: r["diff_opt_minus_live"])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    xs = np.arange(len(leo))
    diffs = [r["diff_opt_minus_live"] for r in leo]
    cilo = [r["diff_ci_lo"] for r in leo]
    cihi = [r["diff_ci_hi"] for r in leo]
    yerr_lo = [d - lo for d, lo in zip(diffs, cilo)]
    yerr_hi = [hi - d for d, hi in zip(diffs, cihi)]
    colors = ["#3aa15a" if d < 0 else "#c44d4d" for d in diffs]
    ax.bar(xs, diffs, color=colors, alpha=0.85)
    ax.errorbar(xs, diffs, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="#666", capsize=3, linewidth=1.0)
    for x, r in zip(xs, leo):
        ax.annotate(f"n={r['n_test']}", (x, r["diff_opt_minus_live"]),
                    textcoords="offset points",
                    xytext=(0, 6 if r["diff_opt_minus_live"] >= 0 else -12),
                    ha="center", fontsize=7, color="#333")
    ax.axhline(0, color="#222", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["event_id"] for r in leo], rotation=45, ha="right",
                       fontsize=8)
    ax.set_ylabel("Brier (optimized) − Brier (live)\n← optimized better       worse →")
    title = (f"Leave-one-event-out CV: per-event Brier delta  "
             f"({out['leave_one_event_out_summary']['n_events_optimized_better']}/"
             f"{out['leave_one_event_out_summary']['n_events']} better)\n"
             f"Verdict: {verdict}")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    chart_path = os.path.join(OUT_DIR, "phase2_cv_per_event.png")
    fig.savefig(chart_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()
