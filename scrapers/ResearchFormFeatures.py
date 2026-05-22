"""
ResearchFormFeatures.py
=======================

Test whether recent-form features improve BenPom pre-match win-probability
predictions for VCT 2024 → 2026 stage 1.

Baseline:
    p_fav = sigmoid(BETA * |winner_before - loser_before|),    BETA = 0.154

Features tested (computed for both sides, then signed difference fav − dog):
    1. last5_winrate      — fraction wins in last 5 series
    2. velocity30         — rating_before − rating_30_days_ago
    3. last_result        — did team win its previous series (0/1)
    4. last_opp_strength  — opponent's rating (at that time) in last series
    5. streak             — signed current win/loss streak length
    6. days_since_last    — days of rest since previous series

For each feature we fit:
    Logit(y) = α + β * logit(p_baseline) + γ * form_diff
and report γ, SE, p-value, 95% CI, Brier with feature, LR-test p-value vs
the baseline-only model (which is α + β * logit(p_baseline)).

Joint model fits all six simultaneously. Bonferroni α = 0.05 / 6 ≈ 0.00833.

Output:  data/projection_research_form.json
"""

import json
import math
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

BETA_BASELINE = 0.154  # per task spec
ALPHA_BONF = 0.05 / 6
N_BOOTSTRAP = 200


# ── helpers ───────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def brier(probs, outcomes):
    if len(probs) == 0:
        return None
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def fit_logit_newton(X, y, max_iter=200, tol=1e-8, ridge=1e-6):
    """Newton-Raphson fit of logistic regression. Returns (beta, cov, ll).
    X already has intercept column if you want one (caller's responsibility).
    `ridge` is a tiny L2 added to the Hessian for numerical stability;
    it has no material effect on coefficients but prevents singular Hessians."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        z = X @ beta
        z = np.clip(z, -30, 30)
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
    # Recompute log-likelihood and covariance at converged beta
    z = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    ll = float(np.sum(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
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


# ── data loading ──────────────────────────────────────────────────────────
def load_data():
    files = [
        ("rating_timeline_2024.json", 2024),
        ("rating_timeline_2025.json", 2025),
        ("rating_timeline.json", 2026),
    ]
    matches = []
    checkpoints = []  # list of (date_str, ratings_dict, season)
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
            mm = dict(m)
            mm["season"] = season
            matches.append(mm)
        for cp in d.get("checkpoints", []):
            checkpoints.append((cp["date"], cp.get("ratings", {}), season))
    matches.sort(key=lambda r: (r["date"], r["match_id"]))
    # Sort checkpoints chronologically too
    checkpoints.sort(key=lambda r: r[0])
    return matches, checkpoints


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


# ── feature engineering ───────────────────────────────────────────────────
def build_feature_table(matches, checkpoints):
    """Walk matches in chronological order, maintaining per-team history.
    Returns list of dicts, one per series, with baseline p + signed features."""

    # Per-team chronological history of (date, opponent, opp_rating_at_time,
    # team_won, rating_after). We append AFTER computing features for this match.
    team_hist = defaultdict(list)

    # Index checkpoints by date for fast 30-day lookback.
    # Each checkpoint date maps to ratings dict.
    cp_dates = [parse_date(c[0]) for c in checkpoints]
    cp_ratings = [c[1] for c in checkpoints]
    cp_dates_arr = np.array([d.toordinal() for d in cp_dates])

    def rating_on_or_before(team, target_date):
        """Return team's rating at most recent checkpoint <= target_date,
        or None if no such checkpoint exists / team not present."""
        target_ord = target_date.toordinal()
        # binary search for rightmost cp_date <= target_ord
        idx = np.searchsorted(cp_dates_arr, target_ord, side="right") - 1
        while idx >= 0:
            r = cp_ratings[idx].get(team)
            if r is not None:
                return float(r)
            idx -= 1
        return None

    rows = []
    for m in matches:
        date = parse_date(m["date"])
        wb = float(m["winner_before"])
        lb = float(m["loser_before"])
        if wb > lb:
            fav, dog = m["winner"], m["loser"]
            fav_before, dog_before = wb, lb
            fav_won = 1
        else:
            fav, dog = m["loser"], m["winner"]
            fav_before, dog_before = lb, wb
            fav_won = 0  # favorite (rating-higher team) lost

        abs_delta = abs(wb - lb)
        p_base = float(sigmoid(BETA_BASELINE * abs_delta))

        def features_for(team, current_rating, current_date):
            hist = team_hist[team]
            # 1. last-5 win rate
            last5 = hist[-5:]
            last5_wr = (sum(h["won"] for h in last5) / len(last5)) if last5 else None
            # 2. velocity30: rating now − rating 30 days ago (most recent cp <= date-30)
            r30 = rating_on_or_before(team, current_date - timedelta(days=30))
            velocity30 = (current_rating - r30) if r30 is not None else None
            # 3. last result
            last_result = hist[-1]["won"] if hist else None
            # 4. last opponent strength: opponent's rating at time of that prior match
            last_opp_strength = hist[-1]["opp_rating"] if hist else None
            # 5. streak: signed length of current run
            streak = 0
            for h in reversed(hist):
                if streak == 0:
                    streak = 1 if h["won"] else -1
                else:
                    if (streak > 0 and h["won"]) or (streak < 0 and not h["won"]):
                        streak += 1 if h["won"] else -1
                    else:
                        break
            # 6. days since last match
            days_since = (current_date - hist[-1]["date"]).days if hist else None
            return {
                "last5_winrate": last5_wr,
                "velocity30": velocity30,
                "last_result": last_result,
                "last_opp_strength": last_opp_strength,
                "streak": streak if hist else None,
                "days_since_last": days_since,
            }

        fav_feat = features_for(fav, fav_before, date)
        dog_feat = features_for(dog, dog_before, date)

        diff = {}
        for k in fav_feat:
            fv, dv = fav_feat[k], dog_feat[k]
            diff[k] = (fv - dv) if (fv is not None and dv is not None) else None

        rows.append({
            "date": m["date"],
            "match_id": m["match_id"],
            "season": m["season"],
            "fav": fav, "dog": dog,
            "fav_before": fav_before, "dog_before": dog_before,
            "abs_delta": abs_delta,
            "p_base": p_base,
            "y": fav_won,
            "fav_feat": fav_feat,
            "dog_feat": dog_feat,
            "diff": diff,
        })

        # AFTER computing features, append this match to both teams' histories.
        team_hist[fav].append({
            "date": date, "opp": dog, "opp_rating": dog_before,
            "won": 1 if fav_won == 1 else 0,
        })
        team_hist[dog].append({
            "date": date, "opp": fav, "opp_rating": fav_before,
            "won": 0 if fav_won == 1 else 1,
        })

    return rows


# ── single-feature analysis ───────────────────────────────────────────────
def analyze_feature(rows, feat_name):
    """Fit baseline-only + baseline+feature on the SAME subset (where the
    feature is defined). Returns dict with stats."""
    # Build vectors only over rows where this feature is non-null
    sub = [r for r in rows if r["diff"].get(feat_name) is not None]
    n = len(sub)
    if n < 30:
        return {"name": feat_name, "n_used": n, "error": "too few samples"}
    y = np.array([r["y"] for r in sub], dtype=float)
    base_logit = np.array([safe_logit(r["p_base"]) for r in sub])
    feat = np.array([r["diff"][feat_name] for r in sub], dtype=float)

    # ---- Baseline-only model on same subset: [1, logit(p_base)]
    X0 = np.column_stack([np.ones(n), base_logit])
    b0, cov0, ll0 = fit_logit_newton(X0, y)
    p0 = predict_logit(X0, b0)
    brier0 = brier(p0, y)

    # ---- With feature: [1, logit(p_base), feat]
    X1 = np.column_stack([np.ones(n), base_logit, feat])
    b1, cov1, ll1 = fit_logit_newton(X1, y)
    p1 = predict_logit(X1, b1)
    brier1 = brier(p1, y)

    coef = float(b1[2])
    se = float(np.sqrt(max(cov1[2, 2], 0.0)))
    z = coef / se if se > 0 else 0.0
    # Two-sided Wald p-value
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    # LR test: 2*(ll1 - ll0) ~ chi2(1)
    lr_stat = 2 * (ll1 - ll0)
    lr_p = float(1 - stats.chi2.cdf(max(lr_stat, 0), df=1))

    return {
        "name": feat_name,
        "n_used": n,
        "coef": coef,
        "se": se,
        "z": float(z),
        "p_value": p_value,
        "ci": [float(ci_lo), float(ci_hi)],
        "baseline_brier_on_subset": float(brier0),
        "brier_with_feature": float(brier1),
        "brier_improvement": float(brier0 - brier1),
        "lr_stat": float(lr_stat),
        "lr_p": lr_p,
        "significant_bonferroni": bool(p_value < ALPHA_BONF),
    }


# ── joint model ───────────────────────────────────────────────────────────
def analyze_joint(rows, feat_names):
    """Fit baseline + all features on the subset where ALL are non-null."""
    sub = [r for r in rows if all(r["diff"].get(k) is not None for k in feat_names)]
    n = len(sub)
    if n < 30:
        return {"n_used": n, "error": "too few complete-case rows"}
    y = np.array([r["y"] for r in sub], dtype=float)
    base_logit = np.array([safe_logit(r["p_base"]) for r in sub])
    feats = np.column_stack([
        [r["diff"][k] for r in sub] for k in feat_names
    ])

    X0 = np.column_stack([np.ones(n), base_logit])
    b0, cov0, ll0 = fit_logit_newton(X0, y)
    p0 = predict_logit(X0, b0)
    brier0 = brier(p0, y)

    X1 = np.column_stack([np.ones(n), base_logit, feats])
    b1, cov1, ll1 = fit_logit_newton(X1, y)
    p1 = predict_logit(X1, b1)
    brier1 = brier(p1, y)

    per_feat = []
    retained = []
    for i, name in enumerate(feat_names):
        c = float(b1[2 + i])
        s = float(np.sqrt(max(cov1[2 + i, 2 + i], 0.0)))
        z = c / s if s > 0 else 0.0
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        per_feat.append({"name": name, "coef": c, "se": s, "p_value": p,
                          "ci": [c - 1.96 * s, c + 1.96 * s],
                          "significant_bonferroni": bool(p < ALPHA_BONF)})
        if p < ALPHA_BONF:
            retained.append(name)

    lr_stat = 2 * (ll1 - ll0)
    lr_p = float(1 - stats.chi2.cdf(max(lr_stat, 0), df=len(feat_names)))

    return {
        "n_used": n,
        "per_feature": per_feat,
        "features_retained": retained,
        "baseline_brier_on_subset": float(brier0),
        "joint_brier": float(brier1),
        "joint_brier_improvement": float(brier0 - brier1),
        "lr_stat": float(lr_stat),
        "lr_p": lr_p,
    }


# ── bootstrap CI on Brier improvement of best feature ─────────────────────
def bootstrap_best_feature(rows, feat_name, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    sub = [r for r in rows if r["diff"].get(feat_name) is not None]
    n = len(sub)
    if n < 30:
        return None
    y = np.array([r["y"] for r in sub], dtype=float)
    base_logit = np.array([safe_logit(r["p_base"]) for r in sub])
    feat = np.array([r["diff"][feat_name] for r in sub], dtype=float)
    improvements = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]; bi = base_logit[idx]; fi = feat[idx]
        try:
            X0 = np.column_stack([np.ones(n), bi])
            b0, _, _ = fit_logit_newton(X0, yi)
            p0 = predict_logit(X0, b0)
            X1 = np.column_stack([np.ones(n), bi, fi])
            b1, _, _ = fit_logit_newton(X1, yi)
            p1 = predict_logit(X1, b1)
            improvements.append(brier(p0, yi) - brier(p1, yi))
        except Exception:
            continue
    if not improvements:
        return None
    arr = np.array(improvements)
    return {
        "feature": feat_name,
        "n_boot": int(len(arr)),
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "pct_positive": float((arr > 0).mean()),
    }


# ── main ──────────────────────────────────────────────────────────────────
def main():
    matches, checkpoints = load_data()
    print(f"Loaded {len(matches)} qualifying series, {len(checkpoints)} checkpoints.")

    rows = build_feature_table(matches, checkpoints)
    n = len(rows)
    print(f"Built feature table: {n} series.")

    # Overall (full-sample) baseline Brier
    y_all = np.array([r["y"] for r in rows], dtype=float)
    p_all = np.array([r["p_base"] for r in rows], dtype=float)
    baseline_brier = brier(p_all, y_all)
    print(f"Full-sample baseline Brier: {baseline_brier:.4f}")

    feat_names = ["last5_winrate", "velocity30", "last_result",
                  "last_opp_strength", "streak", "days_since_last"]

    feature_results = []
    for fn in feat_names:
        res = analyze_feature(rows, fn)
        feature_results.append(res)

    joint = analyze_joint(rows, feat_names)

    # Identify best single feature by Brier improvement
    eligible = [f for f in feature_results if "brier_improvement" in f]
    if eligible:
        best = max(eligible, key=lambda f: f["brier_improvement"])
        boot = bootstrap_best_feature(rows, best["name"])
    else:
        best, boot = None, None

    # Verdict
    any_sig = any(f.get("significant_bonferroni") for f in feature_results)
    any_raw_sig = any(f.get("p_value", 1) < 0.05 for f in feature_results
                      if "p_value" in f)
    best_improve = best["brier_improvement"] if best else 0.0
    if any_sig and best_improve > 0.001:
        verdict = "promising"
    elif any_raw_sig and best_improve > 0:
        verdict = "marginal"
    else:
        verdict = "no signal"

    # Headline
    if best:
        pv_txt = f"{best['p_value']:.4f}"
        bi = best["brier_improvement"]
        headline = (f"Best single feature: {best['name']} (Δ Brier "
                    f"{bi:+.5f}, p={pv_txt}); "
                    f"joint Δ Brier {joint.get('joint_brier_improvement', 0):+.5f}; "
                    f"verdict: {verdict}")
    else:
        headline = f"No feature could be fit; verdict: {verdict}"

    out = {
        "n_series": n,
        "baseline_brier": float(baseline_brier),
        "beta_used": BETA_BASELINE,
        "alpha_bonferroni": ALPHA_BONF,
        "n_features_tested": len(feat_names),
        "features": feature_results,
        "joint_model": joint,
        "best_feature_bootstrap": boot,
        "headline": headline,
        "verdict": verdict,
        "generated": datetime.utcnow().isoformat() + "Z",
    }

    out_path = os.path.join(DATA_DIR, "projection_research_form.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Console summary
    print("\n=== Per-feature results ===")
    for f in feature_results:
        if "error" in f:
            print(f"  {f['name']:<22}  n={f['n_used']:>4}   ERROR: {f['error']}")
            continue
        sig = "*" if f["significant_bonferroni"] else " "
        print(f"  {f['name']:<22}  n={f['n_used']:>4}  "
              f"coef={f['coef']:+.4f} (se={f['se']:.4f})  "
              f"p={f['p_value']:.4f}{sig}  "
              f"ΔBrier={f['brier_improvement']:+.5f}  "
              f"LR-p={f['lr_p']:.4f}")
    print(f"\n* = significant at Bonferroni α = {ALPHA_BONF:.4f}")

    print("\n=== Joint model ===")
    if "error" in joint:
        print(f"  ERROR: {joint['error']}  (n_used={joint['n_used']})")
    else:
        print(f"  n_used={joint['n_used']}  retained (Bonferroni): {joint['features_retained']}")
        print(f"  Joint Brier {joint['joint_brier']:.4f}  vs baseline-on-subset {joint['baseline_brier_on_subset']:.4f}  "
              f"(Δ {joint['joint_brier_improvement']:+.5f})")
        print(f"  LR vs baseline: stat={joint['lr_stat']:.3f}  p={joint['lr_p']:.4f}")

    if boot:
        print(f"\n=== Bootstrap on best feature: {boot['feature']} ===")
        print(f"  Δ Brier mean={boot['mean']:+.5f}  95% CI [{boot['ci_lo']:+.5f}, {boot['ci_hi']:+.5f}]  "
              f"({boot['pct_positive']*100:.1f}% of resamples > 0, n_boot={boot['n_boot']})")

    print(f"\nHEADLINE: {headline}")
    print(f"VERDICT:  {verdict}")


if __name__ == "__main__":
    main()
