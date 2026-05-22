"""
ResearchH2HCommonOpponent.py
-----------------------------
Tests whether head-to-head history and common-opponent bridge features
can statistically improve the BenPom baseline pre-match win-probability
predictions (p_fav = sigmoid(0.154 * |delta_rating|)).

Outputs: data/projection_research_h2h.json
"""

import json
import math
import os
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import chi2, norm
except Exception as e:
    raise SystemExit(f"scipy required: {e}")

try:
    import statsmodels.api as sm
except Exception as e:
    raise SystemExit(f"statsmodels required: {e}")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(DATA_DIR, "projection_research_h2h.json")

BASELINE_K = 0.154  # logistic slope in baseline win probability
ALPHA = 0.05
N_TESTS = 6
BONF_ALPHA = ALPHA / N_TESTS


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_events():
    files = ["rating_timeline_2024.json", "rating_timeline_2025.json", "rating_timeline.json"]
    out = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        with open(path) as fh:
            d = json.load(fh)
        for e in d["match_events"]:
            out.append(e)
    # Sort chronologically; ties broken by match_id
    out.sort(key=lambda e: (e["date"], e["match_id"]))
    return out


def days_between(d1: str, d2: str) -> int:
    a = datetime.strptime(d1, "%Y-%m-%d")
    b = datetime.strptime(d2, "%Y-%m-%d")
    return (b - a).days


def build_features(events):
    """
    For each event (series), compute:
      - p_baseline (favorite win prob from rating diff)
      - y (1 if favorite won)
      - H2H features (90d and ever)
      - common-opponent bridge features (60d window)
    Returns list of dicts.
    """
    # Per-team chronological history of (date, opponent, won_bool, maps_list)
    team_history = defaultdict(list)

    rows = []
    for e in events:
        w, l = e["winner"], e["loser"]
        wb, lb = e["winner_before"], e["loser_before"]
        date = e["date"]

        # Determine favorite by pre-match rating
        if wb >= lb:
            fav, dog = w, l
            fav_rating, dog_rating = wb, lb
            fav_won = 1
        else:
            fav, dog = l, w
            fav_rating, dog_rating = lb, wb
            fav_won = 0
        diff = fav_rating - dog_rating  # >= 0
        p_base = sigmoid(BASELINE_K * diff)

        # --- H2H lookups using prior history ---
        def prior_h2h(a, b, window_days=None):
            """Return list of prior matches between a and b (a-perspective wins/losses)."""
            hist = team_history.get(a, [])
            out = []
            for h in hist:
                if h["opp"] != b:
                    continue
                if window_days is not None:
                    if days_between(h["date"], date) > window_days:
                        continue
                out.append(h)
            return out

        h2h_90 = prior_h2h(fav, dog, 90)
        h2h_ever = prior_h2h(fav, dog, None)

        if h2h_90:
            h2h_count_90d = len(h2h_90)
            h2h_fav_wr_90d = sum(1 for h in h2h_90 if h["won"]) / len(h2h_90)
            h2h_fav_diff_90 = h2h_fav_wr_90d - p_base
        else:
            h2h_count_90d = 0
            h2h_fav_wr_90d = float("nan")
            h2h_fav_diff_90 = float("nan")

        if h2h_ever:
            h2h_count_ever = len(h2h_ever)
            h2h_fav_wr_ever = sum(1 for h in h2h_ever if h["won"]) / len(h2h_ever)
            h2h_fav_diff_ever = h2h_fav_wr_ever - p_base
        else:
            h2h_count_ever = 0
            h2h_fav_wr_ever = float("nan")
            h2h_fav_diff_ever = float("nan")

        # --- Common-opponent bridge in last 60d ---
        # For each team, find opponents played in last 60d with map-level round diffs.
        def recent_opps(team):
            cutoff = date  # date of upcoming series; only strictly prior matches
            res = defaultdict(lambda: 0)  # opp -> cumulative round diff for `team`
            counts = defaultdict(lambda: 0)
            for h in team_history.get(team, []):
                if days_between(h["date"], cutoff) > 60:
                    continue
                if days_between(h["date"], cutoff) <= 0:
                    # strictly prior: dates equal but already-processed (history only contains prior)
                    pass
                opp = h["opp"]
                rd = 0
                for m in h["maps"]:
                    # round diff from team's perspective
                    if m["winner"] == team:
                        rd += (m["wr"] - m["lr"])
                    else:
                        rd += (m["lr"] - m["wr"])
                res[opp] += rd
                counts[opp] += 1
            return res, counts

        fav_opps_rd, fav_opps_cnt = recent_opps(fav)
        dog_opps_rd, dog_opps_cnt = recent_opps(dog)

        common = [
            o for o in fav_opps_rd.keys()
            if o in dog_opps_rd and o != fav and o != dog
        ]
        if common:
            diffs = [fav_opps_rd[c] - dog_opps_rd[c] for c in common]
            bridge_score = float(np.mean(diffs))
            bridge_n = len(common)
        else:
            bridge_score = float("nan")
            bridge_n = 0

        # Binary: did F beat ANY common opponent that U lost to in last 60d?
        # We need not just "common opp" but specifically: F won vs C, U lost vs C.
        fav_recent_wins = set()
        fav_recent_losses = set()
        for h in team_history.get(fav, []):
            if days_between(h["date"], date) > 60:
                continue
            if h["won"]:
                fav_recent_wins.add(h["opp"])
            else:
                fav_recent_losses.add(h["opp"])
        dog_recent_wins = set()
        dog_recent_losses = set()
        for h in team_history.get(dog, []):
            if days_between(h["date"], date) > 60:
                continue
            if h["won"]:
                dog_recent_wins.add(h["opp"])
            else:
                dog_recent_losses.add(h["opp"])
        bridge_sign_fav = int(bool(fav_recent_wins & dog_recent_losses))
        # define "applicable" only when there was at least one common opponent
        bridge_sign_defined = 1 if common else 0

        rows.append({
            "match_id": e["match_id"],
            "date": date,
            "fav": fav, "dog": dog,
            "fav_rating": fav_rating, "dog_rating": dog_rating,
            "p_base": p_base,
            "y": fav_won,
            "h2h_count_90d": h2h_count_90d,
            "h2h_fav_wr_90d": h2h_fav_wr_90d,
            "h2h_fav_diff_90": h2h_fav_diff_90,
            "h2h_count_ever": h2h_count_ever,
            "h2h_fav_wr_ever": h2h_fav_wr_ever,
            "h2h_fav_diff_ever": h2h_fav_diff_ever,
            "bridge_score": bridge_score,
            "bridge_n": bridge_n,
            "bridge_sign_fav": bridge_sign_fav,
            "bridge_sign_defined": bridge_sign_defined,
        })

        # --- Append this match to each team's history AFTER feature compute ---
        team_history[w].append({
            "date": date, "opp": l, "won": True, "maps": e["maps"],
        })
        team_history[l].append({
            "date": date, "opp": w, "won": False, "maps": e["maps"],
        })

    return rows


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def brier(p_arr, y_arr):
    p = np.asarray(p_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    return float(np.mean((p - y) ** 2))


def fit_logit(X_cols, y):
    """X_cols is a 2D numpy array (already including intercept if desired). Returns model."""
    X = sm.add_constant(X_cols, has_constant="add")
    model = sm.Logit(y, X)
    res = model.fit(disp=False, method="newton", maxiter=200)
    return res


def lr_test(ll_full, ll_reduced, df):
    stat = 2 * (ll_full - ll_reduced)
    if stat < 0:
        stat = 0.0
    p = 1 - chi2.cdf(stat, df)
    return float(stat), float(p)


def evaluate_feature(rows, feature_key, mask_func=None):
    """
    Fits logistic regression: y ~ logit(p_base) + feature.
    Returns dict of stats.
    """
    if mask_func is None:
        mask_func = lambda r: not (isinstance(r[feature_key], float) and math.isnan(r[feature_key]))

    used = [r for r in rows if mask_func(r)]
    n_used = len(used)
    n_skipped = len(rows) - n_used

    if n_used < 30:
        return {
            "name": feature_key,
            "n_used": n_used,
            "n_skipped": n_skipped,
            "coef": None, "se": None, "p_value": None, "ci": [None, None],
            "brier_with_feature": None,
            "brier_improvement_on_subset": None,
            "lr_p": None,
            "significant_bonferroni": False,
            "note": "too few samples",
        }

    y = np.array([r["y"] for r in used], dtype=float)
    lp = np.array([logit(r["p_base"]) for r in used], dtype=float)
    feat = np.array([r[feature_key] for r in used], dtype=float)

    # Subset baseline Brier
    p_base_arr = np.array([r["p_base"] for r in used], dtype=float)
    brier_base_subset = brier(p_base_arr, y)

    # Reduced model: y ~ logit(p_base)
    X_red = lp.reshape(-1, 1)
    res_red = fit_logit(X_red, y)
    # Full: y ~ logit(p_base) + feature
    X_full = np.column_stack([lp, feat])
    res_full = fit_logit(X_full, y)

    # coef on feature is the last
    # statsmodels: with sm.add_constant, params order = [const, lp, feat]
    coef = float(res_full.params[2])
    se = float(res_full.bse[2])
    pval = float(res_full.pvalues[2])
    ci_lo = float(res_full.conf_int()[2, 0])
    ci_hi = float(res_full.conf_int()[2, 1])

    p_full = res_full.predict(sm.add_constant(X_full, has_constant="add"))
    brier_full = brier(p_full, y)

    lr_stat, lr_p = lr_test(res_full.llf, res_red.llf, df=1)

    return {
        "name": feature_key,
        "n_used": n_used,
        "n_skipped": n_skipped,
        "coef": coef,
        "se": se,
        "p_value": pval,
        "ci": [ci_lo, ci_hi],
        "brier_baseline_on_subset": brier_base_subset,
        "brier_with_feature": brier_full,
        "brier_improvement_on_subset": brier_base_subset - brier_full,
        "lr_stat": lr_stat,
        "lr_p": lr_p,
        "significant_bonferroni": bool(pval < BONF_ALPHA),
    }


def main():
    events = load_events()
    print(f"Loaded {len(events)} match events")

    rows = build_features(events)
    print(f"Built {len(rows)} feature rows")

    # Overall baseline brier
    p_all = np.array([r["p_base"] for r in rows])
    y_all = np.array([r["y"] for r in rows])
    base_brier = brier(p_all, y_all)
    print(f"Baseline Brier (all): {base_brier:.5f}, fav win rate: {y_all.mean():.3f}")

    features_to_test = [
        ("h2h_count_90d",
         lambda r: True),  # always defined
        ("h2h_fav_wr_90d",
         lambda r: not math.isnan(r["h2h_fav_wr_90d"])),
        ("h2h_fav_diff_90",
         lambda r: not math.isnan(r["h2h_fav_diff_90"])),
        ("h2h_fav_wr_ever",
         lambda r: not math.isnan(r["h2h_fav_wr_ever"])),
        ("bridge_score",
         lambda r: not math.isnan(r["bridge_score"])),
        ("bridge_sign_fav",
         lambda r: r["bridge_sign_defined"] == 1),
    ]

    results = []
    for fname, mask in features_to_test:
        r = evaluate_feature(rows, fname, mask)
        results.append(r)
        if r.get("coef") is None:
            print(f"  {fname}: skipped ({r.get('note')})")
            continue
        sig = " *BONF*" if r["significant_bonferroni"] else ""
        print(f"  {fname:25s} n={r['n_used']:4d}  coef={r['coef']:+.4f}  p={r['p_value']:.4f}  "
              f"BrierΔ={r['brier_improvement_on_subset']:+.5f}{sig}")

    # ----- Joint model: subset where ALL real-valued features are defined -----
    def joint_mask(r):
        return (not math.isnan(r["h2h_fav_diff_90"]) and
                not math.isnan(r["bridge_score"]))

    joint_rows = [r for r in rows if joint_mask(r)]
    joint_result = {}
    if len(joint_rows) >= 30:
        y = np.array([r["y"] for r in joint_rows], dtype=float)
        lp = np.array([logit(r["p_base"]) for r in joint_rows], dtype=float)
        feats = {
            "h2h_fav_diff_90": np.array([r["h2h_fav_diff_90"] for r in joint_rows], dtype=float),
            "h2h_fav_wr_ever": np.array([r["h2h_fav_wr_ever"] if not math.isnan(r["h2h_fav_wr_ever"]) else r["p_base"] for r in joint_rows], dtype=float),
            "bridge_score": np.array([r["bridge_score"] for r in joint_rows], dtype=float),
            "bridge_sign_fav": np.array([r["bridge_sign_fav"] for r in joint_rows], dtype=float),
        }
        X_red = lp.reshape(-1, 1)
        res_red = fit_logit(X_red, y)
        X_full = np.column_stack([lp] + list(feats.values()))
        res_full = fit_logit(X_full, y)
        p_base_arr = np.array([r["p_base"] for r in joint_rows], dtype=float)
        b_base = brier(p_base_arr, y)
        p_full = res_full.predict(sm.add_constant(X_full, has_constant="add"))
        b_full = brier(p_full, y)

        names = list(feats.keys())
        retained = []
        for i, n in enumerate(names):
            # params: [const, lp, feat1, feat2, ...]
            p_i = float(res_full.pvalues[2 + i])
            coef_i = float(res_full.params[2 + i])
            se_i = float(res_full.bse[2 + i])
            retained.append({
                "name": n,
                "coef": coef_i, "se": se_i, "p": p_i,
                "significant_bonferroni": bool(p_i < BONF_ALPHA),
            })
        joint_result = {
            "n_used": len(joint_rows),
            "features": retained,
            "joint_brier_baseline_on_subset": b_base,
            "joint_brier_with_features": b_full,
            "joint_brier_improvement": b_base - b_full,
            "features_retained_bonferroni": [r["name"] for r in retained if r["significant_bonferroni"]],
        }
        print(f"\nJoint model (n={len(joint_rows)}): BrierΔ = {b_base - b_full:+.5f}")
        for r in retained:
            mark = " *BONF*" if r["significant_bonferroni"] else ""
            print(f"  {r['name']:25s} coef={r['coef']:+.4f}  p={r['p']:.4f}{mark}")

    # ----- Verdict / headline -----
    any_bonf = any(r.get("significant_bonferroni") for r in results)
    any_nominal = any((r.get("p_value") is not None and r["p_value"] < 0.05) for r in results)
    best_improve = max(
        (r.get("brier_improvement_on_subset") or -1e9) for r in results
    )
    if any_bonf and best_improve > 0.001:
        verdict = "promising"
    elif any_nominal and best_improve > 0.0005:
        verdict = "marginal"
    else:
        verdict = "no signal"

    # Build headline
    best = None
    for r in results:
        if r.get("p_value") is None:
            continue
        if best is None or r["p_value"] < best["p_value"]:
            best = r
    if best is None:
        headline = "No features could be evaluated."
    else:
        headline = (
            f"Best feature {best['name']}: coef={best['coef']:+.3f}, "
            f"p={best['p_value']:.3f}, BrierΔ={best['brier_improvement_on_subset']:+.5f} "
            f"on n={best['n_used']}; verdict={verdict}."
        )

    out = {
        "n_series_total": len(rows),
        "baseline_brier": base_brier,
        "fav_win_rate": float(y_all.mean()),
        "bonferroni_alpha": BONF_ALPHA,
        "features": results,
        "joint_model": joint_result,
        "headline": headline,
        "verdict": verdict,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o)
    print(f"\nWrote {OUT_PATH}")
    print(f"HEADLINE: {headline}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
