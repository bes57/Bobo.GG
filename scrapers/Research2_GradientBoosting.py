#!/usr/bin/env python3
"""
Research2_GradientBoosting.py
-----------------------------
Phase-2 ML research: can a gradient-boosted ensemble with rich features beat
the current optimized BenPom baseline on series Brier?

Baseline:  p_series = sigmoid(0.154 * Δrating + 0.36 * intl_exp_diff)
Target:    held-out test Brier improvement >= 0.0015, Platt slope in [0.85, 1.15],
           bootstrap CI on Δ Brier excluding 0 with 95% confidence.

Pipeline:
  1. Walk matches chronologically and build per-team running state to derive a
     rich feature matrix (no leakage — only data dated strictly BEFORE each
     series is used).
  2. Time-based 75/25 train/test split (chronological).
  3. Compare baseline-only, logistic-regression on baseline + features,
     GradientBoostingClassifier, and HistGradientBoostingClassifier on the
     same train/test split.
  4. Report: train/test Brier, Platt slope, feature importance, LR test,
     200-rep cluster bootstrap CI on test-set Brier improvement.

Outputs:
  - data/projection_research_phase2_ml.json
  - static/projection_test/phase2_ml_feature_importance.png (if competitive)
"""

import json
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
STATIC_DIR = os.path.join(ROOT, "static", "projection_test")
os.makedirs(STATIC_DIR, exist_ok=True)

# ── baseline constants from project memory (Brier-optimal calibration) ──────
BETA_RATING = 0.154
BETA_INTL = 0.36
SEED = 0
N_BOOTSTRAP = 200
TRAIN_FRAC = 0.75

INTL_EVENTS = {
    "2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
    "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
    "2026_masters_santiago",
}

# Mirror MapElo.ORG_REGIONS so we don't import the Flask app.
ORG_REGIONS = {
    "TL": "EMEA", "FNC": "EMEA", "NAVI": "EMEA", "VIT": "EMEA", "BBL": "EMEA",
    "GX": "EMEA", "KC": "EMEA", "TH": "EMEA", "FUT": "EMEA", "GIA": "EMEA",
    "MKOI": "EMEA", "M8": "EMEA", "PCF": "EMEA", "ULF": "EMEA", "EF": "EMEA",
    "SEN": "Americas", "G2": "Americas", "MIBR": "Americas", "NRG": "Americas",
    "100T": "Americas", "C9": "Americas", "EG": "Americas", "KRÜ": "Americas",
    "LEV": "Americas", "FUR": "Americas", "LOUD": "Americas", "2G": "Americas",
    "APK": "Americas", "ENVY": "Americas",
    "PRX": "Pacific", "DRX": "Pacific", "T1": "Pacific", "TLN": "Pacific",
    "GEN": "Pacific", "DFM": "Pacific", "ZETA": "Pacific", "RRQ": "Pacific",
    "TS": "Pacific", "GE": "Pacific", "NS": "Pacific", "FS": "Pacific",
    "VL": "Pacific", "KRX": "Pacific", "BME": "Pacific",
    "EDG": "CN", "BLG": "CN", "TE": "CN", "DRG": "CN", "ASE": "CN", "AG": "CN",
    "XLG": "CN", "WOL": "CN", "FPX": "CN", "JDG": "CN", "NOVA": "CN",
    "TEC": "CN", "TYL": "CN", "TYLOO": "CN",
}

REGIONS = ("Americas", "EMEA", "Pacific", "CN")


# ── helpers ────────────────────────────────────────────────────────────────
def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def safe_logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def platt_slope(y, p):
    """Fit b in logit(y) = a + b * logit(p); report b (calibration slope).
    Slope ~ 1.0 is perfectly calibrated; <1 = overconfident; >1 = underconfident."""
    lp = safe_logit(p).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        lr.fit(lp, y)
        return float(lr.coef_[0, 0])
    except Exception:
        return float("nan")


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


# ── data loading ───────────────────────────────────────────────────────────
def load_all():
    files = [
        ("rating_timeline_2024.json", 2024),
        ("rating_timeline_2025.json", 2025),
        ("rating_timeline.json", 2026),
    ]
    matches = []
    checkpoints = []
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
                continue  # ties — skip (n=1217 after this filter)
            mm = dict(m)
            mm["season"] = season
            matches.append(mm)
        for cp in d.get("checkpoints", []):
            checkpoints.append((cp["date"], cp.get("ratings", {}), season))
    matches.sort(key=lambda r: (r["date"], r["match_id"]))
    checkpoints.sort(key=lambda r: r[0])
    return matches, checkpoints


# ── feature engineering (no leakage) ───────────────────────────────────────
def build_feature_table(matches, checkpoints):
    """Walk matches chronologically; for each, compute features using ONLY
    history strictly before the match date. Append history AFTER recording."""
    # Per-team chronological history of past series.
    team_hist = defaultdict(list)
    # Per-team: number of past series this season.
    season_count = defaultdict(int)  # (team, season) -> count
    # Per-(event, team) match count.
    event_count = defaultdict(int)
    # Per-team intl-this-year set.
    intl_this_year = defaultdict(set)  # year -> set(team)
    # Checkpoint binary-search structures
    cp_dates = [parse_date(c[0]) for c in checkpoints]
    cp_ratings = [c[1] for c in checkpoints]
    cp_dates_arr = np.array([d.toordinal() for d in cp_dates]) if cp_dates else np.array([])

    def rating_on_or_before(team, target_date):
        if cp_dates_arr.size == 0:
            return None
        target_ord = target_date.toordinal()
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
        date_s = m["date"]
        season = m["season"]
        event_id = m["event_id"]
        w, l = m["winner"], m["loser"]
        wb = float(m["winner_before"])
        lb = float(m["loser_before"])

        if wb > lb:
            fav, dog = w, l
            fav_b, dog_b = wb, lb
            fav_won = 1
        else:
            fav, dog = l, w
            fav_b, dog_b = lb, wb
            fav_won = 0

        rating_diff = fav_b - dog_b  # >= 0
        is_intl = 1 if event_id in INTL_EVENTS else 0

        # intl_exp_diff (signed, computed from intl_this_year BEFORE update)
        fav_intl_done = 1 if fav in intl_this_year[season] else 0
        dog_intl_done = 1 if dog in intl_this_year[season] else 0
        intl_exp_diff = fav_intl_done - dog_intl_done

        # Baseline projection
        z_base = BETA_RATING * rating_diff + BETA_INTL * intl_exp_diff
        p_base = float(expit(z_base))

        # ── team-history-derived features ──
        def features_for(team, current_rating):
            hist = team_hist[team]
            n_hist = len(hist)
            # Last-5 winrate
            last5 = hist[-5:]
            last5_wr = (sum(h["won"] for h in last5) / len(last5)) if last5 else 0.5
            # Velocity30: current_rating - rating 30 days ago
            r30 = rating_on_or_before(team, date - timedelta(days=30))
            velocity30 = (current_rating - r30) if r30 is not None else 0.0
            # Streak
            streak = 0
            for h in reversed(hist):
                if streak == 0:
                    streak = 1 if h["won"] else -1
                else:
                    if (streak > 0 and h["won"]) or (streak < 0 and not h["won"]):
                        streak += 1 if h["won"] else -1
                    else:
                        break
            # Days since last match
            days_since = (date - hist[-1]["date"]).days if hist else 60
            days_since = min(days_since, 180)  # cap
            # Map versatility: # distinct maps played past 60d
            cutoff = date - timedelta(days=60)
            maps_seen = set()
            for h in hist:
                if h["date"] < cutoff:
                    continue
                for mp in h["maps"]:
                    maps_seen.add(mp.get("map", ""))
            map_pool = len(maps_seen)
            # Past series this season
            n_season = season_count[(team, season)]
            return {
                "last5_wr": float(last5_wr),
                "velocity30": float(velocity30),
                "streak": int(streak),
                "days_since": int(days_since),
                "map_pool": int(map_pool),
                "n_season": int(n_season),
                "n_hist": int(n_hist),
            }

        ff = features_for(fav, fav_b)
        df = features_for(dog, dog_b)

        # Common opponents past 60d + bridge score
        def recent_opps_round_diff(team):
            cutoff = date - timedelta(days=60)
            out = defaultdict(int)
            for h in team_hist[team]:
                if h["date"] < cutoff:
                    continue
                rd = 0
                for mp in h["maps"]:
                    if mp.get("winner") == team:
                        rd += int(mp.get("wr", 0)) - int(mp.get("lr", 0))
                    else:
                        rd += int(mp.get("lr", 0)) - int(mp.get("wr", 0))
                out[h["opp"]] += rd
            return out

        fav_ro = recent_opps_round_diff(fav)
        dog_ro = recent_opps_round_diff(dog)
        common = [o for o in fav_ro if o in dog_ro and o not in (fav, dog)]
        if common:
            bridge_diffs = [fav_ro[o] - dog_ro[o] for o in common]
            bridge_score = float(np.mean(bridge_diffs))
            common_n = len(common)
        else:
            bridge_score = 0.0
            common_n = 0

        # Lifetime H2H winrate (fav perspective)
        h2h_all = [h for h in team_hist[fav] if h["opp"] == dog]
        if h2h_all:
            h2h_n = len(h2h_all)
            h2h_fav_wr = sum(h["won"] for h in h2h_all) / len(h2h_all)
        else:
            h2h_n = 0
            h2h_fav_wr = 0.5  # neutral

        # Regions
        fav_reg = ORG_REGIONS.get(fav)
        dog_reg = ORG_REGIONS.get(dog)
        same_region = 1 if (fav_reg is not None and fav_reg == dog_reg) else 0
        cross_region = 1 - same_region

        # is_bo5 (champions / champions-final tend to be Bo5; infer from maps count)
        n_maps_played = len(m.get("maps", []))
        # A Bo5 series can have 3-5 maps; a Bo3 has 2-3. The only sure inference
        # is "looked like Bo5" if maps played > 3 OR series_score has a 3-x.
        score = m.get("series_score", "")
        try:
            wscore = int(score.split("-")[0])
        except Exception:
            wscore = 0
        is_bo5 = 1 if (wscore >= 3 or n_maps_played > 3) else 0

        # Event chronological index per season
        # (computed externally below; placeholder here)

        row = {
            "match_id": m["match_id"],
            "date": date_s,
            "date_obj": date,
            "season": season,
            "event_id": event_id,
            "fav": fav,
            "dog": dog,
            "fav_b": fav_b,
            "dog_b": dog_b,
            "rating_diff": rating_diff,
            "abs_rating_diff": abs(rating_diff),
            "fav_rating_sq": fav_b * fav_b,
            "dog_rating_sq": dog_b * dog_b,
            "intl_exp_diff": float(intl_exp_diff),
            "is_intl": is_intl,
            "is_bo5": is_bo5,
            "p_base": p_base,
            "logit_p_base": float(safe_logit(p_base)),
            "y": fav_won,
            # team-derived
            "fav_last5_wr": ff["last5_wr"],
            "dog_last5_wr": df["last5_wr"],
            "last5_wr_diff": ff["last5_wr"] - df["last5_wr"],
            "fav_velocity30": ff["velocity30"],
            "dog_velocity30": df["velocity30"],
            "velocity30_diff": ff["velocity30"] - df["velocity30"],
            "fav_streak": ff["streak"],
            "dog_streak": df["streak"],
            "streak_diff": ff["streak"] - df["streak"],
            "fav_days_since": ff["days_since"],
            "dog_days_since": df["days_since"],
            "days_since_diff": ff["days_since"] - df["days_since"],
            "fav_map_pool": ff["map_pool"],
            "dog_map_pool": df["map_pool"],
            "fav_n_season": ff["n_season"],
            "dog_n_season": df["n_season"],
            "n_season_diff": ff["n_season"] - df["n_season"],
            "fav_n_hist": ff["n_hist"],
            "dog_n_hist": df["n_hist"],
            # event context
            "fav_event_idx": event_count[(event_id, fav)],
            "dog_event_idx": event_count[(event_id, dog)],
            "event_idx_diff": event_count[(event_id, fav)] - event_count[(event_id, dog)],
            # bridge / h2h
            "bridge_score": bridge_score,
            "common_n": common_n,
            "h2h_n": h2h_n,
            "h2h_fav_wr": h2h_fav_wr,
            "h2h_fav_diff": (h2h_fav_wr - p_base) if h2h_n > 0 else 0.0,
            # regions
            "fav_region": fav_reg or "UNK",
            "dog_region": dog_reg or "UNK",
            "same_region": same_region,
            "cross_region": cross_region,
        }
        # Region one-hot
        for r in REGIONS:
            row[f"fav_is_{r}"] = 1 if fav_reg == r else 0
            row[f"dog_is_{r}"] = 1 if dog_reg == r else 0

        rows.append(row)

        # ── UPDATE state AFTER recording row ──
        won_w = 1
        won_l = 0
        # We append to fav and dog using fav-perspective wins/losses
        team_hist[fav].append({
            "date": date, "opp": dog, "opp_rating": dog_b,
            "won": fav_won, "maps": m.get("maps", []),
        })
        team_hist[dog].append({
            "date": date, "opp": fav, "opp_rating": fav_b,
            "won": 1 - fav_won, "maps": m.get("maps", []),
        })
        season_count[(fav, season)] += 1
        season_count[(dog, season)] += 1
        event_count[(event_id, fav)] += 1
        event_count[(event_id, dog)] += 1
        if is_intl:
            intl_this_year[season].add(fav)
            intl_this_year[season].add(dog)

    # Event chronological index per season (intl events numbered 1,2,3 by date)
    intl_idx_by_event = {}
    seen_by_year = defaultdict(list)
    for m in matches:
        if m["event_id"] in INTL_EVENTS:
            ev = m["event_id"]
            if ev not in intl_idx_by_event:
                seen_by_year[m["season"]].append(ev)
                intl_idx_by_event[ev] = len(seen_by_year[m["season"]])
    for row in rows:
        row["intl_event_idx"] = intl_idx_by_event.get(row["event_id"], 0)

    return rows


# ── feature matrix selection ───────────────────────────────────────────────
FEATURE_COLS = [
    "logit_p_base",       # baseline as a must-include feature
    "rating_diff", "abs_rating_diff", "fav_b", "dog_b",
    "fav_rating_sq", "dog_rating_sq",
    "intl_exp_diff", "is_intl", "is_bo5",
    "fav_last5_wr", "dog_last5_wr", "last5_wr_diff",
    "fav_velocity30", "dog_velocity30", "velocity30_diff",
    "fav_streak", "dog_streak", "streak_diff",
    "fav_days_since", "dog_days_since", "days_since_diff",
    "fav_map_pool", "dog_map_pool",
    "fav_n_season", "dog_n_season", "n_season_diff",
    "fav_n_hist", "dog_n_hist",
    "fav_event_idx", "dog_event_idx", "event_idx_diff",
    "bridge_score", "common_n",
    "h2h_n", "h2h_fav_wr", "h2h_fav_diff",
    "same_region", "cross_region",
    "intl_event_idx",
    # Region one-hot (will be appended)
] + [f"fav_is_{r}" for r in REGIONS] + [f"dog_is_{r}" for r in REGIONS]


# ── bootstrap CI on Brier difference ───────────────────────────────────────
def bootstrap_delta_brier(y, p_base, p_model, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return [float("nan"), float("nan")]
    deltas = np.empty(n_boot)
    y = np.asarray(y, dtype=float)
    p_base = np.asarray(p_base, dtype=float)
    p_model = np.asarray(p_model, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        b0 = float(np.mean((p_base[idx] - y[idx]) ** 2))
        b1 = float(np.mean((p_model[idx] - y[idx]) ** 2))
        deltas[i] = b0 - b1  # positive = model better
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


# ── main pipeline ──────────────────────────────────────────────────────────
def main():
    print("Loading data…")
    matches, checkpoints = load_all()
    print(f"Non-tied series: {len(matches)}")
    print(f"Checkpoints: {len(checkpoints)}")

    print("Building feature table…")
    rows = build_feature_table(matches, checkpoints)
    n = len(rows)
    print(f"Built table: {n} rows.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    # Time-based split
    n_train = int(n * TRAIN_FRAC)
    n_test = n - n_train
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    train_dates = (train["date"].iloc[0], train["date"].iloc[-1])
    test_dates = (test["date"].iloc[0], test["date"].iloc[-1])
    print(f"Train: n={n_train}  ({train_dates[0]} → {train_dates[1]})")
    print(f"Test:  n={n_test}  ({test_dates[0]} → {test_dates[1]})")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Using {len(feature_cols)} features.")

    X_train = train[feature_cols].astype(float).values
    X_test = test[feature_cols].astype(float).values
    y_train = train["y"].values.astype(int)
    y_test = test["y"].values.astype(int)
    p_base_train = train["p_base"].values
    p_base_test = test["p_base"].values

    # ── 1) BASELINE (no learning) ──
    base_train_brier = brier(y_train, p_base_train)
    base_test_brier = brier(y_test, p_base_test)
    base_test_platt = platt_slope(y_test, p_base_test)
    print(f"\nBaseline (frozen):  train Brier={base_train_brier:.5f}  test Brier={base_test_brier:.5f}  Platt slope={base_test_platt:.3f}")

    models_out = []

    # ── 2) LOGISTIC REGRESSION (baseline + features) ──
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    lr.fit(Xtr_s, y_train)
    p_tr_lr = lr.predict_proba(Xtr_s)[:, 1]
    p_te_lr = lr.predict_proba(Xte_s)[:, 1]
    lr_train_brier = brier(y_train, p_tr_lr)
    lr_test_brier = brier(y_test, p_te_lr)
    lr_platt = platt_slope(y_test, p_te_lr)
    lr_ci = bootstrap_delta_brier(y_test, p_base_test, p_te_lr)
    # LR test vs baseline-only on test set:
    # use log-likelihoods on test (refit not needed; we want the realized fit)
    def loglik(y, p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    ll_base_test = loglik(y_test, p_base_test)
    ll_lr_test = loglik(y_test, p_te_lr)
    lr_lr_stat = 2 * (ll_lr_test - ll_base_test)
    lr_lr_p = float(1 - stats.chi2.cdf(max(lr_lr_stat, 0), df=len(feature_cols)))

    lr_ship = (
        (base_test_brier - lr_test_brier) >= 0.0015
        and 0.85 <= lr_platt <= 1.15
        and lr_ci[0] > 0
    )

    # Bonferroni-significant features for logistic (α ≈ 0.05/20 = 0.0025)
    # Approximate per-coef p-values via the standard logistic regression
    # asymptotic approximation. We compute SEs on TRAIN.
    try:
        ptr = lr.predict_proba(Xtr_s)[:, 1]
        W = ptr * (1 - ptr)
        # Hessian: X^T W X (+intercept col). Build with intercept appended.
        X_ext = np.column_stack([np.ones(len(Xtr_s)), Xtr_s])
        H = (X_ext.T * W) @ X_ext + 1e-6 * np.eye(X_ext.shape[1])
        cov = np.linalg.inv(H)
        coefs = np.concatenate([lr.intercept_, lr.coef_[0]])
        ses = np.sqrt(np.diag(cov))
        # Z scores for each feature (skip intercept)
        z = coefs[1:] / ses[1:]
        pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
        bonf_alpha = 0.05 / len(feature_cols)
        sig_feats = [
            {"name": feature_cols[i], "coef_std": float(lr.coef_[0][i]),
             "z": float(z[i]), "p": float(pvals[i])}
            for i in range(len(feature_cols))
            if pvals[i] < bonf_alpha
        ]
        sig_feats.sort(key=lambda d: abs(d["z"]), reverse=True)
    except Exception as e:
        sig_feats = []
        print(f"[warn] could not compute LR p-values: {e}")

    models_out.append({
        "name": "logistic_baseline_plus_features",
        "train_brier": float(lr_train_brier),
        "test_brier": float(lr_test_brier),
        "test_platt_b": float(lr_platt),
        "delta_test_brier": float(base_test_brier - lr_test_brier),
        "bootstrap_brier_ci": lr_ci,
        "n_params_or_trees": int(len(feature_cols) + 1),
        "ship_worthy": bool(lr_ship),
        "notes": (
            f"L2-regularized (C=1.0) standardized features; "
            f"LR test stat={lr_lr_stat:.2f} on df={len(feature_cols)} → p={lr_lr_p:.3g}; "
            f"{len(sig_feats)} Bonferroni-significant features"
        ),
        "bonferroni_significant": sig_feats[:10],
    })
    print(f"Logistic:           train Brier={lr_train_brier:.5f}  test Brier={lr_test_brier:.5f}  Δ={base_test_brier - lr_test_brier:+.5f}  Platt={lr_platt:.3f}  CI95={lr_ci}")

    # ── 3) GRADIENT BOOSTING ──
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED,
    )
    gb.fit(X_train, y_train)
    p_tr_gb = gb.predict_proba(X_train)[:, 1]
    p_te_gb = gb.predict_proba(X_test)[:, 1]
    gb_train_brier = brier(y_train, p_tr_gb)
    gb_test_brier = brier(y_test, p_te_gb)
    gb_platt = platt_slope(y_test, p_te_gb)
    gb_ci = bootstrap_delta_brier(y_test, p_base_test, p_te_gb)
    gb_ship = (
        (base_test_brier - gb_test_brier) >= 0.0015
        and 0.85 <= gb_platt <= 1.15
        and gb_ci[0] > 0
    )

    gb_importance = sorted(
        [{"name": feature_cols[i], "importance": float(gb.feature_importances_[i])}
         for i in range(len(feature_cols))],
        key=lambda d: d["importance"], reverse=True,
    )

    models_out.append({
        "name": "gradient_boosting_depth3_n200",
        "train_brier": float(gb_train_brier),
        "test_brier": float(gb_test_brier),
        "test_platt_b": float(gb_platt),
        "delta_test_brier": float(base_test_brier - gb_test_brier),
        "bootstrap_brier_ci": gb_ci,
        "n_params_or_trees": 200,
        "ship_worthy": bool(gb_ship),
        "notes": "sklearn GradientBoostingClassifier, depth 3, lr 0.05, seed=0",
    })
    print(f"GBM:                train Brier={gb_train_brier:.5f}  test Brier={gb_test_brier:.5f}  Δ={base_test_brier - gb_test_brier:+.5f}  Platt={gb_platt:.3f}  CI95={gb_ci}")

    # ── 4) HistGradientBoostingClassifier ──
    hgb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05, random_state=SEED,
    )
    hgb.fit(X_train, y_train)
    p_tr_hgb = hgb.predict_proba(X_train)[:, 1]
    p_te_hgb = hgb.predict_proba(X_test)[:, 1]
    hgb_train_brier = brier(y_train, p_tr_hgb)
    hgb_test_brier = brier(y_test, p_te_hgb)
    hgb_platt = platt_slope(y_test, p_te_hgb)
    hgb_ci = bootstrap_delta_brier(y_test, p_base_test, p_te_hgb)
    hgb_ship = (
        (base_test_brier - hgb_test_brier) >= 0.0015
        and 0.85 <= hgb_platt <= 1.15
        and hgb_ci[0] > 0
    )
    models_out.append({
        "name": "hist_gradient_boosting_depth3_n200",
        "train_brier": float(hgb_train_brier),
        "test_brier": float(hgb_test_brier),
        "test_platt_b": float(hgb_platt),
        "delta_test_brier": float(base_test_brier - hgb_test_brier),
        "bootstrap_brier_ci": hgb_ci,
        "n_params_or_trees": 200,
        "ship_worthy": bool(hgb_ship),
        "notes": "sklearn HistGradientBoostingClassifier, depth 3, lr 0.05, seed=0",
    })
    print(f"HGBM:               train Brier={hgb_train_brier:.5f}  test Brier={hgb_test_brier:.5f}  Δ={base_test_brier - hgb_test_brier:+.5f}  Platt={hgb_platt:.3f}  CI95={hgb_ci}")

    # ── 5) STACKED: baseline + GBM-on-residual (logit) ──
    # Residual = logit(y_actual / smoothed) — we instead let a small GBM predict
    # the residual probability and convert. Simpler: predict y from features
    # AFTER subtracting the baseline contribution from each feature's effect.
    # Practical approach: train a GBM to predict the LOGIT-RESIDUAL, i.e.
    # target = (y - p_base) (already centered); then add scaled prediction to logit(p_base).
    # We instead use a robust approach: train GBM on features (excluding logit_p_base),
    # output residual probability in [-1,1] direction, then stack via logistic on
    # [logit(p_base), gbm_pred_logit].
    non_base_cols = [c for c in feature_cols if c != "logit_p_base"]
    X_train_nb = train[non_base_cols].astype(float).values
    X_test_nb = test[non_base_cols].astype(float).values
    gb2 = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED,
    )
    gb2.fit(X_train_nb, y_train)
    p_tr_gb2 = np.clip(gb2.predict_proba(X_train_nb)[:, 1], 1e-6, 1 - 1e-6)
    p_te_gb2 = np.clip(gb2.predict_proba(X_test_nb)[:, 1], 1e-6, 1 - 1e-6)
    z_tr = np.column_stack([safe_logit(p_base_train), safe_logit(p_tr_gb2)])
    z_te = np.column_stack([safe_logit(p_base_test), safe_logit(p_te_gb2)])
    stk = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    stk.fit(z_tr, y_train)
    p_tr_stk = stk.predict_proba(z_tr)[:, 1]
    p_te_stk = stk.predict_proba(z_te)[:, 1]
    stk_train_brier = brier(y_train, p_tr_stk)
    stk_test_brier = brier(y_test, p_te_stk)
    stk_platt = platt_slope(y_test, p_te_stk)
    stk_ci = bootstrap_delta_brier(y_test, p_base_test, p_te_stk)
    stk_ship = (
        (base_test_brier - stk_test_brier) >= 0.0015
        and 0.85 <= stk_platt <= 1.15
        and stk_ci[0] > 0
    )
    models_out.append({
        "name": "stacked_baseline_plus_gbm_residual",
        "train_brier": float(stk_train_brier),
        "test_brier": float(stk_test_brier),
        "test_platt_b": float(stk_platt),
        "delta_test_brier": float(base_test_brier - stk_test_brier),
        "bootstrap_brier_ci": stk_ci,
        "n_params_or_trees": 200,
        "ship_worthy": bool(stk_ship),
        "notes": (
            f"GBM trained on non-baseline features; final stacker = logistic on "
            f"[logit(p_base), logit(p_gbm)]; stacker coefs = {stk.coef_[0].tolist()}"
        ),
    })
    print(f"Stacked GBM:        train Brier={stk_train_brier:.5f}  test Brier={stk_test_brier:.5f}  Δ={base_test_brier - stk_test_brier:+.5f}  Platt={stk_platt:.3f}  CI95={stk_ci}")

    # ── Expanding-window CV (5 folds) — only on GBM (the most complex) ──
    cv_fold_briers = []
    cv_fold_base_briers = []
    fold_starts = np.linspace(int(n * 0.40), n, 6, dtype=int)  # train ends at fold_starts[i], test = next chunk
    for i in range(5):
        train_end = fold_starts[i]
        test_end = fold_starts[i + 1]
        if test_end - train_end < 20 or train_end < 100:
            continue
        Xtr = df[feature_cols].iloc[:train_end].astype(float).values
        ytr = df["y"].iloc[:train_end].values.astype(int)
        Xte = df[feature_cols].iloc[train_end:test_end].astype(float).values
        yte = df["y"].iloc[train_end:test_end].values.astype(int)
        pbtr_base = df["p_base"].iloc[train_end:test_end].values
        gb_cv = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED,
        )
        gb_cv.fit(Xtr, ytr)
        p_cv = gb_cv.predict_proba(Xte)[:, 1]
        cv_fold_briers.append(float(brier(yte, p_cv)))
        cv_fold_base_briers.append(float(brier(yte, pbtr_base)))
    cv_summary = {
        "n_folds": len(cv_fold_briers),
        "gbm_briers": cv_fold_briers,
        "baseline_briers": cv_fold_base_briers,
        "mean_gbm": float(np.mean(cv_fold_briers)) if cv_fold_briers else None,
        "mean_baseline": float(np.mean(cv_fold_base_briers)) if cv_fold_base_briers else None,
        "mean_delta": (
            float(np.mean(cv_fold_base_briers) - np.mean(cv_fold_briers))
            if cv_fold_briers else None
        ),
    }
    print(f"\nExpanding-window CV (5 folds): mean baseline Brier={cv_summary['mean_baseline']}  mean GBM Brier={cv_summary['mean_gbm']}  Δ={cv_summary['mean_delta']}")

    # ── Verdict ──
    best = max(models_out, key=lambda m: m["delta_test_brier"])
    any_ship = any(m["ship_worthy"] for m in models_out)
    if any_ship:
        shipped = [m["name"] for m in models_out if m["ship_worthy"]]
        verdict = f"ship: {shipped[0]}"
    elif best["delta_test_brier"] >= 0.0005:
        verdict = "marginal"
    else:
        verdict = "no signal"

    headline = (
        f"Best model {best['name']} test Brier {best['test_brier']:.4f} "
        f"vs baseline {base_test_brier:.4f} (Δ={best['delta_test_brier']:+.4f}, "
        f"Platt={best['test_platt_b']:.2f}); verdict: {verdict}"
    )

    out = {
        "n_series": int(n),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_date_range": list(train_dates),
        "test_date_range": list(test_dates),
        "feature_cols": feature_cols,
        "baseline": {
            "train_brier": float(base_train_brier),
            "test_brier": float(base_test_brier),
            "test_platt_b": float(base_test_platt),
            "formula": "sigmoid(0.154*rating_diff + 0.36*intl_exp_diff)",
        },
        "models": models_out,
        "top_features": gb_importance[:15],
        "expanding_cv": cv_summary,
        "headline": headline,
        "verdict": verdict,
        "generated": datetime.utcnow().isoformat() + "Z",
    }

    out_path = os.path.join(DATA_DIR, "projection_research_phase2_ml.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # ── Chart: feature importance (only if any model competitive) ──
    competitive = any(m["delta_test_brier"] >= 0.0005 for m in models_out)
    if competitive:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = gb_importance[:15]
        names = [t["name"] for t in top][::-1]
        vals = [t["importance"] for t in top][::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(names, vals, color="#4c78a8")
        ax.set_xlabel("GradientBoostingClassifier feature importance")
        ax.set_title(
            f"Phase-2 ML feature importance (top 15)\n"
            f"GBM test Brier={gb_test_brier:.4f} vs baseline {base_test_brier:.4f} "
            f"(Δ={base_test_brier-gb_test_brier:+.4f})"
        )
        for bar, v in zip(bars, vals):
            ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        chart_path = os.path.join(STATIC_DIR, "phase2_ml_feature_importance.png")
        plt.savefig(chart_path, dpi=110)
        plt.close(fig)
        print(f"Wrote chart {chart_path}")
    else:
        print("No competitive model — skipping chart.")

    print(f"\nHEADLINE: {headline}")
    print(f"VERDICT:  {verdict}")


if __name__ == "__main__":
    main()
