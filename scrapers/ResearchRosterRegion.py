"""
ResearchRosterRegion.py
=======================

Test whether ROSTER-STABILITY and REGION-STRENGTH features improve BenPom
pre-match win-probability predictions for VCT 2024 -> 2026 stage 1.

Baseline:
    p_fav = sigmoid(BETA * |winner_before - loser_before|),    BETA = 0.154

Features (computed per side, then signed diff fav - dog):
    1. roster_change_diff           - did either side change starting roster in
                                      last 30 days? (signed: fav_changed - dog_changed)
    2. roster_intl_overlap_diff     - players overlapping with team's most-recent
                                      international roster (count diff)
    3. region_intl_6mo_diff         - region's intl win rate last 6 months
    4. same_region_flag             - 1 if both teams same region (unsigned)
    5. region_intl_ytd_diff         - region's intl W-L this calendar year

For each feature independently:
    Logit(y) = a + b*logit(p_baseline) + g*feature
Report coef, SE, p-value, 95% CI, Brier improvement, LR test p.

Joint model fits all five simultaneously. Bonferroni alpha = 0.05/5 = 0.01.

Output: data/projection_research_roster_region.json
"""

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
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

BETA_BASELINE = 0.154
N_FEATS = 5
ALPHA_BONF = 0.05 / N_FEATS


# ---- helpers ----
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


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


# ---- event metadata ----
INTL_EVENTS = {
    "2024_masters_madrid",
    "2024_masters_shanghai",
    "2024_champions",
    "2025_masters_bangkok",
    "2025_masters_toronto",
    "2025_champions",
    "2026_masters_santiago",
}


# ---- data loading ----
def load_match_events():
    files = [
        ("rating_timeline_2024.json", 2024),
        ("rating_timeline_2025.json", 2025),
        ("rating_timeline.json", 2026),
    ]
    matches = []
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
    matches.sort(key=lambda r: (r["date"], r["match_id"]))
    return matches


def load_event_rosters():
    """For every event CSV, return {event_id: {org: set(players), org_region: {org: region}}}.

    Also returns {team -> region} aggregated from all CSV Region columns.
    """
    rosters_by_event = {}
    team_region = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".csv"):
            continue
        # Skip non-event CSVs that don't match the season_event pattern
        if not (fname[0:4].isdigit()):
            continue
        eid = fname[:-4]
        path = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed to read {fname}: {e}", file=sys.stderr)
            continue
        if "Org" not in df.columns or "Player" not in df.columns:
            continue
        org_players = {}
        org_region = {}
        for org, grp in df.groupby("Org"):
            org_players[org] = set(grp["Player"].dropna().astype(str).tolist())
            if "Region" in df.columns:
                regs = grp["Region"].dropna().unique().tolist()
                if regs:
                    org_region[org] = regs[0]
                    # team_region: last seen wins (most recent event)
                    team_region[org] = regs[0]
        rosters_by_event[eid] = {"rosters": org_players, "region": org_region}
    return rosters_by_event, team_region


def load_event_dates(matches):
    """Map event_id -> sorted list of dates that event covers."""
    ev_dates = defaultdict(list)
    for m in matches:
        ev_dates[m["event_id"]].append(m["date"])
    for k in ev_dates:
        ev_dates[k].sort()
    return ev_dates


# ---- feature construction ----
def build_team_event_history(matches, rosters_by_event):
    """For each team, list of (date_of_first_match_in_event, event_id, roster_set).
    Sorted by date. We use 'first match date of the event for that team' as the
    timestamp the roster is 'observed at'.
    """
    team_event_first_date = defaultdict(dict)  # team -> {event_id: first_date}
    for m in matches:
        for side in (m["winner"], m["loser"]):
            d0 = team_event_first_date[side].get(m["event_id"])
            if d0 is None or m["date"] < d0:
                team_event_first_date[side][m["event_id"]] = m["date"]

    team_hist = defaultdict(list)
    for team, ev_dates in team_event_first_date.items():
        for eid, d in ev_dates.items():
            roster_info = rosters_by_event.get(eid)
            if not roster_info:
                continue
            roster = roster_info["rosters"].get(team)
            if not roster:
                continue
            team_hist[team].append({
                "event_id": eid,
                "date": parse_date(d),
                "roster": roster,
                "is_intl": eid in INTL_EVENTS,
            })
        team_hist[team].sort(key=lambda r: r["date"])
    return team_hist


def get_current_roster(team_hist_team, current_date):
    """Most recent event roster strictly BEFORE current_date (so we use the
    roster observed at the prior event)."""
    prev = None
    for ev in team_hist_team:
        if ev["date"] < current_date:
            prev = ev
        else:
            break
    return prev


def get_prior_intl_roster(team_hist_team, current_date):
    """Most recent international roster strictly BEFORE current_date."""
    prev = None
    for ev in team_hist_team:
        if ev["date"] < current_date and ev["is_intl"]:
            prev = ev
    return prev


def roster_changed_in_30d(team_hist_team, current_date):
    """Did the team's roster change between the event 30+ days ago and the
    most recent event before current_date?

    Returns 1/0, or None if we can't determine (no prior or no 30-day-prior
    comparison available)."""
    # Most recent event before current_date
    recent = None
    earlier = None
    for ev in team_hist_team:
        if ev["date"] >= current_date:
            break
        # earlier = most recent event whose date <= recent["date"] - 30d
        if recent is not None and (recent["date"] - ev["date"]).days >= 30:
            earlier = ev  # rolling — keep updating; final value is most recent that's >=30d before recent
        recent = ev
    if recent is None:
        return None
    # Re-derive 'earlier' as the most recent event with date <= recent.date - 30d
    threshold = recent["date"] - timedelta(days=30)
    earlier = None
    for ev in team_hist_team:
        if ev["date"] >= recent["date"]:
            break
        if ev["date"] <= threshold:
            earlier = ev
    if earlier is None:
        return None
    # Roster change = non-empty symmetric difference of starters
    if recent["roster"] != earlier["roster"]:
        return 1
    return 0


# ---- region features ----
def build_region_intl_records(matches, team_region):
    """For each (date, region), we need: last-6-months intl win rate and YTD
    intl W-L. We compute incrementally during the walk over matches.

    Returns helper that, given a date and region, returns the requested stats.
    Implementation: keep list of intl matches with their date and the region
    of each side and outcome; on demand, filter.
    """
    intl_matches = []
    for m in matches:
        if m["event_id"] not in INTL_EVENTS:
            continue
        wr = team_region.get(m["winner"])
        lr = team_region.get(m["loser"])
        if not wr or not lr:
            continue
        # Skip same-region intl matches (e.g., NA-vs-NA at Champions) — not
        # informative about region-vs-region strength.
        intl_matches.append({
            "date": parse_date(m["date"]),
            "winner_region": wr,
            "loser_region": lr,
        })
    intl_matches.sort(key=lambda r: r["date"])
    return intl_matches


def region_winrate_window(intl_matches, region, end_date, days):
    """Win rate (W / (W+L)) for `region` in cross-region intl matches that
    occurred in (end_date - days, end_date). Returns None if no qualifying
    matches."""
    start = end_date - timedelta(days=days)
    w = 0
    n = 0
    for im in intl_matches:
        if im["date"] >= end_date:
            break
        if im["date"] <= start:
            continue
        if im["winner_region"] == im["loser_region"]:
            continue
        if region == im["winner_region"]:
            w += 1; n += 1
        elif region == im["loser_region"]:
            n += 1
    if n == 0:
        return None
    return w / n


def region_winrate_ytd(intl_matches, region, end_date):
    """Win rate in cross-region intl matches in the same calendar year, up to
    (but not including) end_date."""
    year_start = datetime(end_date.year, 1, 1).date()
    w = 0
    n = 0
    for im in intl_matches:
        if im["date"] >= end_date:
            break
        if im["date"] < year_start:
            continue
        if im["winner_region"] == im["loser_region"]:
            continue
        if region == im["winner_region"]:
            w += 1; n += 1
        elif region == im["loser_region"]:
            n += 1
    if n == 0:
        return None
    return w / n


# ---- master feature table ----
def build_feature_table(matches, rosters_by_event, team_region, team_hist,
                        intl_matches, fallback_regions):
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
            fav_won = 0
        abs_delta = abs(wb - lb)
        p_base = float(sigmoid(BETA_BASELINE * abs_delta))

        def feat(team):
            hist = team_hist.get(team, [])
            # Roster turnover indicator (0/1, or None)
            rc = roster_changed_in_30d(hist, date)
            # Roster overlap with prior international
            cur = get_current_roster(hist, date)
            intl = get_prior_intl_roster(hist, date)
            if cur is not None and intl is not None:
                overlap = len(cur["roster"] & intl["roster"])
            else:
                overlap = None
            return {"roster_changed_30d": rc, "intl_overlap": overlap}

        f_fav = feat(fav)
        f_dog = feat(dog)

        # Region features
        fav_region = team_region.get(fav) or fallback_regions.get(fav)
        dog_region = team_region.get(dog) or fallback_regions.get(dog)
        same_region = None
        region_6mo_diff = None
        region_ytd_diff = None
        if fav_region and dog_region:
            same_region = 1 if fav_region == dog_region else 0
            fav_6 = region_winrate_window(intl_matches, fav_region, date, 183)
            dog_6 = region_winrate_window(intl_matches, dog_region, date, 183)
            if fav_6 is not None and dog_6 is not None:
                region_6mo_diff = fav_6 - dog_6
            fav_ytd = region_winrate_ytd(intl_matches, fav_region, date)
            dog_ytd = region_winrate_ytd(intl_matches, dog_region, date)
            if fav_ytd is not None and dog_ytd is not None:
                region_ytd_diff = fav_ytd - dog_ytd

        diff = {}
        # Roster change diff: fav_changed - dog_changed
        if f_fav["roster_changed_30d"] is not None and f_dog["roster_changed_30d"] is not None:
            diff["roster_change_diff"] = f_fav["roster_changed_30d"] - f_dog["roster_changed_30d"]
        else:
            diff["roster_change_diff"] = None
        # Intl roster overlap diff
        if f_fav["intl_overlap"] is not None and f_dog["intl_overlap"] is not None:
            diff["roster_intl_overlap_diff"] = f_fav["intl_overlap"] - f_dog["intl_overlap"]
        else:
            diff["roster_intl_overlap_diff"] = None
        diff["region_intl_6mo_diff"] = region_6mo_diff
        diff["same_region_flag"] = same_region
        diff["region_intl_ytd_diff"] = region_ytd_diff

        rows.append({
            "date": m["date"],
            "match_id": m["match_id"],
            "season": m["season"],
            "event_id": m["event_id"],
            "fav": fav, "dog": dog,
            "fav_region": fav_region, "dog_region": dog_region,
            "p_base": p_base, "y": fav_won,
            "diff": diff,
        })
    return rows


# ---- single-feature analysis ----
def analyze_feature(rows, feat_name):
    sub = [r for r in rows if r["diff"].get(feat_name) is not None]
    n = len(sub)
    if n < 30:
        return {"name": feat_name, "n_used": n, "error": "too few samples"}
    y = np.array([r["y"] for r in sub], dtype=float)
    base_logit = np.array([safe_logit(r["p_base"]) for r in sub])
    feat = np.array([r["diff"][feat_name] for r in sub], dtype=float)

    X0 = np.column_stack([np.ones(n), base_logit])
    b0, cov0, ll0 = fit_logit_newton(X0, y)
    p0 = predict_logit(X0, b0)
    brier0 = brier(p0, y)

    X1 = np.column_stack([np.ones(n), base_logit, feat])
    b1, cov1, ll1 = fit_logit_newton(X1, y)
    p1 = predict_logit(X1, b1)
    brier1 = brier(p1, y)

    coef = float(b1[2])
    se = float(np.sqrt(max(cov1[2, 2], 0.0)))
    z = coef / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se
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


def analyze_joint(rows, feat_names):
    sub = [r for r in rows if all(r["diff"].get(k) is not None for k in feat_names)]
    n = len(sub)
    if n < 30:
        return {"n_used": n, "error": "too few complete-case rows"}
    y = np.array([r["y"] for r in sub], dtype=float)
    base_logit = np.array([safe_logit(r["p_base"]) for r in sub])
    feats = np.column_stack([[r["diff"][k] for r in sub] for k in feat_names])

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


# ---- main ----
def main():
    matches = load_match_events()
    print(f"Loaded {len(matches)} qualifying series (winner_before != loser_before).")

    rosters_by_event, team_region_csv = load_event_rosters()
    print(f"Loaded rosters for {len(rosters_by_event)} events; CSV gave {len(team_region_csv)} team->region mappings.")

    # Augment region map with MapElo's ORG_REGIONS (canonical) as fallback
    fallback_regions = {}
    try:
        from MapElo import ORG_REGIONS
        fallback_regions = dict(ORG_REGIONS)
        print(f"Loaded MapElo ORG_REGIONS ({len(fallback_regions)} teams).")
    except Exception as e:
        print(f"[warn] MapElo import failed: {e}", file=sys.stderr)

    # Merged: prefer ORG_REGIONS (canonical from MapElo); fallback to CSV
    team_region = dict(team_region_csv)
    for k, v in fallback_regions.items():
        team_region[k] = v  # MapElo wins on conflict

    team_hist = build_team_event_history(matches, rosters_by_event)
    print(f"Built team history for {len(team_hist)} teams.")

    intl_matches = build_region_intl_records(matches, team_region)
    print(f"Indexed {len(intl_matches)} intl matches with known regions.")

    rows = build_feature_table(matches, rosters_by_event, team_region,
                                team_hist, intl_matches, fallback_regions)
    n = len(rows)
    print(f"Built feature table: {n} series.")

    # Full-sample baseline Brier
    y_all = np.array([r["y"] for r in rows], dtype=float)
    p_all = np.array([r["p_base"] for r in rows], dtype=float)
    baseline_brier = brier(p_all, y_all)
    print(f"Full-sample baseline Brier: {baseline_brier:.4f}")

    feat_names = [
        "roster_change_diff",
        "roster_intl_overlap_diff",
        "region_intl_6mo_diff",
        "same_region_flag",
        "region_intl_ytd_diff",
    ]

    # Diagnostics: how often is each feature defined?
    for fn in feat_names:
        defined = sum(1 for r in rows if r["diff"].get(fn) is not None)
        print(f"  feature {fn:<28}  defined in {defined}/{n} rows")

    feature_results = [analyze_feature(rows, fn) for fn in feat_names]
    joint = analyze_joint(rows, feat_names)

    # Verdict
    any_bonf = any(f.get("significant_bonferroni") for f in feature_results)
    any_raw = any(f.get("p_value", 1) < 0.05 for f in feature_results if "p_value" in f)
    best = None
    eligible = [f for f in feature_results if "brier_improvement" in f]
    if eligible:
        best = max(eligible, key=lambda f: f["brier_improvement"])
    best_imp = best["brier_improvement"] if best else 0.0
    if any_bonf and best_imp > 0.001:
        verdict = "promising"
    elif any_raw and best_imp > 0:
        verdict = "marginal"
    else:
        verdict = "no signal"

    if best:
        headline = (f"Best feature: {best['name']} (Δ Brier {best['brier_improvement']:+.5f}, "
                    f"p={best['p_value']:.4f}); joint Δ Brier "
                    f"{joint.get('joint_brier_improvement', 0):+.5f}; verdict: {verdict}")
    else:
        headline = f"No feature fit; verdict: {verdict}"

    # Trim feature results to required schema for output
    feat_out = []
    for f in feature_results:
        if "error" in f:
            feat_out.append({"name": f["name"], "n_used": f["n_used"], "error": f["error"]})
            continue
        feat_out.append({
            "name": f["name"],
            "coef": f["coef"],
            "se": f["se"],
            "p_value": f["p_value"],
            "ci": f["ci"],
            "brier_with_feature": f["brier_with_feature"],
            "brier_improvement": f["brier_improvement"],
            "lr_p": f["lr_p"],
            "n_used": f["n_used"],
            "significant_bonferroni": f["significant_bonferroni"],
            "baseline_brier_on_subset": f["baseline_brier_on_subset"],
        })

    out = {
        "n_series": n,
        "baseline_brier": float(baseline_brier),
        "beta_used": BETA_BASELINE,
        "alpha_bonferroni": ALPHA_BONF,
        "n_features_tested": len(feat_names),
        "features": feat_out,
        "joint_model": {
            "n_used": joint.get("n_used"),
            "features_retained": joint.get("features_retained", []),
            "joint_brier": joint.get("joint_brier"),
            "joint_brier_improvement": joint.get("joint_brier_improvement"),
            "baseline_brier_on_subset": joint.get("baseline_brier_on_subset"),
            "lr_p": joint.get("lr_p"),
            "per_feature": joint.get("per_feature"),
        },
        "headline": headline,
        "verdict": verdict,
        "generated": datetime.utcnow().isoformat() + "Z",
    }

    out_path = os.path.join(DATA_DIR, "projection_research_roster_region.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Console summary
    print("\n=== Per-feature results ===")
    for f in feature_results:
        if "error" in f:
            print(f"  {f['name']:<28}  n={f['n_used']:>4}   ERROR: {f['error']}")
            continue
        sig = "*" if f["significant_bonferroni"] else " "
        print(f"  {f['name']:<28}  n={f['n_used']:>4}  "
              f"coef={f['coef']:+.4f} (se={f['se']:.4f})  "
              f"p={f['p_value']:.4f}{sig}  "
              f"ΔBrier={f['brier_improvement']:+.5f}  "
              f"LR-p={f['lr_p']:.4f}")
    print(f"\n* = significant at Bonferroni alpha = {ALPHA_BONF:.4f}")

    print("\n=== Joint model ===")
    if "error" in joint:
        print(f"  ERROR: {joint['error']}  (n_used={joint['n_used']})")
    else:
        print(f"  n_used={joint['n_used']}  retained: {joint['features_retained']}")
        print(f"  Joint Brier {joint['joint_brier']:.4f}  vs baseline-on-subset {joint['baseline_brier_on_subset']:.4f}  "
              f"(Δ {joint['joint_brier_improvement']:+.5f})  LR-p={joint['lr_p']:.4f}")

    print(f"\nHEADLINE: {headline}")
    print(f"VERDICT:  {verdict}")


if __name__ == "__main__":
    main()
