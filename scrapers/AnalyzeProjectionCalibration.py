"""
AnalyzeProjectionCalibration.py
================================

Deep calibration analysis of BenPom pre-match win probabilities against
historical VCT match outcomes (2024 → present).

For each match in the rating timelines we have winner_before / loser_before
(the BenPom rating JUST before the match).  We turn that into a model
probability via sigmoid(BETA * delta), with BETA = 0.136 (SNAP_BETA).

We evaluate at TWO levels:

  - Map level   : every map of every series, favorite-side win or loss.
  - Series level: closed-form lift of map prob to bo3/bo5 series prob.

Outputs:
  data/projection_calibration.json     # all numbers needed by the page
  static/projection_test/*.png         # charts for each research question
"""

import json
import math
import os
import sys
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# ── Constants ──────────────────────────────────────────────────────────────
# Override via CLI:
#   --beta 0.154         (default 0.136)
#   --suffix _beta154    (default empty)
#   --intl-bonus 0.36    (default 0.0; set positive to enable intl-experience offset)
BETA = 0.136
SUFFIX = ""
INTL_BONUS = 0.0
# CN-DOG OFFSET (Vegas-calibrated): when the underdog is CN at an international,
# apply a flat +CN_DOG_OFFSET logit boost to the non-CN favorite. Calibrated
# against 53 CN-vs-non-CN intl matches from VLR.gg betting lines (de-vigged at
# 5%). The dominant signal is "CN-as-dog" regardless of intl_exp_diff: book vs
# BenPom mean gap +0.466 logit on n=38 (with intl_exp_diff=0). CN-as-favorite
# (n=8) shows ~0 gap, so the offset is asymmetric (only fires when CN is dog).
# The earlier CN_INTL_EXP_BOOST (n=7) is now subsumed: when intl_exp_diff=+1
# stacks with CN_DOG_OFFSET we get +0.22 + 0.47 = +0.69, matching the +0.70
# bucket the small-n MLE estimated.
CN_DOG_OFFSET = 0.0
for i, arg in enumerate(sys.argv):
    if arg == "--beta" and i + 1 < len(sys.argv):
        BETA = float(sys.argv[i + 1])
    elif arg == "--suffix" and i + 1 < len(sys.argv):
        SUFFIX = sys.argv[i + 1]
    elif arg == "--intl-bonus" and i + 1 < len(sys.argv):
        INTL_BONUS = float(sys.argv[i + 1])
    elif arg == "--cn-dog-offset" and i + 1 < len(sys.argv):
        CN_DOG_OFFSET = float(sys.argv[i + 1])
OUT_DIR = os.path.join(ROOT, "static", f"projection_test{SUFFIX}")
os.makedirs(OUT_DIR, exist_ok=True)

INTL_EVENTS = {
    "2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
    "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
    "2026_masters_santiago",
}

# Canonical chronological order of intl events per season
INTL_ORDER = {
    2024: ["2024_masters_madrid", "2024_masters_shanghai", "2024_champions"],
    2025: ["2025_masters_bangkok", "2025_masters_toronto", "2025_champions"],
    2026: ["2026_masters_santiago"],
}

INTL_LABELS = {
    "2024_masters_madrid":   "M. Madrid",
    "2024_masters_shanghai": "M. Shanghai",
    "2024_champions":        "Champions",
    "2025_masters_bangkok":  "M. Bangkok",
    "2025_masters_toronto":  "M. Toronto",
    "2025_champions":        "Champions",
    "2026_masters_santiago": "M. Santiago",
}

TEAM_REGIONS = {
    # EMEA
    "TL": "EMEA", "FNC": "EMEA", "NAVI": "EMEA", "VIT": "EMEA",
    "BBL": "EMEA", "GX": "EMEA", "KC": "EMEA", "TH": "EMEA",
    "FUT": "EMEA", "GIA": "EMEA", "MKOI": "EMEA", "M8": "EMEA",
    "KOI": "EMEA",
    # Americas
    "SEN": "Americas", "G2": "Americas", "MIBR": "Americas",
    "NRG": "Americas", "100T": "Americas", "C9": "Americas",
    "EG": "Americas", "KRÜ": "Americas", "LEV": "Americas",
    "FUR": "Americas", "LOUD": "Americas",
    # Pacific
    "PRX": "Pacific", "DRX": "Pacific", "T1": "Pacific",
    "TLN": "Pacific", "GEN": "Pacific", "DFM": "Pacific",
    "ZETA": "Pacific", "RRQ": "Pacific", "TS": "Pacific", "GE": "Pacific",
    "KRX": "Pacific", "NS": "Pacific", "BOOM": "Pacific",
    # CN
    "EDG": "CN", "BLG": "CN", "TE": "CN", "DRG": "CN", "ASE": "CN",
    "AG": "CN", "XLG": "CN", "WOL": "CN", "FPX": "CN",
    "JDG": "CN", "NOVA": "CN", "TEC": "CN", "TYL": "CN", "TYLOO": "CN",
}

# ── Plot defaults ──────────────────────────────────────────────────────────
PRIMARY   = "#7c3aed"
SECONDARY = "#5a2a7a"
ACCENT_A  = "#0089d0"   # int'l blue
ACCENT_B  = "#f6a821"   # domestic amber
CI_GRAY   = "#9a9aa6"
GRID      = "#dcd6e6"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#3a2e44",
    "axes.labelcolor": "#3a2e44",
    "axes.titlecolor": "#241a2a",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": "#3a2e44",
    "ytick.color": "#3a2e44",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


# ── Helpers ────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def series_prob_from_map(p, bo):
    """Closed-form: probability favorite wins the series given per-map p, bo3/bo5."""
    if bo == 3:
        return (p ** 2) * (3 - 2 * p)
    if bo == 5:
        return (p ** 3) * (10 - 15 * p + 6 * p * p)
    # default: just return p (bo1 — shouldn't really happen in VCT)
    return p


def infer_bo(series_score, n_maps):
    """bo5 if anyone ever reached 3 wins; else bo3."""
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


def wilson_ci(k, n, z=1.96):
    """Wilson-score 95% CI for binomial p = k/n.  Returns (lo, hi)."""
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + (z * z) / n
    centre = (phat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def brier(probs, outcomes):
    if not probs:
        return None
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def safe_logit(p, eps=1e-6):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def platt_fit(probs, outcomes):
    """Fit  logit(p_true) ≈ a + b * logit(p_model).  b<1 → over-confident,
    b>1 → under-confident.  Uses a tiny gradient-descent on log loss."""
    if len(probs) < 30:
        return {"a": None, "b": None, "n": len(probs)}
    x = np.array([safe_logit(p) for p in probs])
    y = np.array(outcomes, dtype=float)
    a, b = 0.0, 1.0
    lr = 0.02
    n = len(x)
    for _ in range(4000):
        z = a + b * x
        # clip z for numerical safety
        z = np.clip(z, -25, 25)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        ga = err.mean()
        gb = (err * x).mean()
        a -= lr * ga
        b -= lr * gb
    return {"a": float(a), "b": float(b), "n": int(n)}


def bin_calibration(probs, outcomes, edges):
    """Bin predictions; return list of {lo, hi, mid, n, k, mean_pred, win_rate, ci_lo, ci_hi}."""
    out = []
    probs = np.asarray(probs)
    outcomes = np.asarray(outcomes)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"lo": lo, "hi": hi, "mid": (lo + hi) / 2, "n": 0,
                        "k": 0, "mean_pred": None, "win_rate": None,
                        "ci_lo": None, "ci_hi": None})
            continue
        k = int(outcomes[mask].sum())
        mean_pred = float(probs[mask].mean())
        win_rate = k / n
        ci_lo, ci_hi = wilson_ci(k, n)
        out.append({"lo": lo, "hi": hi, "mid": (lo + hi) / 2, "n": n,
                    "k": k, "mean_pred": mean_pred, "win_rate": win_rate,
                    "ci_lo": ci_lo, "ci_hi": ci_hi})
    return out


# ── Load data ──────────────────────────────────────────────────────────────
def load_matches():
    """Load match_events from 2024+ timelines, drop matches with no model opinion."""
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
            # Skip matches where the model has no opinion (typically brand-new team)
            if wb == lb:
                continue
            m = dict(m)
            m["season"] = season
            m["_delta"] = wb - lb           # signed; >0 means winner was favored
            m["_abs_delta"] = abs(wb - lb)
            m["_fav_won_series"] = 1 if wb > lb else 0
            out.append(m)
    out.sort(key=lambda r: (r["date"], r["match_id"]))
    return out


# ── Build map-level and series-level samples ───────────────────────────────
def _build_intl_attendance(matches):
    """For each org, the set of intl event_ids they have ALREADY played in,
    keyed by (org, season). The check at prediction time is: 'has team played
    any intl this calendar year STRICTLY BEFORE the current match date?'.
    Returns a dict: (org, season) -> sorted list of (date, event_id).
    """
    attendance = {}  # (org, season) -> list of (date, event_id)
    for m in sorted(matches, key=lambda r: (r["date"], r["match_id"])):
        if m["event_id"] not in INTL_EVENTS:
            continue
        season = m["season"]
        for org in (m["winner"], m["loser"]):
            key = (org, season)
            attendance.setdefault(key, []).append((m["date"], m["event_id"]))
    return attendance


def _intl_exp_diff(fav_org, dog_org, season, match_date, attendance):
    """Returns +1, -1, or 0 per the intl_exp_diff feature."""
    def attended_before(org):
        for d, _ in attendance.get((org, season), []):
            if d < match_date:
                return True
        return False
    f = attended_before(fav_org)
    d = attended_before(dog_org)
    return (1 if f else 0) - (1 if d else 0)


def build_samples(matches):
    """Return (map_rows, series_rows).

    map_rows: list of dicts with p (favorite's per-map prob), y (1 if fav won
              the map), and metadata (date, event_id, season, intl, fav_org,
              dog_org, fav_region, dog_region, abs_delta, ...).
    series_rows: same but at the series level.

    When INTL_BONUS > 0, applies a logit offset to the SERIES probability based
    on `intl_exp_diff` (the favorite has been to an international this season
    and the underdog has not, or vice-versa). This is the lone Bonferroni-
    significant residual feature from the further-optimization research.
    """
    attendance = _build_intl_attendance(matches) if INTL_BONUS > 0 else {}
    map_rows = []
    series_rows = []
    for m in matches:
        wb = float(m["winner_before"])
        lb = float(m["loser_before"])
        delta = wb - lb
        abs_delta = abs(delta)
        # The model favors whoever had the higher rating going in.
        fav_won_series = 1 if wb > lb else 0
        if wb > lb:
            fav_org, dog_org = m["winner"], m["loser"]
        else:
            fav_org, dog_org = m["loser"], m["winner"]
        p_map = sigmoid(BETA * abs_delta)
        n_maps = len(m.get("maps", []))
        bo = infer_bo(m.get("series_score", ""), n_maps)
        p_series = series_prob_from_map(p_map, bo)
        intl = m["event_id"] in INTL_EVENTS

        # Apply the intl-experience offset (research finding ResearchEventContext).
        # Pulls the SERIES logit up by INTL_BONUS when the favorite has been to
        # an intl this season and the underdog hasn't, and pushes it down by
        # the same amount in the reverse case.
        intl_exp_diff = 0
        if INTL_BONUS > 0:
            intl_exp_diff = _intl_exp_diff(fav_org, dog_org, m["season"], m["date"], attendance)
            if intl_exp_diff != 0:
                # logit-space adjustment, then sigmoid back to probability
                ps = max(min(p_series, 1 - 1e-9), 1e-9)
                logit_ps = math.log(ps / (1 - ps)) + INTL_BONUS * intl_exp_diff
                p_series = 1.0 / (1.0 + math.exp(-logit_ps))

        fav_region = TEAM_REGIONS.get(fav_org)
        dog_region = TEAM_REGIONS.get(dog_org)
        cross_region = fav_region is not None and dog_region is not None and fav_region != dog_region

        # CN-DOG offset (Vegas-calibrated). When CN is the underdog at an
        # intl, the non-CN favorite is systematically under-priced — Vegas
        # markets across n=38 CN-as-dog matches show a mean logit gap of
        # +0.47 vs BenPom's prediction (after intl_exp_diff is already in).
        # No symmetric offset for CN-as-favorite (Vegas gap ~0 on n=8).
        # MUST be evaluated after fav_region/dog_region are assigned above,
        # otherwise the condition reads stale values from the prior iteration
        # (the bug that initially shipped — manifested as ~75 mis-predicted
        # intl matches and Brier going the wrong direction).
        cn_dog_applied = False
        if CN_DOG_OFFSET != 0 and intl and dog_region == 'CN' and fav_region != 'CN':
            ps = max(min(p_series, 1 - 1e-9), 1e-9)
            logit_ps = math.log(ps / (1 - ps)) + CN_DOG_OFFSET
            p_series = 1.0 / (1.0 + math.exp(-logit_ps))
            cn_dog_applied = True

        series_rows.append({
            "match_id": m["match_id"],
            "date": m["date"],
            "event_id": m["event_id"],
            "season": m["season"],
            "intl": intl,
            "bo": bo,
            "p": p_series,
            "y": fav_won_series,
            "p_map": p_map,
            "abs_delta": abs_delta,
            "fav_org": fav_org,
            "dog_org": dog_org,
            "fav_region": fav_region,
            "dog_region": dog_region,
            "cross_region": cross_region,
            "winner": m["winner"],
            "loser": m["loser"],
            "series_score": m.get("series_score", ""),
            "intl_exp_diff": intl_exp_diff,
        })

        for mp in m.get("maps", []):
            # Map-level favorite outcome
            map_winner = mp.get("winner")
            if map_winner is None:
                continue
            fav_won_map = 1 if map_winner == fav_org else 0
            map_rows.append({
                "match_id": m["match_id"],
                "date": m["date"],
                "event_id": m["event_id"],
                "season": m["season"],
                "intl": intl,
                "map": mp.get("map"),
                "p": p_map,
                "y": fav_won_map,
                "abs_delta": abs_delta,
                "fav_org": fav_org,
                "dog_org": dog_org,
                "fav_region": fav_region,
                "dog_region": dog_region,
                "cross_region": cross_region,
                "rounds_fav": (mp.get("wr") if map_winner == fav_org else mp.get("lr")),
                "rounds_dog": (mp.get("lr") if map_winner == fav_org else mp.get("wr")),
            })
    return map_rows, series_rows


# ── Plotting helpers ───────────────────────────────────────────────────────
def plot_calibration(bins, title, fname, color=PRIMARY, sub=None):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    xs, ys, ns, lo, hi = [], [], [], [], []
    for b in bins:
        if b["n"] == 0:
            continue
        xs.append(b["mean_pred"])
        ys.append(b["win_rate"])
        ns.append(b["n"])
        lo.append(b["ci_lo"])
        hi.append(b["ci_hi"])
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
    lo, hi = np.array(lo), np.array(hi)

    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="#d04646", linewidth=1.4, label="Perfect calibration")
    if len(xs):
        ax.fill_between(xs, lo, hi, color=CI_GRAY, alpha=0.25, label="95% Wilson CI")
        ax.plot(xs, ys, "-o", color=color, linewidth=2.2, markersize=7, label="Observed win-rate")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(6, 8), fontsize=8, color=SECONDARY)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.30, 1.05)
    ax.set_xlabel("Model predicted P(favorite wins)")
    ax.set_ylabel("Observed favorite win-rate")
    ttl = title
    if sub:
        ttl = f"{title}\n{sub}"
    ax.set_title(ttl)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)


def plot_calibration_dual(bins_a, label_a, bins_b, label_b, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="#d04646", linewidth=1.4, label="Perfect calibration")
    for bins, color, lab in [(bins_a, ACCENT_A, label_a), (bins_b, ACCENT_B, label_b)]:
        xs, ys, lo, hi, ns = [], [], [], [], []
        for b in bins:
            if b["n"] == 0:
                continue
            xs.append(b["mean_pred"]); ys.append(b["win_rate"])
            lo.append(b["ci_lo"]); hi.append(b["ci_hi"]); ns.append(b["n"])
        if not xs:
            continue
        xs = np.array(xs); ys = np.array(ys); lo = np.array(lo); hi = np.array(hi)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
        ax.plot(xs, ys, "-o", color=color, linewidth=2.2, markersize=7,
                label=f"{lab} (n={sum(ns)})")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.30, 1.05)
    ax.set_xlabel("Model predicted P(favorite wins)")
    ax.set_ylabel("Observed favorite win-rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), bbox_inches="tight")
    plt.close(fig)


# ── Main analysis ──────────────────────────────────────────────────────────
def main():
    matches = load_matches()
    map_rows, series_rows = build_samples(matches)
    print(f"Loaded {len(matches)} series with model opinion; "
          f"{len(map_rows)} maps, {len(series_rows)} series.")

    n_intl_diff_applied = sum(1 for r in series_rows if r.get("intl_exp_diff", 0) != 0)
    results = {
        "meta": {
            "beta": BETA,
            "intl_bonus": INTL_BONUS,
            "cn_dog_offset": CN_DOG_OFFSET,
            "intl_bonus_applied_series": n_intl_diff_applied,
            "time_window": "2024-01-01 → latest data in rating_timeline.json",
            "n_series": len(series_rows),
            "n_maps":   len(map_rows),
            "intl_events": sorted(INTL_EVENTS),
            "generated": datetime.utcnow().isoformat() + "Z",
            "filter": "matches with winner_before == loser_before (model had no opinion) are EXCLUDED",
        },
    }

    edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    # ── Q1. Overall calibration (map + series) ────────────────────────────
    map_probs = [r["p"] for r in map_rows]
    map_outs  = [r["y"] for r in map_rows]
    series_probs = [r["p"] for r in series_rows]
    series_outs  = [r["y"] for r in series_rows]

    map_bins = bin_calibration(map_probs, map_outs, edges)
    series_bins = bin_calibration(series_probs, series_outs, edges)

    map_brier = brier(map_probs, map_outs)
    series_brier = brier(series_probs, series_outs)
    map_baseline_brier = brier([0.5] * len(map_outs), map_outs)
    series_baseline_brier = brier([0.5] * len(series_outs), series_outs)

    plot_calibration(map_bins,
                     f"Map-level calibration  (n={len(map_rows)})",
                     "01_calibration_map.png",
                     color=PRIMARY,
                     sub=f"Brier = {map_brier:.4f}   (vs 0.5-baseline {map_baseline_brier:.4f})")
    plot_calibration(series_bins,
                     f"Series-level calibration  (n={len(series_rows)})",
                     "01_calibration_series.png",
                     color=PRIMARY,
                     sub=f"Brier = {series_brier:.4f}   (vs 0.5-baseline {series_baseline_brier:.4f})")

    results["q1_overall_calibration"] = {
        "map":    {"bins": map_bins,    "brier": map_brier,
                   "brier_baseline_0.5": map_baseline_brier},
        "series": {"bins": series_bins, "brier": series_brier,
                   "brier_baseline_0.5": series_baseline_brier},
    }

    # ── Q2. Favorite undervaluation (Platt fit + high-confidence buckets) ─
    platt_map = platt_fit(map_probs, map_outs)
    platt_ser = platt_fit(series_probs, series_outs)

    hi_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    def slice_above(probs, outs, t):
        probs = np.asarray(probs); outs = np.asarray(outs)
        mask = probs >= t
        n = int(mask.sum())
        if n == 0:
            return {"t": t, "n": 0, "mean_pred": None, "win_rate": None,
                    "delta": None, "ci_lo": None, "ci_hi": None}
        k = int(outs[mask].sum())
        mean_pred = float(probs[mask].mean())
        wr = k / n
        ci_lo, ci_hi = wilson_ci(k, n)
        return {"t": t, "n": n, "mean_pred": mean_pred, "win_rate": wr,
                "delta": wr - mean_pred, "ci_lo": ci_lo, "ci_hi": ci_hi}
    map_hi = [slice_above(map_probs, map_outs, t) for t in hi_thresholds]
    ser_hi = [slice_above(series_probs, series_outs, t) for t in hi_thresholds]

    # Q2 plot — residual vs threshold
    fig, ax = plt.subplots(figsize=(10, 5.0))
    for rows, color, label in [(map_hi, PRIMARY, "Map level"),
                                (ser_hi, ACCENT_A, "Series level")]:
        xs = [r["t"] for r in rows if r["n"] > 0]
        ys = [(r["win_rate"] - r["mean_pred"]) for r in rows if r["n"] > 0]
        ns = [r["n"] for r in rows if r["n"] > 0]
        cilo = [(r["ci_lo"] - r["mean_pred"]) for r in rows if r["n"] > 0]
        cihi = [(r["ci_hi"] - r["mean_pred"]) for r in rows if r["n"] > 0]
        ax.fill_between(xs, cilo, cihi, color=color, alpha=0.15)
        ax.plot(xs, ys, "-o", color=color, linewidth=2.0, markersize=7,
                label=f"{label} (max n={max(ns) if ns else 0})")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(5, 6), fontsize=7, color=color)
    ax.axhline(0, color="#d04646", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Threshold:  predicted prob ≥ t")
    ax.set_ylabel("Observed win-rate − mean predicted")
    ax.set_title("Are high-confidence favorites under-priced?\n"
                 "Positive = model is too modest about its favorites")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_undervaluation.png"), bbox_inches="tight")
    plt.close(fig)

    results["q2_favorite_undervaluation"] = {
        "platt_map":    platt_map,
        "platt_series": platt_ser,
        "map_high_conf":    map_hi,
        "series_high_conf": ser_hi,
        "interpretation": (
            "Platt b > 1 ⇒ model is under-confident (squashing toward 0.5); "
            "b < 1 ⇒ over-confident."
        ),
    }

    # ── Q3. International vs domestic ──────────────────────────────────────
    intl_map = [(r["p"], r["y"]) for r in map_rows if r["intl"]]
    dom_map  = [(r["p"], r["y"]) for r in map_rows if not r["intl"]]
    intl_ser = [(r["p"], r["y"]) for r in series_rows if r["intl"]]
    dom_ser  = [(r["p"], r["y"]) for r in series_rows if not r["intl"]]

    def split(pairs):
        return ([p for p, _ in pairs], [y for _, y in pairs])

    ip, iy = split(intl_map); dp, dy = split(dom_map)
    ip_s, iy_s = split(intl_ser); dp_s, dy_s = split(dom_ser)

    intl_bins_map = bin_calibration(ip, iy, edges)
    dom_bins_map  = bin_calibration(dp, dy, edges)
    intl_bins_ser = bin_calibration(ip_s, iy_s, edges)
    dom_bins_ser  = bin_calibration(dp_s, dy_s, edges)

    plot_calibration_dual(
        intl_bins_map, f"International (n={len(ip)})",
        dom_bins_map,  f"Domestic (n={len(dp)})",
        "Map-level calibration: international vs domestic",
        "03_intl_vs_dom_map.png")
    plot_calibration_dual(
        intl_bins_ser, f"International (n={len(ip_s)})",
        dom_bins_ser,  f"Domestic (n={len(dp_s)})",
        "Series-level calibration: international vs domestic",
        "03_intl_vs_dom_series.png")

    results["q3_intl_vs_domestic"] = {
        "map": {
            "intl":    {"n": len(ip),  "brier": brier(ip, iy),
                        "fav_winrate": float(np.mean(iy)) if iy else None,
                        "mean_pred":   float(np.mean(ip)) if ip else None,
                        "bins": intl_bins_map},
            "domestic":{"n": len(dp),  "brier": brier(dp, dy),
                        "fav_winrate": float(np.mean(dy)) if dy else None,
                        "mean_pred":   float(np.mean(dp)) if dp else None,
                        "bins": dom_bins_map},
        },
        "series": {
            "intl":    {"n": len(ip_s),  "brier": brier(ip_s, iy_s),
                        "fav_winrate": float(np.mean(iy_s)) if iy_s else None,
                        "mean_pred":   float(np.mean(ip_s)) if ip_s else None,
                        "bins": intl_bins_ser},
            "domestic":{"n": len(dp_s),  "brier": brier(dp_s, dy_s),
                        "fav_winrate": float(np.mean(dy_s)) if dy_s else None,
                        "mean_pred":   float(np.mean(dp_s)) if dp_s else None,
                        "bins": dom_bins_ser},
        },
    }

    # ── Q4. First-intl vs last-intl per season (Brier per intl event) ─────
    by_event_series = defaultdict(list)
    by_event_map    = defaultdict(list)
    for r in series_rows:
        if r["intl"]:
            by_event_series[r["event_id"]].append((r["p"], r["y"]))
    for r in map_rows:
        if r["intl"]:
            by_event_map[r["event_id"]].append((r["p"], r["y"]))

    per_event = {}
    for ev in INTL_EVENTS:
        sp, sy = split(by_event_series.get(ev, []))
        mp, my = split(by_event_map.get(ev, []))
        per_event[ev] = {
            "label": INTL_LABELS.get(ev, ev),
            "season": int(ev.split("_")[0]),
            "n_series": len(sp), "brier_series": brier(sp, sy),
            "fav_winrate_series": float(np.mean(sy)) if sy else None,
            "mean_pred_series":   float(np.mean(sp)) if sp else None,
            "n_maps":   len(mp), "brier_map":    brier(mp, my),
            "fav_winrate_map":    float(np.mean(my)) if my else None,
            "mean_pred_map":      float(np.mean(mp)) if mp else None,
        }

    # Plot: small-multiples grid, one panel per season
    seasons = sorted(INTL_ORDER.keys())
    n_sea = len(seasons)
    fig, axes = plt.subplots(1, n_sea, figsize=(4.2 * n_sea, 4.4), sharey=True)
    if n_sea == 1:
        axes = [axes]
    for ax, season in zip(axes, seasons):
        order = INTL_ORDER[season]
        labels = [INTL_LABELS[e] for e in order]
        br_s = [per_event[e]["brier_series"] for e in order]
        br_m = [per_event[e]["brier_map"]    for e in order]
        ns_s = [per_event[e]["n_series"]     for e in order]
        x = np.arange(len(order))
        width = 0.36
        bs = [b if b is not None else 0 for b in br_s]
        bm = [b if b is not None else 0 for b in br_m]
        bars1 = ax.bar(x - width / 2, bs, width, color=PRIMARY,  label="Series Brier")
        bars2 = ax.bar(x + width / 2, bm, width, color=ACCENT_A, label="Map Brier")
        for xi, n in zip(x, ns_s):
            ax.annotate(f"n_s={n}", (xi - width / 2, 0.005),
                        ha="center", va="bottom", fontsize=7,
                        color="white", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"{season}")
        ax.set_ylim(0, max(0.32, max(bs + bm) * 1.15 if bs + bm else 0.32))
        ax.grid(axis="x", alpha=0)
        if ax is axes[0]:
            ax.set_ylabel("Brier score (lower = better)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Brier by intl event, in chronological season order", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_intl_brier_by_event.png"),
                bbox_inches="tight")
    plt.close(fig)

    results["q4_intl_event_chronology"] = {
        "order": INTL_ORDER,
        "per_event": per_event,
    }

    # ── Q5. Buckets: cross-region intl matches, by region ─────────────────
    region_buckets = {}
    for r in series_rows:
        if not r["intl"]:
            continue
        fr, dr = r["fav_region"], r["dog_region"]
        if fr is None or dr is None:
            continue
        key = "same-region" if fr == dr else f"{fr} vs {dr}"
        region_buckets.setdefault(key, []).append((r["p"], r["y"]))
    region_summary = []
    for k, pairs in sorted(region_buckets.items(), key=lambda kv: -len(kv[1])):
        ps, ys = split(pairs)
        if not ps:
            continue
        wr = float(np.mean(ys))
        mp = float(np.mean(ps))
        ci_lo, ci_hi = wilson_ci(int(sum(ys)), len(ys))
        region_summary.append({
            "bucket": k,
            "n": len(ps),
            "mean_pred": mp,
            "fav_winrate": wr,
            "residual": wr - mp,
            "brier": brier(ps, ys),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })

    # Also: cross-region Masters vs Champions
    masters_pairs = [(r["p"], r["y"]) for r in series_rows
                     if r["intl"] and "masters" in r["event_id"] and r["cross_region"]]
    champs_pairs  = [(r["p"], r["y"]) for r in series_rows
                     if r["intl"] and "champions" in r["event_id"] and r["cross_region"]]

    def summarize(pairs, label):
        ps, ys = split(pairs)
        if not ps:
            return {"label": label, "n": 0}
        wr = float(np.mean(ys)); mp = float(np.mean(ps))
        cilo, cihi = wilson_ci(int(sum(ys)), len(ys))
        return {"label": label, "n": len(ps), "mean_pred": mp,
                "fav_winrate": wr, "residual": wr - mp,
                "brier": brier(ps, ys), "ci_lo": cilo, "ci_hi": cihi}

    cross_region_summary = {
        "masters_cross_region":   summarize(masters_pairs, "Masters (cross-region)"),
        "champions_cross_region": summarize(champs_pairs,  "Champions (cross-region)"),
    }

    # Plot Q5: residual per region bucket, with CI bars
    rows = [b for b in region_summary if b["n"] >= 4]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    labels = [b["bucket"] for b in rows]
    mids   = [b["fav_winrate"] for b in rows]
    means  = [b["mean_pred"] for b in rows]
    resid  = [b["residual"] for b in rows]
    cilo   = [b["ci_lo"] for b in rows]
    cihi   = [b["ci_hi"] for b in rows]
    x = np.arange(len(labels))
    colors = [PRIMARY if r >= 0 else "#c44d4d" for r in resid]
    ax.bar(x, resid, color=colors, alpha=0.85)
    # CI on residual = CI on win_rate − mean_pred
    yerr_lo = [m - lo for m, lo in zip(mids, cilo)]
    yerr_hi = [hi - m for m, hi in zip(mids, cihi)]
    ax.errorbar(x, resid, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor=CI_GRAY, capsize=3, linewidth=1.0)
    for xi, b in zip(x, rows):
        ax.annotate(f"n={b['n']}", (xi, b["residual"]), textcoords="offset points",
                    xytext=(0, 6 if b["residual"] >= 0 else -12),
                    ha="center", fontsize=8, color=SECONDARY)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Observed favorite win-rate − model predicted")
    ax.set_title("International matchups by region pairing  (residual + 95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_region_buckets.png"), bbox_inches="tight")
    plt.close(fig)

    results["q5_region_buckets"] = {
        "by_region_pairing":  region_summary,
        "cross_region_event": cross_region_summary,
    }

    # ── Q6. CI band already drawn in Q1.  Add a "sample-size landscape" plot
    #        showing every decile's CI half-width vs n.
    fig, ax = plt.subplots(figsize=(10, 5.0))
    for bins, color, label in [(map_bins, PRIMARY,   "Map level"),
                                (series_bins, ACCENT_A, "Series level")]:
        ns = [b["n"] for b in bins if b["n"] > 0]
        half = [((b["ci_hi"] - b["ci_lo"]) / 2) for b in bins if b["n"] > 0]
        mids = [b["mid"] for b in bins if b["n"] > 0]
        ax.scatter(ns, half, color=color, label=label, s=60, alpha=0.85)
        for n, h, m in zip(ns, half, mids):
            ax.annotate(f"{m:.2f}", (n, h), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Bin sample size (log scale)")
    ax.set_ylabel("Wilson 95% CI half-width")
    ax.set_title("How tight is each decile?  Bin labels = bin midpoint")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_sample_size_landscape.png"), bbox_inches="tight")
    plt.close(fig)

    # ── Q7. Free-form: a) biggest model misses, b) abs-delta vs accuracy,
    #                  c) seasonal Brier drift, d) team-level fav residuals
    # a) Biggest series-level misses
    sorted_misses = sorted(series_rows, key=lambda r: r["p"], reverse=True)
    biggest_misses = []
    for r in sorted_misses:
        if r["y"] == 0:   # favorite lost the series
            biggest_misses.append({
                "date": r["date"], "event_id": r["event_id"],
                "fav": r["fav_org"], "dog": r["dog_org"],
                "winner": r["winner"], "loser": r["loser"],
                "series_score": r["series_score"],
                "p_fav": r["p"], "p_fav_map": r["p_map"],
                "abs_delta": r["abs_delta"],
            })
            if len(biggest_misses) >= 8:
                break

    # b) Abs-delta vs accuracy (does the model actually do better when very confident?)
    delta_edges = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    delta_buckets_map = []
    for lo, hi in zip(delta_edges[:-1], delta_edges[1:]):
        bucket = [(r["p"], r["y"]) for r in map_rows
                  if lo <= r["abs_delta"] < hi]
        ps, ys = split(bucket)
        if not ps:
            delta_buckets_map.append({"lo": lo, "hi": hi, "n": 0})
            continue
        wr = float(np.mean(ys)); mp = float(np.mean(ps))
        cilo, cihi = wilson_ci(int(sum(ys)), len(ys))
        delta_buckets_map.append({
            "lo": lo, "hi": hi, "n": len(ps),
            "mean_pred": mp, "win_rate": wr,
            "ci_lo": cilo, "ci_hi": cihi,
            "brier": brier(ps, ys),
        })

    fig, ax = plt.subplots(figsize=(10, 5.0))
    rows7 = [b for b in delta_buckets_map if b["n"] > 0]
    xs = [(b["lo"] + b["hi"]) / 2 for b in rows7]
    pred = [b["mean_pred"] for b in rows7]
    obs  = [b["win_rate"] for b in rows7]
    cilo = [b["ci_lo"] for b in rows7]
    cihi = [b["ci_hi"] for b in rows7]
    ns   = [b["n"] for b in rows7]
    ax.fill_between(xs, cilo, cihi, color=CI_GRAY, alpha=0.25, label="95% Wilson CI on observed")
    ax.plot(xs, pred, "-o", color=PRIMARY,  linewidth=2.2, label="Model predicted (map p)")
    ax.plot(xs, obs,  "-o", color=ACCENT_A, linewidth=2.2, label="Observed fav win-rate (map)")
    for x, y, n in zip(xs, obs, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=8, color=SECONDARY)
    ax.set_xlabel("|winner_before − loser_before|  (BenPom rating gap)")
    ax.set_ylabel("Probability / observed win-rate")
    ax.set_title("Map win-rate vs raw rating gap (does confidence translate to outcomes?)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "07_abs_delta.png"), bbox_inches="tight")
    plt.close(fig)

    # c) Seasonal Brier (series-level), one number per season
    per_season = {}
    for season in (2024, 2025, 2026):
        rows_s = [r for r in series_rows if r["season"] == season]
        ps = [r["p"] for r in rows_s]; ys = [r["y"] for r in rows_s]
        ps_m = [r["p"] for r in map_rows if r["season"] == season]
        ys_m = [r["y"] for r in map_rows if r["season"] == season]
        per_season[season] = {
            "n_series": len(ps), "brier_series": brier(ps, ys),
            "fav_winrate_series": float(np.mean(ys)) if ys else None,
            "mean_pred_series":   float(np.mean(ps)) if ps else None,
            "n_maps":   len(ps_m), "brier_map": brier(ps_m, ys_m),
        }

    fig, ax = plt.subplots(figsize=(8, 4.6))
    seasons2 = sorted(per_season.keys())
    bs = [per_season[s]["brier_series"] or 0 for s in seasons2]
    bm = [per_season[s]["brier_map"]    or 0 for s in seasons2]
    x = np.arange(len(seasons2)); width = 0.36
    ax.bar(x - width / 2, bs, width, color=PRIMARY,  label="Series Brier")
    ax.bar(x + width / 2, bm, width, color=ACCENT_A, label="Map Brier")
    for xi, s in zip(x, seasons2):
        ax.annotate(f"n_s={per_season[s]['n_series']}",
                    (xi - width / 2, 0.005), ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in seasons2])
    ax.set_ylabel("Brier score")
    ax.set_title("Brier score by season  (map and series)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "07_seasonal_brier.png"), bbox_inches="tight")
    plt.close(fig)

    # d) Team-level: when team X was the favorite, did they over/under-perform?
    team_fav = defaultdict(list)
    for r in series_rows:
        team_fav[r["fav_org"]].append((r["p"], r["y"]))
    team_resid = []
    for team, pairs in team_fav.items():
        if len(pairs) < 8:
            continue
        ps, ys = split(pairs)
        wr = float(np.mean(ys)); mp = float(np.mean(ps))
        team_resid.append({
            "team": team, "n": len(pairs),
            "mean_pred": mp, "fav_winrate": wr,
            "residual": wr - mp,
        })
    team_resid.sort(key=lambda r: r["residual"])
    over_priced = team_resid[:8]   # most negative residual = model is too high
    under_priced = team_resid[-8:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False)
    for ax, rows, title, color in [
        (axes[0], over_priced,  "Most OVER-priced as favorite (model too high)", "#c44d4d"),
        (axes[1], under_priced, "Most UNDER-priced as favorite (model too low)", PRIMARY),
    ]:
        labels = [f"{r['team']} (n={r['n']})" for r in rows]
        vals   = [r["residual"] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, vals, color=color, alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0, color="#444", linewidth=0.8)
        for yi, r in zip(y, rows):
            ax.annotate(f"{r['residual']:+.3f}", (r["residual"], yi),
                        xytext=(4 if r["residual"] >= 0 else -4, 0),
                        textcoords="offset points",
                        va="center",
                        ha="left" if r["residual"] >= 0 else "right",
                        fontsize=8, color="#241a2a")
        ax.set_title(title)
        ax.set_xlabel("Observed win-rate − mean predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "07_team_residuals.png"), bbox_inches="tight")
    plt.close(fig)

    results["q7_freeform"] = {
        "biggest_misses_series": biggest_misses,
        "abs_delta_buckets_map": delta_buckets_map,
        "per_season":   per_season,
        "team_residuals_overpriced":  over_priced,
        "team_residuals_underpriced": under_priced,
    }

    # ── Write JSON ─────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, f"projection_calibration{SUFFIX}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    # ── Print a console summary ────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"Map Brier:    {map_brier:.4f} (n={len(map_rows)}, baseline {map_baseline_brier:.4f})")
    print(f"Series Brier: {series_brier:.4f} (n={len(series_rows)}, baseline {series_baseline_brier:.4f})")
    print(f"Platt (map):    a={platt_map['a']}, b={platt_map['b']}")
    print(f"Platt (series): a={platt_ser['a']}, b={platt_ser['b']}")
    for level, label in [("map", "map"), ("series", "series")]:
        intl_b = results["q3_intl_vs_domestic"][level]["intl"]["brier"]
        dom_b  = results["q3_intl_vs_domestic"][level]["domestic"]["brier"]
        print(f"Intl {label} Brier: {intl_b:.4f}    Domestic {label} Brier: {dom_b:.4f}")
    print("\nIntl event Brier (series):")
    for season in sorted(INTL_ORDER):
        for ev in INTL_ORDER[season]:
            e = per_event[ev]
            print(f"  {ev:<28}  n_s={e['n_series']:>2}  Brier={e['brier_series']}")

    return results


if __name__ == "__main__":
    main()
