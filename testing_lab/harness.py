"""Testing Lab harness — walk-forward evaluation of BenPom series probabilities.

Dataset: match_events from data/rating_timeline*.json (leak-free pre-match
ratings solved through the previous match day, real dates, CN shrinkage — the
production model exactly as it stands). Enrichment: format, GF/playoff flags,
regions, intl attendance as-of, MatchName.

Surfaces: closed-form series probability (the backend past-match path) with
swappable beta / offset-gating / calibration layers, so production can be
scored against variants on identical inputs.
"""
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "testing_lab", "out")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, "/Users/benny_es1/VCTMM")
from vctmm.benpom.teams import ORG_REGIONS  # noqa: E402

# The 9 internationals the frontend MC gates offsets on (veto_mc.INTL_EVENTS)
INTL_EVENTS_9 = {
    "2024_masters_madrid", "2024_masters_shanghai", "2024_champions",
    "2025_masters_bangkok", "2025_masters_toronto", "2025_champions",
    "2026_masters_santiago", "2026_masters_london", "2026_champions",
}
# The 3 events the backend past-match path gates on (MapElo.py L7523)
INTL_EVENTS_BACKEND = {"2026_masters_santiago", "2026_masters_london", "2026_champions"}
INTL_EXP_BONUS = 0.40
CN_DOG_OFFSET = 0.35
GF_UPPER_LOGIT = 0.25
BETA_LIVE = 0.170          # what every live surface uses (veto_mc.BETA)
BETA_SWEEP_NOTE = 0.136    # what BuildMapRatings' last sweep says pairs with RD_SCALE=2.5

TIMELINE_FILES = ["rating_timeline_2023.json", "rating_timeline_2024.json",
                  "rating_timeline_2025.json", "rating_timeline.json"]


def _is_intl_event(eid):
    """Exact-shape rules, not substrings. Real internationals are
    YYYY_masters_<city>, YYYY_champions, and YYYY_lock_in. The 2026-07-28
    corpus backfill added off-season ids (2024/2025_shanghai_masters,
    2025_super_champions_cup, 2023_china_champions_qual) that substring
    matching would misclassify as internationals."""
    return bool(re.match(r"^\d{4}_masters(_|$)", eid)
                or re.fullmatch(r"\d{4}_champions", eid)
                or re.fullmatch(r"\d{4}_lock_in", eid))


def _match_names():
    mr = pd.read_csv(os.path.join(DATA, "match_results.csv"),
                     usecols=["MatchID", "MatchName"])
    return dict(mr.drop_duplicates("MatchID").values)


def _stage(match_name):
    s = (match_name or "").lower()
    if "grand final" in s:
        return "grand_final"
    if re.search(r"playoff|bracket|upper|lower|semifinal|quarterfinal|round of|"
                 r"knockout|final", s):
        return "playoffs"
    if re.search(r"group|swiss|league|regular|week ", s):
        return "groups"
    return "other"


def load_series():
    """One row per completed series with leak-free pre-match ratings."""
    names = _match_names()
    rows = []
    for fn in TIMELINE_FILES:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        for me in d["match_events"]:
            w, l = me["winner"], me["loser"]
            if w not in ORG_REGIONS or l not in ORG_REGIONS:
                continue  # junk org parses ('CN', 'INTL', ...)
            try:
                wm, lm = map(int, me["series_score"].split("-"))
            except Exception:
                continue
            if wm <= lm or wm not in (1, 2, 3):
                continue  # incomplete/forfeit/odd series
            mname = names.get(me["match_id"], "")
            fmt = {1: "bo1", 2: "bo3", 3: "bo5"}[wm]
            stage = _stage(mname)
            if fmt == "bo5" and stage == "grand_final":
                fmt = "bo5_gf"
            rows.append({
                "match_id": me["match_id"], "date": me["date"],
                "event_id": me["event_id"], "year": int(me["date"][:4]),
                "winner": w, "loser": l, "w_maps": wm, "l_maps": lm,
                "r_w": me["winner_before"], "r_l": me["loser_before"],
                "fmt": fmt, "stage": stage, "match_name": mname,
                "n_maps_played": len(me.get("maps", [])),
                "intl": _is_intl_event(me["event_id"]),
                "reg_w": ORG_REGIONS.get(w, "?"), "reg_l": ORG_REGIONS.get(l, "?"),
            })
    df = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    # de-dup (2026 file may re-list); keep first
    df = df.drop_duplicates("match_id", keep="first").reset_index(drop=True)
    return df


def intl_attendance_asof(df):
    """org -> date of first intl-event series (any masters/champions/lock_in)."""
    first = {}
    for _, r in df[df["intl"]].iterrows():
        for org in (r["winner"], r["loser"]):
            if org not in first or r["date"] < first[org]:
                first[org] = r["date"]
    return first


# ── probability surfaces ─────────────────────────────────────────────────────

def series_wp(p, fmt):
    if fmt in ("bo5", "bo5_gf"):
        return p ** 3 * (1 + 3 * (1 - p) + 6 * (1 - p) ** 2)
    if fmt == "bo1":
        return p
    return p ** 2 * (3 - 2 * p)


def shift_logit(p, delta):
    if not delta:
        return p
    ps = min(max(p, 1e-9), 1 - 1e-9)
    return 1.0 / (1.0 + math.exp(-(math.log(ps / (1 - ps)) + delta)))


def closed_form_prob(r_a, r_b, fmt, beta):
    p_map = 1.0 / (1.0 + math.exp(-beta * (r_a - r_b)))
    p = series_wp(p_map, fmt)
    if fmt == "bo5_gf":
        p = shift_logit(p, GF_UPPER_LOGIT)  # assumes A = upper-bracket team
    return p


def apply_offsets(p_a, event_id, org_a, org_b, date, attendance, gating,
                  exp_bonus=INTL_EXP_BONUS, cn_dog=CN_DOG_OFFSET):
    gate = (INTL_EVENTS_9 if gating == "frontend9"
            else INTL_EVENTS_BACKEND if gating == "backend" else set())
    if event_id not in gate:
        return p_a

    def attended(org):
        d = attendance.get(org)
        return bool(d and d < date)

    a_fav = p_a >= 0.5
    fav, dog = (org_a, org_b) if a_fav else (org_b, org_a)
    delta = exp_bonus * ((1 if attended(fav) else 0) - (1 if attended(dog) else 0))
    if ORG_REGIONS.get(dog) == "CN" and ORG_REGIONS.get(fav) != "CN":
        delta += cn_dog
    if not delta:
        return p_a
    p_fav = p_a if a_fav else 1 - p_a
    p_fav = shift_logit(p_fav, delta)
    return p_fav if a_fav else 1 - p_fav


def predict(df, beta=BETA_LIVE, gating="backend", attendance=None,
            exp_bonus=INTL_EXP_BONUS, cn_dog=CN_DOG_OFFSET):
    """p assigned to the eventual WINNER of each series (closed form).

    GF caveat: we don't know bracket side here, so bo5_gf logit shift is
    applied toward the pre-shift favorite (upper team is usually the favorite;
    measured separately).
    """
    if attendance is None:
        attendance = intl_attendance_asof(df)
    ps = np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        p_map = 1.0 / (1.0 + math.exp(-beta * (r.r_w - r.r_l)))
        p = series_wp(p_map, r.fmt if r.fmt != "bo5_gf" else "bo5")
        if r.fmt == "bo5_gf":
            p = shift_logit(p, GF_UPPER_LOGIT if p >= 0.5 else -GF_UPPER_LOGIT)
        p = apply_offsets(p, r.event_id, r.winner, r.loser, r.date, attendance,
                          gating, exp_bonus, cn_dog)
        ps[i] = p
    return ps


# ── metrics ──────────────────────────────────────────────────────────────────

def logloss(p_win):
    return float(-np.mean(np.log(np.clip(p_win, 1e-9, 1))))


def brier(p_win):
    return float(np.mean((1.0 - np.clip(p_win, 0, 1)) ** 2))


def summarize(p_win, label=""):
    return {"label": label, "n": int(len(p_win)),
            "logloss": round(logloss(p_win), 5), "brier": round(brier(p_win), 5)}


def paired_bootstrap(p_a, p_b, n_boot=4000, seed=7):
    """P(model A better than B) on log-loss + mean delta CI. Positive delta
    = A better (lower loss)."""
    la = -np.log(np.clip(p_a, 1e-9, 1))
    lb = -np.log(np.clip(p_b, 1e-9, 1))
    d = lb - la  # >0 means A better
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"mean_delta": float(d.mean()),
            "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)),
            "p_better": float((means > 0).mean())}


def reliability(p_fav, fav_won, n_bins=10, lo=0.5, hi=1.0):
    """Favorite-side reliability: bins over predicted favorite prob."""
    edges = np.linspace(lo, hi, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (p_fav >= edges[i]) & (p_fav < edges[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        if m.sum() == 0:
            continue
        k, n = int(fav_won[m].sum()), int(m.sum())
        # Wilson interval
        z, ph = 1.96, k / n
        den = 1 + z * z / n
        ci = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        out.append({"bin_lo": round(float(edges[i]), 3), "bin_hi": round(float(edges[i + 1]), 3),
                    "pred_mean": round(float(p_fav[m].mean()), 4),
                    "emp": round(ph, 4), "n": n,
                    "ci_lo": round((ph + z * z / (2 * n)) / den - ci, 4),
                    "ci_hi": round((ph + z * z / (2 * n)) / den + ci, 4)})
    return out


def calib_slope(p_win, seed=None):
    """Fit y ~ sigmoid(a + b*logit(p)) on winner-side probs by viewing each
    series from both sides (y=1 at p, y=0 at 1-p). b>1 => underconfident
    (probs too close to 50%), b<1 => overconfident."""
    from scipy.optimize import minimize
    p = np.clip(np.concatenate([p_win, 1 - p_win]), 1e-9, 1 - 1e-9)
    y = np.concatenate([np.ones(len(p_win)), np.zeros(len(p_win))])
    x = np.log(p / (1 - p))

    def nll(ab):
        z = np.clip(ab[0] + ab[1] * x, -30, 30)
        q = 1 / (1 + np.exp(-z))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    return {"a": round(float(res.x[0]), 4), "b": round(float(res.x[1]), 4)}
