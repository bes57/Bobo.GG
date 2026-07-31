"""Series win probability from model_snapshot.json (the private v5 model).

CLI:      python3 trading_model/predict.py FNC TH bo3
          python3 trading_model/predict.py LEV EDG bo5_gf --upper LEV
Library:  from predict import load_model, series_probability
          m = load_model()
          p = series_probability(m, "FNC", "TH", "bo3")   # P(FNC wins)

Rules encoded here (do not re-derive elsewhere — this is the reference):
  p_map = sigmoid(beta * (r_a - r_b + xregion_adj))
  xregion_adj = offsets[region_a] - offsets[region_b]  (0 same-region;
      this REPLACES the old intl_exp/cn_dog offsets entirely)
  unknown/new team -> region prior (25th percentile of its region)
  series: bo1 p; bo3 p^2(3-2p); bo5 p^3(1+3q+6q^2), q=1-p
  bo5_gf: +gf_upper_logit on the series prob toward the upper-bracket team
"""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(path=None):
    with open(path or os.path.join(HERE, "model_snapshot.json")) as f:
        return json.load(f)


def team_rating(m, org, region_hint=None):
    if org in m["ratings"]:
        return m["ratings"][org], m["org_regions"].get(org) or region_hint
    reg = region_hint or ""
    return m["region_priors"].get(reg, 0.0), reg


def _series_from_map(p, fmt):
    if fmt == "bo1":
        return p
    if fmt in ("bo5", "bo5_gf"):
        q = 1 - p
        return p ** 3 * (1 + 3 * q + 6 * q * q)
    return p * p * (3 - 2 * p)


def _shift_logit(p, delta):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) + delta)))


def series_probability(m, org_a, org_b, fmt="bo3", upper=None,
                       region_a=None, region_b=None):
    """P(org_a beats org_b). upper: org code of the upper-bracket team,
    only used when fmt == 'bo5_gf'."""
    r_a, reg_a = team_rating(m, org_a, region_a)
    r_b, reg_b = team_rating(m, org_b, region_b)
    adj = 0.0
    if reg_a and reg_b and reg_a != reg_b:
        off = m["xregion_offsets"]
        adj = off.get(reg_a, 0.0) - off.get(reg_b, 0.0)
    p_map = 1.0 / (1.0 + math.exp(-m["beta"] * (r_a - r_b + adj)))
    p = _series_from_map(p_map, fmt)
    if fmt == "bo5_gf" and upper in (org_a, org_b):
        delta = m["gf_upper_logit"] if upper == org_a else -m["gf_upper_logit"]
        p = _shift_logit(p, delta)
    return p


def map_probability(m, org_a, org_b, a_picked=None,
                    region_a=None, region_b=None):
    """Single-map P(org_a wins). a_picked: True if org_a picked the map,
    False if org_b picked it, None for decider/unknown."""
    r_a, reg_a = team_rating(m, org_a, region_a)
    r_b, reg_b = team_rating(m, org_b, region_b)
    adj = 0.0
    if reg_a and reg_b and reg_a != reg_b:
        off = m["xregion_offsets"]
        adj = off.get(reg_a, 0.0) - off.get(reg_b, 0.0)
    z = m["beta"] * (r_a - r_b + adj)
    if a_picked is True:
        z += m["b_pick"]
    elif a_picked is False:
        z -= m["b_pick"]
    return 1.0 / (1.0 + math.exp(-z))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("org_a")
    ap.add_argument("org_b")
    ap.add_argument("fmt", nargs="?", default="bo3",
                    choices=["bo1", "bo3", "bo5", "bo5_gf"])
    ap.add_argument("--upper", help="upper-bracket org (bo5_gf only)")
    ap.add_argument("--region-a", help="region for an unknown org_a")
    ap.add_argument("--region-b", help="region for an unknown org_b")
    a = ap.parse_args()
    m = load_model()
    p = series_probability(m, a.org_a, a.org_b, a.fmt, a.upper,
                           a.region_a, a.region_b)
    print(f"{m['model_version']} (ratings as of {m['ratings_as_of']})")
    print(f"P({a.org_a} beats {a.org_b}, {a.fmt}) = {p:.4f}")
