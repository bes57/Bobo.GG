"""Gate A — v6 surface parity (stage 2, deploy-surfaces).

Backend: MapElo._v6_series_prob (the function every site surface quotes:
hub upcoming/past win_prob_a, alpha home, team profiles) must equal
trading_model/predict.py series_probability to <= 1e-9 on 40 pairs sampled
from current ratings — mixed regions + formats incl. bo5_gf both uppers and
unknown-org (region-prior) cases. predict.py's `m` dict is constructed FROM
data/site_model.json + current timeline ratings (predict.py itself loads
trading_model/model_snapshot.json, which is NOT the site snapshot).

Frontend: python replica of the injected-constants JS path (v6SeriesProbHUB /
xregionAdjHUB / shiftSeriesProb in MAPELO_MODERN_HTML, v6SeriesProbSim in
MAPELO_MATCHUP_HTML — same formula) vs predict.py for 10 pairs to <= 1e-6.

Usage: python3 testing_lab/v9/scratch/deploy/surface_parity.py
Writes: testing_lab/v9/stats/deploy_surface_parity.json
"""
import importlib.util
import json
import math
import os
import random
import sys

ROOT = "/Users/benny_es1/PythonTest"
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "predict", os.path.join(ROOT, "trading_model", "predict.py"))
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)

import MapElo  # noqa: E402  (site backend under test)

site_model = json.load(open(os.path.join(ROOT, "data", "site_model.json")))
tl = json.load(open(os.path.join(ROOT, "data", "rating_timeline.json")))
ratings = tl["checkpoints"][-1]["ratings"]

# predict.py's m dict, built from the SITE snapshot + current timeline ratings
M = {
    "model_version": "parity-check",
    "ratings_as_of": "parity-check",
    "beta": site_model["beta"],
    "xregion_offsets": site_model["xregion_offsets"],
    "region_priors": site_model["region_priors"],
    "gf_upper_logit": site_model["gf_upper_logit"],
    "b_pick": site_model["b_pick"],
    "ratings": ratings,
    "org_regions": MapElo.ORG_REGIONS,
}

by_region = {}
for org in sorted(MapElo.ACTIVE_2026_ORGS):
    if org in ratings:
        by_region.setdefault(MapElo.ORG_REGIONS.get(org, ""), []).append(org)

rng = random.Random(60731)
regions = [r for r in ("Americas", "EMEA", "Pacific", "CN") if by_region.get(r)]

pairs = []


def add_pair(a, b, fmt, upper=None, tag=""):
    pairs.append({"a": a, "b": b, "fmt": fmt, "upper": upper, "tag": tag})


# 16 same-region pairs (4 per region), formats cycled
fmts = ["bo1", "bo3", "bo5", "bo3"]
for reg in regions:
    orgs = by_region[reg]
    for i in range(4):
        a, b = rng.sample(orgs, 2)
        add_pair(a, b, fmts[i % len(fmts)], tag=f"same-{reg}")

# 16 cross-region pairs, every region combination at least once
combos = [(x, y) for xi, x in enumerate(regions) for y in regions[xi + 1:]]
fmts_x = ["bo3", "bo5", "bo1", "bo3"]
k = 0
for _ in range(16):
    rx, ry = combos[k % len(combos)]
    k += 1
    a = rng.choice(by_region[rx])
    b = rng.choice(by_region[ry])
    if rng.random() < 0.5:
        a, b = b, a
    add_pair(a, b, fmts_x[k % len(fmts_x)], tag=f"xreg-{rx}v{ry}")

# 4 bo5_gf pairs — both uppers, same- and cross-region
a, b = rng.sample(by_region["EMEA"], 2)
add_pair(a, b, "bo5_gf", upper=a, tag="gf-upper-a-same")
add_pair(a, b, "bo5_gf", upper=b, tag="gf-upper-b-same")
a = rng.choice(by_region["Americas"])
b = rng.choice(by_region["CN"])
add_pair(a, b, "bo5_gf", upper=a, tag="gf-upper-a-xreg")
add_pair(a, b, "bo5_gf", upper=b, tag="gf-upper-b-xreg")

# 4 unknown-org cases (region-prior path): orgs in ORG_REGIONS but absent
# from the current checkpoint, plus a fully unknown org code.
unknown_known_region = [o for o in MapElo.ORG_REGIONS
                        if o not in ratings][:3] or ["GIA"]
for i, u in enumerate(unknown_known_region[:3]):
    opp = rng.choice(by_region[regions[i % len(regions)]])
    add_pair(u, opp, "bo3", tag=f"unknown-org-{u}")
add_pair("ZZZQ", rng.choice(by_region["EMEA"]), "bo3", tag="unknown-org-noregion")

pairs = pairs[:40]
assert len(pairs) == 40, len(pairs)

# ── Backend parity (<= 1e-9) ────────────────────────────────────────────────
rows = []
max_diff_backend = 0.0
for pr in pairs:
    a, b, fmt, upper = pr["a"], pr["b"], pr["fmt"], pr["upper"]
    p_ref = predict.series_probability(
        M, a, b, fmt, upper,
        region_a=MapElo.ORG_REGIONS.get(a), region_b=MapElo.ORG_REGIONS.get(b))
    p_site = MapElo._v6_series_prob(site_model, ratings, a, b, fmt, upper)
    d = abs(p_ref - p_site)
    max_diff_backend = max(max_diff_backend, d)
    rows.append({**pr, "p_predict": p_ref, "p_site_backend": p_site, "abs_diff": d})

# ── Frontend JS replica parity (<= 1e-6) ────────────────────────────────────
# Line-for-line python replica of the injected-constants JS math
# (v6SeriesProbHUB + xregionAdjHUB + shiftSeriesProb in MAPELO_MODERN_HTML;
# v6SeriesProbSim in MAPELO_MATCHUP_HTML is the same formula with
# upperIsA instead of gfUpperOrg).


def js_xregion_adj_hub(org_a, org_b, org_regions, xoff):
    ra = org_regions.get(org_a)
    rb = org_regions.get(org_b)
    if not ra or not rb or ra == rb:
        return 0
    return xoff.get(ra, 0) - xoff.get(rb, 0)


def js_shift_series_prob(p, delta):
    if not delta:
        return p
    ps = max(min(p, 1 - 1e-9), 1e-9)
    return 1.0 / (1.0 + math.exp(-(math.log(ps / (1 - ps)) + delta)))


def js_v6_series_prob_hub(rA, rB, org_a, org_b, fmt, gf_upper_org,
                          site, org_regions):
    beta = site["beta"]
    p = 1 / (1 + math.exp(-beta * (rA - rB + js_xregion_adj_hub(
        org_a, org_b, org_regions, site["xregion_offsets"]))))
    if fmt == "bo1":
        ps = p
    elif fmt in ("bo5", "bo5_gf"):
        q = 1 - p
        ps = p * p * p * (1 + 3 * q + 6 * q * q)
    else:
        ps = p * p * (3 - 2 * p)
    if fmt == "bo5_gf" and gf_upper_org in (org_a, org_b):
        ps = js_shift_series_prob(
            ps, site["gf_upper_logit"] if gf_upper_org == org_a
            else -site["gf_upper_logit"])
    return ps


max_diff_frontend = 0.0
fe_rows = []
fe_pairs = [p for p in pairs if p["a"] in ratings and p["b"] in ratings][:10]
for pr in fe_pairs:
    a, b, fmt, upper = pr["a"], pr["b"], pr["fmt"], pr["upper"]
    p_ref = predict.series_probability(
        M, a, b, fmt, upper,
        region_a=MapElo.ORG_REGIONS.get(a), region_b=MapElo.ORG_REGIONS.get(b))
    p_js = js_v6_series_prob_hub(ratings[a], ratings[b], a, b, fmt,
                                 upper or "", site_model, MapElo.ORG_REGIONS)
    d = abs(p_ref - p_js)
    max_diff_frontend = max(max_diff_frontend, d)
    fe_rows.append({**pr, "p_predict": p_ref, "p_js_replica": p_js, "abs_diff": d})

result = {
    "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "site_model_version": site_model.get("model_version"),
    "ratings_checkpoint": tl["checkpoints"][-1].get("date"),
    "n_pairs_backend": len(rows),
    "max_abs_diff_backend": max_diff_backend,
    "backend_tolerance": 1e-9,
    "backend_pass": bool(max_diff_backend <= 1e-9),
    "n_pairs_frontend": len(fe_rows),
    "max_abs_diff_frontend": max_diff_frontend,
    "frontend_tolerance": 1e-6,
    "frontend_pass": bool(max_diff_frontend <= 1e-6),
    "pairs_backend": rows,
    "pairs_frontend": fe_rows,
}
out = os.path.join(ROOT, "testing_lab", "v9", "stats", "deploy_surface_parity.json")
with open(out, "w") as f:
    json.dump(result, f, indent=1)
print(f"backend  max|diff| = {max_diff_backend:.3e}  "
      f"({'PASS' if result['backend_pass'] else 'FAIL'} @ 1e-9, n={len(rows)})")
print(f"frontend max|diff| = {max_diff_frontend:.3e}  "
      f"({'PASS' if result['frontend_pass'] else 'FAIL'} @ 1e-6, n={len(fe_rows)})")
print(f"wrote {out}")
if not (result["backend_pass"] and result["frontend_pass"]):
    sys.exit(1)
