"""agent:decay — 5e: subpopulation panel for EVERY verdict-carrying config,
+ decay_curves.json overlay data."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
from decay_lib import (Runner, jlog, build_lineup_tables,  # noqa: E402
                       matches_since_change, rotation_dates, V8, PROBS)

sys.path.insert(0, V8)
import referee  # noqa: E402

STATS = os.path.join(V8, "stats")
FAMILY = {"within": 1.773, "cross": 5.889}

rn = Runner()
s = rn.frame
hold = rn.test_v
v6 = rn.run_cfg("v6_consist_20_12", rn.lam_arrays("consist"))
p6, rd6 = v6["p"], v6["rdiff"]
jlog("5e runner ready")

# ── masks ───────────────────────────────────────────────────────────────────
org_matches, _ = build_lineup_tables(rn)
msc = matches_since_change(org_matches)
# top-up audit artifact (scratch only)
rows = []
for org, seq in org_matches.items():
    for dn, mid, L in seq:
        rows.append({"org": org, "match_id": mid,
                     "n_players": len(L) if L else 0,
                     "msc_expanded": msc[(org, mid)]})
pd.DataFrame(rows).to_csv(os.path.join(SCR, "lineups_expanded_msc.csv"),
                          index=False)
# agreement with the lineups agent's published column (covered rows)
lf = pd.read_csv(os.path.join(V8, "data", "lineup_features.csv"),
                 usecols=["match_id", "org", "matches_since_change"])
n_cmp = n_ok = 0
for r in lf.itertuples(index=False):
    mine = msc.get((r.org, r.match_id))
    if mine is None or pd.isna(r.matches_since_change):
        continue
    n_cmp += 1
    if int(mine) == int(r.matches_since_change):
        n_ok += 1
agree = n_ok / max(n_cmp, 1)
jlog(f"S1 msc agreement with lineup_features: {n_ok}/{n_cmp} = {agree:.4f} "
     "(differences expected: their sequences lack the corpus-addition events)")

msc_w = np.array([msc.get((r.winner, r.match_id))
                  if msc.get((r.winner, r.match_id)) is not None else 10**6
                  for r in s.itertuples(index=False)], dtype=float)
msc_l = np.array([msc.get((r.loser, r.match_id))
                  if msc.get((r.loser, r.match_id)) is not None else 10**6
                  for r in s.itertuples(index=False)], dtype=float)
S1 = (msc_w <= 3) | (msc_l <= 3)

rots, _cl = rotation_dates(rn)
sd = pd.to_datetime(s.date).values.astype("datetime64[D]")
S2 = np.zeros(len(s), dtype=bool)
for r_ in rots:
    r0 = np.datetime64(r_, "D")
    S2 |= (sd >= r0) & ((sd - r0).astype(int) <= 21)

rest_w, rest_l = referee.rest_days(s)
known = ~np.isnan(rest_w) & ~np.isnan(rest_l)
S3 = known & ((rest_w > 45) | (rest_l > 45))

ev_first = s.groupby("event_id")["date"].transform("min").values
S4 = (s.date.values > ev_first)

p_fav6 = np.maximum(p6, 1 - p6)
S5 = p_fav6 <= 0.80

masks = {"all_holdout": np.ones(len(s), dtype=bool),
         "S1_post_roster_change_le3": S1, "S2_post_patch_21d": S2,
         "S3_post_break_45d": S3, "S4_within_event_day2plus": S4,
         "S5_quoted_band_20_55c": S5}

# ── configs ─────────────────────────────────────────────────────────────────
form_sel = json.load(open(os.path.join(STATS, "decay_form.json")))
sel_rd = form_sel["results"]["form_rd_selected"].split(" ")[0]
CONFIGS = [
    ("consist_16_10", "within"), ("sym_16", "cross"), ("sym_20", "cross"),
    ("sym_24", "cross"),
    ("lineup_h24_g2.0", "cross"), ("oppq_m1.67", "within"),
    ("amargin_k1.0", "within"), ("eclass_h20_m0.8", "cross"),
    ("eclass_on_v6_m0.8", "within"), ("rot_h24_g0.7", "cross"),
    ("rot_on_v6_g0.7", "within"),
    ("form_wr3", "within"), (sel_rd, "within"), ("form_side5", "within"),
    ("form_player5", "within"), ("form_combined", "within"),
]

panel = {}
for name, regime in CONFIGS:
    z = np.load(os.path.join(PROBS, f"{name}.npz"))
    p_c = z["p"]
    base = hold & ~np.isnan(p6) & ~np.isnan(p_c)
    rows = []
    for mname, mk in masks.items():
        m = base & mk
        n = int(m.sum())
        if n < 15:
            rows.append({"subpop": mname, "n": n, "note": "n<15, suppressed"})
            continue
        ll_c = float(np.mean(-np.log(np.clip(p_c[m], 1e-9, 1))))
        ll_6 = float(np.mean(-np.log(np.clip(p6[m], 1e-9, 1))))
        dm = (ll_6 - ll_c) * 1000
        bmde = FAMILY[regime] * math.sqrt(1217 / n)
        tag = ("WIN" if dm > 0 else "WORSE") if abs(dm) >= bmde \
            else "INSIDE NOISE FLOOR"
        rows.append({"subpop": mname, "n": n, "ll_v6": round(ll_6, 5),
                     "ll_cand": round(ll_c, 5), "delta_milli": round(dm, 3),
                     "bucket_mde_milli": round(bmde, 3), "tag": tag})
    panel[name] = {"regime": regime, "rows": rows}
    hl = [f"{r['subpop']}:{r.get('delta_milli', 'NA'):+}m/{r.get('tag','-')}"
          if 'delta_milli' in r else f"{r['subpop']}:n={r['n']}" for r in rows]
    jlog(f"5e {name}: " + " | ".join(hl))

res = {
    "preregistered": "testing_lab/v8/preregister.decay.md",
    "mask_definitions": {
        "S1_post_roster_change_le3": "either org's matches_since_change <= 3, "
            "recomputed on the expanded lineup table (lineups-agent rule "
            "verbatim; walk back while lineup identical); unknown lineup -> not in bucket",
        "S2_post_patch_21d": "series date within 21d at-or-after any derived "
            "map-pool rotation date (5b-e mechanical derivation)",
        "S3_post_break_45d": "either team's rest days > 45 (both teams have a "
            "prior series; referee.rest_days)",
        "S4_within_event_day2plus": "series date > event's first series date",
        "S5_quoted_band_20_55c": "v6 favorite-side prob <= 0.80 == either side "
            "priced inside [20,55)c under v6 (referee fallback-band definition)",
    },
    "mask_coverage_holdout": {k: int((hold & v).sum()) for k, v in masks.items()},
    "s1_agreement_with_published": {"n_compared": n_cmp, "n_equal": n_ok,
                                    "rate": round(agree, 4)},
    "bucket_mde_rule": "family_MDE(regime) * sqrt(1217/n_bucket); tags at "
                       "bucket MDE, p_better clause waived in buckets (preregistered)",
    "panel": panel,
}
json.dump(res, open(os.path.join(STATS, "decay_subpops.json"), "w"), indent=1)
jlog("wrote stats/decay_subpops.json")

# ── curves ──────────────────────────────────────────────────────────────────
g = np.arange(0, 61)
def w(hl):
    return np.exp(-math.log(2) / hl * g)
curves = {
    "g_games_ago": g.tolist(),
    "series": [
        {"label": "v6 consistent (HL20)", "family": "v6", "w": w(20).round(5).tolist()},
        {"label": "v6 anomalous (HL12)", "family": "v6", "w": w(12).round(5).tolist()},
        {"label": "sym_20", "family": "sym", "w": w(20).round(5).tolist()},
        {"label": "lineup h24 g2 · overlap 1.0", "family": "lineup",
         "w": (1.0 ** 2 * w(24)).round(5).tolist()},
        {"label": "lineup h24 g2 · overlap 0.8", "family": "lineup",
         "w": (0.8 ** 2 * w(24)).round(5).tolist()},
        {"label": "lineup h24 g2 · overlap 0.6", "family": "lineup",
         "w": (0.6 ** 2 * w(24)).round(5).tolist()},
        {"label": "lineup h24 g2 · overlap 0.4", "family": "lineup",
         "w": (0.4 ** 2 * w(24)).round(5).tolist()},
        {"label": "eclass-on-v6 · vct consistent (HL20)", "family": "eclass",
         "w": w(20).round(5).tolist()},
        {"label": "eclass-on-v6 · vct anomalous (HL12)", "family": "eclass",
         "w": w(12).round(5).tolist()},
        {"label": "eclass-on-v6 · ewc consistent (HL16)", "family": "eclass",
         "w": w(16).round(5).tolist()},
        {"label": "eclass-on-v6 · ewc anomalous (HL9.6)", "family": "eclass",
         "w": w(9.6).round(5).tolist()},
    ],
    "note": "w(g) = solve weight vs games-ago per side, before year/lineup "
            "continuity, champ x2, playoff x1.6 multipliers. Lineup curves "
            "include the (overlap)^gamma factor (floor 0.04). eclass-on-v6 "
            "(+0.24m, the only positive-direction axis) scales both v6 HLs "
            "by 0.8 for ewc_offseason-class games.",
}
json.dump(curves, open(os.path.join(STATS, "decay_curves.json"), "w"), indent=1)
jlog("wrote stats/decay_curves.json")
