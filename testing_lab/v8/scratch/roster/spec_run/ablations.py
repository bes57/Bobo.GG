"""SPEC RUN per-team ablations (spec §4) — chart runs, never performance.

Per case: boost enabled for the FEATURED TEAM ONLY, s=0 globally (A4),
boundaries activated only if dated >= window start (A5). Hard gate enforced
HERE: max|r_v6 - r_ablation| over every solve day strictly before the first
activated boundary must be EXACTLY 0.0 — computed globally over ALL teams
(stronger than the featured team's own path). A nonzero value raises; no
JSON is written (fix the implementation, never explain the number).
No holdout metrics touched (ratings only).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
STATS = os.path.join(V8, "stats")
sys.path.insert(0, HERE)

import runner  # noqa: E402
from speclib import SpecPlan, load_corpus  # noqa: E402

gate = json.load(open(os.path.join(HERE, "gate_decision.json")))
sel = gate["selection"]
if sel["deployed_is_v6"]:
    pol = sel["documentation_policy"]
    a_c, tau_c, s_c = (sel["documentation_a"], sel["documentation_tau"],
                       sel["documentation_s"])
    cfg_label = (f"DOCUMENTATION config (gate did not clear; deployed model "
                 f"IS v6): {pol} a={a_c} tau={tau_c} (boost only, s=0 in "
                 "ablations per ADDENDUM 4 A4)")
else:
    pol = sel["policy"]
    a_c, tau_c, s_c = sel["a"], sel["tau"], sel["s"]
    cfg_label = (f"shipped config {pol} a={a_c} tau={tau_c} "
                 "(boost only, s=0 in ablations per ADDENDUM 4 A4)")
CAP = sel.get("cap")          # set by read_corpus if fitted before ablations
N_MIN = sel.get("n_min", 3)

corpus = load_corpus()
W = int(pol[3]) if pol.startswith("p1w") else 5
C = int(pol.split("c")[1]) if "c" in pol[4:] else 5
plan = SpecPlan("p1", W=W, c=C, corpus=corpus) if pol.startswith("p1") \
    else SpecPlan("p3", W=5, corpus=corpus)

if a_c == 0:
    print("NOTE: configured boost a=0 (P3 documentation case) — ablation "
          "lines will be identical to v6 by design; emitting anyway.")

CASES = [("ENVY", "2025-11-01", None),
         ("LEV", "2025-08-01", "2026-03-01"),
         ("SEN", "2026-05-01", None)]

print("v6 reference daily run...", flush=True)
ref = runner.run_config(None, 0.0, 5.0, 0.0, daily=True)
days = sorted(ref["daily_r"].keys())
tidx = ref["tidx"]

doc = {"written_by": "agent:roster-g SPEC RUN",
       "preregistered": "ADDENDUM 4 (windows, boost-only ablation, exact-0 "
                        "pre-boundary gate enforced at emit time)",
       "config": cfg_label,
       "policy": pol, "a": a_c, "tau": tau_c,
       "s_note": "s=0 globally in ablation runs (A4); the corpus-wide "
                 "scoring run in roster_spec_read.json is separate and "
                 "never mixed (spec §5)",
       "cases": []}

for org, w0, w1 in CASES:
    w1 = w1 or "2026-12-31"
    fin = plan.final(org)
    bounds_all = [b for b in fin["boundaries"]] if fin else []
    bounds_win = [b for b in bounds_all
                  if w0 <= b["date"] <= w1 and not b["provisional"]]
    print(f"--- {org} window {w0}..{w1}: {len(bounds_win)} in-window "
          f"boundaries {[(b['date'], b['k']) for b in bounds_win]}", flush=True)
    abl = runner.run_config(plan, a_c, tau_c, 0.0, n_min=N_MIN, cap=CAP,
                            team_filter=[org], boost_only=True,
                            min_boundary_date=w0, daily=True)
    ti = tidx[org]
    # detection dates (separation starts there, after the vertical)
    if pol.startswith("p1"):
        det_of = {}
        for e in plan.orgs[org]["events"]:
            det_of.setdefault(e["j"], e["det_date"])
        det_dates = [det_of.get(b["j"]) for b in bounds_win]
    else:
        det_dates = []
    b0 = bounds_win[0]["date"] if bounds_win else None
    # ── the hard gate: exact zero before the first activated boundary ──────
    if b0 is not None:
        pre_days = [d for d in days if d < b0]
    else:
        pre_days = days
    gmax = 0.0
    for d in pre_days:
        gmax = max(gmax, float(np.abs(ref["daily_r"][d]
                                      - abl["daily_r"][d]).max()))
    team_pre = max((float(abs(ref["daily_r"][d][ti] - abl["daily_r"][d][ti]))
                    for d in pre_days), default=0.0)
    if gmax != 0.0:
        raise AssertionError(
            f"ZERO-GATE VIOLATION {org}: max|r_v6-r_abl|={gmax} over "
            f"{len(pre_days)} pre-boundary days — implementation bug; "
            "no JSON written")
    print(f"    zero-gate OK: global max pre-boundary diff = {gmax} "
          f"(team path {team_pre}) over {len(pre_days)} days", flush=True)
    win_days = [d for d in days if w0 <= d <= w1]
    v6_path = [{"d": d, "r": round(float(ref["daily_r"][d][ti]), 4)}
               for d in win_days]
    abl_path = [{"d": d, "r": round(float(abl["daily_r"][d][ti]), 4)}
                for d in win_days]
    seq = corpus["team_match_seq"][org]
    match_dates = sorted({d for d, mid in seq if w0 <= d <= w1})
    post_sep = max((float(abs(ref["daily_r"][d][ti] - abl["daily_r"][d][ti]))
                    for d in win_days), default=0.0)
    doc["cases"].append({
        "org": org, "window": [w0, w1],
        "boundaries": [{"date": b["date"], "k": b["k"],
                        "k5": f"{b['k']}/5", "policy": pol,
                        "detected": det_of.get(b["j"]) if pol.startswith("p1")
                        else None}
                       for b in bounds_win],
        "boundaries_pre_window_context": [
            {"date": b["date"], "k": b["k"], "provisional": b["provisional"]}
            for b in bounds_all if b["date"] < w0],
        "v6_path": v6_path, "ablation_path": abl_path,
        "prechange_max_abs_diff": 0.0,
        "prechange_gate": {"global_max": gmax, "team_path_max": team_pre,
                           "n_days_checked": len(pre_days),
                           "boundary_ref": b0 or "NONE (whole timeline)"},
        "max_separation_in_window": round(post_sep, 4),
        "match_dates": match_dates,
        "caption": "Same model; identical until the boundary (separation "
                   "begins at its detection); every point of separation is "
                   "this team's own adaptation.",
    })

sen = next(c for c in doc["cases"] if c["org"] == "SEN")
assert sen["boundaries"] == [], "SEN case must draw no vertical"
assert sen["max_separation_in_window"] == 0.0, \
    "SEN ablation must be identical to v6 in-window"
with open(os.path.join(STATS, "roster_spec_cases.json"), "w") as f:
    json.dump(doc, f, indent=1)
print("stats/roster_spec_cases.json written (3 cases, zero-gates enforced)")
