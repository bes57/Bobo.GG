"""SPEC RUN — roster_flag extension (spec §2.5 'what the bot gets') +
frozen prospective arm H_specrun. Updates stats/roster_integration.json
by ADDING keys (never overwriting prior agents' keys). No bot writes.
"""
import json
import os
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
STATS = os.path.join(V8, "stats")
sys.path.insert(0, HERE)

from speclib import SpecPlan, load_corpus  # noqa: E402

gate = json.load(open(os.path.join(HERE, "gate_decision.json")))
sel = gate["selection"]
corpus = load_corpus()
plan = SpecPlan("p1", W=8, c=5, corpus=corpus)

last_date = max(d for org in corpus["team_match_seq"]
                for d, m in corpus["team_match_seq"][org])
D = (date.fromisoformat(last_date) + timedelta(days=1)).isoformat()


def slugs(mode):
    return sorted(u.rsplit("/", 1)[-1] for u in mode) if mode else None


flags = {}
for org in sorted(plan.orgs):
    ver = plan.version_asof(org, D)
    if ver is None or ver["nvis"] == 0:
        continue
    mm = plan.orgs[org]["matches"]
    last_pos = ver["nvis"] - 1
    last_known = max((i for i in range(ver["nvis"])
                      if mm[i][2] is not None), default=None)
    conf = [b for b in ver["boundaries"] if not b["provisional"]]
    prov = [b for b in ver["boundaries"] if b["provisional"]]
    last_dev = (bool(ver["o"][last_known] < 1.0)
                if last_known is not None else False)
    flags[org] = {
        "as_of": D,
        "modal_five": slugs(ver["mode"]),
        "last_match_date": mm[last_pos][0],
        "last_match_deviated_from_modal": last_dev,
        "matches_since_confirmed_boundary": (int(ver["n"][last_pos])
                                             if ver["n"][last_pos] >= 0
                                             else None),
        "last_confirmed_boundary": ({"date": conf[-1]["date"],
                                     "k": conf[-1]["k"],
                                     "k5": f"{conf[-1]['k']}/5"}
                                    if conf else None),
        "provisional_pending": bool(prov) or last_dev,
        "provisional_detail": ([{"date": b["date"], "k": b["k"]}
                                for b in prov] or None),
        "sizing_note": ("provisional/deviating state = sizing signal, not a "
                        "fair-value change; quote smaller until it settles "
                        "(spec 2.5)"),
    }

integ = json.load(open(os.path.join(STATS, "roster_integration.json")))
integ["spec_run_roster_flag_extension"] = {
    "written_by": "agent:roster-g SPEC RUN",
    "classifier": "P1 W=8 c=5 (gate-selected policy; walk-forward, "
                  "ADDENDUM 4 definitions)",
    "fields": ["modal_five", "last_match_deviated_from_modal",
               "matches_since_confirmed_boundary", "provisional_pending"],
    "teams": flags,
    "note": "design artifact for Tier-1 sizing only; no bot writes; the "
            "fair-value subsystem itself did NOT earn deployment (see "
            "spec_run_verdict)"}
integ["prospective_validation_plan"]["arms_frozen"]["H_specrun"] = (
    "SPEC RUN gate-selected config, FROZEN: P1 W=8 c=5, a=28.0 tau=13.0 "
    "s=0.7 n_min=3 cap=1.5 (lambda_hat=0 by inner-CV; gate fired "
    "+6.42m CV vs 5.34m SE). Exploratory holdout read: -11.595m "
    "[-18.99,-4.39] — FALSIFIER FIRED on the spent frame; expectations LOW; "
    "carried for prospective adjudication per spec 6 (decision rule "
    "unchanged: dLL >= MDE80 at realized n AND p_better >= .95 both CRN "
    "modes AND no G3 regression).")
integ["spec_run_verdict"] = {
    "deployed_model": "v6 EXACTLY (subsystem not deployed)",
    "gate": "FIRED on train inner-CV (+6.416m > SE 5.339m, p1w8c5) but the "
            "preregistered holdout falsifier fired (-11.595m <= -1.773m): "
            "the causal subsystem at the gate-selected config is dead on "
            "this frame; v6 stands.",
    "gate_bar_lesson": "mean>SE at K=5 with lambda from the same inner-CV "
                       "did not protect out-of-sample here; recorded as a "
                       "first-class negative result for the 3.5 guarantee "
                       "chain on this family."}
with open(os.path.join(STATS, "roster_integration.json"), "w") as f:
    json.dump(integ, f, indent=1)
n_pend = sum(1 for v in flags.values() if v["provisional_pending"])
print(f"roster_integration.json updated: {len(flags)} teams, "
      f"{n_pend} provisional/deviating as of {D}; H_specrun frozen")
for org in ("SEN", "ENVY", "LEV", "G2", "TH"):
    if org in flags:
        f_ = flags[org]
        print(f"  {org}: msb={f_['matches_since_confirmed_boundary']} "
              f"dev={f_['last_match_deviated_from_modal']} "
              f"pend={f_['provisional_pending']} "
              f"b={f_['last_confirmed_boundary']}")
