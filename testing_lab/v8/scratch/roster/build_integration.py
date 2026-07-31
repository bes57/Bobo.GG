"""agent:roster step 6 — integration tiers + preregistered prospective plan.

Design document (no solve changes, no new looks). MDE projections are display
arithmetic on stored sigmas (power_mde*.json). Writes stats/roster_integration.json.
"""
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.dirname(os.path.dirname(HERE))
STATS = os.path.join(V8, "stats")

pm = json.load(open(os.path.join(STATS, "power_mde.json")))
pme = json.load(open(os.path.join(STATS, "power_mde_expanded.json")))
rb = next(b for b in pm["buckets"] if b["bucket"].startswith("roster change <=3"))
sd_bucket = rb["sd_within_median"]          # 0.02199
sd_overall = pme["mde"]["within"]["sigma_adj"]  # 0.02207
tr = json.load(open(os.path.join(STATS, "roster_treatments.json")))


def mde80(sd, n):
    return round(2.8016 * sd / math.sqrt(n) * 1000, 2)


proj = {str(n): {"overall_within_milli": mde80(sd_overall, n),
                 "post_change_le3_within_milli": mde80(sd_bucket, n)}
        for n in (50, 75, 100, 150, 200, 300)}

doc = {
    "written_by": "agent:roster",
    "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    "status": "DESIGN + PREREGISTERED PROSPECTIVE PLAN — no code changes made; "
              "VCTMM is hands-off (v8 README rule 5)",

    "tier1_sizing_only": {
        "recommendation": "RECOMMENDED NOW",
        "what": "build_model_snapshot adds a per-team roster_flag block; the "
                "bot sizes down / widens quotes on fresh rosters. Zero "
                "fair-value risk: the model's probabilities are untouched.",
        "snapshot_fields": {
            "overlap_vs_prev": "overlap of the team's current lineup vs its "
                               "pre-change lineup at the most recent "
                               "rotation-guarded change event (engine formula "
                               "|A∩B|/max(|A|,|B|,5))",
            "matches_since_change": "lineups-agent definition (walk-forward)",
            "run_len_alive": "consecutive matches on the current lineup",
            "sustained_alive": "walk-forward sustained-at-day flag "
                               "(confirmed >=3 or still alive)",
            "change_date": "date of the most recent change event",
        },
        "data_path": "data/maps/<event>.csv player rows -> engine."
                     "load_match_lineups (already scraped per event; no new "
                     "scraping); episode logic = preregister.roster.md §1",
        "playbook_rule_sketch": [
            "matches_since_change <= 2 AND overlap_vs_prev <= 0.6 -> "
            "quarter-size, widen quote by one tick",
            "matches_since_change <= 2 AND overlap_vs_prev == 0.8 -> "
            "half-size",
            "matches_since_change >= 3 -> normal sizing",
        ],
        "evidence_basis": "atlas: post-change teams beat the carried rating "
                          "by +4.4/+5.6/+4.8 pp (keep4/keep3/overhaul, first "
                          "3 matches, vs +0.7 stable) — model error is "
                          "ELEVATED and DIRECTIONAL there; ENVY S1->S2 cost "
                          "24-32 pp/match, LEV/Neon -14.7 pp/match. Sizing "
                          "down on |elevated-uncertainty| rows needs no "
                          "directional model claim.",
        "telemetry": "log roster_obs {overlap_vs_prev, matches_since_change} "
                     "per quoted event (fits the existing telemetry spec) so "
                     "the prospective test below can also be read off live "
                     "quoting data",
    },

    "tier2_fair_value": {
        "recommendation": "NOT RECOMMENDED NOW — no treatment earned it",
        "why": "all three exploratory reads were negative on the (spent) "
               "holdout: (b) -3.889m (falsifier fired), (c) -1.423m (inside "
               "the 1.77m floor, no support), (d) phase-reset -19.275m vs v6 "
               "and -7.524m vs its own filter base (falsifier fired; contrast "
               "prediction reversed). v6's year-boundary continuity 0.3 "
               "remains the champion roster mechanism.",
        "if_it_ever_earns_it": {
            "constants_that_change_b": "engine roster structures: year_cont "
                                       "replaced by episode list {change_date, "
                                       "ov, sustained-at-day}; gamma; beta "
                                       "refit (scale-bound, README rule 8)",
            "constants_that_change_c": "post-solve blend (a0, M, ov<=0.6 "
                                       "trigger, region-mean pull) inside "
                                       "build_model_snapshot after the daily "
                                       "solve; beta refit on adjusted rdiff",
            "vendor_sync": ["freeze spec + params in trading_model/ as a NEW "
                            "snapshot version field (never edit the frozen v6 "
                            "model_snapshot.json)",
                            "shadow-quote both surfaces >=2 weeks",
                            "promotion decision ONLY via the prospective test "
                            "below on settled outcomes (never Kalshi "
                            "agreement — README rule 9)"],
        },
    },

    "player_identity_mean_shift_design_note": {
        "status": "DESIGN NOTE ONLY (operator addendum) — no holdout read spent",
        "idea": "at a change event, shift the reference-point mean by a "
                "walk-forward estimate of (incoming - outgoing) player "
                "quality (e.g. decayed per-map rating share or PTD-style "
                "contribution from data/enriched/player_map_advanced.csv), "
                "scaled by (1 - ov). Directional where the phase-reset was "
                "agnostic: Neon-in would shift LEV UP, inspire-out ENVY DOWN.",
        "prior_evidence": "ledger row 6 'lineup-overlap roster reweighting' "
                          "REJECTED pre-v8 but reclassified UNRESOLVED "
                          "(stats/ledger_reclass.json) — that rejection was a "
                          "solve-reweighting, not a mean shift; Phase 3's "
                          "player-rating idea remains unused. CN events lack "
                          "player enrichment (coverage audit required first).",
        "cost": "walk-forward player table + train-fit shift scale; one "
                "prospective arm, zero holdout reads on this frame",
    },

    "prospective_validation_plan": {
        "status": "PREREGISTERED NOW (this file + report page) — the only "
                  "path to adjudication; the 2025-26 holdout is spent",
        "population": "series dated > 2026-07-28 as they settle (Stage 2 "
                      "remainder + Champions 2026 + any 2027 data at "
                      "adjudication time), walk-forward, all params frozen "
                      "as below, beta refit only on data dated <= 2026-07-28",
        "arms_frozen": {
            "A_v6": "champion (reference)",
            "B_continuity": "read-1 spec, gamma=2.0 (exploratory read was "
                            "-3.889m; expectation LOW)",
            "C_coldstart": "read-2 spec, a0=1.0 M=6 (exploratory -1.423m, "
                           "inside floor)",
            "D_phase_reset": "read-3' spec, g=0.5 on h3-1b base (operator "
                             "shape; exploratory read NEGATIVE with falsifier "
                             "fired — carried prospectively because the frame "
                             "evidence is exploratory-only, expectation LOW)",
            "E_atlas_replication": "NON-MODEL check: does the +4-6pp "
                                   "first-3-matches post-change outperformance "
                                   "replicate? (mean(won - p_v6) on "
                                   "post-change team-observations, msc<=2, "
                                   "sustained episodes)",
        },
        "metrics": "mean paired dLL vs v6 (overall + post-change <=3 bucket), "
                   "CRN-style paired bootstrap with NEW seeds registered in "
                   "crn.json at adjudication time; both units via "
                   "referee.expected_roi_of_dll",
        "decision_rule": "an arm is promoted to Wave-3 candidate iff dLL >= "
                         "MDE80 at the realized n AND p_better >= 0.95 in "
                         "BOTH CRN modes AND no G3 major bucket regression "
                         "(referee.promotion_gate bars). Arm E 'replicates' "
                         "iff the pooled first-3 bias is positive with 95% CI "
                         "excluding 0.",
        "mde80_projection_milli": {
            "formula": "2.8016 * sd / sqrt(n); sd_overall_within=0.02207 "
                       "(power_mde_expanded sigma_adj), sd_post_change_le3="
                       "0.02199 (power_mde roster bucket)",
            "by_n": proj,
            "reading": "at a realistic first checkpoint (~100 fresh series, "
                       "~60 in the post-change bucket) the floors are ~6.2m "
                       "overall / ~8.0m bucket — only large effects are "
                       "adjudicable soon; plan to re-check after Champions.",
        },
    },

    "exploratory_reads_context": {k: {"delta_milli": tr[k]["delta_milli"],
                                      "status": "EXPLORATORY"}
                                  for k in ("read1_b_continuity",
                                            "read2_c_coldstart",
                                            "read3_d_phase_reset")},
}
with open(os.path.join(STATS, "roster_integration.json"), "w") as f:
    json.dump(doc, f, indent=1)
print("written; MDE projections:", json.dumps(proj, indent=1))
