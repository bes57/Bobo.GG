# agent:v9-finish — Phase 3: prospective evaluator + the v9 Lab page

Read testing_lab/v9/README.md, DESIGN.md, stats/v9_prospective_protocol.json
(you implement it verbatim), stats/v9_ladder.json (the ladder: v6 alone),
stats/{v9_candidates,v9_search_grid,v9_gate_autopsy,v9_fixtures,v9_cost,
v9_looks}.json, phase_search.md, then wave2_common.md. You are the program's
reporter + evaluator-builder. You compute no new model results.

## 1. The standing prospective evaluator
testing_lab/v9/score_prospective.py — idempotent, run-on-demand:
- Loads every FROZEN arm on disk: the v9 ladder (v6, β=0.128512) PLUS the
  earlier pre-frozen roster arms recorded in testing_lab/v8/stats/
  roster_integration.json (F_v6_overreact, H_specrun, and the b/c/d/atlas
  replication arms if specified there) — all were preregistered with fixed
  configs; score them as REFERENCE arms under the same machinery, clearly
  separated from the (empty) candidate set in output.
- Finds settled series dated > 2026-07-28 from the live data files
  (match_results.csv + match_dates.json + maps CSVs — the same harness
  rules as frame_expanded; walk-forward solves include all history).
- Scores every arm per the protocol: paired ΔLL vs v6, block+iid CRN,
  checkpoint logic n ∈ {100,200,400} with the frozen bars/alpha-spend/kill
  rules; before n=100 it reports "accumulating (n=X)" and evaluates nothing.
- Writes stats/v9_prospective_scoreboard.json (arms, n so far, next
  checkpoint, per-arm status ACCUMULATING/ALIVE/KILLED/PROMOTED) + appends
  a dated line to logs/prospective.log each run. Never refits anything.
- Document the run cadence in the file header (manual or cron weekly; it
  must be safe to run any time — idempotent, read-only over data/).
  Run it once now to initialize the scoreboard (expect n≈0-few).

## 2. The v9 Lab page — /testing/report/v9_lab
gen_v9_report.py in the house pattern (study gen_v8_report.py + the roster
generator incl. its gate style; every number from stats JSONs, download
links via /testing/v8/stats/ route — add a v9 alias route ONLY if trivial,
else copy the JSONs it needs into testing_lab/v8/stats/ with v9_ prefix —
NO: they already live in testing_lab/v9/stats; simplest correct: extend
TestingLab.py's v8_stats route pattern with an equivalent /testing/v9/stats/
route — one small addition, same auth gate, same regex).
Sections, verdict-first per house convention:
§0 THE ANSWER: "The ladder is v6 alone" — zero of 5 candidates advanced;
   the family measured dead across solve-side a=0.5→28, δ1 (nominated
   nothing, best +0.81m in-era), δ2 (train +5.5m → pooled −5.4m transfer,
   the third train-mirage, caught pre-nomination by the red-flag rule),
   hybrid; v6's β frozen; what would change this = the prospective
   scoreboard, embedded.
§1 The motivation (honest): LEV −14.7pp/match underpricing, ENVY +24-32pp,
   atlas +4-6pp — the misses are real; every mechanism tried pays more
   elsewhere than it earns on the change windows (link the case charts on
   the Roster page rather than duplicating).
§2 The validation design: transfer protocol clauses, measured false-advance
   ~2% / power 75-90%, and THE GATE AUTOPSY table (gate said / transfer
   said / holdout said) — the chart that explains why v9's referee is
   trustworthy where the spec run's wasn't.
§3 The search: grid coverage chart (3,240 configs), eligibility funnel
   (231 → 5 evaluated → 0 advanced), candidate table clause-by-clause,
   the N1 red-flag story (74.8% of gain in 10 rows).
§4 The prospective scoreboard (live from the evaluator's JSON): arms,
   n accumulating, checkpoint bars, statuses; caption: this is the only
   confirmatory instrument left, by design.
§5 Looks ledger + integrity notes (design agent's struck-draft disclosure
   included verbatim — it's part of the record), tripwires, ledger entry
   (the family's death is do-not-retest at these definitions; re-open
   triggers: new data source at player level, or a prospective surprise).
Nav: add "v9 Lab" tab across all existing report pages + hub, same
string-patch discipline as before (assert-exactly-once; regenerate only
roster/v9 pages, never the older ones).
Mirror: testing_lab/v9/v9_report.md, answer-first.

## Outputs (yours alone)
score_prospective.py, gen_v9_report.py, the served page + nav patches +
TestingLab.py v9-stats route, stats/v9_prospective_scoreboard.json,
v9_report.md, preregister.finish.md (trivial: you reshape, never compute),
logs/finish.log.

Return ≤300 words: scoreboard init state (n so far), page URL + sections,
nav patch count, route added, artifact paths.
