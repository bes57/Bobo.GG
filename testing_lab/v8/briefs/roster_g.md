# agent:roster-g — THE SPEC RUN (v6 + mid-season roster subsystem)

Your contract, in priority order:
1. testing_lab/v8/briefs/roster_spec_operator.md — THE DEFINITION. Operator-
   authored. If anything below or anything on disk contradicts it, the spec
   wins. Read it twice. Its §7 read-back has already been delivered to the
   operator; your preregistration must open by restating it in your own words
   and flagging any point where you think the spec is ambiguous BEFORE running.
2. testing_lab/v8/briefs/wave2_common.md — frame/CRN/referee/train-only/MDE
   rules still bind wherever the spec doesn't override them.
3. This file — execution pointers only.

## Pointers (context from prior work; artifacts all on disk)
- v6 engine: testing_lab/engine.py (games-decay path computes PER-SIDE w_w/w_l
  per solve day then combines sqrt(w_w·w_l) — the spec's "per-side hook":
  apply the boost to the changed team's side factor before the combine; the
  sub down-weight [1 − s(1−o)] likewise on the fielding team's side).
  v6 config: decay games consistency (20,12), rd power .75 scale 2.5,
  roster_mode year 0.3, ridge .5, region_prior_ridge 1.5, w_custom PO×1.6,
  champ ×2 (exact-shape champions flag). Reuse referee.load_timeline_games-
  compatible machinery / scratch/roster/ v6 baselines; v6-reproduction guard
  to 1e-12 before anything else (prior runs achieved this — see logs).
- Lineups: v8/data/lineups.csv + lineup_features.csv (+ full-corpus top-ups in
  scratch/bias_h3/ and scratch/context/). Per-match fives with ProfileURLs.
  NOTE the spec's modal-five definition (trailing W MATCHES, ties by recency)
  is NOT the earlier 30d-calendar modal — implement the spec's, walk-forward.
- Frame: v8/data/frame_expanded/series.csv (sha in crn.json). Holdout
  > 2024-12-31, n=1217. Within-family MDE 1.773m. CRN for all bootstraps.
- Prior roster artifacts: preregister.roster.md (ADDENDA 1-3 + outcomes),
  stats/roster_*.json, logs/roster.log. Treatments (b)(c)(d)(e) and the
  overlay (f) are HISTORY — keep their records; your run supersedes their
  role. Do not overwrite their JSON keys; add yours.
- SEN/Marved: the spec asserts SEN fielded Marved for one match last week then
  reverted. Verify from data (lineups + maps CSVs; scrape nothing) — if the
  data shows something different, report what the data shows and use the
  closest real SEN deviation for fixture 1, flagging the discrepancy.

## Execution notes (binding)
- PREREGISTER FIRST: preregister.roster.md ADDENDUM 4 (SPEC RUN) — the §7
  restatement, all grids (a ∈ [0,6] discretized incl. 0 interior, τ ∈
  {2,3,5,8,13}, s ∈ [0,1] grid, W ∈ {3,5,8}, c ∈ {0,3,5}, m for P2, n_min +
  cap for the floor), the inner-CV design for λ and the activation-gate bar
  (improvement > its own inner-CV SE), the P1/P2/P3 policy definitions, chain
  merge rule, fixture assertions, falsifiers, and the read plan (below).
  Everything train-only; the spec's nesting/shrinkage/gate/floor are enforced
  IN CODE with assertions whose results are published.
- P2 warning: run it ONLY if your harness reproduces the exact knowable state
  at each T (provisional boundaries included). Otherwise publish P2 = NOT RUN
  with the reason, per the spec.
- FIXTURES BEFORE RUNS (spec §2.5): SEN named case, synthetic revert,
  synthetic sustained (detection at the lag the policy predicts — earlier
  detection = peeking = stop), corpus census per policy per season with
  reverted-deviation counts and the EWC-class sanity check. All hard
  assertions; publish as stats/roster_spec_fixtures.json with pass/fail.
- Holdout reads (exploratory, spent-holdout labels, looks tally kept
  current in roster_looks.json; each disclosed): budget ≤3 — ONE corpus-wide
  headline read for the gate-selected configuration (policy chosen by
  train-side evidence only, disclosed), plus at most TWO pre-registered
  policy-contrast reads if train evidence genuinely cannot separate them.
  Do not spend reads on grid points; grids are train/inner-CV territory.
- Per-team ABLATION runs (spec §4): ENVY, LEV, SEN (+ any additional named
  case you feature): a>0 for the featured team only, a=0 elsewhere. Emit
  stats/roster_spec_cases.json: per case {org, boundaries: [{date, k5,
  policy}], v6_path: [{d,r}], ablation_path: [{d,r}], prechange_max_abs_diff
  (must be exactly 0.0 — if not, FIX THE IMPLEMENTATION), match_dates}.
  SEN must show boundaries: [] under the winning policy (P2: retracted, final
  timeline identical). The page generator (orchestrator-owned) enforces the
  zero-gate and the SEN no-vertical gate at render time and will refuse to
  build on violation — emit data that passes.
- Corpus-wide scoring (spec §5): headline read + slices (change-gated,
  improvement, degradation, retention bands k=4/3/≤2, sub-heavy rows) +
  coupling distribution (non-changing teams' rating movement under the
  corpus-wide run) in its own JSON block + gate outcome (fired?, â after
  shrinkage, inner-CV margin) as a first-class output.
- roster_flag extension (spec §2.5 "what the bot gets"): emit the extended
  per-team fields into stats/roster_integration.json (modal five, last-match
  deviation, matches since confirmed boundary, provisional bit) computed as
  of the data's last date — design artifact for the operator's Tier-1 sizing
  integration, no bot writes.
- Prospective arm: freeze H_specrun (gate-selected config incl. policy,
  â, τ̂, ŝ, W, c, n_min) in roster_integration.json.

## Outputs (yours alone; one writer)
- stats/roster_spec_fixtures.json, roster_spec_census.json,
  roster_spec_cases.json, roster_spec_read.json (headline + slices +
  coupling + gate outcome), roster_looks.json (updated),
  roster_integration.json (updated: roster_flag extension + frozen arm)
- preregister.roster.md ADDENDUM 4 (+outcomes at same resolution)
- testing_lab/v8/roster_adaptation.md: new top section per spec §6 (answer
  first: subsystem yes/no, gate decision, â, effect ± CI next to MDE)
- logs/roster.log; scratch/roster/spec_run/ for all code
- Do NOT touch the HTML generator (orchestrator-owned).

## Return (≤400 words)
Fixture pass/fail line each; census headline (boundaries per policy,
EWC sanity); gate decision + â/τ̂/ŝ/policy; corpus-wide read (Δm, CIs, key
slices) with MDE context; ablation zero-gate values (must be 0.0 ×3);
coupling distribution summary; SEN data verification result; artifact paths.
