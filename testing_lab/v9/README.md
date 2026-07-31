# BenPom v9 lab — v6 + roster subsystem, optimized between the two

Operator-commissioned 2026-07-29. Goal: find the best point in the nested
family spanning pure v6 (a=s=0) to v6+full roster subsystem, with validation
you can trust. Motivation (measured, on record): v6 underpriced LEV by
−14.7pp/match for 10 matches after the Neon change; overpriced ENVY +24–32pp
after its chain; population atlas shows +4–6pp early new-roster
outperformance. The v8 spec run killed ONE configuration (a=28, solve-side,
gate-selected) — not the family.

## The three laws of v9 (learned the hard way)
1. **The 2025-26 holdout is SPENT (403 recorded looks).** It adjudicates
   nothing. Selection NEVER touches it. Budget: ≤3 exploratory sanity reads
   for the entire program, taken only on frozen candidates, disclosed in
   stats/v9_looks.json, never used for selection.
2. **Selection runs on transfer, not on fit.** The recurring failure mode
   ((b), (e), spec run) is train-gain that dies out of era. A candidate
   advances ONLY on era-transfer evidence inside pre-07/28 data:
   fit 2023-24 → validate 2025; fit 2023-25 → validate 2026H1. Win both or
   die. Fragility (drop-top-5%, era-jackknife) applies to every survivor.
3. **The real referee is prospective.** Frozen ladder (v6 + ≤3 candidates,
   conservative → aggressive), β refit pre-07/28 then frozen, decision rules
   fixed in advance, scored on post-2026-07-28 series as they settle
   (Stage 2 is live; Champions lands Sep-Oct). Nothing ships before the
   prospective bar; nothing on the public site regardless (standing).

## Family (both mechanisms first-class)
- **Solve-side** (spec-run machinery, reused): per-side boost
  1 + a(1−k/5)e^(−n/τ), sub down-weight s(1−o), policy P1 W∈{3,5,8} / P3,
  chain c=5, n_min, cap. HARD prior from the failure: a capped at 6.0, no
  widening — the a=28 lesson is a law, not a suggestion.
- **Prediction-layer** (NEW; zero coupling BY CONSTRUCTION): the solve stays
  pure v6 for everyone; a changed team's series logit gets a post-change
  adjustment δ(n, k) (e.g., δ = b·(1−k/5)·e^(−n/τ) applied toward the
  direction of its own post-change evidence, or an atlas-informed intercept).
  Exact pre-change identity, no team ever couples to another, cheap to fit.
  This is the operator's per-team-overlay intuition in its natural home.
- Hybrids allowed if transfer-validated.

## Conventions
v8's stack is the foundation: engine (exact-shape champ flag), referee.py,
crn.json, frame_expanded, lineups tables, spec_run classifier+fixtures
(scratch/roster/spec_run/speclib.py — causal P1/P3, census, fixtures 54/54).
Same rules as v8: preregister before running, one writer per artifact,
journals in logs/, fail loudly, walk-forward always, both units (milli-LL +
ROI translation), MDE context on every number, market data never a target.
Layout mirrors v8: briefs/ preregister.<agent>.md stats/ logs/ scratch/.
Page: /testing/report/v9_lab, house style, gates enforced in the generator.
