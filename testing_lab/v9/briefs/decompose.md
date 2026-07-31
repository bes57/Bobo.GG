# agent:v9-decompose — the case ledger: WHERE exactly the candidates lost

Read testing_lab/v9/README.md, then stats/v9_candidates.json (the frozen
configs + their ledgered transfer evaluations), then scratch/search/ (the
runners that produced them — reuse, don't rebuild), then wave2_common.md.

## Scope (operator question, verbatim intent)
"What was worse? What are examples where the model is worse? It shouldn't
affect teams with no roster changes, so we can only look at teams with
mid-season roster changes." Produce the per-match decomposition of the
ALREADY-LEDGERED evaluations — same rows, same frozen configs, no new
fitting, no new selection, no new looks (append one disclosure line to
stats/v9_looks.json: decomposition-of-existing-reads, zero new information
about any un-evaluated config).

## Work
For each of: N1_delta2 (the best δ), S_a1.0 (solve-side representative),
N2_hybrid6 — on the VAL1 (2025) + VAL2 (2026H1) windows exactly as ledgered:
1. Per-match rows where the candidate differs from v6 (for δ: exactly the
   affected matches; for solve-side: ALL matches — split into
   change-affected vs pure-unchanged): date, event, teams, which side(s)
   had an active phase (org, k/5, n_since, direction of adjustment),
   p_v6, p_candidate, outcome, per-match ΔLL contribution (candidate − v6;
   positive = candidate better).
2. Aggregates per candidate per window: n affected, n better / n worse,
   summed Δ on affected rows vs unaffected rows (for solve-side, the
   unaffected-row sum quantifies pure coupling damage — the operator's
   "shouldn't affect unchanged teams" test, answered numerically).
3. THE EXAMPLES (chart/prose-ready): per candidate, the 10 worst and 10
   best affected matches with a one-line reason field derivable from the
   data (e.g. "δ2 pushed ORG up (k=2/5, n=1) but they lost — post-change
   team was genuinely worse", "boost amplified a noisy early win,
   overshot next match"). For δ2 additionally: the split of affected-match
   Δ by whether the changed team's post-change record was improving vs
   degrading (the LEV-shape vs ENVY-shape ledger — the operator's own
   framing; expect δ2 to win the first and lose the second; report
   whatever the data says).
4. Sanity assertions: for δ candidates, Δ on matches with NO active phase
   is exactly 0 (locality by construction — publish the assertion); the
   summed per-match Δ reproduces each ledgered window total to numerical
   tolerance (tie-out with the published numbers, print the tie-out).

## Outputs (yours alone)
- stats/v9_case_decomposition.json (all of the above, chart-ready)
- logs/decompose.log; scratch/decompose/
- Do NOT touch gen_v9_report.py (orchestrator-owned), the narrative, the
  ledger verdicts, or any preregister outcomes.

Return ≤300 words: tie-out result, the locality assertion, the headline
split (LEV-shape vs ENVY-shape Δ for δ2; unaffected-row damage for
solve-side), and the 3 most instructive named examples.
