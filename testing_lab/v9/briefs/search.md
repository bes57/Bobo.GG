# agent:v9-search — Phase 2: transfer-gated optimization + the frozen ladder

Read testing_lab/v9/README.md, then testing_lab/v9/DESIGN.md + stats/
v9_transfer_protocol.json (the law you execute VERBATIM — advance clauses
A1-A5, C5 concentration flag, one ledgered evaluation per candidate, β
discipline), then briefs/family.md context + scratch/family/v9lib.py (the
implementation you search), then wave2_common.md.

## Priors you inherit (from the gate autopsy, stats/v9_gate_autopsy.json)
- Solve-side transfers NEGATIVE at every tested a including the cap
  (a=6 pooled −5.38m; a=4.5 −4.6/−3.8m; a=28 blocked everywhere). The
  solve-side family is PRESUMED DEAD: verify cheaply at 2-3 small-a points
  (a ∈ {0.5, 1, 2} at the τ/s the train stage prefers) under the protocol,
  and if the presumption holds, record it and spend everything else on the
  prediction layer. Do not burn hours confirming a corpse.
- Prediction-layer configs cost ~0.02s on a shared v6 base (v9_cost.json):
  sweep DENSELY — b, τ, k-scaling exponent, δ1 vs δ2 vs hybrid-δ, policy
  W ∈ {3,5,8}, n_min — full grid on FIT windows, hyperparams frozen on
  ≤2024-12-31 per protocol before any VAL touch.
- Hybrid (small-a solve-side + δ): only if BOTH parents show independent
  transfer signal; otherwise skip (pre-justify in preregister).

## Discipline
- Preregister (preregister.search.md) BEFORE running: the full grid, the
  candidate-selection rule (how grid winners become the ≤N candidates that
  get their ONE transfer evaluation each — protocol's rule), predicted
  signs/sizes per family member (δ2's atlas-prior positive; δ1's
  evidence-conditional; solve-side presumed negative), and falsifiers.
- Every transfer evaluation ledgered in stats/v9_looks.json per protocol.
  NO spent-holdout-as-holdout reads — the 0/3 exploratory budget stays
  untouched for the freeze stage's sanity reads (not yours).
- Fragility per survivor: drop-top-5% + leave-one-event-out (A5) + C5.
- Walk-forward everywhere; δ1's date-strict assert stays hot in every run.

## Deliverables
- stats/v9_search_grid.json (full grid results on FIT windows — train-side,
  chart-ready), v9_candidates.json (each candidate: config, transfer ledger
  A1-A5+C5 clause-by-clause, verdict ADVANCE/DIE), v9_ladder.json (≤3
  survivors, ladder-ordered per protocol, β refit once per arm on
  2023→2026-07-28 and FROZEN, plus v6 as arm 0 — the evaluator consumes
  this file verbatim; if ZERO candidates advance, the ladder is v6 alone
  and that is a publishable outcome stated plainly).
- phase_search.md (answer first: how many advanced, the ladder, what died
  incl. the solve-side verdict), preregister.search.md (+outcomes),
  logs/search.log, scratch/search/.

Return ≤400 words: grid coverage, solve-side verdict, candidates evaluated
with clause-by-clause outcomes, the frozen ladder (or "v6 alone"), looks
ledger state, artifact paths.
