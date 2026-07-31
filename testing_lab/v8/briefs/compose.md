# agent:compose — Wave 3: stacking across what survived

Read briefs/wave2_common.md first; it is law (frame, CRN, referee, MDE
context, both units, train-only fitting, fail loudly). Scope: does any
pre-declared stack of Wave 2's surviving components beat v6 on the expanded
holdout at the promotion bar — or does v6 stand? "v6 stands" is an acceptable,
publishable outcome and must be stated in §12's words if true.

## The survivors you may compose from (and only these)
- **3e stand-in shrink** (context): shrink-toward-0.5, k_standin=+0.347 as
  fit on train; overall floor, EWC bucket +3.46m. stats/context_shrink.json.
- **Event-class fade** (decay 5b-d): off-season results fade faster; +0.24m,
  positive in all six subpops, sub-MDE. stats/decay_axes.json.
- **n_eff-gated 5d state-space** (bias-h3): the NAMED, not-holdout-shopped
  shape — v6 everywhere, 5d state-space only on hard-gated thin-data rows
  (h3's +28..+72m buckets). stats/h3_ensemble.json (gate spec),
  h3_process_noise.json, scratch/bias_h3/ machinery.
- Referee-side control variate is already banked; use it in every CI.
Nothing else: every other Wave 2 config was dead or redundant. Do not
resurrect killed components inside a stack.

## Rules
1. Preregister (preregister.compose.md) BEFORE any holdout scoring:
   at most THREE stacks, exact composition + fitting procedure + predicted
   effect + falsifier for each. Component selection justified from TRAIN
   evidence and the wave artifacts only.
2. Fit every stack's parameters jointly on train (β refit per stack,
   scratch/compose/). One holdout scoring per stack, ever. No iteration
   after seeing holdout numbers — whatever they are, they publish.
3. Judge with referee.promotion_gate (G1 CRN both modes at the Phase-0 MDE,
   G2 max|bias| strictly reduced vs v6's 0.1478, G3 bucket floors) + both
   units + full caterpillar and bucket panels per stack.
4. **Multiple-looks accounting (the adversary will demand it):** count every
   holdout scoring the program has made across all agents (harvest from
   logs/*.log and stats/*.json) and publish the tally with your p-values in
   family-wise context. Do not hide the garden of forking paths — measure it.
5. The unexplained residual (PRX −10.2 / NRG −9.1 under each stack) gets
   reported, not chased: name it future work if it persists. No new
   mechanisms inside compose.

## Outputs (yours alone)
- stats/compose_stacks.json (per stack: spec, train fit, holdout LL, ΔLL,
  CIs both modes, both units, caterpillar, buckets)
- stats/compose_gate.json (promotion_gate verdict objects, clause by clause)
- stats/compose_spec.json (EXACT reproducible spec of the best stack —
  every constant, every gate threshold, data recipe — written so a
  spec-level reimplementation needs nothing else)
- stats/compose_looks.json (the holdout-scoring tally)
- phase_compose.md (verdict up front, in plain words)
- preregister.compose.md, logs/compose.log, scratch/compose/

Done when: ≤3 preregistered stacks scored once each; gate verdicts published;
looks tally published; the one-sentence program verdict written ("stack X
clears/fails the bar" or "v6 stands because …" — if the truth is "we could
not tell", say it in those words).

Return ≤500 words: per-stack ΔLL + gate verdict, the looks tally, the
residual-bias note, artifact paths.
