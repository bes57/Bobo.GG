# agent:bias-h3 — Phase 5 H3: rating uncertainty (state-space) + 5d

Read briefs/wave2_common.md first; it is law. Scope: point estimates ignore
per-team uncertainty — test a state-space rating (Kalman / Glicko-2 analog)
whose match probability integrates over rating variance. You ALSO own 5d
(heterogeneous half-lives): the process-noise parameter IS the half-life, so
partial pooling by team type lives here, not with agent:decay. You are the
heaviest compute in Wave 2; budget accordingly and journal checkpoints so a
restart never redoes a finished sweep.

## Experiments (preregister each)
1. **State-space core.** Per-team state (rating, variance): time/games-update
   with process noise q, observation = margin-transformed game result with
   observation noise fit on train. Match probability = ∫ sigmoid(β·Δr) over
   the joint uncertainty (Gauss-Hermite or MC with crn seeds). Walk-forward
   exactly (state at match uses only earlier games). β and all noise params
   fit train-only. Judge: holdout LL; per-team bias caterpillar (mechanism
   prediction: thin-data teams shrink toward 0.5 WITHOUT shrinking
   established elites — the observed asymmetry); max|bias|; cold-start
   bucket (was n=57 / LL 0.70205); buckets panel.
2. **5d Heterogeneous half-lives via pooled process noise.** q by team type
   with partial pooling (fit the pooling variance, no free-per-team q):
   types from observables (walk-forward): roster stability (lineup_features
   games_since_change distribution), org age (first-game date), historical
   rating volatility. Report fitted q (as implied half-life in games) per
   type with CIs. This is the defensible version of "more recency for some
   teams"; if stable-elite q ⇒ longer memory and rebuilt-roster q ⇒ shorter,
   the operator's instinct gets its quantified form — publish the table
   either way.
3. **n_eff per prediction.** Emit per-match effective-sample /
   posterior-variance so the telemetry spec's confidence-aware sizing has its
   input; include distribution summaries in the stats JSON.
4. **Ensemble-where-it-wins check.** If the state-space model beats v6 only
   on subsets (lineup-change-adjacent, cold-start, EWC stand-ins), quantify a
   gated ensemble on TRAIN, score once on holdout, and report it as the
   candidate shape instead of full replacement.

Publish stats/h3_process_noise.json EARLY (as soon as 5d fits exist) —
agent:decay's writeup references it.

## Outputs (yours alone)
- stats/h3_statespace.json, h3_process_noise.json (half-life by team type,
  CI, chart-ready), h3_neff.json, h3_bias_caterpillar.json, h3_ensemble.json
- phase5_h3.md, preregister.bias_h3.md, logs/bias_h3.log, scratch/bias_h3/

Done when: core model scored with bias tables + cold-start bucket; 5d
half-life table published with pooling described; n_eff emitted; ensemble
answer given; verdicts plain, MDE-contextualized, both units.
