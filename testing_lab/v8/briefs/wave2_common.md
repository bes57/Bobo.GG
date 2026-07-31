# Wave 2 common covenant — read before your own brief

Binds every Wave 2 agent. Your per-agent brief adds scope; this file is the
shared law. v8/README.md standing rules apply on top (esp. rule 9: market data
is never a fitting target; model selection on settled outcomes only).

## The frame (no private corpora)
- Canonical evaluation frame: testing_lab/v8/data/frame_expanded/series.csv
  (+ README.md with sha256). VERIFY the sha256 against crn.json's
  "frame_expanded" block before using it; abort loudly on mismatch.
- Holdout = date > 2024-12-31 (expect n=1217). Train = the rest. Never fit
  anything — β, weights, coefficients, hyperparameters — on holdout rows.
- Engine on the expanded corpus: `from engine import Engine; eng = Engine()`
  now loads the expanded game set automatically (registry + dates landed).
  Then REPLACE the series frame before running:
      eng.series = frame_df.reset_index(drop=True)
      eng.pred_days = sorted(frame_df.date.unique())
  and call eng.run(cfg) as usual (run() rebuilds its per-day index from
  self.series). Do not use harness.load_series() for evaluation — it reads
  the site's stale timeline files.
- engine.py's champions flag is now exact-shape ("YYYY_champions") — off-season
  ids with "champions"/"masters" substrings are NOT internationals. harness
  ._is_intl_event was fixed the same way; the frame's `intl` column is already
  correct. Do not reintroduce substring tests.
- match_results.csv carries a MapNum=="all" aggregate row per match (series
  score). Per-map consumers MUST filter it or they double-count.

## Judging
- Judge with testing_lab/v8/referee.py: paired_bootstrap_crn (iid + block),
  bucketed, per_team_bias (probability-points ×100 — NOT rating points),
  pnl_weighted_ll/expected_roi_of_dll for the ROI translation (REPORTING unit
  only, never selection). crn.json governs all resampling; no private seeds.
- Quote the applicable pair-MDE next to every ΔLL (within-family 1.77m,
  cross-family 5.89m at n=1217 — stats/power_mde_expanded.json). A |Δ| inside
  the noise floor is reported as INSIDE NOISE FLOOR, never as a win or a kill.
- Report every effect in BOTH units: milli-LL and expected ROI on the quoting
  surface (referee.expected_roi_of_dll).
- β is scale-bound: any solve-constant change ⇒ refit β on train only, per
  config, in your own scratch dir testing_lab/v8/scratch/<agent>/ (create it;
  never share engine instances across configs via disk).

## Process
- Preregister BEFORE running: testing_lab/v8/preregister.<agent>.md —
  mechanism, predicted sign, predicted effect size, falsifier, per experiment.
  Append outcomes AFTER, at the same resolution for failures as successes.
- Outputs: only your declared paths (stats/<agent>_*.json + your phase
  markdown + scratch). One writer per artifact. Journal to logs/<agent>.log.
- Local data only. No network. No writes to data/, scrapers/, trading_model/,
  VCTMM, or another agent's outputs.
- Useful local inputs: v8/data/lineup_features.csv + lineups.csv +
  modal5_by_org_date.csv (walk-forward, 100% coverage of the pre-expansion
  corpus; corpus additions may lack rows — check and top up from
  data/maps/<id>.csv with the same definitions if your tests need them,
  writing the top-up ONLY into your scratch dir), data/enriched/
  (round_outcomes.csv, player_map_advanced.csv — CN events have NO round/
  player enrichment; audit coverage before relying on it), data/map_vetos.csv.
- Fail loudly. No silent substitution of samples, metrics, or events.
- Return ≤500 words + artifact paths. No transcripts.
