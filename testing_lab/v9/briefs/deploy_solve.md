# agent:deploy-solve — v6 becomes the SITE's rating engine (operator order)

OPERATOR DIRECTIVE 2026-07-30: "deploy v6 to BenPom in BoboGG on all fronts."
This explicitly reverses the old "v6 stays private" decision. You do stage 1:
the rating solve + the site model snapshot. Stage 2 (prediction surfaces) is
a later agent — do not touch MapElo.py or BobosHome.py.

## The v6 spec (the exact champion; no interpretation)
Massey walk-forward, per prediction day, games strictly before the day:
- games-counted consistency decay: results consistent with the team's decayed
  map winrate (HL16) persist HL 20 games, anomalies HL 12 (per-side ages,
  combined sqrt(w_w·w_l))
- rd target sign(rd)·|rd|^0.75·2.5
- year-boundary roster continuity 0.3 (applied as sqrt(cw·cl) factor)
- ridge 0.5 + region-prior ridge 1.5 toward each region's PREVIOUS-day mean
- playoff/GF solve weight ×1.6; Champions ×2 (EXACT-SHAPE id match
  "YYYY_champions" — the guard already exists, keep it)
- Reference implementation: testing_lab/engine.py run() games-branch +
  trading_model/build_model_snapshot.py. Port it NATIVELY into
  scrapers/BuildRatingTimeline.py (site scrapers must not import testing_lab
  — the harness drags VCTMM sys.path). Vectorize like engine does.

## Work
1. **BuildRatingTimeline.py**: replace the production solve (calendar
   LAMBDA_DECAY massey + intl weights + CN v10 shrinkage pipeline) with the
   v6 solve above for ALL years. PRESERVE the output JSON schema exactly
   (checkpoints[{date, ratings{org:val}}], match_events[{...winner_before,
   winner_after, deltas...}], top-level keys) — every site page and the
   harness read it. Ratings scale changes (v6 scale); that is expected and
   fine. CN cluster shrinkage: GONE from the timeline (v6's region-prior
   ridge + prediction-layer offsets replace it — document in the file
   header). Keep per-year files. Incremental logic: simplify to full
   rebuild per year (engine-style vectorized solve is seconds); delete the
   stale-checkpoint reuse if it fights the new solve.
2. **Site model snapshot**: at the end of the build, emit
   data/site_model.json: {model_version: "benpom-v6-site-YYYY-MM-DD",
   generated_utc, beta (refit via minimize_scalar bounded 0.03-0.6 on ALL
   valid completed series to date, engine-parity math), xregion_offsets
   (Nelder-Mead fit, CN pinned 0, exactly build_model_snapshot's method),
   region_priors (25th pct per region), gf_upper_logit: 0.25, b_pick
   (picker-winrate logit from data/map_vetos.csv, build_model_snapshot's
   method), ratings_as_of}. This is the single source stage 2 wires every
   surface to.
3. **Parity gates (hard, in code, before writing outputs):**
   a. Solve parity vs testing_lab/engine.py on the frozen frame: same-day
      ratings for 20 sampled days across 2023-2026 must match engine's
      daily_r to ≤1e-9 (run engine in a subprocess/venv-safe way for the
      check only, or replicate its numbers from stored v9 scratch npz —
      document which).
   b. β sanity: refit lands within [0.115, 0.145] (v9 protocol refit was
      0.128512 on data through 07/28); outside ⇒ investigate, don't ship.
4. **Rebuild** all rating_timeline files + site_model.json. Do NOT rebuild
   map_ratings/veto (unchanged products). Do NOT restart Flask (stage 2).
5. Update file-header docs: BuildRatingTimeline now BUILDS THE PUBLIC v6.
   Note the operator decision + date.

## Outputs (yours alone)
- scrapers/BuildRatingTimeline.py (rewritten solve), data/rating_timeline*.json
  (rebuilt), data/site_model.json (new)
- testing_lab/v9/stats/deploy_solve_parity.json (both gates' evidence)
- logs at testing_lab/v9/logs/deploy_solve.log
Forbidden: MapElo.py, BobosHome.py, BuildMapRatings.py solve internals,
testing_lab imports into scrapers, git.

Return ≤300 words: parity results, new β + offsets + priors, leaderboard
top-5 before/after (from the JSONs), artifact paths.
