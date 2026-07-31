# Pre-registration — agent:bias-h3 (written 2026-07-28, BEFORE any experiment run)

Scope: Phase 5 H3 state-space rating (Kalman analog, match probability integrates
over rating variance) + 5d heterogeneous half-lives via pooled process noise +
n_eff telemetry + ensemble-where-it-wins. Frame: canonical
`v8/data/frame_expanded/series.csv` (sha256 verified vs crn.json before use:
ff772d41…, n=2058, train 841 = date<=2024-12-31, holdout 1217). All fits train-only;
holdout touched only for final scoring. Randomness: GH quadrature is deterministic;
the single MC cross-check uses crn.json mc_seeds[0]; all bootstraps via
referee.paired_bootstrap_crn. MDEs quoted from stats/power_mde_expanded.json:
within-family 1.773m, cross-family 5.889m.

## Shared machinery (fixed before running)

- **Observation**: per MAP, y = sign(rd)·|rd|^0.75·2.5 (v6's rd transform,
  unchanged), winner-referenced to team A of the pair. One filter update per map,
  in (date_s, match_id) order (engine game order).
- **Filter**: per-team state (r, v). Init r=0, v=V0. Prediction step at each of a
  team's own maps: v += q (games-counted, no calendar accrual — in-family with
  v6's information-replacement decay). Update: e = y − (r_A − r_B),
  S = v_A + v_B + R_i, K_T = v_T/S, r_A += K_A·e, r_B −= K_B·e, v_T −= v_T²/S.
- **Game weights**: R_i = R/w_i with w_i = (2.0 if exact-shape YYYY_champions else
  1.0) × (1.6 if the map's series stage ∈ {playoffs, grand_final} else 1.0) —
  v6's hand-set weights, NOT refit. Stage from the frame; maps of matches not in
  the frame default to weight 1.0 (mirrors run_v7_stage1 stage_by_mid default).
- **Identification**: R fixed := Var(y) over TRAIN games (date<=2024-12-31).
  Fixing R loses no generality (only q/R, V0/R matter up to state scale; β absorbs
  scale). Free params of the core: q, V0. β refit train-only per config
  (bounds 0.03–0.6, minimize_scalar, engine-identical series NLL) — house rule 8.
- **Leak rule**: predictions for day D use the state snapshot after all games with
  date < D (strict day granularity, same as engine m_hist). Same-day games are
  never history for that day's predictions.
- **Match probability**: p_series = ∫ series_wp(sigmoid(β·δ), fmt) N(δ; r_A−r_B,
  v_A+v_B) dδ by 20-node Gauss–Hermite (the SAME δ draw shared across maps of a
  series — rating uncertainty is common across the series, not iid per map).
  bo5_gf uses the plain bo5 closed form (frame carries no bracket side; identical
  treatment for baseline and candidate, so pair deltas are unaffected).
- **Baseline**: v6 = engine cfg {rd 0.75/2.5, roster year/0.3, ridge 0.5,
  champ 2.0, region_prior_ridge 1.5, PO 1.6 w_custom, decay consist(20,12)} run
  on the expanded frame, β refit on train. Judged with referee.py.
- **Fit criterion for (q, V0) and all 5d params**: mean series NLL on TRAIN rows
  only, β refit at every grid point. Log-space grid q/R ∈ [1e-4, 3e-2],
  V0/R ∈ [0.05, 3], coarse 2D grid then one local refinement; every grid point
  journaled to scratch/bias_h3/sweep_*.json (checkpoint; restart skips finished
  points).

## Experiment 1 — state-space core (SS-core)

- Mechanism: point-estimate ratings ignore per-team uncertainty; integrating
  sigmoid over N(Δr, v_A+v_B) shrinks exactly the thin-data predictions toward
  0.5 while leaving established teams sharp — the asymmetry a global β cannot
  express.
- Predicted sign/size: aggregate holdout ΔLL vs v6 positive but small —
  predicted +0 to +3m, i.e. likely INSIDE the cross-family noise floor (5.889m);
  the mechanism should show in the tails: cold-start bucket (old-frame reference
  n=57 / LL 0.70205) improves by ≥10m; per-team bias caterpillar shows reduced
  max|bias| with thin-data teams' bias shrinking and elite teams' bias not
  inflating; favorite-band buckets not degraded.
- Falsifier: cold-start bucket NOT improved, or improved only by shrinking
  everyone (elite/big-gap buckets degrade by more than their bucket noise), or
  max|team bias| increases.
- Variants (each preregistered, β refit, same grid discipline; all train-fit):
  1a. plain core (q, V0) as above.
  1b. + calendar leak: v += q_cal per week elapsed since the team's previous
      game (one extra param; tests whether breaks add real uncertainty).
      Predicted: small train gain, holdout ≈ tie; falsifier: hurts train.
  1c. + debut prior mean: a team's r initialized at its region's trailing mean
      rating of already-rated teams (walk-forward, day before debut) instead
      of 0 (v6 has region_prior_ridge; this is the SS analog). Predicted: helps
      cold-start bucket further; falsifier: no cold-start change.
  Model advanced to judging = best TRAIN NLL among {1a, 1b, 1c}; all three's
  train numbers reported, holdout scored for all three (labelled primary =
  train-selected BEFORE holdout is looked at).
- Judging (referee.py, exactly): holdout LL both units (milli-LL +
  expected_roi_of_dll vs v6 p_ref), paired_bootstrap_crn iid AND block_event,
  cross-family MDE 5.889m quoted, per_team_bias caterpillar (candidate vs v6,
  min_n=25), max|bias|, bucketed() panel incl. cold-start (r_w/r_l attached from
  the v6 engine run; COLD_EPS 5e-4) and EWC-class, favorite bands.
- GH validity check: |GH20 − MC(200k, crn mc_seeds[0])| < 1e-4 on 50 spot
  matches, reported in the stats JSON.

## Experiment 2 — 5d heterogeneous half-lives via pooled process noise

- Mechanism: process noise q IS the memory dial (steady-state gain ⇒ effective
  per-game forgetting (1−K*); HL_games = ln2/−ln(1−K*)). Team types should
  need different q: rebuilt rosters carry more true rating drift than stable
  elites.
- Typing axes (walk-forward observables, all computable at update time; cells
  fixed now):
  - A. roster stability (PRIMARY): at each map of team T, matches_since_change
    (consecutive prior matches with lineup identical to the current one, the
    lineups-agent definition) — cells: change-adjacent msc≤3 / settling
    4≤msc≤10 / stable msc>10. Source: full-corpus recompute of
    lineups/lineup_features into scratch/bias_h3/ (corpus additions lack rows;
    top-up mandated scratch-only). Verification, two modes (pre-run amendment,
    2026-07-28, written before the recompute ran): (i) IMPLEMENTATION check —
    recompute restricted to the lineups agent's own event universe
    (set(lineups.csv.event_id)) must reproduce their matches_since_change /
    games_since_change on ALL overlapping (match, org) rows with 0 mismatches,
    else stop and report; (ii) the full-corpus version used for typing may
    legitimately differ on overlap rows where corpus-addition matches
    interleave an org's history — count and report those rows, never silently.
  - B. org age: days since org's first corpus game at update time — cells
    <180d / 180–540d / >540d. Left-censoring at corpus start (2023) noted as a
    caveat in the output.
  - C. rating volatility: trailing std of the last 12 standardized innovations
    z = e/√S from the FITTED SS-core (train-fit params; walk-forward by
    construction) — cells: train-terciles of the pooled z-std distribution.
- Fit, per axis: q_k free per cell (3 cells), V0 and R and β as in core
  (V0 re-fit jointly? NO — V0 frozen at SS-core's fitted value to keep the
  comparison within-family and the parameter count honest; stated here, before
  fitting). Per-cell MLE q̂_k + curvature/profile SE on TRAIN.
- Partial pooling (the deliverable): random-effects on log q across cells:
  DerSimonian–Laird τ̂² from the K=3 per-axis MLEs and their SEs; pooled
  q̃_k = precision-weighted shrink of log q̂_k toward log q̄. No free-per-team q
  anywhere. Report per cell: q̂, q̃ (pooled), implied HL in games
  (HL = ln2/−ln(1−K*) at R̄=R, w=1), 95% CI from the train profile mapped
  through the HL formula, n_games per cell, τ̂², and the shrink fraction.
- Predicted sign/size: q̂(change-adjacent) > q̂(stable) by ≥2× (HL shorter for
  rebuilt rosters); q̂(young org) > q̂(established); q̂(high vol) > q̂(low vol) —
  the operator's "more recency for some teams" quantified. Holdout ΔLL of
  het-q vs SS-core: predicted +0 to +2m (within-family MDE 1.773m — likely
  INSIDE NOISE FLOOR; the half-life table is the deliverable either way).
- Falsifier: ordering flat or reversed (q̂ ratio < 1.3× between extreme cells),
  or τ̂² ≈ 0 with all cells shrinking to the pooled q (then heterogeneity is
  unsupported at this n — published as such).
- **stats/h3_process_noise.json published as soon as these fits exist** (before
  the rest of the judging suite completes; agent:decay references it).
- Holdout scoring: axis A model (primary), B and C reported as secondary;
  within-family MDE 1.773m quoted on every Δ.

## Experiment 3 — n_eff per prediction

- Emit per frame row (all 2058, holdout flagged): pre-match v_w, v_l,
  σ_Δ = √(v_w+v_l), n_eff_T = R/v_T, harmonic pair n_eff, p_point (β·Δr point
  sigmoid + series form), p_integrated, shrink factor
  (|p_int−0.5|/|p_point−0.5|, 1.0 when p_point=0.5). From the train-selected
  SS-core config. Distribution summaries: quantiles overall / by year / by
  event class (vct, ewc-offseason via EWC_CLASS_PREFIXES + pre-2026 analogs,
  intl) / cold-start rows; correlation of pair n_eff with per-row |ΔLL vs v6|.
  No selection decisions ride on this — telemetry deliverable.

## Experiment 4 — ensemble-where-it-wins

- Question: does SS win only on subsets? Candidate gates (fixed list, chosen on
  TRAIN only): (a) either-team n_eff < θ, θ ∈ {3, 5, 8, 12}; (b) either team
  change-adjacent (msc≤3 at last lineup before the match); (c) event class
  ewc_offseason; (d) soft blend w(x) = σ(a + b·min_neff) fit on train (2 params).
- Selection rule (fixed now): the gate/blend with best TRAIN mean NLL of the
  composite (SS inside gate, v6 outside) wins; ONE composite is then scored
  ONCE on holdout with full referee judging vs v6 AND vs SS-core. Predicted:
  composite beats v6 by more than SS-core does, driven by cold-start/EWC rows.
  Falsifier: no gate's TRAIN composite beats v6's train NLL → "no subset where
  SS pays for itself"; reported plainly, holdout still scored for the
  pre-named primary gate (a, θ=5) for completeness, labelled exploratory.
- MDE context: composite vs v6 is cross-family (5.889m); composite vs SS-core
  within-family (1.773m).

## Outputs (mine alone)

stats/h3_statespace.json, stats/h3_process_noise.json (EARLY),
stats/h3_neff.json, stats/h3_bias_caterpillar.json, stats/h3_ensemble.json,
phase5_h3.md, logs/bias_h3.log, scratch/bias_h3/. Outcomes appended below
AFTER runs, same resolution for failures as successes.

## Not done here

No holdout fitting of anything. No writes outside declared paths. No market
data anywhere in fitting or selection (README rule 9). No network.

---

# OUTCOMES (appended after runs, 2026-07-28; same resolution for failures)

**Exp 1 (SS core).** Predicted +0..+3m vs v6 ⇒ WRONG on aggregate: plain core
−8.87m [LOSS, cross-MDE 5.889m], train-selected primary 1b (+q_cal) −11.75m
[LOSS] — 1b's train win (0.64485, best) was regime overfit; falsifier partially
tripped: cold-start DID improve massively (below) but not "without cost
elsewhere": established buckets degrade 2-12m, and mean|bias| worsens
0.0473→0.052 while max|bias| improves 0.1478→0.1255. Mechanism-in-tails
CONFIRMED: debut +58m, <10 maps +71.7m (n=39), <30 maps +28m (n=117, 5d model);
thin-half sharpness −0.61pp vs established +0.10pp, corr +0.40. GH validation
passed (GH20≡GH40 machine epsilon; MC within 3SE). COLD_EPS bucket empty on
engine ratings (region prior) — prior-map-count definitions substituted,
documented in-artifact.

**Exp 2 (5d).** Roster axis: predicted q(change)>q(stable) ≥2× ⇒ CONFIRMED in
point estimates (11×; HL 7.4 vs 24.8 games; settling cell at q→0 boundary =
non-monotone, not predicted). DL pooling: τ̂²=0 all axes ⇒ the preregistered
"unsupported at this n" clause is the formal verdict; published both. Holdout:
5d vs plain core +3.55m [within-family WIN, MDE 1.773m, p_better 0.946/0.979];
5d vs v6 −5.32m [INSIDE NOISE FLOOR]. Org-age: REVERSED (falsified;
left-censoring caveat as preregistered). Volatility: flat (falsified).
h3_process_noise.json published before the judging suite, as required.

**Exp 3 (n_eff).** Emitted for all 2058 rows from the train-selected core (1b)
+ 5d secondary summaries. corr(pair n_eff, |ΔLL vs v6|) = −0.09.

**Exp 4 (ensemble).** Several composites beat v6 on train (so the "no subset
pays" falsifier did NOT trip); train-selected winner = soft n_eff blend on 1b;
holdout: −12.15m vs v6 [LOSS], −3.28m vs plain core [LOSS] ⇒ the preregistered
selection produced a non-generalizing composite; published as-is, no holdout
shopping for the non-selected hard gates (their subset behavior is already
visible in the preregistered cold buckets). Promotion answer: HOLD.

Deviations from plan: (1) cold-start bucket definition substituted
(COLD_EPS empty on engine ratings — documented in h3_statespace.json);
(2) GH-vs-MC bar restated as GH20-vs-GH40 convergence (original 1e-4 bar was
tighter than MC noise at 200k; both checks pass and are recorded);
(3) exp-4 search space explicitized as gates × {1b, 5d} with selection still
purely on train (documented in h3_ensemble.json). No holdout was used for any
selection anywhere.
